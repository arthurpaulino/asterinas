// SPDX-License-Identifier: MPL-2.0

use alloc::{collections::BinaryHeap, sync::Arc};
use core::{
    cmp::{self, Reverse},
    sync::atomic::{AtomicU64, Ordering},
};

use ostd::{
    cpu::{num_cpus, CpuId},
    task::{
        scheduler::{EnqueueFlags, UpdateFlags},
        Task,
    },
};

use super::{
    time::{base_slice_clocks, min_period_clocks},
    CurrentRuntime, SchedAttr, SchedClassRq,
};
use crate::{
    sched::nice::{Nice, NiceValue},
    thread::AsThread,
};

const WEIGHT_0: u64 = 1024;
const HAS_PENDING: u64 = 1 << (u64::BITS - 1);

pub const fn nice_to_weight(nice: Nice) -> u64 {
    // weight = 1024 * 1.25^(-nice)
    const FACTOR_NUMERATOR: u64 = 5;
    const FACTOR_DENOMINATOR: u64 = 4;
    const NICE_TO_WEIGHT: [u64; 40] = const {
        let mut ret = [0; 40];
        let mut i = 0;
        let mut n = NiceValue::MIN.get();
        while n <= NiceValue::MAX.get() {
            ret[i] = match n {
                0 => WEIGHT_0,
                p @ 1.. => {
                    let num = FACTOR_DENOMINATOR.pow(p as u32);
                    let den = FACTOR_NUMERATOR.pow(p as u32);
                    WEIGHT_0 * num / den
                }
                neg => {
                    let num = FACTOR_NUMERATOR.pow((-neg) as u32);
                    let den = FACTOR_DENOMINATOR.pow((-neg) as u32);
                    WEIGHT_0 * num / den
                }
            };
            assert!(ret[i] & HAS_PENDING == 0);
            i += 1;
            n += 1;
        }
        ret
    };
    NICE_TO_WEIGHT[(nice.value().get() + 20) as usize]
}

/// Per-entity attributes for FAIR/EEVDF.
///
/// We keep existing fields and add EEVDF bookkeeping:
/// - `vstart`: entity’s eligibility time in virtual time
/// - `deadline`: entity’s virtual finish (vfinish)
#[derive(Debug)]
pub struct FairAttr {
    weight: AtomicU64,
    pending_weight: AtomicU64,
    vruntime: AtomicU64,

    // --- EEVDF bookkeeping (internal; does not change external API) ---
    vstart: AtomicU64,
    deadline: AtomicU64,
}

impl FairAttr {
    pub fn new(nice: Nice) -> Self {
        FairAttr {
            weight: nice_to_weight(nice).into(),
            pending_weight: Default::default(),
            vruntime: Default::default(),
            vstart: Default::default(),
            deadline: Default::default(),
        }
    }

    pub fn update(&self, nice: Nice) {
        self.pending_weight
            .store(nice_to_weight(nice), Ordering::Relaxed);
        self.weight.fetch_or(HAS_PENDING, Ordering::Release);
    }

    #[inline]
    fn update_vruntime(&self, delta_sched_clocks: u64, weight: u64) -> u64 {
        // vr += Δt * (WEIGHT_0 / weight)
        let delta_vr = delta_sched_clocks * WEIGHT_0 / weight;
        self.vruntime.fetch_add(delta_vr, Ordering::Relaxed) + delta_vr
    }

    /// Fetch (old,current) weight, applying pending update if needed.
    fn fetch_weight(&self) -> (u64, u64) {
        let mut weight = self.weight.load(Ordering::Acquire);
        if weight & HAS_PENDING == 0 {
            return (weight, weight);
        }
        let mut new_weight = self.pending_weight.load(Ordering::Relaxed);
        loop {
            match self.weight.compare_exchange_weak(
                weight,
                new_weight,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(failure) => {
                    if failure & HAS_PENDING == 0 {
                        return (failure, failure);
                    }
                    weight = failure;
                    new_weight = self.pending_weight.load(Ordering::Relaxed);
                }
            }
        }
        let old_weight = weight & !HAS_PENDING;
        (old_weight, new_weight)
    }
}

/// Items in the **eligible** heap: ordered by **deadline (vfinish)**.
struct EligibleItem {
    entity: Arc<Task>,
    deadline: u64,
    // keep vstart for optional diagnostics; not used for ordering
    vstart: u64,
}

impl core::fmt::Debug for EligibleItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EligibleItem")
            .field("deadline", &self.deadline)
            .field("vstart", &self.vstart)
            .finish()
    }
}

impl EligibleItem {
    #[inline]
    fn key(&self) -> u64 {
        self.deadline
    }
}

impl PartialEq for EligibleItem {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}
impl Eq for EligibleItem {}
impl PartialOrd for EligibleItem {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for EligibleItem {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // earlier deadline = "smaller" = higher priority; heap is Reverse<>
        self.key().cmp(&other.key())
    }
}

/// Items in the **waiting** heap: ordered by **vstart (eligibility time)**.
struct WaitingItem {
    entity: Arc<Task>,
    vstart: u64,
}

impl core::fmt::Debug for WaitingItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WaitingItem")
            .field("vstart", &self.vstart)
            .finish()
    }
}

impl WaitingItem {
    #[inline]
    fn key(&self) -> u64 {
        self.vstart
    }
}
impl PartialEq for WaitingItem {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}
impl Eq for WaitingItem {}
impl PartialOrd for WaitingItem {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for WaitingItem {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

/// EEVDF per-CPU runqueue.
///
/// Internally maintains:
/// - `eligible`: min-heap by **deadline (vfinish)**
/// - `waiting`:  min-heap by **vstart** (becomes eligible when vstart <= min_vruntime)
#[derive(Debug)]
pub(super) struct FairClassRq {
    #[expect(unused)]
    cpu: CpuId,

    eligible: BinaryHeap<Reverse<EligibleItem>>,
    waiting: BinaryHeap<Reverse<WaitingItem>>,

    /// Global virtual time (monotonic, never decreases).
    min_vruntime: u64,

    /// Sum of weights of **queued** entities (excludes the current running one).
    total_weight: u64,

    /// Cached info for the current running entity on this CPU.
    /// We don't need the entity handle here; just its last computed deadline.
    current_deadline: u64,
    current_vstart: u64,
}

impl FairClassRq {
    pub fn new(cpu: CpuId) -> Self {
        Self {
            cpu,
            eligible: BinaryHeap::new(),
            waiting: BinaryHeap::new(),
            min_vruntime: 0,
            total_weight: 0,
            current_deadline: 0,
            current_vstart: 0,
        }
    }

    /// Period (sched clocks). Same interface as before; we keep your design.
    fn period(&self) -> u64 {
        let base_slice_clks = base_slice_clocks();
        let min_period_clks = min_period_clocks();

        let period_single_cpu =
            (base_slice_clks * (self.eligible.len() + self.waiting.len() + 1) as u64)
                .max(min_period_clks);

        period_single_cpu * u64::from((1 + num_cpus()).ilog2())
    }

    /// Convert a *wall-clock* time slice to a **vruntime** delta for a given weight.
    #[inline]
    fn wall_to_vruntime(ts: u64, weight: u64) -> u64 {
        ts * WEIGHT_0 / weight
    }

    /// Target wall-clock slice for an entity with `cur_weight`.
    fn time_slice(&self, cur_weight: u64) -> u64 {
        let total = self.total_weight + cur_weight; // include current when running
        if total == 0 {
            return self.period(); // degenerate; shouldn't happen, but safe
        }
        self.period() * cur_weight / total
    }

    /// Move tasks from waiting → eligible when `vstart <= min_vruntime`.
    fn drain_eligible(&mut self) {
        while let Some(Reverse(head)) = self.waiting.peek() {
            if head.vstart > self.min_vruntime {
                break;
            }
            let Reverse(WaitingItem { entity, vstart }) = self.waiting.pop().unwrap();

            // Compute deadline = vstart + vruntime_slice(weight)
            let fair = &entity.as_thread().unwrap().sched_attr().fair;
            let (_old_w, w) = fair.fetch_weight();
            let vr_slice = Self::wall_to_vruntime(self.time_slice(w), w);
            let deadline = vstart.saturating_add(vr_slice);

            fair.vstart.store(vstart, Ordering::Relaxed);
            fair.deadline.store(deadline, Ordering::Relaxed);

            self.eligible.push(Reverse(EligibleItem {
                entity,
                deadline,
                vstart,
            }));
        }
    }

    /// Recompute the current entity's EEVDF window (vstart, deadline).
    fn refresh_current_window(&mut self, attr: &FairAttr) {
        let (_, w) = attr.fetch_weight();
        let vr = attr.vruntime.load(Ordering::Relaxed);
        let vstart = vr.max(self.min_vruntime);
        let vr_slice = Self::wall_to_vruntime(self.time_slice(w), w);
        let deadline = vstart.saturating_add(vr_slice);

        self.current_vstart = vstart;
        self.current_deadline = deadline;

        attr.vstart.store(vstart, Ordering::Relaxed);
        attr.deadline.store(deadline, Ordering::Relaxed);
    }

    /// Helper: minimum vstart among queued tasks (waiting head) or current vruntime.
    fn recompute_min_vruntime_with_current(&mut self, current_vr: u64) {
        // Linux keeps min_vruntime monotonic and near the min of all entities.
        // We approximate: min(current_vr, waiting_head.vstart) and never decrease.
        let next_min = if let Some(Reverse(w)) = self.waiting.peek() {
            current_vr.min(w.vstart)
        } else {
            current_vr
        };
        if next_min > self.min_vruntime {
            self.min_vruntime = next_min;
        }
    }
}

impl SchedClassRq for FairClassRq {
    fn enqueue(&mut self, entity: Arc<Task>, flags: Option<EnqueueFlags>) {
        let fair = &entity.as_thread().unwrap().sched_attr().fair;
        let (_old_w, w) = fair.fetch_weight();

        // Compute vstart from the entity’s current vruntime and rq.min_vruntime.
        let vr = fair.vruntime.load(Ordering::Relaxed);
        // New entities (Spawn) often get slight boost in CFS; here we stay canonical EEVDF:
        // vstart = max(vr, min_vruntime). (You can bias Spawn by adding a small vruntime slice.)
        let vstart_base = vr.max(self.min_vruntime);

        // Decide whether it's immediately eligible.
        // For Wake/Spawn we consider it eligible iff vstart <= min_vruntime.
        // (That is true by construction; we keep the branch for clarity and future tweaks.)
        let vstart = match flags {
            Some(EnqueueFlags::Spawn) | Some(EnqueueFlags::Wake) | _ => vstart_base,
        };

        // It contributes its weight to the queued total immediately.
        self.total_weight += w;

        if vstart <= self.min_vruntime {
            // Eligible now: compute deadline and push to eligible heap.
            let vr_slice = Self::wall_to_vruntime(self.time_slice(w), w);
            let deadline = vstart.saturating_add(vr_slice);

            fair.vstart.store(vstart, Ordering::Relaxed);
            fair.deadline.store(deadline, Ordering::Relaxed);

            self.eligible.push(Reverse(EligibleItem {
                entity,
                deadline,
                vstart,
            }));
        } else {
            // Not yet eligible: goes to waiting heap keyed by vstart.
            fair.vstart.store(vstart, Ordering::Relaxed);
            self.waiting.push(Reverse(WaitingItem { entity, vstart }));
        }

        // Some tasks might become eligible after this enqueue (if min_vruntime advanced).
        self.drain_eligible();
    }

    fn len(&self) -> usize {
        self.eligible.len() + self.waiting.len()
    }

    fn is_empty(&self) -> bool {
        self.eligible.is_empty() && self.waiting.is_empty()
    }

    fn pick_next(&mut self) -> Option<Arc<Task>> {
        // Ensure all newly eligible tasks are visible.
        self.drain_eligible();

        let Reverse(EligibleItem { entity, .. }) = self.eligible.pop()?;

        // Remove its weight from queued sum; it becomes the running entity.
        let sched_attr = entity.as_thread().unwrap().sched_attr();
        let (old_w, _w_now) = sched_attr.fair.fetch_weight();
        self.total_weight = self.total_weight.saturating_sub(old_w);

        // Set current EEVDF window for the running entity (used for preemption tests).
        self.refresh_current_window(&sched_attr.fair);

        Some(entity)
    }

    fn update_current(
        &mut self,
        rt: &CurrentRuntime,
        attr: &SchedAttr,
        flags: UpdateFlags,
    ) -> bool {
        match flags {
            UpdateFlags::Tick | UpdateFlags::Yield | UpdateFlags::Wait => {
                // Advance current entity’s vruntime in proportion to service.
                let (_old_w, w) = attr.fair.fetch_weight();
                let vr = attr.fair.update_vruntime(rt.delta, w);

                // Keep rq->min_vruntime monotonic and near the true minimum.
                self.recompute_min_vruntime_with_current(vr);

                // Some waiting tasks may have become eligible as min_vruntime advanced.
                self.drain_eligible();

                // Recompute current entity’s own EEVDF window.
                self.refresh_current_window(&attr.fair);

                // Preemption tests:
                // 1) If the entity is going to sleep (WAIT), we must switch.
                if matches!(flags, UpdateFlags::Wait) {
                    return !self.is_empty();
                }

                // 2) If the entity consumed its (wall clock) slice, switch.
                let time_exhausted = rt.period_delta > self.time_slice(w);

                // 3) EEVDF rule: if there exists an eligible task with an **earlier deadline**,
                //    it should preempt the current entity.
                let earlier_deadline_exists = if let Some(Reverse(head)) = self.eligible.peek() {
                    head.deadline < self.current_deadline
                } else {
                    false
                };

                // 4) If the current voluntarily yields, also switch (but we still update vruntime above).
                let voluntary_yield = matches!(flags, UpdateFlags::Yield);

                time_exhausted || earlier_deadline_exists || voluntary_yield
            }
            UpdateFlags::Exit => !self.is_empty(),
        }
    }
}
