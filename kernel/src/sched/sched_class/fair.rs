// SPDX-License-Identifier: MPL-2.0

use alloc::{boxed::Box, sync::Arc};
use core::{
    cmp, mem,
    sync::atomic::{AtomicI64, AtomicU64, Ordering},
    u64,
};

use ostd::{
    cpu::{num_cpus, CpuId},
    task::{
        scheduler::{EnqueueFlags, UpdateFlags},
        Task,
    },
};

use crate::{
    sched::{
        nice::{Nice, NiceValue},
        sched_class::{
            time::{base_slice_clocks, tick_period_clocks},
            CurrentRuntime, SchedAttr, SchedClassRq,
        },
    },
    thread::AsThread,
};

const WEIGHT_0: i64 = 1024;

pub const fn nice_to_weight(nice: Nice) -> i64 {
    // Calculated by the formula below:
    //
    //     weight = 1024 * 1.25^(-nice)
    //
    // We propose that every increment of the nice value results
    // in 12.5% change of the CPU load weight.
    const FACTOR_NUMERATOR: i64 = 5;
    const FACTOR_DENOMINATOR: i64 = 4;

    const NICE_TO_WEIGHT: [i64; 40] = const {
        let mut ret = [0; 40];

        let mut index = 0;
        let mut nice = NiceValue::MIN.get();
        while nice <= NiceValue::MAX.get() {
            ret[index] = match nice {
                0 => WEIGHT_0,
                nice @ 1.. => {
                    let numerator = FACTOR_DENOMINATOR.pow(nice as u32);
                    let denominator = FACTOR_NUMERATOR.pow(nice as u32);
                    WEIGHT_0 * numerator / denominator
                }
                nice => {
                    let numerator = FACTOR_NUMERATOR.pow((-nice) as u32);
                    let denominator = FACTOR_DENOMINATOR.pow((-nice) as u32);
                    WEIGHT_0 * numerator / denominator
                }
            };

            index += 1;
            nice += 1;
        }
        ret
    };

    NICE_TO_WEIGHT[(nice.value().get() + 20) as usize]
}

fn wall_to_virtual(delta: i64, weight: i64) -> i64 {
    if weight != WEIGHT_0 {
        delta * WEIGHT_0 / weight
        // TODO: set as cold path.
    } else {
        delta // Avoid unnecessary math most of the times.
    }
}

fn avg_vruntime(mut weighted_vruntime_offsets: i64, total_weight: i64, min_vruntime: i64) -> i64 {
    if weighted_vruntime_offsets < 0 {
        // Sign flips effective floor/ceiling.
        weighted_vruntime_offsets -= total_weight - 1;
    }
    weighted_vruntime_offsets / total_weight + min_vruntime
}

fn is_eligible(
    vruntime: i64,
    min_vruntime: i64,
    total_weight: i64,
    weighted_vruntime_offsets: i64,
) -> bool {
    (vruntime - min_vruntime) * total_weight <= weighted_vruntime_offsets
}

#[derive(Debug)]
pub struct FairAttr {
    weight: AtomicI64,
    vlag: AtomicI64,
    id: AtomicU64,
}

impl FairAttr {
    pub fn new(nice: Nice) -> Self {
        FairAttr {
            weight: nice_to_weight(nice).into(),
            vlag: AtomicI64::new(0),
            id: AtomicU64::new(u64::MAX),
        }
    }

    pub fn update(&self, nice: Nice) {
        self.weight.store(nice_to_weight(nice), Ordering::Release);
    }
}

#[derive(Debug)]
pub(super) struct FairClassRq {
    #[expect(unused)]
    cpu: CpuId,
    queue: EligibilityTree,
    queue_len: usize,
    queued_weight: i64,
    weighted_vruntime_offsets: i64,
    current_task_data: Option<TaskData>,
    next_id: u64,
    base_slice_clocks: i64,
    lag_limit_clocks: i64,
}

#[derive(Debug)]
struct TaskData {
    task: Arc<Task>,
    id: u64,
    deadline: i64,
    weight: i64,
    vruntime: i64,
    is_exiting: bool,
}

impl FairClassRq {
    pub fn new(cpu: CpuId) -> Self {
        let base_slice_clocks = base_slice_clocks() as i64;
        let lag_limit_clocks = (tick_period_clocks() as i64).max(2 * base_slice_clocks);
        Self {
            cpu,
            queue: EligibilityTree::new(),
            queue_len: 0,
            queued_weight: 0,
            weighted_vruntime_offsets: 0,
            current_task_data: None,
            next_id: 0,
            base_slice_clocks,
            lag_limit_clocks,
        }
    }

    fn min_vruntime(&self) -> i64 {
        match (&self.current_task_data, self.queue.min_vruntime()) {
            (None, None) => 0,
            (None, Some(x)) => x,
            (Some(current_task_data), None) => current_task_data.vruntime,
            (Some(current_task_data), Some(y)) => current_task_data.vruntime.min(y),
        }
    }

    fn total_weight(&self) -> i64 {
        match &self.current_task_data {
            Some(current_task_data) => current_task_data.weight + self.queued_weight,
            None => self.queued_weight,
        }
    }

    fn slice(&self) -> i64 {
        self.base_slice_clocks * (num_cpus().ilog2() as i64 + 1)
    }
}

impl SchedClassRq for FairClassRq {
    fn enqueue(&mut self, task: Arc<Task>, flags: Option<EnqueueFlags>) {
        // Φ' = ∑{i ∈ S∪⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ')]
        //    = ∑{i ∈ S}[wᵢ(ρᵢ - ρₘᵢₙ')] + wₜ(ρₜ - ρₘᵢₙ')
        //    = ∑{i ∈ S}[wᵢ(ρᵢ - ρₘᵢₙ)] - ∑{i ∈ S}[wᵢ(ρₘᵢₙ' - ρₘᵢₙ)] + wₜ(ρₜ - ρₘᵢₙ')
        //    = Φ + W(ρₘᵢₙ - ρₘᵢₙ') + wₜ(ρₜ - ρₘᵢₙ')

        let fair_attr = &task.as_thread().unwrap().sched_attr().fair;

        let weight = fair_attr.weight.load(Ordering::Relaxed);
        let mut vslice = wall_to_virtual(self.slice(), weight);

        let (id, vlag) = match flags {
            Some(EnqueueFlags::Spawn) => {
                // Define the ID for newly spawned tasks.
                let id = self.next_id;
                self.next_id += 1;
                fair_attr.id.store(id, Ordering::Relaxed);

                // Spawned tasks don't have lag.
                let vlag = 0;

                // When joining the competition; the existing tasks will be,
                // on average, halfway through their slice, as such start tasks
                // off with half a slice to ease into the competition.
                // Reference: https://elixir.bootlin.com/linux/v6.16.8/source/kernel/sched/fair.c#L5300
                vslice /= 2;

                (id, vlag)
            }
            _ => {
                // Load the already defined ID.
                let id = fair_attr.id.load(Ordering::Relaxed);
                debug_assert_ne!(id, u64::MAX);

                // Load the stored virtual lag.
                let vlag = fair_attr.vlag.load(Ordering::Relaxed);

                (id, vlag)
            }
        };

        let min_vruntime = self.min_vruntime();
        let total_weight = self.total_weight();
        let vruntime = if total_weight != 0 {
            let avg_vruntime =
                avg_vruntime(self.weighted_vruntime_offsets, total_weight, min_vruntime);
            if vlag != 0 {
                // If we want to place a task and preserve lag, we have to
                // consider the effect of the new entity on the weighted
                // average and compensate for this, otherwise lag can quickly
                // evaporate.
                // Reference: https://elixir.bootlin.com/linux/v6.16.8/source/kernel/sched/fair.c#L5230
                let vlag_adjusted = (total_weight + weight) * vlag / total_weight;
                avg_vruntime - vlag_adjusted
            } else {
                avg_vruntime
            }
        } else {
            self.weighted_vruntime_offsets + min_vruntime
        };

        if vruntime < min_vruntime {
            // ρₜ = ρₘᵢₙ' => Φ' = Φ + W(ρₘᵢₙ - ρₘᵢₙ')
            self.weighted_vruntime_offsets += total_weight * (min_vruntime - vruntime);
        } else {
            // ρₘᵢₙ' = ρₘᵢₙ => Φ' = Φ + wₜ(ρₜ - ρₘᵢₙ')
            self.weighted_vruntime_offsets += weight * (vruntime - min_vruntime);
        }

        let deadline = vruntime + vslice;
        self.queue.insert(TaskData {
            task,
            deadline,
            id,
            weight,
            vruntime,
            is_exiting: false,
        });

        self.queue_len += 1;
        self.queued_weight += weight;
    }

    fn pick_next(&mut self) -> Option<Arc<Task>> {
        // Φ' = ∑{i ∈ S\⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ')]
        //    = ∑{i ∈ S\⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ)] - ∑{i ∈ S\⦃t⦄}[wᵢ(ρₘᵢₙ' - ρₘᵢₙ)]
        //    = Φ - wₜ(ρₜ - ρₘᵢₙ) - W'(ρₘᵢₙ' - ρₘᵢₙ)

        let min_vruntime = self.min_vruntime();
        let total_weight = self.total_weight();
        let TaskData {
            task,
            id,
            deadline,
            weight,
            vruntime,
            is_exiting,
        } = self
            .queue
            .pop_next(min_vruntime, total_weight, self.weighted_vruntime_offsets)?;

        if let Some(current_task_data) = &self.current_task_data {
            let TaskData {
                task: preempted_task,
                vruntime: preempted_vruntime,
                weight: preempted_weight,
                is_exiting: preempted_is_exiting,
                ..
            } = current_task_data;

            if !preempted_is_exiting {
                // Store the virtual lag for the preempted task.
                let avg_vruntime = if total_weight != 0 {
                    avg_vruntime(self.weighted_vruntime_offsets, total_weight, min_vruntime)
                } else {
                    self.weighted_vruntime_offsets + min_vruntime
                };
                // Limit this to either double the slice length with a minimum of TICK_NSEC
                // since that is the timing granularity.
                // Reference: https://elixir.bootlin.com/linux/v6.16.8/source/kernel/sched/fair.c#L686
                let vlimit = wall_to_virtual(self.lag_limit_clocks, weight);
                let vlag = (avg_vruntime - preempted_vruntime).clamp(-vlimit, vlimit);
                let preempted_fair_attr = &preempted_task.as_thread().unwrap().sched_attr().fair;
                preempted_fair_attr.vlag.store(vlag, Ordering::Relaxed);
            }

            // This is the minimum queued vruntime *before* popping.
            let min_queued_vruntime = self.queue.min_vruntime_against(vruntime);

            if *preempted_vruntime < min_queued_vruntime {
                // `preempted_vruntime` was the minimum vruntime and now it's moved
                // forward to `min_queued_vruntime`.
                // ρₜ = ρₘᵢₙ => Φ' = Φ - W'(ρₘᵢₙ' - ρₘᵢₙ)
                self.weighted_vruntime_offsets -=
                    self.queued_weight * (min_queued_vruntime - preempted_vruntime);
            } else {
                // `min_queued_vruntime` remains as the minimum vruntime.
                // ρₘᵢₙ' = ρₘᵢₙ =>  Φ' = Φ - wₜ(ρₜ - ρₘᵢₙ)
                self.weighted_vruntime_offsets -=
                    preempted_weight * (preempted_vruntime - min_queued_vruntime);
            }
        } else {
            // This is not a dequeue. `weighted_vruntime_offsets` doesn't change.
        }

        self.current_task_data = Some(TaskData {
            task: task.clone(),
            id,
            vruntime,
            weight,
            deadline,
            is_exiting,
        });

        self.queue_len -= 1;
        self.queued_weight -= weight;

        Some(task)
    }

    fn update_current(
        &mut self,
        rt: &CurrentRuntime,
        attr: &SchedAttr,
        flags: UpdateFlags,
    ) -> bool {
        // Φ' = ∑{i ∈ S}[wᵢ(ρᵢ' - ρₘᵢₙ')]
        //    = ∑{i ∈ S\⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ')] + wₜ(ρₜ + Δ - ρₘᵢₙ')
        //    = ∑{i ∈ S\⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ + ρₘᵢₙ - ρₘᵢₙ')] + wₜ(ρₜ - ρₘᵢₙ') + wₜΔ
        //    = ∑{i ∈ S\⦃t⦄}[wᵢ(ρᵢ - ρₘᵢₙ)] + (ρₘᵢₙ - ρₘᵢₙ')∑{i ∈ S\⦃t⦄}[wᵢ] + wₜ(ρₜ - ρₘᵢₙ') + wₜΔ
        //    = Φ - wₜ(ρₜ - ρₘᵢₙ) + (ρₘᵢₙ - ρₘᵢₙ')(W - wₜ) + wₜ(ρₜ - ρₘᵢₙ') + wₜΔ
        //    = Φ + wₜΔ - W(ρₘᵢₙ' - ρₘᵢₙ)

        // The data for the current task must have been set in `pick_next`.
        let current_task_data = self.current_task_data.as_mut().unwrap();
        debug_assert_eq!(current_task_data.id, attr.fair.id.load(Ordering::Relaxed));

        let weight = current_task_data.weight;
        let vdelta = wall_to_virtual(rt.delta as i64, weight);
        let deadline = current_task_data.deadline;

        let old_vruntime = current_task_data.vruntime;
        let new_vruntime = old_vruntime + vdelta;
        current_task_data.vruntime = new_vruntime;

        // Adjust `weighted_vruntime_offsets`.
        let total_weight = weight + self.queued_weight;
        if let Some(min_queued_vruntime) = self.queue.min_vruntime() {
            // Advance `weighted_vruntime_offsets` with the contribution of the current task.
            self.weighted_vruntime_offsets += weight * vdelta;

            if old_vruntime < min_queued_vruntime {
                // The old task's vruntime was the minimum vruntime and now the
                // minimum vruntime is the minimum between the minimum queued
                // vruntime and the new task's vruntime.
                let new_min_vruntime = min_queued_vruntime.min(new_vruntime);
                self.weighted_vruntime_offsets -= total_weight * (new_min_vruntime - old_vruntime);
            }
        } else {
            // The queue is empty so the current task is the only one at play.
            // `weighted_vruntime_offsets` is zero and doesn't change because the
            // vruntime offset for the (only) task doesn't change, since it's also
            // the task with the minimum vruntime.
            debug_assert_eq!(self.weighted_vruntime_offsets, 0);
        }

        if self.queue.is_empty() {
            return false; // There's no competing task.
        }

        match flags {
            UpdateFlags::Tick | UpdateFlags::Yield => new_vruntime >= deadline,
            UpdateFlags::Wait => true,
            UpdateFlags::Exit => {
                // Avoid computing and storing vlag in `pick_next`.
                current_task_data.is_exiting = true;
                true
            }
        }
    }

    fn len(&self) -> usize {
        self.queue_len
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// The [`EligibilityTree`] is a balanced binary search tree in which tasks are
/// ordered by their deadlines.
///
/// This data structure currently behaves as an AVL tree but an RB tree is likely
/// better.
enum EligibilityTree {
    Node {
        data: TaskData,
        min_vruntime: i64,
        height: i8,
        left: Box<Self>,
        right: Box<Self>,
    },
    Leaf,
}

impl core::fmt::Debug for EligibilityTree {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Leaf => write!(f, "Tree::Leaf"),
            Self::Node { data, .. } => write!(f, "Tree::Node[id={}]", data.id),
        }
    }
}

impl Ord for TaskData {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.deadline.cmp(&other.deadline) {
            cmp::Ordering::Equal => self.id.cmp(&other.id),
            ord => ord,
        }
    }
}
impl PartialOrd for TaskData {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for TaskData {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}
impl Eq for TaskData {}

impl EligibilityTree {
    fn new() -> Self {
        Self::Leaf
    }

    fn is_empty(&self) -> bool {
        self.is_leaf()
    }

    fn insert(&mut self, new_data: TaskData) {
        match self {
            Self::Leaf => {
                let min_vruntime = new_data.vruntime;
                *self = Self::Node {
                    data: new_data,
                    min_vruntime,
                    height: 0,
                    left: Box::new(Self::new()),
                    right: Box::new(Self::new()),
                }
            }
            Self::Node {
                data, left, right, ..
            } => {
                match new_data.cmp(data) {
                    cmp::Ordering::Equal => {
                        // *data = new_data;
                        unreachable!()
                    }
                    cmp::Ordering::Less => left.insert(new_data),
                    cmp::Ordering::Greater => right.insert(new_data),
                }
                self.update_and_rebalance();
            }
        }
    }

    fn min_vruntime(&self) -> Option<i64> {
        match self {
            Self::Leaf => None,
            Self::Node { min_vruntime, .. } => Some(*min_vruntime),
        }
    }

    fn pop_next(
        &mut self,
        global_min_vruntime: i64,
        total_weight: i64,
        weighted_vruntime_offsets: i64,
    ) -> Option<TaskData> {
        match self {
            Self::Leaf => None,
            Self::Node {
                data, left, right, ..
            } => {
                if left.has_eligible_task(
                    global_min_vruntime,
                    total_weight,
                    weighted_vruntime_offsets,
                ) {
                    let res =
                        left.pop_next(global_min_vruntime, total_weight, weighted_vruntime_offsets);
                    if res.is_some() {
                        self.update_and_rebalance();
                    }
                    return res;
                }

                if is_eligible(
                    data.vruntime,
                    global_min_vruntime,
                    total_weight,
                    weighted_vruntime_offsets,
                ) {
                    // Take ownership of this node so we can move its children around.
                    let node = mem::replace(self, Self::Leaf);
                    let Self::Node {
                        data: task_data,
                        left,
                        mut right,
                        ..
                    } = node
                    else {
                        unreachable!();
                    };

                    // Case: no left child -> replace this node with right subtree.
                    if left.is_leaf() {
                        *self = *right;
                        if !self.is_leaf() {
                            self.update_and_rebalance();
                        }
                        return Some(task_data);
                    }

                    // Case: no right child -> replace this node with left subtree.
                    if right.is_leaf() {
                        *self = *left;
                        if !self.is_leaf() {
                            self.update_and_rebalance();
                        }
                        return Some(task_data);
                    }

                    // Case: two children -> replace with in-order successor (min of right).
                    let successor = right.pop_min().unwrap();
                    *self = Self::Node {
                        data: successor,
                        min_vruntime: 0, // Fixed by `update_and_rebalance`.
                        height: 0,
                        left,
                        right,
                    };
                    self.update_and_rebalance();
                    return Some(task_data);
                }

                if right.has_eligible_task(
                    global_min_vruntime,
                    total_weight,
                    weighted_vruntime_offsets,
                ) {
                    let res = right.pop_next(
                        global_min_vruntime,
                        total_weight,
                        weighted_vruntime_offsets,
                    );
                    if res.is_some() {
                        self.update_and_rebalance();
                    }
                    return res;
                }

                let res = self.pop_min();
                if res.is_some() {
                    self.update_and_rebalance();
                }
                res
            }
        }
    }

    fn has_eligible_task(
        &self,
        global_min_vruntime: i64,
        total_weight: i64,
        weighted_vruntime_offsets: i64,
    ) -> bool {
        match self {
            Self::Leaf => false,
            Self::Node {
                min_vruntime: tree_min_vruntime,
                ..
            } => is_eligible(
                *tree_min_vruntime,
                global_min_vruntime,
                total_weight,
                weighted_vruntime_offsets,
            ),
        }
    }

    fn pop_min(&mut self) -> Option<TaskData> {
        match self {
            Self::Leaf => None,
            Self::Node { left, .. } => {
                if left.is_leaf() {
                    let old_self = mem::replace(self, Self::Leaf);
                    if let Self::Node { data, right, .. } = old_self {
                        *self = *right;
                        return Some(data);
                    }
                    None
                } else {
                    let result = left.pop_min();
                    self.update_and_rebalance();
                    result
                }
            }
        }
    }

    fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf)
    }

    fn update(&mut self) {
        self.update_height();
        self.update_min_vruntime();
    }

    fn height(&self) -> i8 {
        match self {
            Self::Leaf => -1,
            Self::Node { height, .. } => *height,
        }
    }

    fn update_height(&mut self) {
        if let Self::Node {
            height,
            left,
            right,
            ..
        } = self
        {
            *height = 1 + left.height().max(right.height());
        }
    }

    fn update_min_vruntime(&mut self) {
        if let Self::Node {
            data,
            min_vruntime,
            left,
            right,
            ..
        } = self
        {
            *min_vruntime = right.min_vruntime_against(left.min_vruntime_against(data.vruntime));
        }
    }

    fn min_vruntime_against(&self, vruntime: i64) -> i64 {
        match self {
            Self::Leaf => vruntime,
            Self::Node { min_vruntime, .. } => vruntime.min(*min_vruntime),
        }
    }

    fn balance_factor(&self) -> i8 {
        match self {
            Self::Leaf => 0,
            Self::Node { left, right, .. } => left.height() - right.height(),
        }
    }

    fn rotate_left(&mut self) {
        if let Self::Node { right, .. } = self {
            let right_node = mem::replace(right, Box::new(Self::Leaf));
            if let Self::Node {
                data: rdata,
                min_vruntime: rmin_vruntime,
                height: rheight,
                left: rleft,
                right: rright,
            } = *right_node
            {
                let old = mem::replace(self, Self::Leaf);
                if let Self::Node {
                    data,
                    min_vruntime,
                    height,
                    left,
                    right: _,
                } = old
                {
                    *self = Self::Node {
                        data: rdata,
                        min_vruntime: rmin_vruntime,
                        height: rheight,
                        left: Box::new(Self::Node {
                            data,
                            min_vruntime,
                            height,
                            left,
                            right: rleft,
                        }),
                        right: rright,
                    };
                }
            }
        }
        if let Self::Node { left, right, .. } = self {
            left.update();
            right.update();
        }
        self.update();
    }

    fn rotate_right(&mut self) {
        if let Self::Node { left, .. } = self {
            let left_node = mem::replace(left, Box::new(Self::Leaf));
            if let Self::Node {
                data: ldata,
                min_vruntime: lmin_vruntime,
                height: lheight,
                left: lleft,
                right: lright,
            } = *left_node
            {
                let old = mem::replace(self, Self::Leaf);
                if let Self::Node {
                    data,
                    min_vruntime,
                    height,
                    left: _,
                    right,
                } = old
                {
                    *self = Self::Node {
                        data: ldata,
                        min_vruntime: lmin_vruntime,
                        height: lheight,
                        left: lleft,
                        right: Box::new(Self::Node {
                            data,
                            min_vruntime,
                            height,
                            left: lright,
                            right,
                        }),
                    };
                }
            }
        }
        if let Self::Node { left, right, .. } = self {
            left.update();
            right.update();
        }
        self.update();
    }

    fn update_and_rebalance(&mut self) {
        self.update();

        let bf = self.balance_factor();
        if bf > 1 {
            let Self::Node { left, .. } = self else {
                return;
            };
            if left.balance_factor() < 0 {
                left.rotate_left();
            }
            self.rotate_right();
        } else if bf < -1 {
            let Self::Node { right, .. } = self else {
                return;
            };
            if right.balance_factor() > 0 {
                right.rotate_right();
            }
            self.rotate_left();
        }
    }
}
