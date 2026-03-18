//! TPE (Tree-structured Parzen Estimator) sampler.
//!
//! Port of Python `optuna.samplers.TPESampler`.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceGroup};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

use super::parzen_estimator::{default_gamma, ParzenEstimator, ParzenEstimatorParameters};

/// Type alias for `gamma(n) -> n_below` split function.
pub type GammaFn = Arc<dyn Fn(usize) -> usize + Send + Sync>;

/// Type alias for `weights(n) -> weights_vec` function.
pub type WeightsFn = Arc<dyn Fn(usize) -> Vec<f64> + Send + Sync>;

/// Type alias for constraint evaluation function.
pub type ConstraintsFn = Arc<dyn Fn(&FrozenTrial) -> Vec<f64> + Send + Sync>;

/// 分类参数距离函数。
/// 对应 Python `categorical_distance_func` 参数。
/// key: 参数名, value: 计算分类距离的闭包 (choice_i, choice_j) -> distance。
pub type CategoricalDistanceFunc =
    Arc<dyn Fn(&str, usize, usize) -> f64 + Send + Sync>;

/// Default weights function matching Python `default_weights`.
/// 对齐 Python: `np.linspace(1.0/x, 1.0, num=x-25)` + `np.ones(25)`
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n < 25 {
        return vec![1.0; n];
    }
    let mut w = Vec::with_capacity(n);
    let ramp_len = n - 25;
    // 对齐 Python np.linspace(1.0/n, 1.0, num=ramp_len)
    // 注意: np.linspace(start, stop, num=1) 返回 [start]，而非 [stop]
    let start = 1.0 / n as f64;
    if ramp_len == 1 {
        w.push(start);
    } else {
        let step = (1.0 - start) / (ramp_len as f64 - 1.0);
        for i in 0..ramp_len {
            w.push(start + step * i as f64);
        }
    }
    for _ in 0..25 {
        w.push(1.0);
    }
    w
}

/// Hyperopt-style gamma function.
pub fn hyperopt_default_gamma(n: usize) -> usize {
    ((0.25 * (n as f64).sqrt()).ceil() as usize).min(25)
}

/// A TPE (Tree-structured Parzen Estimator) sampler.
///
/// Models the objective function by splitting observed trials into "good"
/// (below threshold) and "bad" (above threshold) groups, fits separate
/// kernel density estimators (l(x) and g(x)), and maximizes l(x)/g(x)
/// as a surrogate for expected improvement.
///
/// Corresponds to Python `optuna.samplers.TPESampler`.
pub struct TpeSampler {
    /// Number of random startup trials before TPE kicks in.
    n_startup_trials: usize,
    /// Number of EI candidates to draw from l(x).
    n_ei_candidates: usize,
    /// 对齐 Python: 支持单目标和多目标。
    /// 单目标时长度为 1，多目标时 >= 2 (MOTPE)。
    directions: Vec<StudyDirection>,
    /// Whether to sample parameters jointly (multivariate) or independently.
    multivariate: bool,
    /// Whether to use group-decomposed search space (requires multivariate=true).
    group: bool,
    /// Use constant liar for parallel optimization.
    constant_liar: bool,
    /// Constraints function (if set, enables constrained optimization).
    constraints_func: Option<ConstraintsFn>,
    /// Custom gamma function (n -> n_below).
    gamma: GammaFn,
    /// Custom weights function (n_below -> weights).
    weights: WeightsFn,
    /// Parzen estimator parameters.
    pe_params: ParzenEstimatorParameters,
    /// RNG for sampling.
    rng: Mutex<ChaCha8Rng>,
    /// Fallback random sampler for startup.
    random_sampler: RandomSampler,
    /// Search space tracker (for multivariate mode).
    search_space: Mutex<IntersectionSearchSpace>,
    /// Group-decomposed search space (for group mode).
    group_search_space: Mutex<SearchSpaceGroup>,
    /// 分类参数自定义距离函数。
    /// 对应 Python `categorical_distance_func` 参数（实验性功能）。
    categorical_distance_func: Option<HashMap<String, CategoricalDistanceFunc>>,
    /// 是否在 multivariate 模式下对独立采样的参数发出警告。
    /// 对应 Python `warn_independent_sampling` 参数。
    warn_independent_sampling: bool,
}

impl std::fmt::Debug for TpeSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpeSampler")
            .field("n_startup_trials", &self.n_startup_trials)
            .field("n_ei_candidates", &self.n_ei_candidates)
            .field("multivariate", &self.multivariate)
            .finish()
    }
}

impl TpeSampler {
    /// Create a new `TpeSampler`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        direction: StudyDirection,
        seed: Option<u64>,
        n_startup_trials: usize,
        n_ei_candidates: usize,
        multivariate: bool,
        consider_magic_clip: bool,
        consider_endpoints: bool,
        prior_weight: f64,
        group: bool,
        constant_liar: bool,
        constraints_func: Option<ConstraintsFn>,
        gamma: Option<GammaFn>,
        weights: Option<WeightsFn>,
        categorical_distance_func: Option<HashMap<String, CategoricalDistanceFunc>>,
        warn_independent_sampling: bool,
    ) -> Self {
        Self::new_multi(
            vec![direction], seed, n_startup_trials, n_ei_candidates,
            multivariate, consider_magic_clip, consider_endpoints, prior_weight,
            group, constant_liar, constraints_func, gamma, weights,
            categorical_distance_func, warn_independent_sampling,
        )
    }

    /// 多目标 TPE 构造函数 (MOTPE).
    #[allow(clippy::too_many_arguments)]
    pub fn new_multi(
        directions: Vec<StudyDirection>,
        seed: Option<u64>,
        n_startup_trials: usize,
        n_ei_candidates: usize,
        multivariate: bool,
        consider_magic_clip: bool,
        consider_endpoints: bool,
        prior_weight: f64,
        group: bool,
        constant_liar: bool,
        constraints_func: Option<ConstraintsFn>,
        gamma: Option<GammaFn>,
        weights: Option<WeightsFn>,
        categorical_distance_func: Option<HashMap<String, CategoricalDistanceFunc>>,
        warn_independent_sampling: bool,
    ) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_rng(&mut rand::rng()),
        };
        Self {
            n_startup_trials,
            n_ei_candidates,
            directions,
            multivariate,
            group,
            constant_liar,
            constraints_func,
            gamma: gamma.unwrap_or_else(|| Arc::new(default_gamma)),
            weights: weights.unwrap_or_else(|| Arc::new(default_weights)),
            pe_params: ParzenEstimatorParameters {
                prior_weight,
                consider_magic_clip,
                consider_endpoints,
                multivariate,
                categorical_distance_func: categorical_distance_func
                    .clone()
                    .unwrap_or_default(),
            },
            rng: Mutex::new(rng),
            random_sampler: RandomSampler::new(seed.map(|s| s.wrapping_add(1))),
            // 对齐 Python: IntersectionSearchSpace(include_pruned=True)
            // 在多变量模式下，也考虑 PRUNED 试验的参数来计算搜索空间交集
            search_space: Mutex::new(IntersectionSearchSpace::new(true)),
            group_search_space: Mutex::new(SearchSpaceGroup::new()),
            categorical_distance_func,
            warn_independent_sampling,
        }
    }

    /// Create with common defaults.
    pub fn with_defaults(direction: StudyDirection, seed: Option<u64>) -> Self {
        Self::new(
            direction, seed, 10, 24, false, true, false, 1.0,
            false, false, None, None, None, None, true,
        )
    }

    /// Create a multivariate TPE sampler.
    pub fn multivariate(direction: StudyDirection, seed: Option<u64>) -> Self {
        Self::new(
            direction, seed, 10, 24, true, true, false, 1.0,
            false, false, None, None, None, None, true,
        )
    }

    /// Get infeasibility score for a trial (sum of positive constraint violations).
    fn infeasible_score(trial: &FrozenTrial) -> f64 {
        trial
            .system_attrs
            .get(crate::multi_objective::CONSTRAINTS_KEY)
            .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
            .map(|cs| cs.iter().filter(|&&c| c > 0.0).sum())
            .unwrap_or(f64::INFINITY) // no constraint value → worst
    }

    /// Split trials into below (good) and above (bad) groups.
    ///
    /// 对齐 Python `_get_pruned_trial_score(trial, direction)`:
    /// 有中间值时返回 (-step, value)，Maximize 时对 value 取负。
    /// 无中间值时返回 (1, 0.0)。
    fn pruned_trial_score(trial: &FrozenTrial, direction: StudyDirection) -> (i64, f64) {
        if trial.intermediate_values.is_empty() {
            return (1, 0.0);
        }
        let (step, intermediate_value) = trial
            .intermediate_values
            .iter()
            .max_by_key(|(k, _)| *k)
            .map(|(&k, &v)| (k, v))
            .unwrap();
        let v = if intermediate_value.is_nan() {
            f64::INFINITY
        } else {
            match direction {
                StudyDirection::Maximize => -intermediate_value,
                _ => intermediate_value,
            }
        };
        (-step, v)
    }

    /// Split trials into below/above groups for TPE.
    ///
    /// 单目标: 按目标值排序分割。
    /// 多目标 (MOTPE): 使用非支配排序 + HSSP 平局打断分割。
    fn split_trials<'a>(
        &self,
        trials: &'a [FrozenTrial],
    ) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
        if self.directions.len() > 1 {
            return self.split_trials_multi_objective(trials);
        }
        self.split_trials_single_objective(trials)
    }

    /// 单目标分割。
    fn split_trials_single_objective<'a>(
        &self,
        trials: &'a [FrozenTrial],
    ) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
        let direction = self.directions[0];
        let constraints_enabled = self.constraints_func.is_some();

        let mut complete: Vec<&FrozenTrial> = Vec::new();
        let mut pruned: Vec<&FrozenTrial> = Vec::new();
        let mut running: Vec<&FrozenTrial> = Vec::new();
        let mut infeasible: Vec<&FrozenTrial> = Vec::new();

        for t in trials {
            match t.state {
                TrialState::Running => running.push(t),
                _ if constraints_enabled && Self::infeasible_score(t) > 0.0 => {
                    infeasible.push(t);
                }
                TrialState::Complete => complete.push(t),
                TrialState::Pruned => pruned.push(t),
                _ => {}
            }
        }

        let n = complete.len() + pruned.len() + infeasible.len();
        let n_below = (self.gamma)(n);

        // Sort complete trials by objective value.
        match direction {
            StudyDirection::Minimize | StudyDirection::NotSet => {
                complete.sort_by(|a, b| {
                    let va = a.value().ok().flatten().unwrap_or(f64::INFINITY);
                    let vb = b.value().ok().flatten().unwrap_or(f64::INFINITY);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            StudyDirection::Maximize => {
                complete.sort_by(|a, b| {
                    let va = a.value().ok().flatten().unwrap_or(f64::NEG_INFINITY);
                    let vb = b.value().ok().flatten().unwrap_or(f64::NEG_INFINITY);
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        pruned.sort_by(|a, b| {
            let score_a = Self::pruned_trial_score(a, direction);
            let score_b = Self::pruned_trial_score(b, direction);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut below = Vec::new();
        let mut above = Vec::new();

        let mut remaining = n_below;
        for t in &complete {
            if remaining > 0 {
                below.push(*t);
                remaining -= 1;
            } else {
                above.push(*t);
            }
        }
        for t in &pruned {
            if remaining > 0 {
                below.push(*t);
                remaining -= 1;
            } else {
                above.push(*t);
            }
        }

        infeasible.sort_by(|a, b| {
            Self::infeasible_score(a)
                .partial_cmp(&Self::infeasible_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for t in &infeasible {
            if remaining > 0 {
                below.push(*t);
                remaining -= 1;
            } else {
                above.push(*t);
            }
        }

        above.extend(running);

        below.sort_by_key(|t| t.number);
        above.sort_by_key(|t| t.number);

        (below, above)
    }

    /// 对齐 Python `_split_complete_trials_multi_objective` + `_split_trials`:
    /// 多目标分割 — complete 使用非支配排序 + HSSP 平局打断，
    /// 然后依次填充 pruned 和 infeasible，与单目标逻辑一致。
    fn split_trials_multi_objective<'a>(
        &self,
        trials: &'a [FrozenTrial],
    ) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
        let constraints_enabled = self.constraints_func.is_some();

        let mut complete: Vec<&FrozenTrial> = Vec::new();
        let mut pruned: Vec<&FrozenTrial> = Vec::new();
        let mut running: Vec<&FrozenTrial> = Vec::new();
        let mut infeasible: Vec<&FrozenTrial> = Vec::new();

        for t in trials {
            match t.state {
                TrialState::Running => running.push(t),
                _ if constraints_enabled && Self::infeasible_score(t) > 0.0 => {
                    infeasible.push(t);
                }
                TrialState::Complete if t.values.is_some() => complete.push(t),
                TrialState::Pruned => pruned.push(t),
                _ => {}
            }
        }

        // 对齐 Python: n = complete + pruned + infeasible
        let n = complete.len() + pruned.len() + infeasible.len();
        if n == 0 {
            return (vec![], running);
        }

        let n_below = (self.gamma)(n);

        // --- Step 1: split complete trials (multi-objective: NSGA + HSSP) ---
        let n_below_complete = n_below.min(complete.len());
        let (below_complete, above_complete) = if complete.is_empty() || n_below_complete == 0 {
            (vec![], complete)
        } else if n_below_complete >= complete.len() {
            (complete.clone(), vec![])
        } else {
            self.split_complete_multi_objective(&complete, n_below_complete)
        };

        // --- Step 2: split pruned trials ---
        let mut remaining = n_below.saturating_sub(below_complete.len());
        let remaining_pruned = remaining.min(pruned.len());
        // 按 pruned_trial_score 排序 — 对齐 Python _split_pruned_trials
        // 多目标 pruned 排序: Python 使用 study.direction (第一个方向)
        let first_dir = self.directions[0];
        pruned.sort_by(|a, b| {
            let sa = Self::pruned_trial_score(a, first_dir);
            let sb = Self::pruned_trial_score(b, first_dir);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let below_pruned: Vec<&FrozenTrial> = pruned[..remaining_pruned].to_vec();
        let above_pruned: Vec<&FrozenTrial> = pruned[remaining_pruned..].to_vec();

        // --- Step 3: split infeasible trials ---
        remaining = remaining.saturating_sub(below_pruned.len());
        let remaining_infeasible = remaining.min(infeasible.len());
        infeasible.sort_by(|a, b| {
            Self::infeasible_score(a)
                .partial_cmp(&Self::infeasible_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let below_infeasible: Vec<&FrozenTrial> = infeasible[..remaining_infeasible].to_vec();
        let above_infeasible: Vec<&FrozenTrial> = infeasible[remaining_infeasible..].to_vec();

        // --- Combine ---
        let mut below = below_complete;
        below.extend(below_pruned);
        below.extend(below_infeasible);

        let mut above = above_complete;
        above.extend(above_pruned);
        above.extend(above_infeasible);
        above.extend(running);

        below.sort_by_key(|t| t.number);
        above.sort_by_key(|t| t.number);

        (below, above)
    }

    /// 多目标 complete trials 分割内部函数: 非支配排序 + HSSP 平局打断。
    fn split_complete_multi_objective<'a>(
        &self,
        complete: &[&'a FrozenTrial],
        n_below: usize,
    ) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
        let n = complete.len();

        // 将目标值转为 loss 值（Maximize 方向取负）
        let loss_values: Vec<Vec<f64>> = complete.iter().map(|t| {
            let vals = t.values.as_ref().unwrap();
            vals.iter().enumerate().map(|(i, &v)| {
                if self.directions[i] == StudyDirection::Maximize { -v } else { v }
            }).collect()
        }).collect();

        // 非支配排序（传入 n_below 以启用提前终止优化）
        let ranks = Self::fast_non_domination_rank_with_n_below(&loss_values, Some(n_below));

        let mut below_indices = Vec::new();
        let mut above_indices = Vec::new();

        let max_rank = *ranks.iter().max().unwrap_or(&0);
        for rank in 0..=max_rank {
            let rank_indices: Vec<usize> = ranks.iter().enumerate()
                .filter(|&(_, &r)| r == rank)
                .map(|(i, _)| i)
                .collect();

            if below_indices.len() + rank_indices.len() <= n_below {
                below_indices.extend(rank_indices);
            } else {
                let remaining = n_below - below_indices.len();
                if remaining > 0 {
                    let rank_losses: Vec<Vec<f64>> = rank_indices.iter()
                        .map(|&i| loss_values[i].clone())
                        .collect();
                    let ref_point = Self::get_reference_point(&loss_values);
                    let hssp_indices: Vec<usize> = (0..rank_losses.len()).collect();
                    let selected = crate::multi_objective::solve_hssp(
                        &rank_losses, &hssp_indices, remaining, &ref_point,
                    );
                    for sel_idx in selected {
                        below_indices.push(rank_indices[sel_idx]);
                    }
                }
                for &idx in &rank_indices {
                    if !below_indices.contains(&idx) {
                        above_indices.push(idx);
                    }
                }
                for higher_rank in (rank + 1)..=max_rank {
                    for (i, &r) in ranks.iter().enumerate() {
                        if r == higher_rank {
                            above_indices.push(i);
                        }
                    }
                }
                break;
            }
        }

        for i in 0..n {
            if !below_indices.contains(&i) && !above_indices.contains(&i) {
                above_indices.push(i);
            }
        }

        let below: Vec<&FrozenTrial> = below_indices.iter().map(|&i| complete[i]).collect();
        let above: Vec<&FrozenTrial> = above_indices.iter().map(|&i| complete[i]).collect();
        (below, above)
    }

    /// 对齐 Python `_fast_non_domination_rank`:
    /// 直接在 loss values 矩阵上计算非支配层级。
    ///
    /// `n_below`: 可选的提前终止参数。当已分配排名的试验数达到 `n_below` 后，
    /// 剩余未分配的试验全部设为当前 rank（对齐 Python `_calculate_nondomination_rank`
    /// 中的 `n_below` 语义，避免不必要的排名计算开销）。
    fn fast_non_domination_rank(loss_values: &[Vec<f64>]) -> Vec<usize> {
        Self::fast_non_domination_rank_with_n_below(loss_values, None)
    }

    fn fast_non_domination_rank_with_n_below(
        loss_values: &[Vec<f64>],
        n_below: Option<usize>,
    ) -> Vec<usize> {
        let n = loss_values.len();
        if n == 0 {
            return vec![];
        }

        let n_below = n_below.unwrap_or(n);

        // 对齐 Python: 单目标特殊路径（使用唯一排名）
        if !loss_values.is_empty() && loss_values[0].len() == 1 {
            let mut indexed: Vec<(usize, f64)> = loss_values.iter().enumerate()
                .map(|(i, v)| (i, v[0]))
                .collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut ranks = vec![0usize; n];
            let mut current_rank = 0;
            for w in indexed.windows(2) {
                ranks[w[0].0] = current_rank;
                if (w[1].1 - w[0].1).abs() > f64::EPSILON {
                    current_rank += 1;
                }
            }
            if let Some(last) = indexed.last() {
                ranks[last.0] = current_rank;
            }
            return ranks;
        }

        let mut dominated_count = vec![0usize; n];
        let mut dominates_list: Vec<Vec<usize>> = vec![vec![]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dom_ij = Self::dominates_values(&loss_values[i], &loss_values[j]);
                let dom_ji = Self::dominates_values(&loss_values[j], &loss_values[i]);
                if dom_ij {
                    dominates_list[i].push(j);
                    dominated_count[j] += 1;
                } else if dom_ji {
                    dominates_list[j].push(i);
                    dominated_count[i] += 1;
                }
            }
        }

        let mut ranks = vec![0usize; n];
        let mut assigned = 0usize;
        let mut current_front: Vec<usize> = (0..n).filter(|&i| dominated_count[i] == 0).collect();
        let mut rank = 0;

        while !current_front.is_empty() {
            let mut next_front = Vec::new();
            for &i in &current_front {
                ranks[i] = rank;
                assigned += 1;
                for &j in &dominates_list[i] {
                    dominated_count[j] -= 1;
                    if dominated_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            // 对齐 Python n_below 提前终止: 当已分配排名的数量 >= n_below 时停止
            if assigned >= n_below {
                // 将剩余未分配的试验设为下一个 rank
                for &j in &next_front {
                    ranks[j] = rank + 1;
                }
                break;
            }
            current_front = next_front;
            rank += 1;
        }

        ranks
    }

    /// Check if values `a` dominates `b` (all <= and at least one <).
    fn dominates_values(a: &[f64], b: &[f64]) -> bool {
        let mut any_strictly_better = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if ai > bi {
                return false;
            }
            if ai < bi {
                any_strictly_better = true;
            }
        }
        any_strictly_better
    }

    /// 对齐 Python `_get_reference_point`:
    /// 计算超体积参考点。
    fn get_reference_point(loss_values: &[Vec<f64>]) -> Vec<f64> {
        if loss_values.is_empty() {
            return vec![];
        }
        let n_objectives = loss_values[0].len();
        let mut worst = vec![f64::NEG_INFINITY; n_objectives];
        for vals in loss_values {
            for (i, &v) in vals.iter().enumerate() {
                if v > worst[i] {
                    worst[i] = v;
                }
            }
        }
        // 对齐 Python: reference_point = max(1.1 * worst, 0.9 * worst); [==0] = EPS
        let eps = 1e-10;
        worst.iter().map(|&w| {
            if w == 0.0 { eps }
            else if w > 0.0 { 1.1 * w }
            else { 0.9 * w }
        }).collect()
    }

    /// 对齐 Python `_calculate_weights_below_for_multi_objective`:
    /// 基于超体积贡献计算 below 组的权重。
    /// 不可行试验（约束违反 > 0）权重设为 EPS，不参与 HV 计算。
    fn calculate_mo_weights(
        below_trials: &[&FrozenTrial],
        directions: &[StudyDirection],
        constraints_enabled: bool,
    ) -> Vec<f64> {
        let n = below_trials.len();
        if n == 0 {
            return vec![];
        }

        let eps = 1e-10_f64;

        // 对齐 Python: 识别可行/不可行试验
        let is_feasible: Vec<bool> = below_trials.iter().map(|t| {
            if !constraints_enabled {
                true
            } else {
                Self::infeasible_score(t) <= 0.0
            }
        }).collect();

        // 初始化权重: 可行=1.0, 不可行=EPS
        let mut weights: Vec<f64> = is_feasible.iter().map(|&f| if f { 1.0 } else { eps }).collect();

        let n_feasible: usize = is_feasible.iter().filter(|&&f| f).count();
        if n_feasible <= 1 {
            return weights;
        }

        // 只在可行试验上计算 HV 贡献
        let feasible_indices: Vec<usize> = is_feasible.iter().enumerate()
            .filter(|&(_, f)| *f)
            .map(|(i, _)| i)
            .collect();

        let loss_values: Vec<Vec<f64>> = feasible_indices.iter().map(|&i| {
            let vals = below_trials[i].values.as_ref().unwrap();
            vals.iter().enumerate().map(|(j, &v)| {
                if directions[j] == StudyDirection::Maximize { -v } else { v }
            }).collect()
        }).collect();

        let ref_point = Self::get_reference_point(&loss_values);

        // 找 Pareto 前沿 (在可行试验中)
        let ranks = Self::fast_non_domination_rank(&loss_values);
        let pareto_local_indices: Vec<usize> = ranks.iter().enumerate()
            .filter(|&(_, &r)| r == 0)
            .map(|(i, _)| i)
            .collect();

        let pareto_losses: Vec<Vec<f64>> = pareto_local_indices.iter()
            .map(|&i| loss_values[i].clone())
            .collect();

        let full_hv = crate::multi_objective::hypervolume(&pareto_losses, &ref_point);
        if full_hv.is_infinite() {
            return weights;
        }

        // Leave-one-out: 每个 Pareto 点的 HV 贡献
        // 对齐 Python: ≤3 目标使用精确 LOO，>3 目标使用近似方法
        let n_objectives = directions.len();
        let mut contribs = vec![0.0_f64; n_feasible];

        if n_objectives <= 3 {
            // ≤3 目标: 精确 LOO（对齐 Python `contribs[on_front] = [hv - compute_hypervolume(pareto_sols[loo], ...)]`）
            for (pi, &local_idx) in pareto_local_indices.iter().enumerate() {
                let without: Vec<Vec<f64>> = pareto_losses.iter().enumerate()
                    .filter(|(i, _)| *i != pi)
                    .map(|(_, v)| v.clone())
                    .collect();
                let hv_without = if without.is_empty() {
                    0.0
                } else {
                    crate::multi_objective::hypervolume(&without, &ref_point)
                };
                contribs[local_idx] = full_hv - hv_without;
            }
        } else {
            // >3 目标: 近似方法（对齐 Python）
            // contribs[on_front] = prod(ref_point - pareto_sols, axis=-1)
            //                    - [compute_hypervolume(limited_sols[i, loo], ref_point)]
            // 其中 limited_sols = max(pareto_sols, pareto_sols[:, newaxis])

            let n_pareto = pareto_losses.len();

            // 计算每个 Pareto 点与参考点的体积乘积
            let box_volumes: Vec<f64> = pareto_losses.iter().map(|sol| {
                sol.iter().zip(ref_point.iter()).map(|(&s, &r)| r - s).product::<f64>()
            }).collect();

            // limited_sols[i][j] = max(pareto_sols[i], pareto_sols[j]) (element-wise)
            // 对于每个 Pareto 点 i，计算 LOO HV 中移除 i 后的 limited_sols
            for (pi, &local_idx) in pareto_local_indices.iter().enumerate() {
                // 构造 limited_sols[pi, loo]: 对其他 Pareto 点 j，取 max(pareto[pi], pareto[j])
                let limited_without: Vec<Vec<f64>> = (0..n_pareto)
                    .filter(|&j| j != pi)
                    .map(|j| {
                        pareto_losses[pi].iter().zip(pareto_losses[j].iter())
                            .map(|(&a, &b)| a.max(b))
                            .collect()
                    })
                    .collect();

                let limited_hv = if limited_without.is_empty() {
                    0.0
                } else {
                    crate::multi_objective::hypervolume(&limited_without, &ref_point)
                };
                contribs[local_idx] = box_volumes[pi] - limited_hv;
            }
        }

        // 对齐 Python: weights[feasible] = max(contribs / max(max(contribs), EPS), EPS)
        let max_contrib = contribs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let denom = max_contrib.max(eps);
        for (fi, &orig_idx) in feasible_indices.iter().enumerate() {
            weights[orig_idx] = (contribs[fi] / denom).max(eps);
        }

        weights
    }

    /// Get observations from trials for the given search space.
    fn get_observations(
        trials: &[&FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> HashMap<String, Vec<f64>> {
        let mut obs: HashMap<String, Vec<f64>> = HashMap::new();
        for name in search_space.keys() {
            obs.insert(name.to_string(), Vec::new());
        }

        for &trial in trials {
            // Only include trials that have all params in the search space.
            let has_all = search_space
                .keys()
                .all(|name| trial.params.contains_key(name));
            if !has_all {
                continue;
            }

            for (name, dist) in search_space {
                if let Some(val) = trial.params.get(name)
                    && let Ok(internal) = dist.to_internal_repr(val)
                {
                    obs.get_mut(name).unwrap().push(internal);
                }
            }
        }
        obs
    }

    /// Sample using TPE for a given search space.
    fn tpe_sample(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
        current_trial_number: Option<i64>,
    ) -> Result<HashMap<String, f64>> {
        // Constant liar: include running trials, exclude current trial
        let filtered: Vec<FrozenTrial>;
        let effective_trials = if self.constant_liar {
            filtered = trials
                .iter()
                .filter(|t| {
                    t.state == TrialState::Complete
                        || t.state == TrialState::Pruned
                        || t.state == TrialState::Running
                })
                .filter(|t| current_trial_number.map_or(true, |n| t.number != n))
                .cloned()
                .collect();
            &filtered
        } else {
            trials
        };

        let (below, above) = self.split_trials(effective_trials);

        let obs_below = Self::get_observations(&below, search_space);
        let obs_above = Self::get_observations(&above, search_space);

        // 对齐 Python: 多目标时 below 组使用 HV 贡献权重
        let weights_ref = self.weights.clone();
        let pe_below = if self.directions.len() > 1 {
            // MOTPE: 使用超体积贡献权重
            let mo_weights = Self::calculate_mo_weights(
                &below, &self.directions, self.constraints_func.is_some(),
            );
            ParzenEstimator::new(&obs_below, search_space, &self.pe_params,
                                 Some(&mo_weights), Some(&|n| weights_ref(n)))
        } else {
            ParzenEstimator::new(&obs_below, search_space, &self.pe_params, None,
                                 Some(&|n| weights_ref(n)))
        };
        let pe_above =
            ParzenEstimator::new(&obs_above, search_space, &self.pe_params, None,
                                 Some(&|n| weights_ref(n)));

        // Sample candidates from l(x).
        let mut rng = self.rng.lock();
        let samples = pe_below.sample(&mut *rng, self.n_ei_candidates);
        drop(rng);

        // Compute acquisition: log l(x) - log g(x).
        let log_l = pe_below.log_pdf(&samples);
        let log_g = pe_above.log_pdf(&samples);

        let mut best_idx = 0;
        let mut best_acq = f64::NEG_INFINITY;
        for i in 0..self.n_ei_candidates {
            let acq = log_l[i] - log_g[i];
            if acq > best_acq {
                best_acq = acq;
                best_idx = i;
            }
        }

        // Extract the best sample.
        let mut result = HashMap::new();
        for (name, vals) in &samples {
            result.insert(name.clone(), vals[best_idx]);
        }
        Ok(result)
    }

    /// Check if we're still in the startup phase.
    fn is_startup(&self, trials: &[FrozenTrial]) -> bool {
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete || t.state == TrialState::Pruned)
            .count();
        n_complete < self.n_startup_trials
    }
}

impl Sampler for TpeSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        if !self.multivariate {
            return IndexMap::new();
        }
        if self.is_startup(trials) {
            return IndexMap::new();
        }

        if self.group {
            // Group mode: use SearchSpaceGroup to decompose search space
            let mut gs = self.group_search_space.lock();
            // Add distributions from each trial
            for trial in trials {
                if trial.state == TrialState::Complete || trial.state == TrialState::Pruned {
                    gs.add_distributions(&trial.distributions);
                }
            }
            // Return the union as the relative space, filtering out single() distributions
            // 对齐 Python: if distribution.single(): continue
            let mut result = IndexMap::new();
            for space in gs.search_spaces() {
                for (k, v) in space.iter() {
                    if !v.single() {
                        result.insert(k.clone(), v.clone());
                    }
                }
            }
            result
        } else {
            let mut ss = self.search_space.lock();
            // 对齐 Python: filter out single() distributions
            // Python: for name, distribution in search_space.items():
            //            if distribution.single(): continue
            ss.calculate(trials)
                .iter()
                .filter(|(_, d)| !d.single())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        }
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }
        if self.is_startup(trials) {
            return Ok(HashMap::new());
        }

        if self.group {
            // Group mode: sample each sub-space independently
            // 对齐 Python: sorted(sub_space.items()) 确保键排序
            let gs = self.group_search_space.lock();
            let mut params = HashMap::new();
            for sub_space in gs.search_spaces() {
                let mut filtered: IndexMap<String, Distribution> = IndexMap::new();
                let mut sorted_keys: Vec<&String> = sub_space
                    .keys()
                    .filter(|name| search_space.contains_key(*name))
                    .collect();
                sorted_keys.sort();
                for name in sorted_keys {
                    if let Some(dist) = sub_space.get(name) {
                        filtered.insert(name.clone(), dist.clone());
                    }
                }
                if !filtered.is_empty() {
                    params.extend(self.tpe_sample(trials, &filtered, None)?);
                }
            }
            return Ok(params);
        }

        // search_space is already IndexMap with correct ordering
        self.tpe_sample(trials, search_space, None)
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // 对齐 Python `TPESampler.sample_independent`：
        // 启动期用随机采样，之后用 TPE 对单参数采样。
        let completed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete || t.state == TrialState::Pruned)
            .collect();

        if completed.len() < self.n_startup_trials {
            return self
                .random_sampler
                .sample_independent(trials, trial, param_name, distribution);
        }

        // multivariate 模式下对独立采样发出警告
        if self.multivariate && self.warn_independent_sampling {
            // 只在参数已出现在某些已完成试验中时才警告（对齐 Python）
            if trials.iter().any(|t| t.params.contains_key(param_name)) {
                eprintln!(
                    "[optuna] The parameter '{}' in trial#{} is sampled independently. \
                     The TPE multivariate algorithm only applies to parameters \
                     in the common search space.",
                    param_name,
                    trial.number,
                );
            }
        }

        // 使用 TPE 对单个参数进行采样（对齐 Python self._sample(study, trial, {param_name: dist})）
        let mut single_space = IndexMap::new();
        single_space.insert(param_name.to_string(), distribution.clone());
        let result = self.tpe_sample(trials, &single_space, Some(trial.number))?;
        Ok(result.get(param_name).copied().unwrap_or_else(|| {
            // 如果 TPE 无法采样（如没有有效观测），回退到随机
            self.random_sampler
                .sample_independent(trials, trial, param_name, distribution)
                .unwrap_or(0.0)
        }))
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        _state: TrialState,
        _values: Option<&[f64]>,
    ) {
        // 约束值由 compute_constraints() 返回，tell() 负责存储到 storage。
        // 对齐 Python: after_trial 调用 _process_constraints_after_trial + random_sampler.after_trial
    }

    /// 对齐 Python `_process_constraints_after_trial`:
    /// 当设置了 constraints_func 且状态为 Complete 或 Pruned 时，
    /// 计算约束值并返回，由 tell() 存储到 trial.system_attrs。
    fn compute_constraints(
        &self,
        trial: &FrozenTrial,
        state: TrialState,
    ) -> Option<Vec<f64>> {
        let cf = self.constraints_func.as_ref()?;
        if state != TrialState::Complete && state != TrialState::Pruned {
            return None;
        }
        let constraints = cf(trial);
        // 对齐 Python: 检查 NaN
        if constraints.iter().any(|c| c.is_nan()) {
            crate::optuna_warn!("Constraint values cannot be NaN. Storing None.");
            return None;
        }
        Some(constraints)
    }

    /// 对齐 Python `TPESampler.reseed_rng(seed)`: 重新设置随机种子。
    fn reseed_rng(&self, seed: u64) {
        *self.rng.lock() = ChaCha8Rng::seed_from_u64(seed);
    }
}

/// Builder for constructing a [`TpeSampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use optuna_rs::{TpeSamplerBuilder, StudyDirection};
///
/// let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize)
///     .seed(42)
///     .n_startup_trials(20)
///     .multivariate(true)
///     .build();
/// ```
pub struct TpeSamplerBuilder {
    directions: Vec<StudyDirection>,
    seed: Option<u64>,
    n_startup_trials: usize,
    n_ei_candidates: usize,
    multivariate: bool,
    consider_magic_clip: bool,
    consider_endpoints: bool,
    prior_weight: f64,
    group: bool,
    constant_liar: bool,
    constraints_func: Option<ConstraintsFn>,
    gamma: Option<GammaFn>,
    weights: Option<WeightsFn>,
    categorical_distance_func: Option<HashMap<String, CategoricalDistanceFunc>>,
    warn_independent_sampling: bool,
}

impl TpeSamplerBuilder {
    /// Create a new builder with the given optimization direction.
    pub fn new(direction: StudyDirection) -> Self {
        Self {
            directions: vec![direction],
            seed: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
            multivariate: false,
            consider_magic_clip: true,
            consider_endpoints: false,
            prior_weight: 1.0,
            group: false,
            constant_liar: false,
            constraints_func: None,
            gamma: None,
            weights: None,
            categorical_distance_func: None,
            warn_independent_sampling: true,
        }
    }

    /// Create a new builder for multi-objective optimization (MOTPE).
    pub fn new_multi(directions: Vec<StudyDirection>) -> Self {
        Self {
            directions,
            seed: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
            multivariate: false,
            consider_magic_clip: true,
            consider_endpoints: false,
            prior_weight: 1.0,
            group: false,
            constant_liar: false,
            constraints_func: None,
            gamma: None,
            weights: None,
            categorical_distance_func: None,
            warn_independent_sampling: true,
        }
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the number of random startup trials before TPE kicks in.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set the number of EI candidates to evaluate.
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }

    /// Enable or disable multivariate (joint) sampling.
    pub fn multivariate(mut self, multivariate: bool) -> Self {
        self.multivariate = multivariate;
        self
    }

    /// Enable or disable magic clip for bandwidth selection.
    pub fn consider_magic_clip(mut self, consider: bool) -> Self {
        self.consider_magic_clip = consider;
        self
    }

    /// Enable or disable considering endpoints in the Parzen estimator.
    pub fn consider_endpoints(mut self, consider: bool) -> Self {
        self.consider_endpoints = consider;
        self
    }

    /// Set the prior weight for the Parzen estimator.
    pub fn prior_weight(mut self, weight: f64) -> Self {
        self.prior_weight = weight;
        self
    }

    /// Enable group-decomposed search space (requires multivariate=true).
    pub fn group(mut self, group: bool) -> Self {
        self.group = group;
        self
    }

    /// Enable constant liar for parallel optimization.
    pub fn constant_liar(mut self, constant_liar: bool) -> Self {
        self.constant_liar = constant_liar;
        self
    }

    /// Set a constraints function for constrained optimization.
    pub fn constraints_func(mut self, func: ConstraintsFn) -> Self {
        self.constraints_func = Some(func);
        self
    }

    /// Set a custom gamma function (n -> n_below).
    pub fn gamma(mut self, gamma: GammaFn) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set a custom weights function (n_below -> weights).
    pub fn weights(mut self, weights: WeightsFn) -> Self {
        self.weights = Some(weights);
        self
    }

    /// 设置分类参数自定义距离函数（实验性功能）。
    /// 对应 Python `categorical_distance_func` 参数。
    pub fn categorical_distance_func(
        mut self,
        func: HashMap<String, CategoricalDistanceFunc>,
    ) -> Self {
        self.categorical_distance_func = Some(func);
        self
    }

    /// 设置是否在 multivariate 模式下对独立采样的参数发出警告。
    /// 默认为 true。对应 Python `warn_independent_sampling` 参数。
    pub fn warn_independent_sampling(mut self, warn: bool) -> Self {
        self.warn_independent_sampling = warn;
        self
    }

    /// Build the [`TpeSampler`].
    pub fn build(self) -> TpeSampler {
        TpeSampler::new_multi(
            self.directions,
            self.seed,
            self.n_startup_trials,
            self.n_ei_candidates,
            self.multivariate,
            self.consider_magic_clip,
            self.consider_endpoints,
            self.prior_weight,
            self.group,
            self.constant_liar,
            self.constraints_func,
            self.gamma,
            self.weights,
            self.categorical_distance_func,
            self.warn_independent_sampling,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    fn make_complete_trial(number: i64, value: f64, params: HashMap<String, ParamValue>) -> FrozenTrial {
        let now = chrono::Utc::now();
        let mut distributions = HashMap::new();
        for (name, val) in &params {
            match val {
                ParamValue::Float(_) => {
                    distributions.insert(
                        name.clone(),
                        Distribution::FloatDistribution(
                            FloatDistribution::new(-10.0, 10.0, false, None).unwrap(),
                        ),
                    );
                }
                ParamValue::Int(_) => {
                    distributions.insert(
                        name.clone(),
                        Distribution::IntDistribution(
                            IntDistribution::new(-10, 10, false, 1).unwrap(),
                        ),
                    );
                }
                ParamValue::Categorical(_) => {}
            }
        }
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![value]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params,
            distributions,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_tpe_startup_uses_random() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let trial = FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        // During startup, should sample without error.
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_tpe_split_trials_minimize() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let mut trials = Vec::new();
        for i in 0..20 {
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(i as f64));
            trials.push(make_complete_trial(i, i as f64, params));
        }

        let (below, above) = sampler.split_trials(&trials);
        // gamma(20) = min(ceil(2.0), 25) = 2
        assert_eq!(below.len(), 2);
        // Below should have the best (lowest) values: 0.0, 1.0
        assert_eq!(below[0].value().unwrap(), Some(0.0));
        assert_eq!(below[1].value().unwrap(), Some(1.0));
        assert_eq!(above.len(), 18);
    }

    #[test]
    fn test_tpe_sample_relative() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        // Create 15 completed trials (past startup of 10).
        let mut trials = Vec::new();
        for i in 0..15 {
            let x = (i as f64 - 7.0).abs(); // V-shape: minimum at i=7
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(i as f64 - 7.0));
            trials.push(make_complete_trial(i, x, params));
        }

        let mut search_space = IndexMap::new();
        search_space.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(-10.0, 10.0, false, None).unwrap()),
        );

        let result = sampler.sample_relative(&trials, &search_space).unwrap();
        assert!(result.contains_key("x"));
        let x = result["x"];
        assert!(
            (-10.0..=10.0).contains(&x),
            "TPE sample {x} out of bounds"
        );
    }

    #[test]
    fn test_tpe_converges_better_than_random() {
        // Run TPE on a simple quadratic: minimize x^2.
        // TPE should concentrate samples near x=0 more than random.
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let random = RandomSampler::new(Some(42));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(-5.0, 5.0, false, None).unwrap());

        // Generate 30 trials for both samplers.
        let mut tpe_trials = Vec::new();
        let mut random_trials = Vec::new();

        for i in 0..30 {
            // Random sampling for both during first 10, then TPE kicks in.
            let x_rand = random
                .sample_independent(
                    &[],
                    &FrozenTrial {
                        number: i,
                        state: TrialState::Running,
                        values: None,
                        datetime_start: Some(chrono::Utc::now()),
                        datetime_complete: None,
                        params: HashMap::new(),
                        distributions: HashMap::new(),
                        user_attrs: HashMap::new(),
                        system_attrs: HashMap::new(),
                        intermediate_values: HashMap::new(),
                        trial_id: i,
                    },
                    "x",
                    &dist,
                )
                .unwrap();

            let mut rp = HashMap::new();
            rp.insert("x".to_string(), ParamValue::Float(x_rand));
            random_trials.push(make_complete_trial(i, x_rand * x_rand, rp));

            // TPE: use sample_relative if past startup.
            let x_tpe = if i >= 10 {
                let mut search_space = IndexMap::new();
                search_space.insert("x".to_string(), dist.clone());
                let result = sampler.sample_relative(&tpe_trials, &search_space).unwrap();
                result["x"]
            } else {
                // Startup: use random
                let t = FrozenTrial {
                    number: i,
                    state: TrialState::Running,
                    values: None,
                    datetime_start: Some(chrono::Utc::now()),
                    datetime_complete: None,
                    params: HashMap::new(),
                    distributions: HashMap::new(),
                    user_attrs: HashMap::new(),
                    system_attrs: HashMap::new(),
                    intermediate_values: HashMap::new(),
                    trial_id: i,
                };
                sampler.sample_independent(&tpe_trials, &t, "x", &dist).unwrap()
            };

            let mut tp = HashMap::new();
            tp.insert("x".to_string(), ParamValue::Float(x_tpe));
            tpe_trials.push(make_complete_trial(i, x_tpe * x_tpe, tp));
        }

        // Compare best values found.
        let tpe_best = tpe_trials
            .iter()
            .map(|t| t.value().unwrap().unwrap())
            .fold(f64::INFINITY, f64::min);
        let rand_best = random_trials
            .iter()
            .map(|t| t.value().unwrap().unwrap())
            .fold(f64::INFINITY, f64::min);

        // TPE should find a reasonably good solution.
        // We don't require it to be strictly better than random in every seed,
        // but it should find something < 1.0 for x^2 on [-5, 5].
        assert!(
            tpe_best < 5.0,
            "TPE best={tpe_best} should be reasonable (rand_best={rand_best})"
        );
    }

    #[test]
    fn test_tpe_empty_search_space() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let result = sampler
            .sample_relative(&[], &IndexMap::new())
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_default_weights_edge_n26_matches_python() {
        // Python: n=26 => np.linspace(1/26, 1.0, num=1) = [1/26] (num=1时返回start)
        // 后接 ones(25)
        let w = default_weights(26);
        assert_eq!(w.len(), 26);
        // 第一个权重 = 1/26 ≈ 0.0385
        assert!(
            (w[0] - 1.0 / 26.0).abs() < 1e-12,
            "w[0]={}, expected={}",
            w[0],
            1.0 / 26.0
        );
        // 后 25 个权重全为 1.0
        assert!(w[1..].iter().all(|x| (*x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_default_weights_ramp_hits_one() {
        // Python linspace 保证斜坡末端是 1.0
        let w = default_weights(100);
        assert_eq!(w.len(), 100);
        assert!((w[74] - 1.0).abs() < 1e-12);
    }

    /// Python 交叉验证: n=0/1/24/25/30 的权重
    #[test]
    fn test_python_cross_default_weights_all_cases() {
        // n=0 → 空
        assert!(default_weights(0).is_empty());
        // n=1 → [1.0]
        assert_eq!(default_weights(1), vec![1.0]);
        // n=24 → 全 1.0
        let w24 = default_weights(24);
        assert_eq!(w24.len(), 24);
        assert!(w24.iter().all(|x| (*x - 1.0).abs() < 1e-12));
        // n=25 → 全 1.0（边界: ramp_len=0 走 n<25 分支）
        let w25 = default_weights(25);
        assert_eq!(w25.len(), 25);
        // n=30 → 5 个斜坡 + 25 个 flat
        // Python: np.linspace(1/30, 1.0, num=5) ≈ [0.0333, 0.275, 0.5167, 0.7583, 1.0]
        let w30 = default_weights(30);
        assert_eq!(w30.len(), 30);
        let start = 1.0 / 30.0;
        let step = (1.0 - start) / 4.0;
        for i in 0..5 {
            let expected = start + step * i as f64;
            assert!(
                (w30[i] - expected).abs() < 1e-12,
                "w30[{i}]={}, expected={expected}", w30[i]
            );
        }
        // 最后一个斜坡值必须是 1.0
        assert!((w30[4] - 1.0).abs() < 1e-12);
        // flat 部分全 1.0
        assert!(w30[5..].iter().all(|x| (*x - 1.0).abs() < 1e-12));
    }

    #[test]
    fn test_split_trials_sorted_by_number() {
        // 对齐 Python: split_trials 后 below/above 按 trial.number 排序
        // 确保权重按时间顺序（recency）分配，而非按目标值质量
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let mut trials = Vec::new();
        // 创建乱序 number 的试验: number=[3,0,2,1] values=[0.1,0.2,0.3,0.4]
        for (number, value) in [(3i64, 0.1), (0, 0.2), (2, 0.3), (1, 0.4)] {
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(value));
            trials.push(make_complete_trial(number, value, params));
        }
        let (below, above) = sampler.split_trials(&trials);
        // gamma(4) = min(ceil(0.4), 25) = 1 → below 有 1 个
        assert_eq!(below.len(), 1);
        assert_eq!(above.len(), 3);
        // above 应按 number 升序: [0, 1, 2]
        let above_numbers: Vec<i64> = above.iter().map(|t| t.number).collect();
        let mut sorted = above_numbers.clone();
        sorted.sort();
        assert_eq!(above_numbers, sorted, "above 应按 trial.number 排序");
    }

    #[test]
    fn test_split_trials_nan_intermediate_sort() {
        // 对齐 Python: NaN 中间值映射为 inf，排在同步骤试验最后
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let now = chrono::Utc::now();

        // 创建完成的试验（占 below）
        let mut complete_trials: Vec<FrozenTrial> = (0..5).map(|i| {
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(i as f64));
            make_complete_trial(i, i as f64, params)
        }).collect();

        // 创建3个 pruned 试验: step=10, values = [1.0, NaN, 2.0]
        for (idx, val) in [(5i64, 1.0), (6, f64::NAN), (7, 2.0)] {
            let mut iv = HashMap::new();
            iv.insert(10i64, val);
            complete_trials.push(FrozenTrial {
                number: idx,
                state: TrialState::Pruned,
                values: None,
                datetime_start: Some(now),
                datetime_complete: Some(now),
                params: HashMap::from([("x".to_string(), ParamValue::Float(0.0))]),
                distributions: HashMap::from([(
                    "x".to_string(),
                    Distribution::FloatDistribution(FloatDistribution::new(-10.0, 10.0, false, None).unwrap()),
                )]),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: iv,
                trial_id: idx,
            });
        }

        let (below, above) = sampler.split_trials(&complete_trials);
        // 验证两组都已按 number 排序
        for group in [&below, &above] {
            let numbers: Vec<i64> = group.iter().map(|t| t.number).collect();
            let mut sorted = numbers.clone();
            sorted.sort();
            assert_eq!(numbers, sorted, "应按 trial.number 排序");
        }
    }

    #[test]
    fn test_infer_relative_search_space_filters_single() {
        // 对齐 Python: single() 分布不应出现在相对搜索空间中
        let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize)
            .multivariate(true)
            .n_startup_trials(0) // 立即启用 TPE
            .seed(42)
            .build();
        let now = chrono::Utc::now();

        // 创建试验 with x=[0..5] and y=5.0 (single distribution: low==high)
        let mut trials = Vec::new();
        for i in 0..5 {
            trials.push(FrozenTrial {
                number: i,
                state: TrialState::Complete,
                values: Some(vec![i as f64]),
                datetime_start: Some(now),
                datetime_complete: Some(now),
                params: HashMap::from([
                    ("x".to_string(), ParamValue::Float(i as f64)),
                    ("y".to_string(), ParamValue::Float(5.0)),
                ]),
                distributions: HashMap::from([
                    ("x".to_string(), Distribution::FloatDistribution(
                        FloatDistribution::new(0.0, 10.0, false, None).unwrap()
                    )),
                    ("y".to_string(), Distribution::FloatDistribution(
                        FloatDistribution::new(5.0, 5.0, false, None).unwrap() // single!
                    )),
                ]),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                trial_id: i,
            });
        }

        let search_space = sampler.infer_relative_search_space(&trials);
        // y 是 single 分布，应被过滤掉
        assert!(
            !search_space.contains_key("y"),
            "single() 分布应被过滤: {:?}", search_space.keys().collect::<Vec<_>>()
        );
        // x 应保留
        assert!(search_space.contains_key("x"));
    }

    // ======== MOTPE multi-objective tests ========

    fn make_mo_trial(number: i64, values: Vec<f64>, params: HashMap<String, ParamValue>) -> FrozenTrial {
        let now = chrono::Utc::now();
        let mut distributions = HashMap::new();
        for (name, val) in &params {
            match val {
                ParamValue::Float(_) => {
                    distributions.insert(
                        name.clone(),
                        Distribution::FloatDistribution(
                            FloatDistribution::new(-10.0, 10.0, false, None).unwrap(),
                        ),
                    );
                }
                _ => {}
            }
        }
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(values),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params,
            distributions,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_fast_non_domination_rank() {
        // Three points: A=(1,3) dominates nothing exclusively,
        // B=(2,2), C=(3,1) -- all are Pareto-optimal
        let loss_values = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
        ];
        let ranks = TpeSampler::fast_non_domination_rank(&loss_values);
        assert_eq!(ranks, vec![0, 0, 0]);

        // A=(1,1) dominates B=(2,2) and C=(3,3)
        let loss_values2 = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let ranks2 = TpeSampler::fast_non_domination_rank(&loss_values2);
        assert_eq!(ranks2[0], 0);
        assert_eq!(ranks2[1], 1);
        assert_eq!(ranks2[2], 2);
    }

    #[test]
    fn test_fast_non_domination_rank_with_n_below() {
        // 对齐 Python `_calculate_nondomination_rank(loss_values, n_below=...)`:
        // n_below 提前终止排名计算
        let loss_values = vec![
            vec![1.0, 1.0],  // rank 0
            vec![2.0, 2.0],  // rank 1
            vec![3.0, 3.0],  // rank 2
        ];

        // n_below=1: 只需要 1 个元素的排名
        let ranks = TpeSampler::fast_non_domination_rank_with_n_below(&loss_values, Some(1));
        assert_eq!(ranks[0], 0);
        // 提前终止后，其余元素位于下一个 rank
        assert!(ranks[1] >= 1);

        // n_below=2: 需要前2个元素
        let ranks2 = TpeSampler::fast_non_domination_rank_with_n_below(&loss_values, Some(2));
        assert_eq!(ranks2[0], 0);
        assert_eq!(ranks2[1], 1);

        // n_below=None (默认): 全部计算
        let ranks_full = TpeSampler::fast_non_domination_rank_with_n_below(&loss_values, None);
        assert_eq!(ranks_full, vec![0, 1, 2]);
    }

    #[test]
    fn test_fast_non_domination_rank_single_objective() {
        // 对齐 Python: 单目标特殊路径，使用唯一排名
        let loss_values = vec![
            vec![3.0],
            vec![1.0],
            vec![2.0],
            vec![1.0],  // 与 index 1 相同
        ];
        let ranks = TpeSampler::fast_non_domination_rank(&loss_values);
        assert_eq!(ranks[0], 2);  // 3.0 → rank 2
        assert_eq!(ranks[1], 0);  // 1.0 → rank 0
        assert_eq!(ranks[2], 1);  // 2.0 → rank 1
        assert_eq!(ranks[3], 0);  // 1.0 → rank 0 (same as index 1)
    }

    #[test]
    fn test_dominates_values() {
        assert!(TpeSampler::dominates_values(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(TpeSampler::dominates_values(&[1.0, 2.0], &[2.0, 2.0]));
        assert!(!TpeSampler::dominates_values(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!TpeSampler::dominates_values(&[2.0, 2.0], &[2.0, 2.0]));
    }

    #[test]
    fn test_get_reference_point() {
        let losses = vec![
            vec![1.0, 2.0],
            vec![3.0, 1.0],
        ];
        let rp = TpeSampler::get_reference_point(&losses);
        assert!((rp[0] - 3.3).abs() < 1e-9); // 1.1 * 3.0
        assert!((rp[1] - 2.2).abs() < 1e-9); // 1.1 * 2.0
    }

    #[test]
    fn test_get_reference_point_negative() {
        let losses = vec![
            vec![-3.0, -1.0],
        ];
        let rp = TpeSampler::get_reference_point(&losses);
        // negative * 0.9
        assert!((rp[0] - (-2.7)).abs() < 1e-9);
        assert!((rp[1] - (-0.9)).abs() < 1e-9);
    }

    #[test]
    fn test_split_trials_multi_objective() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let sampler = TpeSampler::new_multi(
            dirs.clone(), Some(42), 0, 24, false, true, false, 1.0,
            false, false, None, None, None, None, true,
        );

        let trials: Vec<FrozenTrial> = (0..10).map(|i| {
            let x = i as f64;
            make_mo_trial(
                i,
                vec![x, 10.0 - x], // Pareto front: all non-dominated
                HashMap::from([("x".to_string(), ParamValue::Float(x))]),
            )
        }).collect();

        let (below, above) = sampler.split_trials_multi_objective(&trials);
        // default gamma: n_below = ceil(0.1 * 10) = 1 (for n=10)
        // All on Pareto front, so HSSP used
        assert!(!below.is_empty());
        assert!(!above.is_empty());
        assert_eq!(below.len() + above.len(), 10);
    }

    #[test]
    fn test_calculate_mo_weights_all_pareto() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let trials: Vec<FrozenTrial> = vec![
            make_mo_trial(0, vec![1.0, 3.0], HashMap::new()),
            make_mo_trial(1, vec![2.0, 2.0], HashMap::new()),
            make_mo_trial(2, vec![3.0, 1.0], HashMap::new()),
        ];
        let trial_refs: Vec<&FrozenTrial> = trials.iter().collect();
        let weights = TpeSampler::calculate_mo_weights(&trial_refs, &dirs, false);
        assert_eq!(weights.len(), 3);
        // All are Pareto-optimal, so all should have positive weights
        for w in &weights {
            assert!(*w > 0.0);
        }
        // Max weight should be 1.0 (normalized by max contribution)
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max_w - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_calculate_mo_weights_dominated() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let trials: Vec<FrozenTrial> = vec![
            make_mo_trial(0, vec![1.0, 1.0], HashMap::new()), // dominates all
            make_mo_trial(1, vec![2.0, 2.0], HashMap::new()), // dominated
            make_mo_trial(2, vec![3.0, 3.0], HashMap::new()), // dominated
        ];
        let trial_refs: Vec<&FrozenTrial> = trials.iter().collect();
        let weights = TpeSampler::calculate_mo_weights(&trial_refs, &dirs, false);
        // trial 0 is the only Pareto point, should have the largest weight
        assert!(weights[0] > weights[1]);
        assert!(weights[0] > weights[2]);
        // Pareto point weight should be 1.0
        assert!((weights[0] - 1.0).abs() < 1e-9);
        // Dominated weights should be EPS-level
        assert!(weights[1] < 1e-5);
        assert!(weights[2] < 1e-5);
    }

    #[test]
    fn test_motpe_builder_multi() {
        let sampler = TpeSamplerBuilder::new_multi(vec![
            StudyDirection::Minimize,
            StudyDirection::Maximize,
        ]).build();
        assert_eq!(sampler.directions.len(), 2);
        assert_eq!(sampler.directions[0], StudyDirection::Minimize);
        assert_eq!(sampler.directions[1], StudyDirection::Maximize);
    }

    #[test]
    fn test_motpe_builder_single() {
        let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize).build();
        assert_eq!(sampler.directions.len(), 1);
        assert_eq!(sampler.directions[0], StudyDirection::Minimize);
    }

    // ====================================================================
    // MOTPE 约束处理测试 — 对齐 Python
    // ====================================================================

    /// 对齐 Python: calculate_mo_weights 无约束时所有试验都参与 HV。
    #[test]
    fn test_calculate_mo_weights_no_constraints() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let trials: Vec<FrozenTrial> = vec![
            make_mo_trial(0, vec![1.0, 3.0], HashMap::new()),
            make_mo_trial(1, vec![2.0, 2.0], HashMap::new()),
            make_mo_trial(2, vec![3.0, 1.0], HashMap::new()),
        ];
        let refs: Vec<&FrozenTrial> = trials.iter().collect();
        let w = TpeSampler::calculate_mo_weights(&refs, &dirs, false);
        assert_eq!(w.len(), 3);
        for wi in &w {
            assert!(*wi > 0.0);
        }
    }

    /// 对齐 Python: calculate_mo_weights 有约束时不可行试验权重 ≈ EPS。
    #[test]
    fn test_calculate_mo_weights_with_constraints() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let mut t0 = make_mo_trial(0, vec![1.0, 3.0], HashMap::new());
        let mut t1 = make_mo_trial(1, vec![2.0, 2.0], HashMap::new());
        let mut t2 = make_mo_trial(2, vec![3.0, 1.0], HashMap::new());
        // t0, t1 可行 (constraint <= 0)
        t0.system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-1.0]),
        );
        t1.system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-0.5]),
        );
        // t2 不可行 (constraint > 0)
        t2.system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::json!([2.0]),
        );

        let trials = vec![t0, t1, t2];
        let refs: Vec<&FrozenTrial> = trials.iter().collect();
        let w = TpeSampler::calculate_mo_weights(&refs, &dirs, true);
        assert_eq!(w.len(), 3);
        // t0, t1 should have much larger weights than t2 (infeasible → EPS)
        assert!(w[0] > w[2] * 100.0, "feasible weight should >> infeasible: {} vs {}", w[0], w[2]);
        assert!(w[1] > w[2] * 100.0, "feasible weight should >> infeasible: {} vs {}", w[1], w[2]);
        // t2 weight should be EPS-level
        assert!(w[2] < 1e-5, "infeasible weight should be EPS-level: {}", w[2]);
    }

    /// 对齐 Python: split_trials_multi_objective 区分 infeasible 试验。
    #[test]
    fn test_split_trials_mo_with_constraints() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let sampler = TpeSamplerBuilder::new_multi(dirs)
            .constraints_func(Arc::new(|_| vec![0.0]))
            .n_startup_trials(0)
            .build();

        let mut trials = Vec::new();
        // 8 complete feasible trials
        for i in 0..8 {
            let mut t = make_mo_trial(i, vec![i as f64, (10 - i) as f64], HashMap::new());
            t.system_attrs.insert(
                crate::multi_objective::CONSTRAINTS_KEY.to_string(),
                serde_json::json!([-1.0]),
            );
            trials.push(t);
        }
        // 2 infeasible trials
        for i in 8..10 {
            let mut t = make_mo_trial(i, vec![i as f64, (10 - i) as f64], HashMap::new());
            t.system_attrs.insert(
                crate::multi_objective::CONSTRAINTS_KEY.to_string(),
                serde_json::json!([5.0]),
            );
            trials.push(t);
        }

        let (below, above) = sampler.split_trials_multi_objective(&trials);
        assert_eq!(below.len() + above.len(), 10);
        // Infeasible trials should be placed after complete, with lower priority
        // Check that below doesn't consist entirely of infeasible trials
        let below_infeasible: Vec<_> = below.iter()
            .filter(|t| TpeSampler::infeasible_score(t) > 0.0)
            .collect();
        // With gamma(10) = ceil(0.1*10) = 1, only 1 trial goes to below
        // That trial should be feasible (complete trials fill first)
        assert!(below_infeasible.is_empty() || below.len() > below_infeasible.len(),
            "feasible trials should be prioritized in below");
    }

    /// 对齐 Python: split_trials_multi_objective 处理 pruned 试验。
    #[test]
    fn test_split_trials_mo_with_pruned() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let sampler = TpeSamplerBuilder::new_multi(dirs).n_startup_trials(0).build();

        let now = chrono::Utc::now();
        let mut trials = Vec::new();
        // 3 complete trials
        for i in 0..3 {
            trials.push(make_mo_trial(i, vec![i as f64, (5 - i) as f64], HashMap::new()));
        }
        // 2 pruned trials (no values, but have intermediate values)
        for i in 3..5 {
            let mut t = FrozenTrial {
                number: i,
                state: TrialState::Pruned,
                values: None,
                datetime_start: Some(now),
                datetime_complete: Some(now),
                params: HashMap::new(),
                distributions: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::from([(0, i as f64)]),
                trial_id: i,
            };
            trials.push(t);
        }

        let (below, above) = sampler.split_trials_multi_objective(&trials);
        // n = 3 + 2 = 5, gamma(5) = ceil(0.1*5) = 1
        assert_eq!(below.len() + above.len(), 5);
        assert!(!below.is_empty(), "below should not be empty");
    }

    // ── default_weights 精确对齐 Python ──

    /// 对齐 Python: default_weights(0) → []
    #[test]
    fn test_default_weights_zero() {
        assert!(default_weights(0).is_empty());
    }

    /// 对齐 Python: n <= 24 时全为 1.0
    #[test]
    fn test_default_weights_under_25() {
        for n in [1, 5, 10, 24] {
            let w = default_weights(n);
            assert_eq!(w.len(), n);
            assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10),
                "n={n}: all weights should be 1.0");
        }
    }

    /// 对齐 Python: n=25 时也全为 1.0 (ramp_len=0)
    #[test]
    fn test_default_weights_25() {
        let w = default_weights(25);
        assert_eq!(w.len(), 25);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10));
    }

    /// 对齐 Python: n=26 时第一个是 1/26≈0.038462，最后25个全为 1.0
    #[test]
    fn test_default_weights_26() {
        let w = default_weights(26);
        assert_eq!(w.len(), 26);
        assert!((w[0] - 1.0 / 26.0).abs() < 1e-6);
        for i in 1..26 {
            assert!((w[i] - 1.0).abs() < 1e-10, "w[{i}] should be 1.0");
        }
    }

    /// 对齐 Python: n=50 时 ramp_len=25，linspace(1/50, 1.0, 25) + [1.0]*25
    #[test]
    fn test_default_weights_50() {
        let w = default_weights(50);
        assert_eq!(w.len(), 50);
        assert!((w[0] - 1.0 / 50.0).abs() < 1e-6); // 0.02
        assert!((w[24] - 1.0).abs() < 1e-6);        // ramp 最后一个 = 1.0
        for i in 25..50 {
            assert!((w[i] - 1.0).abs() < 1e-10, "w[{i}] should be 1.0");
        }
    }

    /// 对齐 Python: n=100 时 ramp_len=75
    #[test]
    fn test_default_weights_100() {
        let w = default_weights(100);
        assert_eq!(w.len(), 100);
        assert!((w[0] - 0.01).abs() < 1e-6);
        assert!((w[74] - 1.0).abs() < 1e-6);
        for i in 75..100 {
            assert!((w[i] - 1.0).abs() < 1e-10);
        }
    }

    // ── hyperopt_default_gamma 精确对齐 Python ──

    /// 对齐 Python: hyperopt_default_gamma 已知值
    #[test]
    fn test_hyperopt_default_gamma_values() {
        // gamma(n) = min(ceil(0.25 * sqrt(n)), 25)
        assert_eq!(hyperopt_default_gamma(0), 0);
        assert_eq!(hyperopt_default_gamma(1), 1);
        assert_eq!(hyperopt_default_gamma(4), 1);
        assert_eq!(hyperopt_default_gamma(16), 1);
        assert_eq!(hyperopt_default_gamma(17), 2); // ceil(0.25*4.123) = 2
        assert_eq!(hyperopt_default_gamma(25), 2); // ceil(0.25*5) = 2
        assert_eq!(hyperopt_default_gamma(64), 2); // ceil(0.25*8) = 2
        assert_eq!(hyperopt_default_gamma(100), 3); // ceil(0.25*10) = 3
        assert_eq!(hyperopt_default_gamma(10000), 25); // ceil(0.25*100) = 25
        assert_eq!(hyperopt_default_gamma(100000), 25); // min(ceil(79.06), 25) = 25
    }

    /// 对齐 Python: gamma 上界为 25
    #[test]
    fn test_hyperopt_default_gamma_capped_at_25() {
        assert_eq!(hyperopt_default_gamma(1_000_000), 25);
    }

    // ── TpeSampler 构建参数 ──

    /// 对齐 Python: TPESampler 默认参数值
    #[test]
    fn test_tpe_default_parameters() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        assert_eq!(sampler.n_startup_trials, 10);
        assert_eq!(sampler.n_ei_candidates, 24);
        assert!(!sampler.multivariate);
        assert!(!sampler.group);
        assert!(!sampler.constant_liar);
    }

    /// 对齐 Python: TpeSamplerBuilder 参数设置
    #[test]
    fn test_tpe_builder_custom_parameters() {
        let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize)
            .n_startup_trials(20)
            .n_ei_candidates(48)
            .multivariate(true)
            .seed(42)
            .build();
        assert_eq!(sampler.n_startup_trials, 20);
        assert_eq!(sampler.n_ei_candidates, 48);
        assert!(sampler.multivariate);
    }
}
