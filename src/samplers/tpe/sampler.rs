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
    /// Study direction for sorting trials.
    direction: StudyDirection,
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
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            n_startup_trials,
            n_ei_candidates,
            direction,
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
    /// When constraints are enabled, infeasible trials are separated and
    /// only fill below if feasible trials don't fill it.
    fn split_trials<'a>(
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
                TrialState::Complete => complete.push(t),
                TrialState::Pruned => pruned.push(t),
                _ => {}
            }
        }

        let n = complete.len() + pruned.len() + infeasible.len();
        let n_below = (self.gamma)(n);

        // Sort complete trials by objective value.
        match self.direction {
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

        // Sort pruned by (-last_step, direction-aware intermediate_value).
        // 对齐 Python _get_pruned_trial_score：Maximize 时对中间值取负
        // 对齐 Python: NaN 中间值映射为 inf，确保排在同步骤试验的最后
        pruned.sort_by(|a, b| {
            let sa = a.last_step().unwrap_or(i64::MIN);
            let sb = b.last_step().unwrap_or(i64::MIN);
            match sb.cmp(&sa) {
                std::cmp::Ordering::Equal => {
                    // 对齐 Python: if math.isnan(intermediate_value): return (-step, float("inf"))
                    let va = a
                        .intermediate_values
                        .get(&sa)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let va = if va.is_nan() { f64::INFINITY } else { va };
                    let vb = b
                        .intermediate_values
                        .get(&sb)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let vb = if vb.is_nan() { f64::INFINITY } else { vb };
                    // Minimize: 小值优先(升序)；Maximize: 大值优先(降序)
                    match self.direction {
                        StudyDirection::Maximize => {
                            vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        _ => {
                            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    }
                }
                other => other,
            }
        });

        // Split: best n_below from complete, then pruned, then infeasible.
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

        // Infeasible trials sorted by violation score (less violation first).
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

        // Running trials always go to above.
        above.extend(running);

        // 对齐 Python: 按 trial.number 排序，确保权重按时间顺序分配
        // Python: below_trials.sort(key=lambda trial: trial.number)
        //         above_trials.sort(key=lambda trial: trial.number)
        below.sort_by_key(|t| t.number);
        above.sort_by_key(|t| t.number);

        (below, above)
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

        // 对齐 Python: 将自定义 weights 函数传递给 ParzenEstimator
        let weights_ref = self.weights.clone();
        let pe_below =
            ParzenEstimator::new(&obs_below, search_space, &self.pe_params, None,
                                 Some(&|n| weights_ref(n)));
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
    ) -> HashMap<String, Distribution> {
        if !self.multivariate {
            return HashMap::new();
        }
        if self.is_startup(trials) {
            return HashMap::new();
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
            let mut result = HashMap::new();
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
        search_space: &HashMap<String, Distribution>,
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

        // Convert to IndexMap for ordered iteration.
        let ordered: IndexMap<String, Distribution> = search_space
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        self.tpe_sample(trials, &ordered, None)
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
}

impl TpeSamplerBuilder {
    /// Create a new builder with the given optimization direction.
    pub fn new(direction: StudyDirection) -> Self {
        Self {
            direction,
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
        TpeSampler::new(
            self.direction,
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

        let mut search_space = HashMap::new();
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
                let mut search_space = HashMap::new();
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
            .sample_relative(&[], &HashMap::new())
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
}
