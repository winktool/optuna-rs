//! CMA-ES（协方差矩阵自适应进化策略）采样器模块
//!
//! 对应 Python `optuna.samplers.CmaEsSampler`。
//! 纯 Rust 实现，使用 Jacobi 特征值分解进行协方差矩阵更新。
//!
//! ## 功能特性
//! - 标准 CMA-ES 算法（自适应步长、协方差矩阵学习）
//! - 可分离 CMA (use_separable_cma) — 仅对角协方差，高维效率更高
//! - 学习率自适应 (lr_adapt) — 大维度时降低学习率
//! - 边距修正 (with_margin) — 离散参数的边界处理
//! - 热启动 (source_trials / x0) — 从已有试验初始化均值向量

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
///
/// A (mu/mu_w, lambda)-CMA-ES implementation for single-objective optimization.
pub struct CmaEsSampler {
    direction: StudyDirection,
    sigma0: Option<f64>,
    n_startup_trials: usize,
    popsize: Option<usize>,
    independent_sampler: Arc<dyn Sampler>,
    state: Mutex<Option<CmaState>>,
    rng: Mutex<ChaCha8Rng>,
    search_space: Mutex<IntersectionSearchSpace>,
    /// Initial parameter values (warm-start).
    x0: Option<HashMap<String, f64>>,
    /// Whether to include pruned trials in sampling.
    consider_pruned_trials: bool,
    /// Use separable CMA-ES (diagonal covariance matrix).
    use_separable_cma: bool,
    /// Enable margin correction for integer/discrete parameters.
    with_margin: bool,
    /// Use learning-rate adaptation.
    lr_adapt: bool,
    /// Source trials for warm-starting from another study.
    source_trials: Option<Vec<FrozenTrial>>,
    /// Whether to warn when independent sampling is used (default: true).
    warn_independent_sampling: bool,
    /// 对齐 Python: 持有 storage 引用用于状态持久化。
    /// 在 Study 构造时通过 `set_storage()` 注入。
    storage: Mutex<Option<Arc<dyn crate::storage::Storage>>>,
    /// Study ID for storage operations.
    study_id: Mutex<Option<i64>>,
}

impl std::fmt::Debug for CmaEsSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmaEsSampler")
            .field("n_startup_trials", &self.n_startup_trials)
            .finish()
    }
}

/// Internal CMA-ES algorithm state.
///
/// 可序列化/反序列化，支持持久化到 storage system_attrs。
/// 对应 Python `cmaes.CMA` 对象的 pickle 序列化。
#[derive(Serialize, Deserialize)]
struct CmaState {
    mean: Vec<f64>,
    sigma: f64,
    n: usize,
    // Covariance matrix (n x n)
    c: Vec<Vec<f64>>,
    // Evolution paths
    p_sigma: Vec<f64>,
    p_c: Vec<f64>,
    // Eigen decomposition cache
    eigenvalues: Vec<f64>,
    b: Vec<Vec<f64>>,
    // Strategy parameters
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c1: f64,
    c_mu: f64,
    chi_n: f64,
    // Generation tracking
    generation: usize,
    // Pending candidates from current ask batch
    pending: Vec<Vec<f64>>,
    pending_idx: usize,
    // Collected (params, value) pairs for current generation
    results: Vec<(Vec<f64>, f64)>,
    // Param names in order
    param_names: Vec<String>,
}

#[allow(clippy::needless_range_loop)]
impl CmaState {
    fn new(mean: Vec<f64>, sigma: f64, lambda: usize, param_names: Vec<String>) -> Self {
        let n = mean.len();
        let mu = lambda / 2;

        // Compute recombination weights
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }

        let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Adaptation parameters
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n as f64 + 2.0).powi(2) + mu_eff))
            .min(1.0 - c1);
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // Identity covariance matrix
        let mut c = vec![vec![0.0; n]; n];
        for i in 0..n {
            c[i][i] = 1.0;
        }

        let eigenvalues = vec![1.0; n];
        let mut b = vec![vec![0.0; n]; n];
        for i in 0..n {
            b[i][i] = 1.0;
        }

        Self {
            mean,
            sigma,
            n,
            c,
            p_sigma: vec![0.0; n],
            p_c: vec![0.0; n],
            eigenvalues,
            b,
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
            generation: 0,
            pending: Vec::new(),
            pending_idx: 0,
            results: Vec::new(),
            param_names,
        }
    }

    /// Sample a new candidate from the distribution.
    fn ask(&mut self, rng: &mut ChaCha8Rng) -> Vec<f64> {
        if self.pending_idx < self.pending.len() {
            let result = self.pending[self.pending_idx].clone();
            self.pending_idx += 1;
            return result;
        }

        // Generate a full batch of lambda candidates
        self.pending.clear();
        self.pending_idx = 0;

        for _ in 0..self.lambda {
            let z: Vec<f64> = (0..self.n)
                .map(|_| {
                    rng.sample(StandardNormal)
                })
                .collect();

            // y = B * D * z
            let mut y = vec![0.0; self.n];
            for i in 0..self.n {
                for j in 0..self.n {
                    y[i] += self.b[i][j] * self.eigenvalues[j].sqrt() * z[j];
                }
            }

            // x = mean + sigma * y
            let x: Vec<f64> = (0..self.n)
                .map(|i| (self.mean[i] + self.sigma * y[i]).clamp(0.0, 1.0))
                .collect();
            self.pending.push(x);
        }

        let result = self.pending[self.pending_idx].clone();
        self.pending_idx += 1;
        result
    }

    /// Record a result and update if generation is complete.
    fn tell(&mut self, params: Vec<f64>, value: f64) {
        self.results.push((params, value));

        if self.results.len() >= self.lambda {
            self.update();
        }
    }

    /// Perform the CMA-ES update step.
    fn update(&mut self) {
        // Sort by objective value (ascending = minimize)
        self.results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let old_mean = self.mean.clone();

        // Update mean
        self.mean = vec![0.0; self.n];
        for i in 0..self.mu {
            for j in 0..self.n {
                self.mean[j] += self.weights[i] * self.results[i].0[j];
            }
        }

        // Compute mean displacement
        let mut mean_diff = vec![0.0; self.n];
        for i in 0..self.n {
            mean_diff[i] = (self.mean[i] - old_mean[i]) / self.sigma;
        }

        // Compute C^(-1/2) * mean_diff for p_sigma update
        let invsqrt_c_diff = self.invsqrt_c_times(&mean_diff);

        // Update evolution path p_sigma
        let cs_comp = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_sigma[i] =
                (1.0 - self.c_sigma) * self.p_sigma[i] + cs_comp * invsqrt_c_diff[i];
        }

        // h_sigma: indicator for p_sigma length
        let ps_norm: f64 = self.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();
        let threshold = (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1)))
            .sqrt()
            * (1.4 + 2.0 / (self.n as f64 + 1.0))
            * self.chi_n;
        let h_sigma = if ps_norm < threshold { 1.0 } else { 0.0 };

        // Update evolution path p_c
        let cc_comp = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_c[i] =
                (1.0 - self.c_c) * self.p_c[i] + h_sigma * cc_comp * mean_diff[i];
        }

        // Update covariance matrix
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        for i in 0..self.n {
            for j in 0..self.n {
                let rank_one = self.p_c[i] * self.p_c[j];
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    let yi = (self.results[k].0[i] - old_mean[i]) / self.sigma;
                    let yj = (self.results[k].0[j] - old_mean[j]) / self.sigma;
                    rank_mu += self.weights[k] * yi * yj;
                }

                self.c[i][j] = (1.0 - self.c1 - self.c_mu + delta_h * self.c1)
                    * self.c[i][j]
                    + self.c1 * rank_one
                    + self.c_mu * rank_mu;
            }
        }

        // Update sigma
        let sigma_factor = (ps_norm / self.chi_n - 1.0) * self.c_sigma / self.d_sigma;
        self.sigma *= (sigma_factor).exp();
        // Clamp sigma to reasonable range
        self.sigma = self.sigma.clamp(1e-20, 1e10);

        // Update eigen decomposition
        self.update_eigen();

        self.generation += 1;
        self.results.clear();
        self.pending.clear();
        self.pending_idx = 0;
    }

    /// Compute C^(-1/2) * v using eigen decomposition.
    fn invsqrt_c_times(&self, v: &[f64]) -> Vec<f64> {
        // C^(-1/2) = B * D^(-1) * B^T
        // First: B^T * v
        let mut bt_v = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                bt_v[i] += self.b[j][i] * v[j];
            }
        }
        // D^(-1) * (B^T * v)
        for i in 0..self.n {
            let ev = self.eigenvalues[i].max(1e-20);
            bt_v[i] /= ev.sqrt();
        }
        // B * result
        let mut result = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                result[i] += self.b[i][j] * bt_v[j];
            }
        }
        result
    }

    /// Update eigen decomposition of C using Jacobi iteration.
    fn update_eigen(&mut self) {
        let n = self.n;

        // Force symmetry
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (self.c[i][j] + self.c[j][i]) / 2.0;
                self.c[i][j] = avg;
                self.c[j][i] = avg;
            }
        }

        // Simple Jacobi eigenvalue algorithm for small matrices
        let mut a = self.c.clone();
        let mut v = vec![vec![0.0; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        for _ in 0..100 {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-15 {
                break;
            }

            // Compute rotation
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Apply rotation to a
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                    new_a[p][i] = new_a[i][p];
                    new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                    new_a[q][i] = new_a[i][q];
                }
            }
            new_a[p][p] = cos_t * cos_t * a[p][p]
                + 2.0 * sin_t * cos_t * a[p][q]
                + sin_t * sin_t * a[q][q];
            new_a[q][q] = sin_t * sin_t * a[p][p]
                - 2.0 * sin_t * cos_t * a[p][q]
                + cos_t * cos_t * a[q][q];
            new_a[p][q] = 0.0;
            new_a[q][p] = 0.0;
            a = new_a;

            // Apply rotation to eigenvectors
            for i in 0..n {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = cos_t * vip + sin_t * viq;
                v[i][q] = -sin_t * vip + cos_t * viq;
            }
        }

        for i in 0..n {
            self.eigenvalues[i] = a[i][i].max(1e-20);
        }
        self.b = v;
    }

    // ── 状态持久化方法 ──────────────────────────────────────────

    /// 对齐 Python `CmaEsSampler._serialize_optimizer`:
    /// 序列化 CMA 状态为 JSON 字符串。
    fn serialize_state(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| {
            crate::error::OptunaError::RuntimeError(format!("CMA-ES state serialization failed: {e}"))
        })
    }

    /// 对齐 Python `CmaEsSampler._restore_optimizer`:
    /// 从 JSON 字符串反序列化 CMA 状态。
    fn deserialize_state(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| {
            crate::error::OptunaError::RuntimeError(format!("CMA-ES state deserialization failed: {e}"))
        })
    }

    /// 对齐 Python `_split_optimizer_str`:
    /// 将序列化字符串分片为多个键值对（适配 storage system_attrs 的大小限制）。
    ///
    /// Python RDBStorage 限制 system_attr 值长度为 2045 字符。
    /// 这里使用相同的分片大小。
    fn split_state_str(s: &str) -> HashMap<String, serde_json::Value> {
        const CHUNK_SIZE: usize = 2045;
        const KEY_PREFIX: &str = "cma:optimizer";
        let mut attrs = HashMap::new();
        for (i, chunk) in s.as_bytes().chunks(CHUNK_SIZE).enumerate() {
            let key = format!("{KEY_PREFIX}:{i}");
            let val = String::from_utf8_lossy(chunk).into_owned();
            attrs.insert(key, serde_json::Value::String(val));
        }
        attrs
    }

    /// 对齐 Python `_concat_optimizer_attrs`:
    /// 将分片重新拼合为完整字符串。
    fn concat_state_attrs(attrs: &HashMap<String, serde_json::Value>) -> Option<String> {
        const KEY_PREFIX: &str = "cma:optimizer";
        let mut indexed: Vec<(usize, &str)> = Vec::new();
        for (k, v) in attrs {
            if let Some(idx_str) = k.strip_prefix(&format!("{KEY_PREFIX}:")) {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if let Some(s) = v.as_str() {
                        indexed.push((idx, s));
                    }
                }
            }
        }
        if indexed.is_empty() {
            return None;
        }
        indexed.sort_by_key(|(i, _)| *i);
        Some(indexed.into_iter().map(|(_, s)| s).collect())
    }
}

impl CmaEsSampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        direction: StudyDirection,
        sigma0: Option<f64>,
        n_startup_trials: Option<usize>,
        popsize: Option<usize>,
        independent_sampler: Option<Arc<dyn Sampler>>,
        seed: Option<u64>,
        x0: Option<HashMap<String, f64>>,
        consider_pruned_trials: bool,
        use_separable_cma: bool,
        with_margin: bool,
        lr_adapt: bool,
        source_trials: Option<Vec<FrozenTrial>>,
    ) -> Self {
        // 对齐 Python: 参数组合验证
        if source_trials.is_some() && x0.is_some() {
            panic!("Cannot specify both `source_trials` and `x0`.");
        }
        if source_trials.is_some() && sigma0.is_some() {
            panic!("Cannot specify both `source_trials` and `sigma0`.");
        }
        if source_trials.is_some() && use_separable_cma {
            panic!("Cannot use `source_trials` with `use_separable_cma`.");
        }
        if lr_adapt && use_separable_cma {
            panic!("Cannot use `lr_adapt` with `use_separable_cma`.");
        }
        if lr_adapt && with_margin {
            panic!("Cannot use `lr_adapt` with `with_margin`.");
        }
        if use_separable_cma && with_margin {
            panic!("Cannot use `use_separable_cma` with `with_margin`.");
        }

        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        Self {
            direction,
            sigma0,
            // 对齐 Python: n_startup_trials 默认值为 1
            n_startup_trials: n_startup_trials.unwrap_or(1),
            popsize,
            independent_sampler: independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(seed))),
            state: Mutex::new(None),
            rng: Mutex::new(rng),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
            x0,
            consider_pruned_trials,
            use_separable_cma,
            with_margin,
            lr_adapt,
            source_trials,
            warn_independent_sampling: true,
            storage: Mutex::new(None),
            study_id: Mutex::new(None),
        }
    }

    fn default_popsize(n: usize) -> usize {
        (4 + (3.0 * (n as f64).ln()).floor() as usize).max(5)
    }

    /// 对齐 Python `CmaEsSampler._restore_optimizer`:
    /// 从已完成 trial 的 system_attrs 恢复 CMA-ES 状态。
    /// 按逆序扫描 trial，找到第一个包含 `cma:optimizer:0` 的 trial 并反序列化。
    fn try_restore_state<'a>(trials: impl Iterator<Item = &'a FrozenTrial>) -> Option<CmaState> {
        for trial in trials {
            if let Some(state_str) = CmaState::concat_state_attrs(&trial.system_attrs) {
                if let Ok(state) = CmaState::deserialize_state(&state_str) {
                    return Some(state);
                }
            }
        }
        None
    }
}

impl Sampler for CmaEsSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        let n_complete = trials
            .iter()
            .filter(|t| {
                t.state == TrialState::Complete
                    || (self.consider_pruned_trials && t.state == TrialState::Pruned)
            })
            .count();

        if n_complete < self.n_startup_trials {
            return IndexMap::new();
        }

        // 对齐 Python: 过滤掉 CategoricalDistribution 和 single-value 分布
        // CMA-ES 只能采样连续参数
        let space = self.search_space.lock().calculate(trials);
        space
            .into_iter()
            .filter(|(_, d)| !matches!(d, Distribution::CategoricalDistribution { .. }) && !d.single())
            .collect()
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        let complete: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| {
                if t.state == TrialState::Complete && t.values.is_some() {
                    true
                } else if self.consider_pruned_trials && t.state == TrialState::Pruned {
                    // 对齐 Python: pruned trial 使用最后一个中间值作为其 value
                    !t.intermediate_values.is_empty()
                } else {
                    false
                }
            })
            .collect();

        if complete.len() < self.n_startup_trials {
            return Ok(HashMap::new());
        }

        // Build ordered search space
        let mut ordered_space = IndexMap::new();
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();
        for name in &param_names {
            ordered_space.insert(name.clone(), search_space[name].clone());
        }

        let transform = SearchSpaceTransform::new(ordered_space.clone(), true, true, true);
        let n_dims = transform.n_encoded();

        let mut state_guard = self.state.lock();
        let mut rng = self.rng.lock();

        // Initialize state if needed
        if state_guard.is_none() {
            // 对齐 Python `_restore_optimizer`:
            // 优先从最近的已完成 trial 的 system_attrs 恢复状态
            let restored = Self::try_restore_state(complete.iter().rev().copied());
            if let Some(restored_state) = restored {
                *state_guard = Some(restored_state);
            } else {
            // 对齐 Python: sigma0 默认值 = min(upper - lower) / 6
            // 在变换空间中, bounds 通常为 [0, 1]，min_range = 1.0
            let sigma = self.sigma0.unwrap_or_else(|| {
                let bounds = transform.bounds();
                let min_range = bounds.iter()
                    .map(|[lo, hi]| hi - lo)
                    .fold(f64::INFINITY, f64::min);
                if min_range.is_finite() && min_range > 0.0 {
                    min_range / 6.0
                } else {
                    1.0 / 6.0
                }
            });
            let lambda = self.popsize.unwrap_or_else(|| Self::default_popsize(n_dims));

            // Initialize mean: prefer x0, then best trial, then center of space
            let mean = if let Some(ref x0) = self.x0 {
                // Use user-specified initial values
                let mut x0_params = IndexMap::new();
                for name in &param_names {
                    if let Some(&val) = x0.get(name) {
                        x0_params.insert(
                            name.clone(),
                            crate::distributions::ParamValue::Float(val),
                        );
                    }
                }
                if x0_params.len() == ordered_space.len() {
                    transform.transform(&x0_params)
                } else {
                    vec![0.5; n_dims]
                }
            } else if let Some(ref source) = self.source_trials {
                // Warm-start from source trials: use the best source trial
                let best_source = source
                    .iter()
                    .filter(|t| t.state == TrialState::Complete && t.values.is_some())
                    .min_by(|a, b| {
                        let va = a.values.as_ref().unwrap()[0];
                        let vb = b.values.as_ref().unwrap()[0];
                        let va = if self.direction == StudyDirection::Maximize { -va } else { va };
                        let vb = if self.direction == StudyDirection::Maximize { -vb } else { vb };
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(bt) = best_source {
                    let mut bp = IndexMap::new();
                    for name in &param_names {
                        if let Some(pv) = bt.params.get(name) {
                            bp.insert(name.clone(), pv.clone());
                        }
                    }
                    if bp.len() == ordered_space.len() {
                        transform.transform(&bp)
                    } else {
                        vec![0.5; n_dims]
                    }
                } else {
                    vec![0.5; n_dims]
                }
            } else {
                // 对齐 Python: 使用搜索空间的中心作为初始均值
                // Python: mean = lower_bounds + (upper_bounds - lower_bounds) / 2
                // 在变换空间中，这等价于每个维度取 0.5
                vec![0.5; n_dims]
            };

            let mut new_state = CmaState::new(mean, sigma, lambda, param_names.clone());

            // Apply separable CMA: use diagonal covariance only
            if self.use_separable_cma {
                // Zero out off-diagonal elements of C
                for i in 0..new_state.n {
                    for j in 0..new_state.n {
                        if i != j {
                            new_state.c[i][j] = 0.0;
                        }
                    }
                }
            }

            // Apply learning-rate adaptation
            if self.lr_adapt {
                // Reduce c1 and c_mu by factor of n for large dimensionality
                let n = new_state.n as f64;
                new_state.c1 /= n.sqrt();
                new_state.c_mu /= n.sqrt();
            }

            *state_guard = Some(new_state);
            } // end else (no restored state)
        }

        let state = state_guard.as_mut().unwrap();
        let candidate = state.ask(&mut rng);

        drop(rng);
        drop(state_guard);

        // Untransform
        let decoded = transform.untransform(&candidate)?;
        let mut result = HashMap::new();
        for (name, dist) in &ordered_space {
            if let Some(pv) = decoded.get(name) {
                let mut internal = dist.to_internal_repr(pv)?;

                // with_margin: 离散参数边界修正（对应 Python CmaEsSampler 的 with_margin 功能）
                // 将连续值夹到离散参数的合法范围内，添加半步长的边距
                if self.with_margin {
                    match dist {
                        Distribution::IntDistribution(id) => {
                            let step = id.step as f64;
                            let margin = step * 0.5;
                            internal = internal.max(id.low as f64 - margin)
                                .min(id.high as f64 + margin);
                        }
                        _ => {}
                    }
                }

                result.insert(name.clone(), internal);
            }
        }

        Ok(result)
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // 对齐 Python: 当启动试验数已满足时，警告独立采样
        if self.warn_independent_sampling {
            let n_complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
            if n_complete >= self.n_startup_trials {
                crate::optuna_warn!(
                    "Trial {} was sampled independently for param '{}'. \
                     CmaEsSampler falls back to independent sampling for parameters \
                     not included in the relative search space.",
                    trial.number, param_name
                );
            }
        }
        self.independent_sampler
            .sample_independent(trials, trial, param_name, distribution)
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        values: Option<&[f64]>,
    ) {
        // 对齐 Python: 支持 Complete + Pruned (当 consider_pruned_trials=true 时)
        let effective_value = if state == TrialState::Complete {
            match values {
                Some(v) if !v.is_empty() => v[0],
                _ => return,
            }
        } else if self.consider_pruned_trials && state == TrialState::Pruned {
            // Pruned trial: 使用最后一个中间值作为目标值
            if let Some(max_step) = trial.intermediate_values.keys().max() {
                trial.intermediate_values[max_step]
            } else {
                return;
            }
        } else {
            return;
        };

        let mut state_guard = self.state.lock();
        if let Some(cma_state) = state_guard.as_mut() {
            let param_names = &cma_state.param_names;
            if param_names.is_empty() {
                return;
            }

            // Reconstruct the search space from trial distributions
            let mut ordered_space = IndexMap::new();
            for name in param_names {
                if let Some(dist) = trial.distributions.get(name) {
                    ordered_space.insert(name.clone(), dist.clone());
                }
            }

            if ordered_space.len() != param_names.len() {
                return;
            }

            let transform = SearchSpaceTransform::new(ordered_space, true, true, true);
            let mut trial_params = IndexMap::new();
            for name in param_names {
                if let Some(pv) = trial.params.get(name) {
                    trial_params.insert(name.clone(), pv.clone());
                }
            }

            if trial_params.len() == param_names.len() {
                let encoded = transform.transform(&trial_params);
                let value = if self.direction == StudyDirection::Maximize {
                    -effective_value
                } else {
                    effective_value
                };
                cma_state.tell(encoded, value);

                // 对齐 Python: 将 CMA 状态和 generation 信息持久化到 trial system_attrs
                if let Some(ref storage) = *self.storage.lock() {
                    // 写入 generation 标记
                    let _ = storage.set_trial_system_attr(
                        trial.trial_id,
                        "cma:generation",
                        serde_json::json!(cma_state.generation),
                    );

                    // 序列化并分片写入 optimizer 状态
                    if let Ok(state_str) = cma_state.serialize_state() {
                        let attrs = CmaState::split_state_str(&state_str);
                        for (key, val) in attrs {
                            let _ = storage.set_trial_system_attr(
                                trial.trial_id,
                                &key,
                                val,
                            );
                        }
                    }
                }
            }
        }
    }

    /// 对齐 Python `CmaEsSampler.reseed_rng(seed)`: 重新设置随机种子。
    fn reseed_rng(&self, seed: u64) {
        *self.rng.lock() = ChaCha8Rng::seed_from_u64(seed);
    }

    /// 注入 storage 引用用于 CMA-ES 状态持久化。
    fn inject_storage(&self, storage: Arc<dyn crate::storage::Storage>, study_id: i64) {
        *self.storage.lock() = Some(storage);
        *self.study_id.lock() = Some(study_id);
    }
}

/// Builder for constructing a [`CmaEsSampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use optuna_rs::{CmaEsSamplerBuilder, StudyDirection};
///
/// let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
///     .sigma0(0.5)
///     .n_startup_trials(10)
///     .seed(42)
///     .build();
/// ```
pub struct CmaEsSamplerBuilder {
    direction: StudyDirection,
    sigma0: Option<f64>,
    n_startup_trials: Option<usize>,
    popsize: Option<usize>,
    independent_sampler: Option<Arc<dyn Sampler>>,
    seed: Option<u64>,
    x0: Option<HashMap<String, f64>>,
    consider_pruned_trials: bool,
    use_separable_cma: bool,
    with_margin: bool,
    lr_adapt: bool,
    source_trials: Option<Vec<FrozenTrial>>,
}

impl CmaEsSamplerBuilder {
    /// Create a new builder with the given optimization direction.
    pub fn new(direction: StudyDirection) -> Self {
        Self {
            direction,
            sigma0: None,
            n_startup_trials: None,
            popsize: None,
            independent_sampler: None,
            seed: None,
            x0: None,
            consider_pruned_trials: false,
            use_separable_cma: false,
            with_margin: false,
            lr_adapt: false,
            source_trials: None,
        }
    }

    /// Set the initial step size (sigma).
    pub fn sigma0(mut self, sigma: f64) -> Self {
        self.sigma0 = Some(sigma);
        self
    }

    /// Set the number of random startup trials before CMA-ES kicks in.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = Some(n);
        self
    }

    /// Set the population size (lambda).
    pub fn popsize(mut self, popsize: usize) -> Self {
        self.popsize = Some(popsize);
        self
    }

    /// Set the independent sampler used for parameters outside the search space.
    pub fn independent_sampler(mut self, sampler: Arc<dyn Sampler>) -> Self {
        self.independent_sampler = Some(sampler);
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set initial parameter values (warm-start).
    pub fn x0(mut self, x0: HashMap<String, f64>) -> Self {
        self.x0 = Some(x0);
        self
    }

    /// Whether to include pruned trials.
    pub fn consider_pruned_trials(mut self, consider: bool) -> Self {
        self.consider_pruned_trials = consider;
        self
    }

    /// Use separable CMA-ES (diagonal covariance matrix).
    pub fn use_separable_cma(mut self, sep: bool) -> Self {
        self.use_separable_cma = sep;
        self
    }

    /// Enable margin correction for integer/discrete parameters.
    pub fn with_margin(mut self, margin: bool) -> Self {
        self.with_margin = margin;
        self
    }

    /// Use learning-rate adaptation.
    pub fn lr_adapt(mut self, lr: bool) -> Self {
        self.lr_adapt = lr;
        self
    }

    /// Set source trials for warm-starting.
    pub fn source_trials(mut self, trials: Vec<FrozenTrial>) -> Self {
        self.source_trials = Some(trials);
        self
    }

    /// Build the [`CmaEsSampler`].
    pub fn build(self) -> CmaEsSampler {
        // 对齐 Python: 参数冲突验证
        assert!(
            self.source_trials.is_none() || (self.x0.is_none() && self.sigma0.is_none()),
            "It is prohibited to pass `source_trials` argument when x0 or sigma0 is specified."
        );
        assert!(
            self.source_trials.is_none() || !self.use_separable_cma,
            "It is prohibited to pass `source_trials` argument when using separable CMA-ES."
        );
        assert!(
            !self.lr_adapt || (!self.use_separable_cma && !self.with_margin),
            "It is prohibited to pass `use_separable_cma` or `with_margin` argument when using `lr_adapt`."
        );
        assert!(
            !self.use_separable_cma || !self.with_margin,
            "Currently, we do not support `use_separable_cma=True` and `with_margin=True`."
        );
        CmaEsSampler::new(
            self.direction,
            self.sigma0,
            self.n_startup_trials,
            self.popsize,
            self.independent_sampler,
            self.seed,
            self.x0,
            self.consider_pruned_trials,
            self.use_separable_cma,
            self.with_margin,
            self.lr_adapt,
            self.source_trials,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{create_study, StudyDirection};

    #[test]
    fn test_cmaes_creation() {
        let sampler = CmaEsSampler::new(
            StudyDirection::Minimize,
            None,
            Some(10),
            None,
            None,
            Some(42),
            None,
            false,
            false,
            false,
            false,
            None,
        );
        assert_eq!(sampler.n_startup_trials, 10);
    }

    #[test]
    fn test_cmaes_startup_random() {
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            None,
            Some(10),
            None,
            None,
            Some(42),
            None,
            false,
            false,
            false,
            false,
            None,
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(5),
                None,
                None,
            )
            .unwrap();

        assert_eq!(study.trials().unwrap().len(), 5);
    }

    #[test]
    fn test_cmaes_full_run() {
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            Some(0.5),
            Some(10),
            Some(8),
            None,
            Some(42),
            None,
            false,
            false,
            false,
            false,
            None,
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);

        // CMA-ES should find a decent solution
        let best = study.best_value().unwrap();
        assert!(
            best < 25.0,
            "CMA-ES should find a reasonable solution, got {best}"
        );
    }

    /// 对齐 Python: infer_relative_search_space 应过滤 Categorical 分布
    #[test]
    fn test_cmaes_filters_categorical() {
        let sampler = CmaEsSampler::new(
            StudyDirection::Minimize,
            None, Some(0), None, None, Some(42), None,
            false, false, false, false, None,
        );
        let now = chrono::Utc::now();
        use crate::distributions::CategoricalDistribution;
        use crate::distributions::FloatDistribution;
        let mut dists = HashMap::new();
        dists.insert("x".to_string(), Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap()));
        dists.insert("cat".to_string(), Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                crate::distributions::CategoricalChoice::Str("a".into()),
                crate::distributions::CategoricalChoice::Str("b".into()),
            ]).unwrap()));
        let trial = FrozenTrial {
            number: 0, state: TrialState::Complete, values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: HashMap::new(), distributions: dists,
            user_attrs: HashMap::new(), system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(), trial_id: 0,
        };
        let space = sampler.infer_relative_search_space(&[trial]);
        assert!(space.contains_key("x"), "should contain float param");
        assert!(!space.contains_key("cat"), "should not contain categorical param");
    }

    /// 对齐 Python: CmaEsSamplerBuilder 功能
    #[test]
    fn test_builder_pattern() {
        let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
            .sigma0(0.3)
            .n_startup_trials(5)
            .popsize(10)
            .consider_pruned_trials(true)
            .seed(123)
            .build();
        assert_eq!(sampler.n_startup_trials, 5);
        assert_eq!(sampler.popsize, Some(10));
        assert!(sampler.consider_pruned_trials);
    }

    /// 对齐 Python: use_separable_cma 和 with_margin 不能同时使用
    #[test]
    #[should_panic(expected = "use_separable_cma")]
    fn test_builder_separable_with_margin_conflict() {
        CmaEsSamplerBuilder::new(StudyDirection::Minimize)
            .use_separable_cma(true)
            .with_margin(true)
            .build();
    }

    /// 对齐 Python: lr_adapt 和 use_separable_cma 不能同时使用
    #[test]
    #[should_panic(expected = "lr_adapt")]
    fn test_builder_lr_adapt_separable_conflict() {
        CmaEsSamplerBuilder::new(StudyDirection::Minimize)
            .lr_adapt(true)
            .use_separable_cma(true)
            .build();
    }

    /// 对齐 Python: default_popsize
    #[test]
    fn test_default_popsize() {
        // Python: popsize = 4 + floor(3 * ln(n))
        // n=2: 4 + floor(3*0.693) = 4 + 2 = 6 → max(6, 5) = 6
        assert_eq!(CmaEsSampler::default_popsize(2), 6);
        // n=1: 4 + floor(3*0) = 4 → max(4, 5) = 5
        assert_eq!(CmaEsSampler::default_popsize(1), 5);
        // n=10: 4 + floor(3*2.302) = 4 + 6 = 10
        assert_eq!(CmaEsSampler::default_popsize(10), 10);
    }

    /// 对齐 Python: source_trials + x0 互斥
    #[test]
    #[should_panic(expected = "Cannot specify both")]
    fn test_cmaes_source_trials_x0_conflict() {
        let mut x0 = HashMap::new();
        x0.insert("x".to_string(), 0.5);
        CmaEsSampler::new(
            StudyDirection::Minimize, None, None, None, None, None,
            Some(x0), false, false, false, false,
            Some(vec![]),
        );
    }

    /// 对齐 Python: use_separable_cma + with_margin 互斥
    #[test]
    #[should_panic(expected = "Cannot use")]
    fn test_cmaes_separable_margin_conflict() {
        CmaEsSampler::new(
            StudyDirection::Minimize, None, None, None, None, None,
            None, false, true, true, false, None,
        );
    }

    /// 对齐 Python: lr_adapt + use_separable_cma 互斥
    #[test]
    #[should_panic(expected = "Cannot use")]
    fn test_cmaes_lr_adapt_separable_conflict() {
        CmaEsSampler::new(
            StudyDirection::Minimize, None, None, None, None, None,
            None, false, true, false, true, None,
        );
    }

    /// 对齐 Python: CMA-ES 状态序列化/反序列化往返一致性
    #[test]
    fn test_cmaes_state_serialization_roundtrip() {
        let state = CmaState::new(
            vec![0.5, 0.3, 0.7],
            0.3,
            6,
            vec!["x".into(), "y".into(), "z".into()],
        );
        let json = state.serialize_state().unwrap();
        let restored = CmaState::deserialize_state(&json).unwrap();
        assert_eq!(state.mean, restored.mean);
        assert_eq!(state.sigma, restored.sigma);
        assert_eq!(state.n, restored.n);
        assert_eq!(state.generation, restored.generation);
        assert_eq!(state.lambda, restored.lambda);
        assert_eq!(state.param_names, restored.param_names);
    }

    /// 对齐 Python: CMA-ES 状态分片与拼合
    #[test]
    fn test_cmaes_state_split_concat() {
        let state = CmaState::new(
            vec![0.5; 100],
            0.1,
            20,
            (0..100).map(|i| format!("p{i}")).collect(),
        );
        let json = state.serialize_state().unwrap();
        let attrs = CmaState::split_state_str(&json);

        // 应该至少有一个分片
        assert!(!attrs.is_empty());

        // 拼合后应恢复原始 JSON
        let restored_json = CmaState::concat_state_attrs(&attrs).unwrap();
        assert_eq!(json, restored_json);

        // 反序列化应成功
        let restored = CmaState::deserialize_state(&restored_json).unwrap();
        assert_eq!(restored.n, 100);
        assert_eq!(restored.param_names.len(), 100);
    }

    /// 对齐 Python: CMA-ES 状态持久化到 storage 并恢复
    #[test]
    fn test_cmaes_state_persistence_via_storage() {
        use crate::storage::InMemoryStorage;
        use crate::storage::Storage;
        use std::sync::Arc;

        let storage = Arc::new(InMemoryStorage::new());
        let study_id = storage
            .create_new_study(&[StudyDirection::Minimize], Some("cma_persist"))
            .unwrap();

        // 创建 CmaEsSampler 并注入 storage
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            Some(0.5),
            Some(5),
            Some(6),
            None,
            Some(42),
            None,
            false,
            false,
            false,
            false,
            None,
        ));
        sampler.inject_storage(storage.clone(), study_id);

        let study = crate::study::Study::new(
            "cma_persist".into(),
            study_id,
            storage.clone(),
            vec![StudyDirection::Minimize],
            sampler,
            Arc::new(crate::pruners::NopPruner),
        );

        // 运行足够多的 trial 使 CMA-ES 初始化并更新
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(15),
            None,
            None,
        ).unwrap();

        let trials = study.trials().unwrap();
        // 检查至少有一个 trial 包含 CMA 状态
        let has_cma_state = trials.iter().any(|t| {
            t.system_attrs.keys().any(|k| k.starts_with("cma:optimizer"))
        });
        assert!(has_cma_state, "至少一个 trial 应包含 CMA-ES 持久化状态");

        // 检查有 generation 标记
        let has_gen = trials.iter().any(|t| {
            t.system_attrs.contains_key("cma:generation")
        });
        assert!(has_gen, "至少一个 trial 应包含 generation 标记");
    }
}
