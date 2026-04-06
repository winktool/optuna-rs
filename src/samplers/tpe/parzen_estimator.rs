//! Parzen (KDE) estimator for TPE.
//!
//! Fits a mixture of kernels to observations, then can sample and compute log-PDF.
//! Port of Python optuna's `_ParzenEstimator`.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use rand::RngExt;

use crate::distributions::Distribution;

use super::truncnorm;

const EPS: f64 = 1e-12;

/// 分类参数距离函数类型。
/// 对应 Python `categorical_distance_func` 中的 value: (choice_a, choice_b) -> distance。
pub type CategoricalDistanceFn = Arc<dyn Fn(&str, usize, usize) -> f64 + Send + Sync>;

/// Parameters for the Parzen estimator.
pub struct ParzenEstimatorParameters {
    pub prior_weight: f64,
    pub consider_magic_clip: bool,
    pub consider_endpoints: bool,
    pub multivariate: bool,
    /// 分类参数距离函数映射: param_name -> distance_fn(choice_i, choice_j) -> dist。
    /// 对应 Python `_ParzenEstimatorParameters.categorical_distance_func`。
    pub categorical_distance_func: HashMap<String, CategoricalDistanceFn>,
}

impl std::fmt::Debug for ParzenEstimatorParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParzenEstimatorParameters")
            .field("prior_weight", &self.prior_weight)
            .field("consider_magic_clip", &self.consider_magic_clip)
            .field("consider_endpoints", &self.consider_endpoints)
            .field("multivariate", &self.multivariate)
            .field("categorical_distance_func_keys", &self.categorical_distance_func.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Clone for ParzenEstimatorParameters {
    fn clone(&self) -> Self {
        Self {
            prior_weight: self.prior_weight,
            consider_magic_clip: self.consider_magic_clip,
            consider_endpoints: self.consider_endpoints,
            multivariate: self.multivariate,
            categorical_distance_func: self.categorical_distance_func.clone(),
        }
    }
}

impl Default for ParzenEstimatorParameters {
    fn default() -> Self {
        Self {
            prior_weight: 1.0,
            consider_magic_clip: true,
            consider_endpoints: false,
            multivariate: false,
            categorical_distance_func: HashMap::new(),
        }
    }
}

/// A fitted kernel density estimator as a mixture model.
///
/// Supports numerical (truncated normal) and categorical parameters.
pub struct ParzenEstimator {
    /// Mixture component weights, length = n_kernels (n_obs + 1 for numerical).
    weights: Vec<f64>,
    /// Per-parameter distribution info.
    param_dists: Vec<ParamKernels>,
    /// Ordered parameter names matching param_dists.
    param_names: Vec<String>,
}

/// Kernel info for a single parameter dimension.
enum ParamKernels {
    /// Truncated normal kernels for continuous/discrete numerical params.
    Numerical {
        mus: Vec<f64>,    // kernel centers, len = n_kernels
        sigmas: Vec<f64>, // kernel widths, len = n_kernels
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>, // if Some, discrete rounding
    },
    /// Categorical kernels: weights[kernel][choice].
    Categorical {
        cat_weights: Vec<Vec<f64>>, // n_kernels × n_choices, row-normalized
        n_choices: usize,
    },
}

impl ParzenEstimator {
    /// 对齐 Python _call_weights_func: 验证权重向量有效性
    fn validate_weights(weights: &[f64]) {
        // 先检查 NaN/Inf（否则后续比较会因 NaN 语义产生误导性消息）
        assert!(
            weights.iter().all(|w| w.is_finite()),
            "Weights must be finite. Got {:?}",
            weights
        );
        // 检查负权重
        assert!(
            !weights.iter().any(|&w| w < 0.0),
            "Weights must be non-negative. Got {:?}",
            weights
        );
        // 检查全零
        assert!(
            weights.iter().any(|&w| w > 0.0),
            "At least one weight must be positive. Got {:?}",
            weights
        );
    }

    /// Build a Parzen estimator from observations and search space.
    ///
    /// `observations`: param_name → Vec<f64> of internal repr values.
    /// `search_space`: param_name → Distribution, same order (use IndexMap).
    /// `params`: estimator configuration.
    /// `predetermined_weights`: if Some, used instead of default_weights.
    /// `weights_func`: if Some, use this function instead of default_weights when
    ///   `predetermined_weights` is None. 对齐 Python `parameters.weights` 字段。
    pub fn new(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &IndexMap<String, Distribution>,
        params: &ParzenEstimatorParameters,
        predetermined_weights: Option<&[f64]>,
        weights_func: Option<&dyn Fn(usize) -> Vec<f64>>,
    ) -> Self {
        // 对齐 Python: prior_weight < 0 时 raise ValueError
        assert!(
            params.prior_weight >= 0.0,
            "A non-negative value must be specified for `prior_weight`. Got {}.",
            params.prior_weight
        );

        let n_obs = search_space
            .keys()
            .next()
            .and_then(|k| observations.get(k))
            .map(|v| v.len())
            .unwrap_or(0);

        // Compute mixture weights.
        // 对齐 Python: predetermined_weights → weights_func(parameters.weights) → default_weights
        let mut weights = if let Some(pw) = predetermined_weights {
            pw.to_vec()
        } else if let Some(wf) = weights_func {
            // 对齐 Python _call_weights_func: 调用自定义权重函数并截断到 n_obs
            let w = wf(n_obs);
            let w: Vec<f64> = w.into_iter().take(n_obs).collect();
            // 对齐 Python _call_weights_func: 验证权重有效性
            Self::validate_weights(&w);
            w
        } else {
            default_weights(n_obs)
        };

        // 对齐 Python: predetermined_weights 长度验证
        if let Some(pw) = predetermined_weights {
            assert_eq!(
                pw.len(),
                n_obs,
                "predetermined_weights length ({}) != observations length ({})",
                pw.len(),
                n_obs
            );
        }

        if weights.is_empty() {
            weights = vec![1.0];
        } else {
            weights.push(params.prior_weight);
        }

        // Normalize weights.
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        let n_kernels = weights.len();

        // Build per-parameter kernel distributions.
        let mut param_dists = Vec::with_capacity(search_space.len());
        let mut param_names = Vec::with_capacity(search_space.len());

        for (name, dist) in search_space {
            param_names.push(name.clone());
            let obs = observations.get(name).cloned().unwrap_or_default();
            let kernels = match dist {
                Distribution::CategoricalDistribution(cd) => {
                    Self::build_categorical_kernels(name, &obs, cd.choices.len(), n_kernels, params)
                }
                Distribution::FloatDistribution(fd) => Self::build_numerical_kernels(
                    &obs,
                    fd.low,
                    fd.high,
                    fd.log,
                    fd.step,
                    n_kernels,
                    search_space,
                    params,
                ),
                Distribution::IntDistribution(id) => Self::build_numerical_kernels(
                    &obs,
                    id.low as f64,
                    id.high as f64,
                    id.log,
                    Some(id.step as f64),
                    n_kernels,
                    search_space,
                    params,
                ),
            };
            param_dists.push(kernels);
        }

        Self {
            weights,
            param_dists,
            param_names,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_numerical_kernels(
        obs: &[f64],
        mut low: f64,
        mut high: f64,
        log: bool,
        step: Option<f64>,
        n_kernels: usize,
        search_space: &IndexMap<String, Distribution>,
        params: &ParzenEstimatorParameters,
    ) -> ParamKernels {
        // Extend bounds for discrete distributions.
        if let Some(s) = step {
            low -= s / 2.0;
            high += s / 2.0;
        }

        let mut mus: Vec<f64> = obs.to_vec();

        // Apply log transform.
        // For log distributions, low > 0 and all observations > 0 are guaranteed by
        // distribution validation, so no clamping is needed (matching Python's np.log).
        if log {
            mus = mus.iter().map(|&v| v.ln()).collect();
            low = low.ln();
            high = high.ln();
        }

        // Compute sigmas.
        let sigmas = if params.multivariate {
            let n_params = search_space.len();
            let sigma0_mag = 0.2;
            let sigma = sigma0_mag
                * (mus.len().max(1) as f64).powf(-1.0 / (n_params as f64 + 4.0))
                * (high - low);
            vec![sigma; mus.len()]
        } else {
            Self::compute_univariate_sigmas(&mus, low, high, params)
        };

        // Clip sigmas.
        let max_sigma = high - low;
        let min_sigma = if params.consider_magic_clip {
            (high - low) / (100.0_f64).min(1.0 + n_kernels as f64)
        } else {
            EPS
        };

        let mut clipped_sigmas: Vec<f64> = sigmas
            .iter()
            .map(|&s| s.clamp(min_sigma, max_sigma))
            .collect();

        // Append prior kernel: center at midpoint, sigma = full range.
        let prior_mu = 0.5 * (low + high);
        let prior_sigma = high - low;
        mus.push(prior_mu);
        clipped_sigmas.push(prior_sigma);

        ParamKernels::Numerical {
            mus,
            sigmas: clipped_sigmas,
            low,
            high,
            log,
            step,
        }
    }

    /// Compute univariate kernel bandwidths from sorted neighbor distances.
    fn compute_univariate_sigmas(
        mus: &[f64],
        low: f64,
        high: f64,
        params: &ParzenEstimatorParameters,
    ) -> Vec<f64> {
        if mus.is_empty() {
            return vec![];
        }

        // Include prior center in sorting.
        let prior_mu = 0.5 * (low + high);
        let mut mus_with_prior: Vec<f64> = mus.to_vec();
        mus_with_prior.push(prior_mu);

        // Sort and compute index mapping.
        let mut indices: Vec<usize> = (0..mus_with_prior.len()).collect();
        indices.sort_by(|&a, &b| {
            mus_with_prior[a]
                .partial_cmp(&mus_with_prior[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sorted_mus = vec![0.0; mus_with_prior.len()];
        for (rank, &orig_idx) in indices.iter().enumerate() {
            sorted_mus[rank] = mus_with_prior[orig_idx];
        }

        // Pad with low and high.
        let n = sorted_mus.len();
        let mut padded = Vec::with_capacity(n + 2);
        padded.push(low);
        padded.extend_from_slice(&sorted_mus);
        padded.push(high);

        // sigma[i] = max(gap_left, gap_right)
        let mut sorted_sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let gap_left = padded[i + 1] - padded[i];
            let gap_right = padded[i + 2] - padded[i + 1];
            sorted_sigmas.push(gap_left.max(gap_right));
        }

        // Consider endpoints: clip boundary sigmas to nearest interior neighbor.
        if !params.consider_endpoints && padded.len() >= 4 {
            sorted_sigmas[0] = padded[2] - padded[1];
            let last = sorted_sigmas.len() - 1;
            sorted_sigmas[last] = padded[padded.len() - 2] - padded[padded.len() - 3];
        }

        // Unsort to original order, excluding the prior (last in mus_with_prior).
        let mut inv_indices = vec![0usize; mus_with_prior.len()];
        for (rank, &orig_idx) in indices.iter().enumerate() {
            inv_indices[orig_idx] = rank;
        }

        let mut sigmas = Vec::with_capacity(mus.len());
        for i in 0..mus.len() {
            sigmas.push(sorted_sigmas[inv_indices[i]]);
        }
        sigmas
    }

    fn build_categorical_kernels(
        param_name: &str,
        obs: &[f64],
        n_choices: usize,
        n_kernels: usize,
        params: &ParzenEstimatorParameters,
    ) -> ParamKernels {
        if obs.is_empty() {
            // Single uniform prior kernel.
            let uniform = vec![1.0 / n_choices as f64; n_choices];
            return ParamKernels::Categorical {
                cat_weights: vec![uniform],
                n_choices,
            };
        }

        let base_weight = params.prior_weight / n_kernels as f64;
        let mut cat_weights = vec![vec![base_weight; n_choices]; n_kernels];

        if let Some(dist_fn) = params.categorical_distance_func.get(param_name) {
            // 使用距离函数计算分类权重。
            // 对应 Python `_calculate_categorical_distributions` 中 categorical_distance_func 分支。
            // 1. 收集唯一的 observed 索引和 reverse 映射
            let observed_indices: Vec<usize> = obs.iter().map(|&v| v as usize).collect();
            let mut unique_indices: Vec<usize> = observed_indices.clone();
            unique_indices.sort();
            unique_indices.dedup();

            // 2. 计算距离矩阵: unique_indices × n_choices
            let mut dists: Vec<Vec<f64>> = Vec::with_capacity(unique_indices.len());
            for &ui in &unique_indices {
                let row: Vec<f64> = (0..n_choices)
                    .map(|c| dist_fn(param_name, ui, c))
                    .collect();
                dists.push(row);
            }

            // 3. 归一化 + 指数衰减: coef = ln(n_kernels/prior_weight) * ln(n_choices) / ln(6)
            let coef = (n_kernels as f64 / params.prior_weight).ln()
                * (n_choices as f64).ln()
                / 6.0_f64.ln();

            for (i, &oi) in observed_indices.iter().enumerate() {
                // 找到 oi 在 unique_indices 中的位置
                let ui_pos = unique_indices.iter().position(|&u| u == oi).unwrap();
                let row = &dists[ui_pos];
                let max_d = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let max_d = if max_d < 1e-14 { 1.0 } else { max_d };
                for c in 0..n_choices {
                    let norm_d = row[c] / max_d;
                    cat_weights[i][c] = (-norm_d * norm_d * coef).exp();
                }
            }
        } else {
            // 默认: 每个观测 kernel 对其观测类别 +1
            for (i, &v) in obs.iter().enumerate() {
                let idx = v as usize;
                if idx < n_choices {
                    cat_weights[i][idx] += 1.0;
                }
            }
        }

        // Normalize each row.
        for row in &mut cat_weights {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for w in row.iter_mut() {
                    *w /= sum;
                }
            }
        }

        ParamKernels::Categorical {
            cat_weights,
            n_choices,
        }
    }

    /// Sample `size` parameter vectors from the mixture.
    ///
    /// Returns: param_name → Vec<f64> of internal-repr values, each of length `size`.
    pub fn sample(
        &self,
        rng: &mut impl RngExt,
        size: usize,
    ) -> HashMap<String, Vec<f64>> {
        // Choose which mixture component each sample comes from.
        let active_indices = self.sample_component_indices(rng, size);

        let mut result = HashMap::new();
        for (d, name) in self.param_dists.iter().zip(self.param_names.iter()) {
            let samples = match d {
                ParamKernels::Numerical {
                    mus,
                    sigmas,
                    low,
                    high,
                    log,
                    step,
                } => {
                    let mut a_vec = Vec::with_capacity(size);
                    let mut b_vec = Vec::with_capacity(size);
                    let mut loc_vec = Vec::with_capacity(size);
                    let mut scale_vec = Vec::with_capacity(size);

                    for &k in &active_indices {
                        loc_vec.push(mus[k]);
                        scale_vec.push(sigmas[k]);
                        a_vec.push(*low);
                        b_vec.push(*high);
                    }

                    let mut samples = truncnorm::rvs(&a_vec, &b_vec, &loc_vec, &scale_vec, rng);

                    // Undo log transform.
                    if *log {
                        for s in &mut samples {
                            *s = s.exp();
                        }
                    }

                    // Round discrete.
                    if let Some(st) = step {
                        // Recover original low/high before step extension.
                        // non-log: low was extended to (orig - step/2), so orig = low + step/2
                        // log: low was extended then log'd: low = ln(orig - step/2),
                        //      so orig = exp(low) + step/2
                        let orig_low = if *log {
                            low.exp() + st / 2.0
                        } else {
                            low + st / 2.0
                        };
                        let orig_high = if *log {
                            high.exp() - st / 2.0
                        } else {
                            high - st / 2.0
                        };
                        for s in &mut samples {
                            *s = crate::search_space::round_ties_even((*s - orig_low) / st) * st + orig_low;
                            *s = s.clamp(orig_low, orig_high);
                        }
                    }

                    samples
                }
                ParamKernels::Categorical {
                    cat_weights,
                    n_choices,
                } => {
                    let mut samples = Vec::with_capacity(size);
                    for &k in &active_indices {
                        let weights = &cat_weights[k];
                        let q: f64 = rng.random();
                        let mut cum = 0.0;
                        let mut choice = *n_choices - 1;
                        for (i, &w) in weights.iter().enumerate() {
                            cum += w;
                            if q < cum {
                                choice = i;
                                break;
                            }
                        }
                        samples.push(choice as f64);
                    }
                    samples
                }
            };
            result.insert(name.clone(), samples);
        }
        result
    }

    /// Compute log-PDF of the mixture for given samples.
    ///
    /// `samples`: param_name → Vec<f64> of internal-repr values, all same length.
    /// Returns: Vec<f64> of log-pdf values, one per sample.
    pub fn log_pdf(&self, samples: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let n_samples = samples
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(0);
        let n_kernels = self.weights.len();

        // weighted_log_pdf[sample][kernel] = log(w_k) + sum_d log p_k^d(x_d)
        let mut weighted_log_pdf = vec![vec![0.0; n_kernels]; n_samples];

        for (d, name) in self.param_dists.iter().zip(self.param_names.iter()) {
            let xs = match samples.get(name) {
                Some(v) => v,
                None => continue,
            };

            match d {
                ParamKernels::Numerical {
                    mus,
                    sigmas,
                    low,
                    high,
                    log,
                    step,
                } => {
                    for (si, &x) in xs.iter().enumerate() {
                        for k in 0..n_kernels {
                            let lp = if let Some(st) = step {
                                // 离散分布: 计算 [x - step/2, x + step/2] 区间的概率质量
                                if *log {
                                    // 离散 log-normal: 原始空间 [x-step/2, x+step/2]
                                    // 映射到 log 空间 [ln(x-step/2), ln(x+step/2)]
                                    let a_log = (x - st / 2.0).ln();
                                    let b_log = (x + st / 2.0).ln();
                                    let a_norm = (a_log - mus[k]) / sigmas[k];
                                    let b_norm = (b_log - mus[k]) / sigmas[k];
                                    let mass = truncnorm::log_gauss_mass(a_norm, b_norm);
                                    let total_a = (*low - mus[k]) / sigmas[k];
                                    let total_b = (*high - mus[k]) / sigmas[k];
                                    let total = truncnorm::log_gauss_mass(total_a, total_b);
                                    mass - total
                                } else {
                                    let a_norm = (x - st / 2.0 - mus[k]) / sigmas[k];
                                    let b_norm = (x + st / 2.0 - mus[k]) / sigmas[k];
                                    let mass = truncnorm::log_gauss_mass(a_norm, b_norm);
                                    let total_a = (*low - mus[k]) / sigmas[k];
                                    let total_b = (*high - mus[k]) / sigmas[k];
                                    let total = truncnorm::log_gauss_mass(total_a, total_b);
                                    mass - total
                                }
                            } else {
                                let x_val = if *log { x.ln() } else { x };
                                truncnorm::logpdf(x_val, *low, *high, mus[k], sigmas[k])
                            };
                            weighted_log_pdf[si][k] += lp;
                        }
                    }
                }
                ParamKernels::Categorical {
                    cat_weights,
                    ..
                } => {
                    for (si, &x) in xs.iter().enumerate() {
                        let idx = x as usize;
                        for k in 0..n_kernels {
                            let w = cat_weights[k]
                                .get(idx)
                                .copied()
                                .unwrap_or(0.0);
                            weighted_log_pdf[si][k] += w.ln();
                        }
                    }
                }
            }
        }

        // Add log mixture weights and logsumexp over kernels.
        // Use ln() directly (matching Python's np.log) — zero weights become -inf,
        // which logsumexp handles correctly.
        let mut result = Vec::with_capacity(n_samples);
        for row in &mut weighted_log_pdf {
            for (k, cell) in row.iter_mut().enumerate() {
                *cell += self.weights[k].ln();
            }
            result.push(logsumexp(row));
        }
        result
    }

    /// Sample mixture component indices.
    fn sample_component_indices(&self, rng: &mut impl RngExt, size: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(size);
        for _ in 0..size {
            let q: f64 = rng.random();
            let mut cum = 0.0;
            let mut idx = self.weights.len() - 1;
            for (i, &w) in self.weights.iter().enumerate() {
                cum += w;
                if q < cum {
                    idx = i;
                    break;
                }
            }
            indices.push(idx);
        }
        indices
    }

    /// Get mixture weights (for cross-validation tests).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get numerical kernel mus/sigmas/low/high for a parameter (for cross-validation tests).
    /// Returns None if the parameter is categorical.
    pub fn numerical_kernels(&self, param: &str) -> Option<(Vec<f64>, Vec<f64>, f64, f64)> {
        let idx = self.param_names.iter().position(|n| n == param)?;
        match &self.param_dists[idx] {
            ParamKernels::Numerical { mus, sigmas, low, high, .. } => {
                Some((mus.clone(), sigmas.clone(), *low, *high))
            }
            _ => None,
        }
    }

    /// Get categorical kernel weights for a parameter (for cross-validation tests).
    pub fn categorical_kernels(&self, param: &str) -> Option<Vec<Vec<f64>>> {
        let idx = self.param_names.iter().position(|n| n == param)?;
        match &self.param_dists[idx] {
            ParamKernels::Categorical { cat_weights, .. } => Some(cat_weights.clone()),
            _ => None,
        }
    }
}

/// Default weights function: uniform for n < 25, linear ramp + flat for n >= 25.
/// 对齐 Python: `np.linspace(1.0/n, 1.0, num=n-25)` + `np.ones(25)`
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp_len = n - 25;
    let mut weights = Vec::with_capacity(n);
    // 对齐 Python np.linspace(1.0/n, 1.0, num=ramp_len)
    if ramp_len == 1 {
        weights.push(1.0 / n as f64);
    } else {
        let start = 1.0 / n as f64;
        let step = (1.0 - start) / (ramp_len as f64 - 1.0);
        for i in 0..ramp_len {
            weights.push(start + step * i as f64);
        }
    }
    weights.extend(std::iter::repeat_n(1.0, 25));
    weights
}

/// Default gamma function: split point for above/below.
pub fn default_gamma(n: usize) -> usize {
    (((n as f64) * 0.1).ceil() as usize).min(25)
}

fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;
    use rand::SeedableRng;

    fn make_search_space() -> IndexMap<String, Distribution> {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
        );
        ss
    }

    #[test]
    fn test_default_weights_small() {
        let w = default_weights(5);
        assert_eq!(w.len(), 5);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_default_weights_large() {
        let w = default_weights(30);
        assert_eq!(w.len(), 30);
        // First element: 1/30, last ramp element: 1.0 (linspace)
        assert!((w[0] - 1.0 / 30.0).abs() < 1e-10);
        // Ramp length = 5, last ramp = w[4] = 1.0
        assert!((w[4] - 1.0).abs() < 1e-10);
        // Last 25 are 1.0
        assert!((w[29] - 1.0).abs() < 1e-10);

        // n=100: ramp = linspace(1/100, 1.0, 75), last ramp = 1.0
        let w100 = default_weights(100);
        assert_eq!(w100.len(), 100);
        assert!((w100[0] - 0.01).abs() < 1e-10);
        assert!((w100[74] - 1.0).abs() < 1e-10); // 对齐 Python: 旜坡最后一个=1.0
    }

    #[test]
    fn test_default_gamma() {
        assert_eq!(default_gamma(10), 1);
        assert_eq!(default_gamma(20), 2);
        assert_eq!(default_gamma(100), 10);
        assert_eq!(default_gamma(300), 25); // capped at 25
    }

    #[test]
    fn test_parzen_estimator_no_observations() {
        let ss = make_search_space();
        let obs: HashMap<String, Vec<f64>> = HashMap::new();
        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);

        assert_eq!(pe.weights.len(), 1); // Just prior
        assert!((pe.weights[0] - 1.0).abs() < 1e-10);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 10);
        assert_eq!(samples["x"].len(), 10);
        for &v in &samples["x"] {
            assert!(v >= 0.0 && v <= 10.0, "sample {v} out of bounds");
        }
    }

    #[test]
    fn test_parzen_estimator_with_observations() {
        let ss = make_search_space();
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);

        // 3 obs + 1 prior = 4 kernels
        assert_eq!(pe.weights.len(), 4);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["x"] {
            assert!(v >= 0.0 && v <= 10.0, "sample {v} out of bounds");
        }

        // log_pdf should return finite values for in-range samples
        let lp = pe.log_pdf(&samples);
        assert_eq!(lp.len(), 100);
        for &v in &lp {
            assert!(v.is_finite(), "log_pdf returned non-finite: {v}");
        }
    }

    #[test]
    fn test_parzen_estimator_categorical() {
        let mut ss = IndexMap::new();
        ss.insert(
            "opt".to_string(),
            Distribution::CategoricalDistribution(
                CategoricalDistribution::new(vec![
                    CategoricalChoice::Str("a".into()),
                    CategoricalChoice::Str("b".into()),
                    CategoricalChoice::Str("c".into()),
                ])
                .unwrap(),
            ),
        );

        let mut obs = HashMap::new();
        obs.insert("opt".to_string(), vec![0.0, 0.0, 1.0]); // mostly "a"

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["opt"] {
            assert!(
                v >= 0.0 && v < 3.0 && (v - v.round()).abs() < 1e-10,
                "categorical sample {v} invalid"
            );
        }
    }

    #[test]
    fn test_parzen_estimator_log_scale() {
        let mut ss = IndexMap::new();
        ss.insert(
            "lr".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.001, 1.0, true, None).unwrap(),
            ),
        );

        let mut obs = HashMap::new();
        obs.insert("lr".to_string(), vec![0.01, 0.1]);

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["lr"] {
            assert!(
                v >= 0.001 - 1e-10 && v <= 1.0 + 1e-10,
                "log-scale sample {v} out of bounds"
            );
        }
    }

    #[test]
    fn test_parzen_estimator_int_step() {
        let mut ss = IndexMap::new();
        ss.insert(
            "n".to_string(),
            Distribution::IntDistribution(IntDistribution::new(0, 10, false, 2).unwrap()),
        );

        let mut obs = HashMap::new();
        obs.insert("n".to_string(), vec![2.0, 4.0, 6.0]);

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["n"] {
            let iv = v as i64;
            assert!(
                iv >= 0 && iv <= 10 && iv % 2 == 0,
                "int step sample {v} invalid"
            );
        }
    }

    #[test]
    fn test_logsumexp() {
        assert!((logsumexp(&[0.0, 0.0]) - 2.0_f64.ln()).abs() < 1e-10);
        assert!((logsumexp(&[-1000.0, -1000.0]) - (-1000.0 + 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_log_pdf_discrete_log_normal() {
        // 验证离散 log-normal 分布的 log_pdf 使用 ln(x ± step/2) 而非 ln(x) ± step/2
        // 数学性质: ln(x - s/2) ≠ ln(x) - s/2
        // 例: x=10, s=2 → ln(9) ≈ 2.197 vs ln(10) - 1.0 ≈ 1.303
        use crate::distributions::IntDistribution;
        let ss = {
            let mut m = IndexMap::new();
            m.insert(
                "x".to_string(),
                Distribution::IntDistribution(
                    IntDistribution::new(1, 100, true, 1).unwrap(),
                ),
            );
            m
        };
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![10.0, 20.0, 50.0]);

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None, None);
        // log_pdf 应该返回有限值（非 NaN）
        let log_pdf_vals = pe.log_pdf(&HashMap::from([("x".to_string(), vec![10.0, 20.0, 50.0])]));
        for &lp in &log_pdf_vals {
            assert!(lp.is_finite(), "log_pdf 返回非有限值: {}", lp);
        }
    }

    // ========================================================================
    // 验证对齐 Python 的校验逻辑
    // ========================================================================

    /// prior_weight < 0 应 panic（对齐 Python ValueError）
    #[test]
    #[should_panic(expected = "non-negative")]
    fn test_negative_prior_weight_panics() {
        let ss = IndexMap::new();
        let obs = HashMap::new();
        let mut params = ParzenEstimatorParameters::default();
        params.prior_weight = -1.0;
        let _pe = ParzenEstimator::new(&obs, &ss, &params, None, None);
    }

    /// 自定义权重函数返回负权重应 panic
    #[test]
    #[should_panic(expected = "non-negative")]
    fn test_weights_func_negative_panics() {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![0.5]);
        let params = ParzenEstimatorParameters::default();
        let bad_weights = |_n: usize| vec![-1.0];
        let _pe = ParzenEstimator::new(&obs, &ss, &params, None, Some(&bad_weights));
    }

    /// 自定义权重函数返回全零权重应 panic
    #[test]
    #[should_panic(expected = "positive")]
    fn test_weights_func_all_zero_panics() {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![0.5]);
        let params = ParzenEstimatorParameters::default();
        let zero_weights = |_n: usize| vec![0.0];
        let _pe = ParzenEstimator::new(&obs, &ss, &params, None, Some(&zero_weights));
    }

    /// 自定义权重函数返回 NaN 应 panic
    #[test]
    #[should_panic(expected = "finite")]
    fn test_weights_func_nan_panics() {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![0.5]);
        let params = ParzenEstimatorParameters::default();
        let nan_weights = |_n: usize| vec![f64::NAN];
        let _pe = ParzenEstimator::new(&obs, &ss, &params, None, Some(&nan_weights));
    }

    /// predetermined_weights 长度不匹配应 panic
    #[test]
    #[should_panic(expected = "predetermined_weights length")]
    fn test_predetermined_weights_length_mismatch() {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![0.5, 0.3]);
        let params = ParzenEstimatorParameters::default();
        // 2 observations but 3 weights
        let _pe = ParzenEstimator::new(&obs, &ss, &params, Some(&[1.0, 1.0, 1.0]), None);
    }

    /// 对齐 Python: 离散 log-scale 采样后 rounding 使用正确的原始 bounds。
    /// Bug: 旧代码使用 (low + step/2).exp() 恢复原始 low，
    /// 正确应为 low.exp() + step/2 = exp(ln(orig-step/2)) + step/2 = orig
    #[test]
    fn test_log_discrete_sample_bounds_recovery() {
        use crate::distributions::IntDistribution;
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::IntDistribution(
                IntDistribution::new(1, 100, true, 1).unwrap(),
            ),
        );
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![10.0, 20.0, 50.0]);

        let pe = ParzenEstimator::new(
            &obs, &ss, &ParzenEstimatorParameters::default(), None, None,
        );

        // 采样 1000 次，所有值应在 [1, 100] 范围内且为整数
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 1000);
        let x_samples = &samples["x"];
        for &v in x_samples {
            assert!(
                v >= 1.0 && v <= 100.0,
                "log-discrete sample {v} 超出 [1, 100] 范围"
            );
            assert!(
                (v - v.round()).abs() < 1e-10,
                "log-discrete sample {v} 不是整数"
            );
        }
    }
}
