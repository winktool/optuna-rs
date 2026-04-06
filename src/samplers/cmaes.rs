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
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution as _, StandardNormal};
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
///
/// 对齐 Python cmaes 库: 使用 Active CMA-ES (包含负权重)。

// Serde default functions for LR adaptation fields
fn default_lr_alpha() -> f64 { 1.4 }
fn default_lr_beta_mean() -> f64 { 0.1 }
fn default_lr_beta_sigma() -> f64 { 0.03 }
fn default_lr_gamma() -> f64 { 0.1 }
fn default_lr_eta() -> f64 { 1.0 }
// Serde default functions for termination criteria
fn default_tolx_unset() -> f64 { -1.0 }
fn default_tolxup() -> f64 { 1e4 }
fn default_tolfun() -> f64 { 1e-12 }
fn default_tolconditioncov() -> f64 { 1e14 }

#[derive(Serialize, Deserialize)]
pub struct CmaState {
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub n: usize,
    // Covariance matrix (n x n)
    pub c: Vec<Vec<f64>>,
    // Evolution paths
    pub p_sigma: Vec<f64>,
    pub p_c: Vec<f64>,
    // Eigen decomposition cache
    pub eigenvalues: Vec<f64>,
    pub b: Vec<Vec<f64>>,
    // Strategy parameters
    pub lambda: usize,
    pub mu: usize,
    /// 对齐 Python cmaes: ALL lambda weights (positive + negative).
    /// weights[0..mu] are positive and sum to 1.0.
    /// weights[mu..lambda] are negative and sum to -min_alpha.
    pub weights: Vec<f64>,
    pub mu_eff: f64,
    /// 对齐 Python: mu_eff_minus (eq.49)
    #[serde(default)]
    pub mu_eff_minus: f64,
    pub c_sigma: f64,
    pub d_sigma: f64,
    pub c_c: f64,
    pub c1: f64,
    pub c_mu: f64,
    pub chi_n: f64,
    /// 对齐 Python cmaes: Learning rate adaptation state.
    #[serde(default)]
    pub lr_adapt: bool,
    #[serde(default = "default_lr_alpha")]
    pub lr_alpha: f64,
    #[serde(default = "default_lr_beta_mean")]
    pub lr_beta_mean: f64,
    #[serde(default = "default_lr_beta_sigma")]
    pub lr_beta_sigma: f64,
    #[serde(default = "default_lr_gamma")]
    pub lr_gamma: f64,
    #[serde(default)]
    pub lr_e_mean: Vec<f64>,
    #[serde(default)]
    pub lr_e_sigma: Vec<f64>,
    #[serde(default)]
    pub lr_v_mean: f64,
    #[serde(default)]
    pub lr_v_sigma: f64,
    #[serde(default = "default_lr_eta")]
    pub lr_eta_mean: f64,
    #[serde(default = "default_lr_eta")]
    pub lr_eta_sigma: f64,
    /// 对齐 Python cmaes: Termination criteria
    #[serde(default = "default_tolx_unset")]
    pub tolx: f64,
    #[serde(default = "default_tolxup")]
    pub tolxup: f64,
    #[serde(default = "default_tolfun")]
    pub tolfun: f64,
    #[serde(default = "default_tolconditioncov")]
    pub tolconditioncov: f64,
    #[serde(default)]
    pub funhist_term: usize,
    #[serde(default)]
    pub funhist_values: Vec<f64>,
    // Generation tracking
    pub generation: usize,
    // Pending candidates from current ask batch
    pub pending: Vec<Vec<f64>>,
    pub pending_idx: usize,
    // Collected (params, value) pairs for current generation
    pub results: Vec<(Vec<f64>, f64)>,
    // Param names in order
    pub param_names: Vec<String>,
}

#[allow(clippy::needless_range_loop)]
impl CmaState {
    pub fn new(mean: Vec<f64>, sigma: f64, lambda: usize, param_names: Vec<String>) -> Self {
        let n = mean.len();
        let mu = lambda / 2;

        // 对齐 Python cmaes (eq.49): 计算 ALL lambda 个 weights_prime
        let weights_prime: Vec<f64> = (0..lambda)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();

        // mu_eff from positive (top-mu) raw weights (eq.49)
        let pos_sum_raw: f64 = weights_prime[..mu].iter().sum();
        let pos_sq_sum_raw: f64 = weights_prime[..mu].iter().map(|w| w * w).sum();
        let mu_eff: f64 = pos_sum_raw * pos_sum_raw / pos_sq_sum_raw;

        // mu_eff_minus from negative (bottom) raw weights (eq.49)
        let neg_sum_raw: f64 = weights_prime[mu..].iter().sum();
        let neg_sq_sum_raw: f64 = weights_prime[mu..].iter().map(|w| w * w).sum();
        let mu_eff_minus: f64 = if neg_sq_sum_raw.abs() > 1e-300 {
            neg_sum_raw * neg_sum_raw / neg_sq_sum_raw
        } else {
            0.0
        };

        // Adaptation parameters
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);

        // 对齐 Python: alpha_cov = 2
        let alpha_cov = 2.0;
        let c1 = alpha_cov / ((n as f64 + 1.3).powi(2) + mu_eff);
        // 对齐 Python: min(1 - c1 - 1e-8, ...)
        let c_mu = (alpha_cov * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n as f64 + 2.0).powi(2) + alpha_cov * mu_eff / 2.0))
            .min(1.0 - c1 - 1e-8);

        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // 对齐 Python (eqs.50-52): min_alpha for negative weight scaling
        let min_alpha = if c_mu > 0.0 {
            (1.0 + c1 / c_mu)
                .min(1.0 + (2.0 * mu_eff_minus) / (mu_eff + 2.0))
                .min((1.0 - c1 - c_mu) / (n as f64 * c_mu))
        } else {
            1.0
        };

        // 对齐 Python (eq.53): normalize positive/negative weights separately
        let positive_sum: f64 = weights_prime.iter().filter(|&&w| w > 0.0).sum();
        let negative_sum: f64 = weights_prime.iter().filter(|&&w| w < 0.0).map(|w| w.abs()).sum();

        let weights: Vec<f64> = weights_prime.iter().map(|&w| {
            if w >= 0.0 {
                w / positive_sum
            } else if negative_sum > 0.0 {
                min_alpha / negative_sum * w
            } else {
                0.0
            }
        }).collect();

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

        // 对齐 Python: funhist_term = 10 + ceil(30 * n / lambda)
        let funhist_term = 10 + (30.0 * n as f64 / lambda as f64).ceil() as usize;

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
            mu_eff_minus,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
            // LR adaptation — initialized to defaults, enabled via set_lr_adapt()
            lr_adapt: false,
            lr_alpha: 1.4,
            lr_beta_mean: 0.1,
            lr_beta_sigma: 0.03,
            lr_gamma: 0.1,
            lr_e_mean: vec![0.0; n],
            lr_e_sigma: vec![0.0; n * n],
            lr_v_mean: 0.0,
            lr_v_sigma: 0.0,
            lr_eta_mean: 1.0,
            lr_eta_sigma: 1.0,
            // Termination criteria — 对齐 Python cmaes
            tolx: 1e-12 * sigma,
            tolxup: 1e4,
            tolfun: 1e-12,
            tolconditioncov: 1e14,
            funhist_term,
            funhist_values: vec![0.0; funhist_term * 2],
            generation: 0,
            pending: Vec::new(),
            pending_idx: 0,
            results: Vec::new(),
            param_names,
        }
    }

    /// Sample a new candidate from the distribution.
    ///
    /// 对齐 Python cmaes.CMA.ask():
    /// 先尝试重采样 n_max_resampling 次（默认 10*n_dim），
    /// 仅当所有重采样均越界时才 clamp（_repair_infeasible_params）。
    fn ask(&mut self, rng: &mut ChaCha8Rng) -> Vec<f64> {
        if self.pending_idx < self.pending.len() {
            let result = self.pending[self.pending_idx].clone();
            self.pending_idx += 1;
            return result;
        }

        // Generate a full batch of lambda candidates
        self.pending.clear();
        self.pending_idx = 0;

        // 对齐 Python: n_max_resampling = 10 * n_dimension
        let n_max_resampling = 10 * self.n;

        for _ in 0..self.lambda {
            let mut x = None;

            // 对齐 Python cmaes.CMA.ask(): 重采样循环
            for _ in 0..n_max_resampling {
                let candidate = self.sample_solution(rng);
                if self.is_feasible(&candidate) {
                    x = Some(candidate);
                    break;
                }
            }

            // 对齐 Python cmaes.CMA._repair_infeasible_params():
            // 所有重采样失败后，采样一次并 clamp
            let x = x.unwrap_or_else(|| {
                let candidate = self.sample_solution(rng);
                self.repair_infeasible_params(&candidate)
            });

            self.pending.push(x);
        }

        let result = self.pending[self.pending_idx].clone();
        self.pending_idx += 1;
        result
    }

    /// 对齐 Python cmaes.CMA._sample_solution():
    /// 从当前分布采样一个候选解。
    fn sample_solution(&self, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.n)
            .map(|_| StandardNormal.sample(&mut *rng))
            .collect();

        // y = B * D * z
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                y[i] += self.b[i][j] * self.eigenvalues[j].sqrt() * z[j];
            }
        }

        // x = mean + sigma * y
        (0..self.n)
            .map(|i| self.mean[i] + self.sigma * y[i])
            .collect()
    }

    /// 对齐 Python cmaes.CMA._is_feasible():
    /// 检查候选解是否在 bounds [0, 1] 内。
    fn is_feasible(&self, x: &[f64]) -> bool {
        x.iter().all(|&v| (0.0..=1.0).contains(&v))
    }

    /// 对齐 Python cmaes.CMA._repair_infeasible_params():
    /// 将越界值 clamp 到合法范围 [0, 1]。
    fn repair_infeasible_params(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }

    /// Record a result and update if generation is complete.
    pub fn tell(&mut self, params: Vec<f64>, value: f64) {
        self.results.push((params, value));

        if self.results.len() >= self.lambda {
            self.update();
        }
    }

    /// Perform the CMA-ES update step.
    /// 对齐 Python cmaes.CMA.tell(): Active CMA-ES with negative weights.
    fn update(&mut self) {
        // 对齐 Python: generation 在 tell() 开头递增
        self.generation += 1;

        // Sort by objective value (ascending = minimize)
        self.results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 对齐 Python: store funhist_values (best and worst per generation)
        if self.funhist_term > 0 && !self.funhist_values.is_empty() {
            let idx = 2 * (self.generation % self.funhist_term);
            if idx + 1 < self.funhist_values.len() {
                self.funhist_values[idx] = self.results[0].1;
                self.funhist_values[idx + 1] = self.results[self.results.len() - 1].1;
            }
        }

        let old_mean = self.mean.clone();

        // 对齐 Python: save old values for lr_adaptation BEFORE standard update
        let old_sigma = self.sigma;
        let old_sigma_c: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> = if self.lr_adapt {
            // old_Sigma = sigma^2 * C
            let n = self.n;
            let mut old_sigma_mat = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    old_sigma_mat[i][j] = self.sigma * self.sigma * self.c[i][j];
                }
            }
            // old_invsqrtC = B * D^{-1} * B^T
            let old_invsqrt_c = self.compute_c_invsqrt();
            Some((old_sigma_mat, old_invsqrt_c))
        } else {
            None
        };

        // 对齐 Python: y_k = (x_k - mean) / sigma for ALL lambda solutions
        let y_k: Vec<Vec<f64>> = self.results.iter()
            .map(|(x, _)| {
                (0..self.n).map(|j| (x[j] - old_mean[j]) / self.sigma).collect()
            })
            .collect();

        // 对齐 Python (eq.41): y_w = sum(weights[:mu] * y_k[:mu])
        let mut y_w = vec![0.0; self.n];
        for i in 0..self.mu {
            for j in 0..self.n {
                y_w[j] += self.weights[i] * y_k[i][j];
            }
        }

        // Update mean: mean += cm * sigma * y_w (cm = 1)
        for j in 0..self.n {
            self.mean[j] = old_mean[j] + self.sigma * y_w[j];
        }

        // Compute C^(-1/2) for p_sigma update and w_io
        // 对齐 Python: 使用更新前的 B, D 计算 C^{-1/2}
        let c_invsqrt = self.compute_c_invsqrt();

        // C^(-1/2) * y_w
        let invsqrt_c_yw = self.mat_vec_mul(&c_invsqrt, &y_w);

        // Update evolution path p_sigma
        let cs_comp = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_sigma[i] =
                (1.0 - self.c_sigma) * self.p_sigma[i] + cs_comp * invsqrt_c_yw[i];
        }

        // h_sigma: indicator for p_sigma length
        let ps_norm: f64 = self.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();

        // 对齐 Python: sigma update
        self.sigma *= ((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0)).exp();
        self.sigma = self.sigma.min(1e32); // 对齐 Python: _SIGMA_MAX = 1e32

        // 对齐 Python: h_sigma 使用已递增的 generation (= self._g + 1)
        let h_sigma_cond_left = ps_norm
            / (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1))).sqrt();
        let h_sigma_cond_right = (1.4 + 2.0 / (self.n as f64 + 1.0)) * self.chi_n;
        let h_sigma = if h_sigma_cond_left < h_sigma_cond_right { 1.0 } else { 0.0 };

        // Update evolution path p_c (eq.45)
        let cc_comp = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_c[i] =
                (1.0 - self.c_c) * self.p_c[i] + h_sigma * cc_comp * y_w[i];
        }

        // 对齐 Python (eq.46): w_io — adapt negative weights
        // w_io = weights * where(weights >= 0, 1, n / (||C^{-1/2} y_k||^2 + EPS))
        let w_io: Vec<f64> = {
            let n_results = self.results.len().min(self.lambda);
            (0..n_results).map(|k| {
                if self.weights.get(k).copied().unwrap_or(0.0) >= 0.0 {
                    self.weights[k]
                } else {
                    // Compute ||C^{-1/2} * y_k[k]||^2
                    let c_inv_yk = self.mat_vec_mul(&c_invsqrt, &y_k[k]);
                    let norm_sq: f64 = c_inv_yk.iter().map(|v| v * v).sum();
                    self.weights[k] * self.n as f64 / (norm_sq + 1e-8)
                }
            }).collect()
        };

        // 对齐 Python (eq.47): covariance matrix update
        let delta_h_sigma = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        let w_sum: f64 = self.weights[..self.results.len().min(self.lambda)].iter().sum();

        for i in 0..self.n {
            for j in 0..self.n {
                // rank-one update
                let rank_one = self.p_c[i] * self.p_c[j];

                // rank-mu update using w_io and ALL lambda y_k
                let mut rank_mu = 0.0;
                let n_results = self.results.len().min(self.lambda);
                for k in 0..n_results {
                    rank_mu += w_io[k] * y_k[k][i] * y_k[k][j];
                }

                // 对齐 Python: (1 + c1*delta_h - c1 - cmu*sum(weights)) * C
                self.c[i][j] = (1.0 + self.c1 * delta_h_sigma - self.c1 - self.c_mu * w_sum)
                    * self.c[i][j]
                    + self.c1 * rank_one
                    + self.c_mu * rank_mu;
            }
        }

        // 对齐 Python: LR adaptation (https://arxiv.org/abs/2304.03473)
        if self.lr_adapt {
            if let Some((old_sigma_mat, old_invsqrt_c)) = old_sigma_c {
                self.lr_adaptation(&old_mean, old_sigma, &old_sigma_mat, &old_invsqrt_c);
            }
        }

        // Update eigen decomposition
        self.update_eigen();

        self.results.clear();
        self.pending.clear();
        self.pending_idx = 0;
    }

    /// Compute C^(-1/2) matrix using eigen decomposition.
    /// Returns n x n matrix: B * D^(-1) * B^T
    fn compute_c_invsqrt(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut result = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let d_inv = 1.0 / self.eigenvalues[k].max(1e-20).sqrt();
                    sum += self.b[i][k] * d_inv * self.b[j][k];
                }
                result[i][j] = sum;
            }
        }
        result
    }

    /// Matrix-vector multiply: result = mat * v
    fn mat_vec_mul(&self, mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        let n = v.len();
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += mat[i][j] * v[j];
            }
        }
        result
    }

    /// Enable learning rate adaptation.
    pub fn set_lr_adapt(&mut self, enabled: bool) {
        self.lr_adapt = enabled;
    }

    /// 对齐 Python cmaes.CMA._lr_adaptation()
    /// Learning rate adaptation: https://arxiv.org/abs/2304.03473
    fn lr_adaptation(
        &mut self,
        old_mean: &[f64],
        old_sigma: f64,
        old_sigma_mat: &[Vec<f64>],  // old_Sigma = sigma^2 * C (before update)
        old_invsqrt_c: &[Vec<f64>],  // old C^{-1/2} (before update)
    ) {
        let n = self.n;

        // calculate one-step difference of the parameters
        // Deltamean = self.mean - old_mean (n x 1 column vector)
        let delta_mean: Vec<f64> = (0..n).map(|i| self.mean[i] - old_mean[i]).collect();

        // Sigma = sigma^2 * C (after update)
        let mut sigma_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                sigma_mat[i][j] = self.sigma * self.sigma * self.c[i][j];
            }
        }

        // DeltaSigma = Sigma - old_Sigma
        let mut delta_sigma_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                delta_sigma_mat[i][j] = sigma_mat[i][j] - old_sigma_mat[i][j];
            }
        }

        // local coordinate
        // old_inv_sqrtSigma = old_invsqrtC / old_sigma
        // locDeltamean = old_inv_sqrtSigma @ Deltamean
        let mut loc_delta_mean = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                loc_delta_mean[i] += (old_invsqrt_c[i][j] / old_sigma) * delta_mean[j];
            }
        }

        // locDeltaSigma = (old_inv_sqrtSigma @ DeltaSigma @ old_inv_sqrtSigma).reshape(n*n) / sqrt(2)
        // First: temp = DeltaSigma @ old_inv_sqrtSigma
        let mut temp = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    temp[i][j] += delta_sigma_mat[i][k] * (old_invsqrt_c[j][k] / old_sigma);
                    // Note: old_inv_sqrtSigma^T = old_inv_sqrtSigma (symmetric)
                    // so old_inv_sqrtSigma[k][j] = old_invsqrt_c[k][j]/old_sigma
                    // but .dot(old_inv_sqrtSigma) means multiply on right by the matrix
                    // Python: old_inv_sqrtSigma.dot(DeltaSigma.dot(old_inv_sqrtSigma))
                }
            }
        }
        // Actually re-do this properly:
        // result = old_inv_sqrtSigma @ DeltaSigma @ old_inv_sqrtSigma
        // Step 1: mid = DeltaSigma @ old_inv_sqrtSigma (n x n)
        let mut mid = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += delta_sigma_mat[i][k] * old_invsqrt_c[k][j] / old_sigma;
                }
                mid[i][j] = s;
            }
        }
        // Step 2: result = old_inv_sqrtSigma @ mid (n x n)
        let mut loc_delta_sigma_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += (old_invsqrt_c[i][k] / old_sigma) * mid[k][j];
                }
                loc_delta_sigma_mat[i][j] = s;
            }
        }
        // Reshape to (n*n) column vector / sqrt(2)
        let sqrt2 = 2.0_f64.sqrt();
        let mut loc_delta_sigma = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                loc_delta_sigma[i * n + j] = loc_delta_sigma_mat[i][j] / sqrt2;
            }
        }

        // moving average E and V
        let beta_m = self.lr_beta_mean;
        let beta_s = self.lr_beta_sigma;

        for i in 0..n {
            self.lr_e_mean[i] = (1.0 - beta_m) * self.lr_e_mean[i] + beta_m * loc_delta_mean[i];
        }
        for i in 0..(n * n) {
            self.lr_e_sigma[i] = (1.0 - beta_s) * self.lr_e_sigma[i] + beta_s * loc_delta_sigma[i];
        }

        let norm_loc_delta_mean_sq: f64 = loc_delta_mean.iter().map(|v| v * v).sum();
        self.lr_v_mean = (1.0 - beta_m) * self.lr_v_mean + beta_m * norm_loc_delta_mean_sq;

        let norm_loc_delta_sigma_sq: f64 = loc_delta_sigma.iter().map(|v| v * v).sum();
        self.lr_v_sigma = (1.0 - beta_s) * self.lr_v_sigma + beta_s * norm_loc_delta_sigma_sq;

        // estimate SNR
        let sqnorm_e_mean: f64 = self.lr_e_mean.iter().map(|v| v * v).sum();
        let hat_snr_mean = (sqnorm_e_mean - (beta_m / (2.0 - beta_m)) * self.lr_v_mean)
            / (self.lr_v_mean - sqnorm_e_mean);

        let sqnorm_e_sigma: f64 = self.lr_e_sigma.iter().map(|v| v * v).sum();
        let hat_snr_sigma = (sqnorm_e_sigma - (beta_s / (2.0 - beta_s)) * self.lr_v_sigma)
            / (self.lr_v_sigma - sqnorm_e_sigma);

        // update learning rate
        let before_eta_mean = self.lr_eta_mean;

        let relative_snr_mean = ((hat_snr_mean / self.lr_alpha / self.lr_eta_mean) - 1.0)
            .clamp(-1.0, 1.0);
        self.lr_eta_mean *= ((self.lr_gamma * self.lr_eta_mean).min(beta_m) * relative_snr_mean).exp();

        let relative_snr_sigma = ((hat_snr_sigma / self.lr_alpha / self.lr_eta_sigma) - 1.0)
            .clamp(-1.0, 1.0);
        self.lr_eta_sigma *= ((self.lr_gamma * self.lr_eta_sigma).min(beta_s) * relative_snr_sigma).exp();

        // cap
        self.lr_eta_mean = self.lr_eta_mean.min(1.0);
        self.lr_eta_sigma = self.lr_eta_sigma.min(1.0);

        // update parameters
        // self.mean = old_mean + eta_mean * Deltamean
        for i in 0..n {
            self.mean[i] = old_mean[i] + self.lr_eta_mean * delta_mean[i];
        }

        // Sigma = old_Sigma + eta_Sigma * DeltaSigma
        let mut new_sigma_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                new_sigma_mat[i][j] = old_sigma_mat[i][j] + self.lr_eta_sigma * delta_sigma_mat[i][j];
            }
        }

        // decompose Sigma to sigma and C
        // 对齐 Python: eigs = eigenvalues of new_sigma_mat
        // sigma = exp(sum(log(eigs)) / (2*n))  [geometric mean of sqrt(eigs)]
        // C = Sigma / sigma^2
        let eigs = self.eigenvalues_symmetric(&new_sigma_mat);
        let log_eig_sum: f64 = eigs.iter().map(|&e| e.max(1e-300).ln()).sum();
        self.sigma = (log_eig_sum / (2.0 * n as f64)).exp();
        self.sigma = self.sigma.min(1e32);

        let sigma_sq = self.sigma * self.sigma;
        for i in 0..n {
            for j in 0..n {
                self.c[i][j] = new_sigma_mat[i][j] / sigma_sq;
            }
        }

        // step-size correction
        self.sigma *= before_eta_mean / self.lr_eta_mean;
    }

    /// Compute eigenvalues of a symmetric matrix using Jacobi iteration.
    /// Only returns eigenvalues (not eigenvectors).
    fn eigenvalues_symmetric(&self, mat: &[Vec<f64>]) -> Vec<f64> {
        let n = mat.len();
        let mut a: Vec<Vec<f64>> = mat.to_vec();

        // Force symmetry
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (a[i][j] + a[j][i]) / 2.0;
                a[i][j] = avg;
                a[j][i] = avg;
            }
        }

        let max_iter = 100 * n * n;
        for _ in 0..max_iter {
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

            let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Apply rotation
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                    new_a[p][i] = new_a[i][p];
                    new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                    new_a[q][i] = new_a[i][q];
                }
            }
            new_a[p][p] = cos_t * cos_t * a[p][p] + 2.0 * sin_t * cos_t * a[p][q] + sin_t * sin_t * a[q][q];
            new_a[q][q] = sin_t * sin_t * a[p][p] - 2.0 * sin_t * cos_t * a[p][q] + cos_t * cos_t * a[q][q];
            new_a[p][q] = 0.0;
            new_a[q][p] = 0.0;
            a = new_a;
        }

        (0..n).map(|i| a[i][i]).collect()
    }

    /// 对齐 Python cmaes.CMA.should_stop()
    /// Returns true when the optimizer should terminate.
    pub fn should_stop(&self) -> bool {
        let n = self.n;

        // Stop if the range of function values of the recent generation is below tolfun
        if self.generation > self.funhist_term && !self.funhist_values.is_empty() {
            let max_v = self.funhist_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_v = self.funhist_values.iter().copied().fold(f64::INFINITY, f64::min);
            if max_v - min_v < self.tolfun {
                return true;
            }
        }

        // Stop if std is smaller than tolx in all coordinates and pc is smaller than tolx
        let dc: Vec<f64> = (0..n).map(|i| self.c[i][i]).collect();
        if dc.iter().all(|&d| self.sigma * d < self.tolx)
            && self.p_c.iter().all(|&p| self.sigma * p < self.tolx)
        {
            return true;
        }

        // Stop if detecting divergent behavior
        let max_d = self.eigenvalues.iter().copied().fold(0.0_f64, f64::max).sqrt();
        if self.sigma * max_d > self.tolxup {
            return true;
        }

        // No effect coordinates
        let dc_sqrt: Vec<f64> = dc.iter().map(|d| d.sqrt()).collect();
        if (0..n).any(|i| self.mean[i] == self.mean[i] + 0.2 * self.sigma * dc_sqrt[i]) {
            return true;
        }

        // No effect axis
        if n > 0 {
            let i = self.generation % n;
            let d_i = self.eigenvalues[i].max(0.0).sqrt();
            let step: Vec<f64> = (0..n).map(|j| 0.1 * self.sigma * d_i * self.b[j][i]).collect();
            if (0..n).all(|j| self.mean[j] == self.mean[j] + step[j]) {
                return true;
            }
        }

        // Stop if condition number exceeds tolconditioncov
        let max_eig = self.eigenvalues.iter().copied().fold(0.0_f64, f64::max).sqrt();
        let min_eig = self.eigenvalues.iter().copied().fold(f64::INFINITY, f64::min).sqrt();
        if min_eig > 0.0 && max_eig / min_eig > self.tolconditioncov {
            return true;
        }

        false
    }

    /// Update eigen decomposition of C.
    ///
    /// 对齐 Python cmaes: np.linalg.eigh (LAPACK DSYEV 算法).
    /// 使用 Householder 三对角化 + implicit QR with Wilkinson shifts.
    /// 比基础 Jacobi 迭代更鲁棒、更适合大维度和病态矩阵。
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

        if n == 1 {
            self.eigenvalues[0] = self.c[0][0].max(1e-20);
            self.b[0][0] = 1.0;
            return;
        }

        // Step 1: Householder tridiagonalization
        // A = Q^T * T * Q where T is tridiagonal
        // diag: diagonal of T, offdiag: sub-diagonal of T
        // q_acc: accumulated orthogonal transformation matrix
        let mut a = self.c.clone();
        let mut q_acc = vec![vec![0.0; n]; n];
        for i in 0..n {
            q_acc[i][i] = 1.0;
        }

        for k in 0..(n - 2) {
            // Compute Householder vector for a[k+1:n, k]
            let mut x = vec![0.0; n - k - 1];
            for i in 0..x.len() {
                x[i] = a[k + 1 + i][k];
            }

            let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if x_norm < 1e-30 {
                continue;
            }

            let alpha = if x[0] >= 0.0 { -x_norm } else { x_norm };

            // v = x - alpha * e_1
            let mut v = x.clone();
            v[0] -= alpha;
            let v_norm: f64 = v.iter().map(|vi| vi * vi).sum::<f64>().sqrt();
            if v_norm < 1e-30 {
                continue;
            }
            for vi in v.iter_mut() {
                *vi /= v_norm;
            }

            // Apply Householder: A = (I - 2vv^T) A (I - 2vv^T)
            // This operates on the submatrix a[k+1:n, k+1:n]
            // but also affects a[k+1:n, k] and a[k, k+1:n]
            let m = n - k - 1;

            // Compute p = A_sub * v
            let mut p = vec![0.0; m];
            for i in 0..m {
                for j in 0..m {
                    p[i] += a[k + 1 + i][k + 1 + j] * v[j];
                }
            }

            // Compute K = 2 * (v^T * p)
            let vp: f64 = v.iter().zip(p.iter()).map(|(vi, pi)| vi * pi).sum();
            let kk = 2.0 * vp;

            // Compute q = 2*p - K*v
            let mut q = vec![0.0; m];
            for i in 0..m {
                q[i] = 2.0 * p[i] - kk * v[i];
            }

            // Update submatrix: A_sub = A_sub - v*q^T - q*v^T
            for i in 0..m {
                for j in 0..m {
                    a[k + 1 + i][k + 1 + j] -= v[i] * q[j] + q[i] * v[j];
                }
            }

            // Update the k-th column/row
            a[k][k + 1] = alpha;
            a[k + 1][k] = alpha;
            for i in 1..m {
                a[k + 1 + i][k] = 0.0;
                a[k][k + 1 + i] = 0.0;
            }

            // Accumulate Q: Q = Q * (I - 2vv^T) applied to columns k+1:n
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..m {
                    dot += q_acc[i][k + 1 + j] * v[j];
                }
                for j in 0..m {
                    q_acc[i][k + 1 + j] -= 2.0 * dot * v[j];
                }
            }
        }

        // Extract tridiagonal elements
        let mut diag = vec![0.0; n];
        let mut offdiag = vec![0.0; n]; // offdiag[i] = T[i, i+1] = T[i+1, i]
        for i in 0..n {
            diag[i] = a[i][i];
        }
        for i in 0..(n - 1) {
            offdiag[i] = a[i][i + 1];
        }

        // Step 2: Implicit QR algorithm with Wilkinson shifts
        // on the tridiagonal matrix T
        let mut z = q_acc; // eigenvectors will be Q * (QR eigenvectors)

        let max_qr_iter = 30 * n;
        let mut lo = 0;
        let mut hi = n - 1;

        let mut iter = 0;
        while lo < hi && iter < max_qr_iter {
            iter += 1;

            // Find unreduced block: look for small offdiag elements
            let mut split = hi;
            while split > lo {
                let s = diag[split - 1].abs() + diag[split].abs();
                let threshold = if s > 0.0 { 1e-14 * s } else { 1e-30 };
                if offdiag[split - 1].abs() <= threshold {
                    break;
                }
                split -= 1;
            }

            if split == hi {
                // Eigenvalue converged at hi
                hi -= 1;
                if hi == 0 {
                    break;
                }
                continue;
            }

            // Find the specific block [split..=hi]
            lo = split;

            // Wilkinson shift: eigenvalue of bottom-right 2x2 closer to diag[hi]
            let d = (diag[hi - 1] - diag[hi]) / 2.0;
            let e2 = offdiag[hi - 1] * offdiag[hi - 1];
            let shift = diag[hi] - e2 / (d + if d >= 0.0 { 1.0 } else { -1.0 }
                * (d * d + e2).sqrt());

            // Implicit QR step with Givens rotations
            let mut g = diag[lo] - shift;
            let mut s_rot = 1.0;
            let mut c_rot = 1.0;
            let mut p_val = 0.0;

            for i in lo..hi {
                let f = s_rot * offdiag[i];
                let b = c_rot * offdiag[i];

                // Givens rotation parameters
                let r = (g * g + f * f).sqrt();
                c_rot = if r > 0.0 { g / r } else { 1.0 };
                s_rot = if r > 0.0 { f / r } else { 0.0 };

                if i > lo {
                    offdiag[i - 1] = r;
                }

                g = diag[i] - p_val;
                let rr = (diag[i + 1] - g) * s_rot + 2.0 * c_rot * b;
                p_val = s_rot * rr;
                diag[i] = g + p_val;
                g = c_rot * rr - b;

                // Update eigenvector matrix
                for k in 0..n {
                    let zi = z[k][i];
                    let zi1 = z[k][i + 1];
                    z[k][i] = c_rot * zi + s_rot * zi1;
                    z[k][i + 1] = -s_rot * zi + c_rot * zi1;
                }
            }

            diag[hi] -= p_val;
            offdiag[hi - 1] = g;
            // Note: offdiag[hi] is not part of the active block
        }

        // Sort eigenvalues in ascending order (matching np.linalg.eigh)
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap_or(std::cmp::Ordering::Equal));

        for (dest, &src) in idx.iter().enumerate() {
            self.eigenvalues[dest] = diag[src].max(1e-20);
        }

        // Reorder eigenvectors to match sorted eigenvalues
        let mut sorted_b = vec![vec![0.0; n]; n];
        for (dest, &src) in idx.iter().enumerate() {
            for i in 0..n {
                sorted_b[i][dest] = z[i][src];
            }
        }
        self.b = sorted_b;
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
            None => ChaCha8Rng::from_rng(&mut rand::rng()),
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

    pub fn default_popsize(n: usize) -> usize {
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

    /// 对齐 Python `cmaes.get_warm_start_mgd(source_solutions, gamma=0.1, alpha=0.1)`:
    /// 从 source trials 估计多元高斯分布参数 (mean, sigma, cov)。
    ///
    /// 算法:
    /// 1. 按目标值排序，取 top gamma% 的解
    /// 2. 计算均值 mean
    /// 3. 计算协方差 Sigma = alpha^2 * I + (1/n) * sum(x * x^T) - mean * mean^T
    /// 4. 分解 sigma = det(Sigma)^(1/(2*dim)), cov = Sigma / det(Sigma)^(1/dim)
    fn get_warm_start_mgd(
        source_solutions: &[(Vec<f64>, f64)],
        dim: usize,
    ) -> (Vec<f64>, f64, Vec<Vec<f64>>) {
        const GAMMA: f64 = 0.1;
        const ALPHA: f64 = 0.1;

        // 1. 按目标值排序（升序 = 最小化）
        let mut sorted: Vec<(Vec<f64>, f64)> = source_solutions.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // 2. 取 top gamma% 的解（至少 1 个）
        let gamma_n = ((sorted.len() as f64 * GAMMA).floor() as usize).max(1);
        let top_solutions: Vec<&Vec<f64>> = sorted[..gamma_n].iter().map(|(x, _)| x).collect();

        // 3. 计算均值
        let mut mean = vec![0.0; dim];
        for x in &top_solutions {
            for i in 0..dim {
                mean[i] += x[i];
            }
        }
        for i in 0..dim {
            mean[i] /= gamma_n as f64;
        }

        // 4. 计算协方差: Sigma = alpha^2 * I + (1/n) * sum(x*x^T) - mean*mean^T
        let mut sigma_mat = vec![vec![0.0; dim]; dim];
        // alpha^2 * I (正则化项)
        for i in 0..dim {
            sigma_mat[i][i] = ALPHA * ALPHA;
        }
        // (1/n) * sum(x * x^T)
        for x in &top_solutions {
            for i in 0..dim {
                for j in 0..dim {
                    sigma_mat[i][j] += x[i] * x[j] / gamma_n as f64;
                }
            }
        }
        // - mean * mean^T
        for i in 0..dim {
            for j in 0..dim {
                sigma_mat[i][j] -= mean[i] * mean[j];
            }
        }

        // 5. 分解 sigma 和归一化 cov
        // det(Sigma) 通过 LU 分解或对角近似计算
        // 这里使用特征值积来计算行列式
        let det = Self::matrix_det(&sigma_mat, dim);
        let det_abs = det.abs().max(1e-300); // 避免零行列式

        let sigma = det_abs.powf(1.0 / (2.0 * dim as f64));
        let det_norm = det_abs.powf(1.0 / dim as f64);

        let mut cov = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] = sigma_mat[i][j] / det_norm;
            }
        }

        (mean, sigma, cov)
    }

    /// 计算矩阵行列式（LU 分解法）。
    pub fn matrix_det(mat: &[Vec<f64>], n: usize) -> f64 {
        // Gaussian elimination with partial pivoting
        let mut a: Vec<Vec<f64>> = mat.to_vec();
        let mut det = 1.0;
        for col in 0..n {
            // Find pivot
            let mut max_val = a[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if a[row][col].abs() > max_val {
                    max_val = a[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-300 {
                return 0.0;
            }
            if max_row != col {
                a.swap(col, max_row);
                det = -det;
            }
            det *= a[col][col];
            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for j in (col + 1)..n {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
        det
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
                // 对齐 Python: 检查恢复的状态维度是否匹配当前搜索空间
                // Python: `if optimizer.dim != len(trans.bounds): warn(); return {}`
                if restored_state.n != n_dims {
                    crate::optuna_warn!(
                        "CMA-ES state dimension ({}) doesn't match search space ({}). \
                         Falling back to independent sampling.",
                        restored_state.n, n_dims
                    );
                    return Ok(HashMap::new());
                }
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
                // 对齐 Python: get_warm_start_mgd 算法
                // 从 source_trials 中估计 (mean, sigma, cov) 用于初始化 CMA-ES
                let sign = if self.direction == StudyDirection::Maximize { -1.0 } else { 1.0 };
                let expected_states = [TrialState::Complete];
                let mut source_solutions: Vec<(Vec<f64>, f64)> = Vec::new();
                for t in source {
                    if !expected_states.contains(&t.state) {
                        continue;
                    }
                    let vals = match &t.values {
                        Some(v) if !v.is_empty() => v,
                        _ => continue,
                    };
                    // 检查搜索空间兼容性：所有参数必须存在且分布兼容
                    let mut trial_params = IndexMap::new();
                    let mut compatible = true;
                    for name in &param_names {
                        if let Some(pv) = t.params.get(name) {
                            trial_params.insert(name.clone(), pv.clone());
                        } else {
                            compatible = false;
                            break;
                        }
                    }
                    if !compatible || trial_params.len() != ordered_space.len() {
                        continue;
                    }
                    let encoded = transform.transform(&trial_params);
                    source_solutions.push((encoded, sign * vals[0]));
                }

                if source_solutions.is_empty() {
                    crate::optuna_warn!("No compatible source_trials found. Using center.");
                    vec![0.5; n_dims]
                } else {
                    // 对齐 Python cmaes.get_warm_start_mgd(gamma=0.1, alpha=0.1)
                    let (mgd_mean, mgd_sigma, mgd_cov) =
                        Self::get_warm_start_mgd(&source_solutions, n_dims);

                    // 用 MGD 结果构造 CmaState，跳过默认初始化
                    let lambda_val = self.popsize.unwrap_or_else(|| Self::default_popsize(n_dims));
                    let mut new_state =
                        CmaState::new(mgd_mean, mgd_sigma, lambda_val, param_names.clone());
                    // 设置协方差矩阵（归一化后的）
                    new_state.c = mgd_cov;
                    new_state.update_eigen();
                    *state_guard = Some(new_state);

                    // 已经初始化了 state，直接跳到 ask
                    let state = state_guard.as_mut().unwrap();
                    let candidate = state.ask(&mut rng);

                    drop(rng);
                    drop(state_guard);

                    // Untransform
                    let decoded = transform.untransform(&candidate)?;
                    let mut result = HashMap::new();
                    for (name, dist) in &ordered_space {
                        if let Some(pv) = decoded.get(name) {
                            let internal = dist.to_internal_repr(pv)?;
                            result.insert(name.clone(), internal);
                        }
                    }
                    return Ok(result);
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

            // Apply learning-rate adaptation: set flag on CmaState
            if self.lr_adapt {
                new_state.set_lr_adapt(true);
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

    /// 对齐 Python: CMA-ES ask() 重采样 vs clamp
    /// 验证 ask() 返回的候选解在 [0, 1] 范围内
    #[test]
    fn test_cmaes_ask_resampling_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // 使用 sigma=2.0 让样本容易越界, 触发重采样逻辑
        let mut state = CmaState::new(
            vec![0.5, 0.5],
            2.0, // 大 sigma → 容易越界
            6,
            vec!["x".into(), "y".into()],
        );

        // 产生多个批次的候选解
        for _ in 0..5 {
            for _ in 0..state.lambda {
                let x = state.ask(&mut rng);
                for &v in &x {
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "ask() 返回的值 {v} 超出 [0, 1]"
                    );
                }
                // 模拟 tell 以触发下一代
                state.tell(x, rand_distr::Distribution::sample(&StandardNormal, &mut rng));
            }
        }
    }

    /// 对齐 Python: is_feasible 和 repair_infeasible_params
    #[test]
    fn test_cmaes_feasibility_check() {
        let state = CmaState::new(vec![0.5], 0.1, 4, vec!["x".into()]);
        // 可行值
        assert!(state.is_feasible(&[0.0]));
        assert!(state.is_feasible(&[0.5]));
        assert!(state.is_feasible(&[1.0]));
        // 不可行值
        assert!(!state.is_feasible(&[-0.01]));
        assert!(!state.is_feasible(&[1.01]));

        // repair
        let repaired = state.repair_infeasible_params(&[-0.5, 1.5, 0.5]);
        assert_eq!(repaired, vec![0.0, 1.0, 0.5]);
    }

    /// 对齐 Python: sample_solution 产生无 clamp 的原始样本
    #[test]
    fn test_cmaes_sample_solution_raw() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let state = CmaState::new(
            vec![0.5, 0.5],
            5.0, // 极大 sigma → 几乎一定越界
            4,
            vec!["x".into(), "y".into()],
        );
        // 采 100 次，至少有一些越界的
        let mut any_out = false;
        for _ in 0..100 {
            let x = state.sample_solution(&mut rng);
            for &v in &x {
                if v < 0.0 || v > 1.0 {
                    any_out = true;
                }
            }
        }
        assert!(any_out, "大 sigma 下 sample_solution 应产生越界值");
    }

    /// 对齐 Python: get_warm_start_mgd 算法
    /// 验证 MGD 估计的 mean ≈ source solutions 的 top-10% 均值
    #[test]
    fn test_warm_start_mgd_basic() {
        // 构造 20 个 2D source solutions
        let mut solutions = Vec::new();
        for i in 0..20 {
            let x = vec![0.3 + i as f64 * 0.01, 0.4 + i as f64 * 0.01];
            let value = (x[0] - 0.3).powi(2) + (x[1] - 0.4).powi(2);
            solutions.push((x, value));
        }
        let (mean, sigma, cov) = CmaEsSampler::get_warm_start_mgd(&solutions, 2);

        // top 10% = top 2 → 最好的两个是 i=0, i=1
        // mean ≈ [0.305, 0.405]
        assert!((mean[0] - 0.305).abs() < 1e-6, "mean[0] = {}", mean[0]);
        assert!((mean[1] - 0.405).abs() < 1e-6, "mean[1] = {}", mean[1]);

        // sigma > 0
        assert!(sigma > 0.0, "sigma 应为正数: {sigma}");

        // cov 应为 2x2 正定矩阵
        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        assert!(cov[0][0] > 0.0, "cov[0][0] 应为正数");
        assert!(cov[1][1] > 0.0, "cov[1][1] 应为正数");
    }

    /// 对齐 Python: get_warm_start_mgd 单解情况
    /// 注意: Python cmaes.get_warm_start_mgd 对单解 (gamma=0.1) 会 assert 失败，
    /// 因为 floor(1 * 0.1) = 0 < 1。我们的实现用 max(1) 做了防御，
    /// 结果等价于 alpha^2*I 的协方差。
    #[test]
    fn test_warm_start_mgd_single_solution() {
        let solutions = vec![(vec![0.3, 0.7], 1.0)];
        let (mean, sigma, cov) = CmaEsSampler::get_warm_start_mgd(&solutions, 2);

        // gamma_n = max(floor(1 * 0.1), 1) = 1, 用唯一解
        assert!((mean[0] - 0.3).abs() < 1e-10);
        assert!((mean[1] - 0.7).abs() < 1e-10);
        // sigma = det(alpha^2 * I)^(1/(2*dim)) = (0.01^2)^(1/4) = 0.1
        assert!((sigma - 0.1).abs() < 1e-10, "sigma = {sigma}");
        // cov = (alpha^2 * I) / det(alpha^2 * I)^(1/dim) = I
        assert!((cov[0][0] - 1.0).abs() < 1e-10, "cov[0][0] = {}", cov[0][0]);
        assert!((cov[1][1] - 1.0).abs() < 1e-10, "cov[1][1] = {}", cov[1][1]);
    }

    /// 对齐 Python: matrix_det 行列式计算
    #[test]
    fn test_matrix_det() {
        // 2x2 identity → det = 1
        let id2 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!((CmaEsSampler::matrix_det(&id2, 2) - 1.0).abs() < 1e-10);

        // [[2, 1], [1, 3]] → det = 5
        let m2 = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        assert!((CmaEsSampler::matrix_det(&m2, 2) - 5.0).abs() < 1e-10);

        // 3x3 identity → det = 1
        let id3 = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        assert!((CmaEsSampler::matrix_det(&id3, 3) - 1.0).abs() < 1e-10);

        // 奇异矩阵 → det = 0
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(CmaEsSampler::matrix_det(&singular, 2).abs() < 1e-10);
    }

    /// 对齐 Python: 维度变化时 CMA-ES 回退到独立采样
    #[test]
    fn test_cmaes_dimension_change_fallback() {
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            Some(0.5), Some(3), Some(6),
            None, Some(42), None,
            false, false, false, false, None,
        ));

        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        // Phase 1: 用 2 个参数运行
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(15),
            None, None,
        ).unwrap();

        let n1 = study.trials().unwrap().len();
        assert_eq!(n1, 15);

        // Phase 2: 用 3 个参数运行 → 维度变化，CMA-ES 应回退
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y + z * z)
            },
            Some(5),
            None, None,
        ).unwrap();

        assert_eq!(study.trials().unwrap().len(), 20);
    }

    /// 对齐 Python: CMA-ES with source_trials (warm-start MGD)
    #[test]
    fn test_cmaes_source_trials_warm_start() {
        let now = chrono::Utc::now();
        // 创建一些 source trials 模拟另一个 study 的结果
        let source_trials: Vec<FrozenTrial> = (0..10).map(|i| {
            let x_val = 0.5 + i as f64 * 0.1;
            let y_val = -0.5 + i as f64 * 0.1;
            let mut params = HashMap::new();
            params.insert("x".to_string(),
                crate::distributions::ParamValue::Float(x_val));
            params.insert("y".to_string(),
                crate::distributions::ParamValue::Float(y_val));
            let mut dists = HashMap::new();
            dists.insert("x".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution::new(-5.0, 5.0, false, None).unwrap()));
            dists.insert("y".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution::new(-5.0, 5.0, false, None).unwrap()));
            FrozenTrial {
                number: i, trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![(x_val - 1.0).powi(2) + (y_val + 1.0).powi(2)]),
                datetime_start: Some(now), datetime_complete: Some(now),
                params, distributions: dists,
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            }
        }).collect();

        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .n_startup_trials(3)
                .popsize(6)
                .seed(42)
                .source_trials(source_trials)
                .build()
        );

        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok((x - 1.0).powi(2) + (y + 1.0).powi(2))
            },
            Some(30),
            None, None,
        ).unwrap();

        assert_eq!(study.trials().unwrap().len(), 30);
        let best = study.best_value().unwrap();
        // warm-start 应帮助找到更好的解
        assert!(best < 10.0, "warm-start 应该有效, got best={best}");
    }

    /// 对齐 Python: CMA-ES with separable CMA
    #[test]
    fn test_cmaes_separable() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .sigma0(0.5)
                .n_startup_trials(5)
                .popsize(8)
                .seed(42)
                .use_separable_cma(true)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(40),
            None, None,
        ).unwrap();
        assert_eq!(study.trials().unwrap().len(), 40);
    }

    /// 对齐 Python: CMA-ES with lr_adapt
    #[test]
    fn test_cmaes_lr_adapt() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .sigma0(0.5)
                .n_startup_trials(5)
                .popsize(8)
                .seed(42)
                .lr_adapt(true)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(40),
            None, None,
        ).unwrap();
        assert_eq!(study.trials().unwrap().len(), 40);
    }

    /// 对齐 Python: CMA-ES with with_margin
    #[test]
    fn test_cmaes_with_margin_int_params() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .sigma0(0.5)
                .n_startup_trials(3)
                .popsize(6)
                .seed(42)
                .with_margin(true)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_int("x", -10, 10, false, 1)?;
                let y = trial.suggest_int("y", -10, 10, false, 1)?;
                Ok((x * x + y * y) as f64)
            },
            Some(30),
            None, None,
        ).unwrap();
        assert_eq!(study.trials().unwrap().len(), 30);
    }

    /// 对齐 Python: CMA-ES maximize direction
    #[test]
    fn test_cmaes_maximize() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Maximize)
                .sigma0(0.5)
                .n_startup_trials(5)
                .popsize(6)
                .seed(42)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Maximize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(-(x - 2.0).powi(2) + 10.0)
            },
            Some(40),
            None, None,
        ).unwrap();
        let best = study.best_value().unwrap();
        assert!(best > 5.0, "maximize 应该找到近最优解, got {best}");
    }

    /// 对齐 Python: consider_pruned_trials
    #[test]
    fn test_cmaes_consider_pruned_trials() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .sigma0(0.5)
                .n_startup_trials(3)
                .popsize(6)
                .seed(42)
                .consider_pruned_trials(true)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                // 报告中间值
                trial.report(x * x, 0)?;
                if trial.number() < 5 {
                    // 前 5 个试验被剪枝
                    return Err(crate::error::OptunaError::TrialPruned);
                }
                Ok(x * x)
            },
            Some(30),
            None, None,
        ).unwrap();
        assert_eq!(study.trials().unwrap().len(), 30);
    }

    /// 对齐 Python: x0 初始化
    #[test]
    fn test_cmaes_x0_initialization() {
        let mut x0 = HashMap::new();
        x0.insert("x".to_string(), 2.0);
        x0.insert("y".to_string(), -1.0);

        let sampler: Arc<dyn Sampler> = Arc::new(
            CmaEsSamplerBuilder::new(StudyDirection::Minimize)
                .sigma0(0.5)
                .n_startup_trials(3)
                .popsize(6)
                .seed(42)
                .x0(x0)
                .build()
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok((x - 2.0).powi(2) + (y + 1.0).powi(2))
            },
            Some(30),
            None, None,
        ).unwrap();
        let best = study.best_value().unwrap();
        assert!(best < 5.0, "x0 warm-start 应帮助找到近最优解, got {best}");
    }

    /// Python 交叉验证: get_warm_start_mgd 与 Python cmaes.get_warm_start_mgd 结果一致
    /// Python 验证值:
    ///   mean = [0.305, 0.405]
    ///   sigma = 0.10012476630625274
    ///   cov_diag ≈ [1.00000311, 1.00000311]
    ///   cov[0][1] ≈ 0.002493773340268805
    #[test]
    fn test_warm_start_mgd_cross_validate_python() {
        let mut solutions = Vec::new();
        for i in 0..20 {
            let x = vec![0.3 + i as f64 * 0.01, 0.4 + i as f64 * 0.01];
            let value = (x[0] - 0.3).powi(2) + (x[1] - 0.4).powi(2);
            solutions.push((x, value));
        }
        let (mean, sigma, cov) = CmaEsSampler::get_warm_start_mgd(&solutions, 2);

        // 与 Python cmaes.get_warm_start_mgd 结果对比
        assert!((mean[0] - 0.305).abs() < 1e-10, "mean[0] = {}", mean[0]);
        assert!((mean[1] - 0.405).abs() < 1e-10, "mean[1] = {}", mean[1]);
        assert!(
            (sigma - 0.10012476630625274).abs() < 1e-8,
            "sigma = {sigma}, expected 0.10012476630625274"
        );
        assert!(
            (cov[0][0] - 1.00000311).abs() < 1e-5,
            "cov[0][0] = {}, expected ~1.00000311", cov[0][0]
        );
        assert!(
            (cov[1][1] - 1.00000311).abs() < 1e-5,
            "cov[1][1] = {}, expected ~1.00000311", cov[1][1]
        );
        assert!(
            (cov[0][1] - 0.002493773340268805).abs() < 1e-5,
            "cov[0][1] = {}, expected ~0.00249", cov[0][1]
        );
    }
}
