//! 高斯过程 (GP) 采样器模块 — Matern 5/2 核 + logEI 采集函数
//!
//! 对应 Python `optuna.samplers.GPSampler`。
//! 纯 Rust 实现，无需 PyTorch / scipy 依赖。
//!
//! ## 算法概述
//! 1. 使用 Matern 5/2 核的高斯过程回归拟合已完成试验
//! 2. 通过 L-BFGS-B 优化核超参数（最大化边际似然）
//! 3. 使用 logEI（对数期望改善）采集函数选择下一组参数
//! 4. 支持自动相关性确定 (ARD) — 每个维度独立的长度尺度

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::{RandomSampler, Sampler};
use crate::search_space::IntersectionSearchSpace;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

// ════════════════════════════════════════════════════════════════════════
// 数学常量
// ════════════════════════════════════════════════════════════════════════

const PI: f64 = std::f64::consts::PI;
const _SQRT_5: f64 = 2.2360679774997896964091736687747631;
const EPS: f64 = 1e-10;
pub(crate) const DEFAULT_MINIMUM_NOISE_VAR: f64 = 1e-6;

// ════════════════════════════════════════════════════════════════════════
// Matern 5/2 核函数
// ════════════════════════════════════════════════════════════════════════

/// Matern 5/2 核: k(d²) = exp(-√(5d²)) * (5/3 * d² + √(5d²) + 1)
pub(crate) fn matern52(squared_distance: f64) -> f64 {
    if squared_distance < 1e-30 {
        return 1.0;
    }
    let sqrt5d = (5.0 * squared_distance).sqrt();
    let exp_part = (-sqrt5d).exp();
    exp_part * ((5.0 / 3.0) * squared_distance + sqrt5d + 1.0)
}

// ════════════════════════════════════════════════════════════════════════
// 线性代数工具 — Cholesky 分解 / 三角求解
// ════════════════════════════════════════════════════════════════════════

/// Cholesky 分解: A = L * L^T，返回下三角矩阵 L
/// 要求 A 为正定对称矩阵
pub(crate) fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag <= 0.0 {
            return None; // 非正定
        }
        l[j][j] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - sum) / l[j][j];
        }
    }

    Some(l)
}

/// 前向替代: L * x = b（L 为下三角）
pub(crate) fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// 后向替代: L^T * x = b（L 为下三角）
pub(crate) fn solve_upper(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

// ════════════════════════════════════════════════════════════════════════
// GP 回归器
// ════════════════════════════════════════════════════════════════════════

/// 高斯过程回归器
pub(crate) struct GPRegressor {
    /// 训练输入 [n_points, n_params]（归一化后）
    pub(crate) x_train: Vec<Vec<f64>>,
    /// 训练目标 [n_points]（标准化后）
    pub(crate) y_train: Vec<f64>,
    /// 是否为分类参数（Hamming 距离）
    pub(crate) is_categorical: Vec<bool>,
    /// 每个维度的逆平方长度尺度 (ARD)
    pub(crate) inverse_squared_lengthscales: Vec<f64>,
    /// 核尺度
    pub(crate) kernel_scale: f64,
    /// 噪声方差
    pub(crate) noise_var: f64,
    /// Cholesky 分解缓存: L
    pub(crate) chol_l: Option<Vec<Vec<f64>>>,
    /// C^{-1} * y 缓存
    pub(crate) alpha: Option<Vec<f64>>,
}

impl GPRegressor {
    pub(crate) fn new(
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
        is_categorical: Vec<bool>,
        inverse_squared_lengthscales: Vec<f64>,
        kernel_scale: f64,
        noise_var: f64,
    ) -> Self {
        let mut gpr = Self {
            x_train,
            y_train,
            is_categorical,
            inverse_squared_lengthscales,
            kernel_scale,
            noise_var,
            chol_l: None,
            alpha: None,
        };
        gpr.cache_cholesky();
        gpr
    }

    /// 计算核矩阵 K(X1, X2)
    pub(crate) fn kernel_matrix(&self, x1: &[Vec<f64>], x2: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n1 = x1.len();
        let n2 = x2.len();
        let mut k = vec![vec![0.0; n2]; n1];

        for i in 0..n1 {
            for j in 0..n2 {
                let mut sqdist = 0.0;
                for d in 0..x1[i].len() {
                    let diff = if self.is_categorical[d] {
                        // Hamming 距离
                        if (x1[i][d] - x2[j][d]).abs() > 0.5 { 1.0 } else { 0.0 }
                    } else {
                        let delta = x1[i][d] - x2[j][d];
                        delta * delta
                    };
                    sqdist += diff * self.inverse_squared_lengthscales[d];
                }
                k[i][j] = self.kernel_scale * matern52(sqdist);
            }
        }

        k
    }

    /// 训练核矩阵 K(X_train, X_train)
    pub(crate) fn train_kernel_matrix(&self) -> Vec<Vec<f64>> {
        self.kernel_matrix(&self.x_train, &self.x_train)
    }

    /// 缓存 Cholesky 分解
    pub(crate) fn cache_cholesky(&mut self) {
        let n = self.x_train.len();
        if n == 0 { return; }

        let mut k = self.train_kernel_matrix();
        // 加入噪声方差
        for i in 0..n {
            k[i][i] += self.noise_var;
        }

        if let Some(l) = cholesky(&k) {
            // α = K^{-1} y = L^{-T} L^{-1} y
            let u = solve_lower(&l, &self.y_train);
            let alpha = solve_upper(&l, &u);
            self.chol_l = Some(l);
            self.alpha = Some(alpha);
        }
    }

    /// 后验预测 mean 和 variance
    pub(crate) fn posterior(&self, x: &[f64]) -> (f64, f64) {
        let alpha = match &self.alpha {
            Some(a) => a,
            None => return (0.0, self.kernel_scale),
        };
        let chol_l = self.chol_l.as_ref().unwrap();

        // k_star = K(x, X_train) [1, n]
        let x_vec = vec![x.to_vec()];
        let k_star_mat = self.kernel_matrix(&x_vec, &self.x_train);
        let k_star = &k_star_mat[0];

        // mean = k_star^T * alpha
        let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum();

        // variance = k(x, x) - k_star^T * K^{-1} * k_star
        // v = L^{-1} * k_star
        let v = solve_lower(chol_l, k_star);
        let v_squared: f64 = v.iter().map(|vi| vi * vi).sum();
        let k_xx = self.kernel_scale; // Matern52(0) * kernel_scale = kernel_scale
        let var = (k_xx - v_squared).max(0.0);

        (mean, var)
    }

    /// 边际对数似然: log p(y | X, θ)
    pub(crate) fn log_marginal_likelihood(&self) -> f64 {
        let chol_l = match &self.chol_l {
            Some(l) => l,
            None => return f64::NEG_INFINITY,
        };
        let alpha = self.alpha.as_ref().unwrap();
        let n = self.y_train.len() as f64;

        // log|K| = 2 * sum(log(diag(L)))
        let log_det: f64 = chol_l.iter().enumerate()
            .map(|(i, row)| row[i].ln())
            .sum::<f64>() * 2.0;

        // y^T K^{-1} y = y^T alpha
        let quad: f64 = self.y_train.iter().zip(alpha.iter())
            .map(|(y, a)| y * a).sum();

        -0.5 * log_det - 0.5 * quad - 0.5 * n * (2.0 * PI).ln()
    }

    /// 返回当前核参数缓存，用于下一轮拟合的初始值。
    /// 对齐 Python `_gprs_cache_list` 的缓存机制。
    pub(crate) fn params_cache(&self) -> Option<KernelParamsCache> {
        Some(KernelParamsCache {
            inverse_squared_lengthscales: self.inverse_squared_lengthscales.clone(),
            kernel_scale: self.kernel_scale,
            noise_var: self.noise_var,
        })
    }
}

/// 缓存的核参数: 用于跨调用复用上一轮拟合结果作为初始值。
/// 对应 Python `_gprs_cache_list` 中的 GPRegressor 参数。
#[derive(Clone)]
pub(crate) struct KernelParamsCache {
    pub(crate) inverse_squared_lengthscales: Vec<f64>,
    pub(crate) kernel_scale: f64,
    pub(crate) noise_var: f64,
}

/// Python `optuna.samplers._gp.prior.default_log_prior` 的忠实移植。
///
/// 先验分布:
/// - `inverse_squared_lengthscales`: 自定义惩罚 `-(0.1/x + 0.1*x)` ← 鼓励 ≈ 1
/// - `kernel_scale`: Gamma(concentration=2, rate=1) → `log(x) - x`
/// - `noise_var`: Gamma(concentration=1.1, rate=30) → `0.1*log(x) - 30*x`
pub(crate) fn default_log_prior(
    inverse_squared_lengthscales: &[f64],
    kernel_scale: f64,
    noise_var: f64,
) -> f64 {
    let ls_prior: f64 = inverse_squared_lengthscales.iter()
        .map(|&x| -(0.1 / x + 0.1 * x))
        .sum();
    // Gamma(α=2, β=1): (α-1)*ln(x) - β*x
    let ks_prior = kernel_scale.ln() - kernel_scale;
    // Gamma(α=1.1, β=30): (α-1)*ln(x) - β*x
    let nv_prior = 0.1 * noise_var.ln() - 30.0 * noise_var;
    ls_prior + ks_prior + nv_prior
}

/// 拟合 GP 超参数 — 对齐 Python: 梯度优化 (有限差分 + 准牛顿法)。
///
/// 对应 Python `gp.fit_kernel_params`:
/// - 在 log 空间中编码超参数，使用 L-BFGS-B 优化 -(LML + log_prior)
/// - 先用 cache 初始值尝试，失败则用默认值再试
///
/// `gpr_cache`: 上一轮拟合的核参数缓存，作为初始起点。
/// `deterministic_objective`: 是否为确定性目标（若 true，固定 noise_var）。
/// 对应 Python `gp.fit_kernel_params(..., gpr_cache=cache)`。
pub(crate) fn fit_kernel_params(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    _seed: u64,
    gpr_cache: Option<&KernelParamsCache>,
    deterministic_objective: bool,
) -> GPRegressor {
    let n_params = if x_train.is_empty() { 0 } else { x_train[0].len() };

    if x_train.is_empty() || y_train.is_empty() {
        return GPRegressor::new(
            x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
            vec![1.0; n_params], 1.0, DEFAULT_MINIMUM_NOISE_VAR,
        );
    }

    // 对齐 Python 的 log 空间编码:
    // raw_params = [log(inv_sq_ls[0..d]), log(kernel_scale), log(noise_var - 0.99*min_noise)]
    let encode = |cache: &KernelParamsCache| -> Vec<f64> {
        let mut params: Vec<f64> = cache.inverse_squared_lengthscales
            .iter().map(|&x| x.max(1e-12).ln()).collect();
        params.push(cache.kernel_scale.max(1e-12).ln());
        if !deterministic_objective {
            let raw = (cache.noise_var - 0.99 * DEFAULT_MINIMUM_NOISE_VAR).max(1e-12);
            params.push(raw.ln());
        }
        params
    };

    let decode = |params: &[f64]| -> (Vec<f64>, f64, f64) {
        let inv_sq_ls: Vec<f64> = params[..n_params].iter().map(|&x| x.exp()).collect();
        let kernel_scale = params[n_params].exp();
        let noise_var = if deterministic_objective {
            DEFAULT_MINIMUM_NOISE_VAR
        } else {
            DEFAULT_MINIMUM_NOISE_VAR + params[n_params + 1].exp()
        };
        (inv_sq_ls, kernel_scale, noise_var)
    };

    // 负对数后验 (损失函数): -(LML + log_prior)
    let neg_log_posterior = |params: &[f64]| -> f64 {
        let (inv_sq_ls, kernel_scale, noise_var) = decode(params);
        if inv_sq_ls.iter().any(|&x| !x.is_finite() || x > 1e10)
            || !kernel_scale.is_finite() || kernel_scale > 1e10
            || !noise_var.is_finite() || noise_var > 1e10
        {
            return 1e30;
        }
        let gpr = GPRegressor::new(
            x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
            inv_sq_ls.clone(), kernel_scale, noise_var,
        );
        let lml = gpr.log_marginal_likelihood();
        if !lml.is_finite() { return 1e30; }
        -(lml + default_log_prior(&inv_sq_ls, kernel_scale, noise_var))
    };

    // 有限差分梯度 (中心差分)
    let gradient = |params: &[f64], f0: f64| -> Vec<f64> {
        let h = 1e-5;
        let n = params.len();
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut p_fwd = params.to_vec();
            let mut p_bwd = params.to_vec();
            p_fwd[i] += h;
            p_bwd[i] -= h;
            let f_fwd = neg_log_posterior(&p_fwd);
            let f_bwd = neg_log_posterior(&p_bwd);
            grad[i] = (f_fwd - f_bwd) / (2.0 * h);
        }
        let _ = f0; // f0 unused in central diff, but available for forward diff
        grad
    };

    // L-BFGS 优化 (有限记忆 BFGS, 对齐 Python scipy L-BFGS-B)
    let optimize = |init_params: &[f64]| -> Option<(Vec<f64>, f64)> {
        let n = init_params.len();
        let m = 5; // L-BFGS memory size
        let max_iter = 50;
        let gtol = 1e-2;

        let mut x = init_params.to_vec();
        let mut f = neg_log_posterior(&x);
        if !f.is_finite() { return None; }
        let mut g = gradient(&x, f);

        // L-BFGS 历史存储
        let mut s_hist: Vec<Vec<f64>> = Vec::new(); // x_{k+1} - x_k
        let mut y_hist: Vec<Vec<f64>> = Vec::new(); // g_{k+1} - g_k
        let mut rho_hist: Vec<f64> = Vec::new();    // 1 / (y_k^T s_k)

        for _ in 0..max_iter {
            // 检查收敛: ||g||_inf < gtol
            let g_norm = g.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
            if g_norm < gtol { break; }

            // L-BFGS 两环递推求搜索方向 d = -H_k * g
            let mut q = g.clone();
            let k = s_hist.len();
            let mut alpha_save = vec![0.0; k];

            // 第一环: 从最新到最旧
            for i in (0..k).rev() {
                let alpha_i: f64 = rho_hist[i] * s_hist[i].iter()
                    .zip(q.iter()).map(|(s, q)| s * q).sum::<f64>();
                alpha_save[i] = alpha_i;
                for j in 0..n {
                    q[j] -= alpha_i * y_hist[i][j];
                }
            }

            // 初始 Hessian 近似 H_0 = gamma * I
            let gamma = if k > 0 {
                let yk = &y_hist[k - 1];
                let sk = &s_hist[k - 1];
                let ys: f64 = yk.iter().zip(sk.iter()).map(|(y, s)| y * s).sum();
                let yy: f64 = yk.iter().map(|y| y * y).sum();
                if yy > 0.0 { ys / yy } else { 1.0 }
            } else {
                1.0
            };
            let mut d: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

            // 第二环: 从最旧到最新
            for i in 0..k {
                let beta: f64 = rho_hist[i] * y_hist[i].iter()
                    .zip(d.iter()).map(|(y, dd)| y * dd).sum::<f64>();
                for j in 0..n {
                    d[j] += s_hist[i][j] * (alpha_save[i] - beta);
                }
            }

            // d = -d (搜索方向)
            for v in d.iter_mut() { *v = -*v; }

            // Armijo 回溯线搜索
            let c1 = 1e-4;
            let dg: f64 = g.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();
            if dg >= 0.0 { break; } // 非下降方向

            let mut step = 1.0;
            let mut x_new;
            let mut f_new;
            loop {
                x_new = x.iter().zip(d.iter()).map(|(&xi, &di)| xi + step * di).collect::<Vec<_>>();
                f_new = neg_log_posterior(&x_new);
                if f_new <= f + c1 * step * dg || step < 1e-10 {
                    break;
                }
                step *= 0.5;
            }

            if !f_new.is_finite() { break; }

            let g_new = gradient(&x_new, f_new);

            // 更新 L-BFGS 历史
            let s_k: Vec<f64> = x_new.iter().zip(x.iter()).map(|(xn, xo)| xn - xo).collect();
            let y_k: Vec<f64> = g_new.iter().zip(g.iter()).map(|(gn, go)| gn - go).collect();
            let ys: f64 = y_k.iter().zip(s_k.iter()).map(|(y, s)| y * s).sum();

            if ys > 1e-10 {
                if s_hist.len() >= m {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                s_hist.push(s_k);
                y_hist.push(y_k);
                rho_hist.push(1.0 / ys);
            }

            x = x_new;
            f = f_new;
            g = g_new;
        }

        Some((x, f))
    };

    // 对齐 Python: 先用 cache 初始值，再用默认值 — 共 2 次尝试
    let default_cache = KernelParamsCache {
        inverse_squared_lengthscales: vec![1.0; n_params],
        kernel_scale: 1.0,
        noise_var: DEFAULT_MINIMUM_NOISE_VAR + 0.01,
    };

    let attempts: Vec<Vec<f64>> = if let Some(cache) = gpr_cache {
        vec![encode(cache), encode(&default_cache)]
    } else {
        vec![encode(&default_cache)]
    };

    let mut best_gpr: Option<GPRegressor> = None;
    let mut best_loss = f64::MAX;

    for init_params in &attempts {
        if let Some((opt_params, loss)) = optimize(init_params) {
            let (inv_sq_ls, kernel_scale, noise_var) = decode(&opt_params);
            if inv_sq_ls.iter().all(|&v| v.is_finite())
                && kernel_scale.is_finite() && noise_var.is_finite()
                && loss < best_loss
            {
                best_loss = loss;
                best_gpr = Some(GPRegressor::new(
                    x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
                    inv_sq_ls, kernel_scale, noise_var,
                ));
            }
        }
    }

    // 所有优化都失败则使用默认参数 (对齐 Python 行为)
    best_gpr.unwrap_or_else(|| GPRegressor::new(
        x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
        vec![1.0; n_params], 1.0, DEFAULT_MINIMUM_NOISE_VAR,
    ))
}

// ════════════════════════════════════════════════════════════════════════
// 采集函数: Log Expected Improvement (logEI)
// ════════════════════════════════════════════════════════════════════════

/// 标准正态 CDF: Φ(z)
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// 标准正态 PDF: φ(z)
pub(crate) fn normal_pdf(z: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * z * z).exp()
}

/// Log Expected Improvement
/// logEI = log(σ * (z * Φ(z) + φ(z)))
/// 其中 z = (μ - f₀) / σ
fn log_ei(mean: f64, var: f64, f0: f64) -> f64 {
    if var < 1e-30 {
        // 方差为零时，EI 为 max(mean - f0, 0)
        return if mean > f0 { (mean - f0).ln() } else { f64::NEG_INFINITY };
    }

    let sigma = var.sqrt();
    let z = (mean - f0) / sigma;

    // 数值稳定的 logEI 计算
    let ei = z * normal_cdf(z) + normal_pdf(z);
    if ei > 1e-30 {
        ei.ln() + sigma.ln()
    } else {
        // 尾部近似（z 很小时）
        -0.5 * z * z - (2.0 * PI).sqrt().ln() + sigma.ln()
    }
}

// ════════════════════════════════════════════════════════════════════════
// 搜索空间归一化
// ════════════════════════════════════════════════════════════════════════

/// 归一化参数到 [0, 1]
pub(crate) fn normalize_param(
    value: f64,
    dist: &Distribution,
) -> f64 {
    match dist {
        Distribution::FloatDistribution(d) => {
            let (low, high) = if d.log {
                (d.low.ln(), d.high.ln())
            } else {
                (d.low, d.high)
            };
            let v = if d.log { value.ln() } else { value };
            let range = high - low;
            if range < 1e-14 { 0.5 } else { (v - low) / range }
        }
        Distribution::IntDistribution(d) => {
            let (low, high) = if d.log {
                ((d.low as f64).ln(), (d.high as f64).ln())
            } else {
                (d.low as f64, d.high as f64)
            };
            let v = if d.log { (value).ln() } else { value };
            let range = high - low;
            if range < 1e-14 { 0.5 } else { (v - low) / range }
        }
        Distribution::CategoricalDistribution(_) => value,
    }
}

/// 反归一化参数
pub(crate) fn unnormalize_param(
    value: f64,
    dist: &Distribution,
) -> f64 {
    match dist {
        Distribution::FloatDistribution(d) => {
            let (low, high) = if d.log {
                (d.low.ln(), d.high.ln())
            } else {
                (d.low, d.high)
            };
            let v = value * (high - low) + low;
            if d.log { v.exp() } else { v }
        }
        Distribution::IntDistribution(d) => {
            let (low, high) = if d.log {
                ((d.low as f64).ln(), (d.high as f64).ln())
            } else {
                (d.low as f64, d.high as f64)
            };
            let v = value * (high - low) + low;
            if d.log { v.exp().round() } else { v.round() }
        }
        Distribution::CategoricalDistribution(_) => value,
    }
}

// ════════════════════════════════════════════════════════════════════════
// GPSampler — 高斯过程采样器
// ════════════════════════════════════════════════════════════════════════

/// 约束函数类型: 接收 FrozenTrial，返回约束值向量。所有值 ≤ 0 表示可行。
pub type ConstraintsFn = Arc<dyn Fn(&FrozenTrial) -> Vec<f64> + Send + Sync>;

/// 高斯过程 (GP) 采样器。
///
/// 对应 Python `optuna.samplers.GPSampler`。
/// 使用 Matern 5/2 核 + logEI 采集函数 + ARD。
pub struct GpSampler {
    /// 随机种子
    seed: Option<u64>,
    /// 优化方向（对齐 Python _sign 符号翻转）
    direction: StudyDirection,
    /// 独立采样器（启动阶段使用）
    independent_sampler: Arc<dyn Sampler>,
    /// 最少启动试验数（使用随机采样）
    n_startup_trials: usize,
    /// 预选候选点数量
    n_preliminary_samples: usize,
    /// 局部搜索次数
    n_local_search: usize,
    /// 是否为确定性目标
    deterministic_objective: bool,
    /// 约束函数
    constraints_func: Option<ConstraintsFn>,
    /// 搜索空间计算器
    search_space: Mutex<IntersectionSearchSpace>,
    /// RNG 共享状态
    rng: Mutex<ChaCha8Rng>,
    /// 目标 GP 核参数缓存: 跨 sample_relative 调用复用上一轮拟合结果。
    /// 对应 Python `self._gprs_cache_list`。
    gprs_cache: Mutex<Option<Vec<KernelParamsCache>>>,
    /// 约束 GP 核参数缓存: 对应 Python `self._constraints_gprs_cache_list`。
    constraints_gprs_cache: Mutex<Option<Vec<KernelParamsCache>>>,
}

impl GpSampler {
    /// 创建 GP 采样器
    ///
    /// # 参数
    /// * `seed` - 随机种子
    /// * `direction` - 优化方向（默认 Minimize）
    /// * `n_startup_trials` - 启动试验数（默认 10）
    /// * `deterministic_objective` - 目标函数是否确定性（默认 false）
    /// * `constraints_func` - 约束函数, 返回值 ≤ 0 表示可行
    /// * `independent_sampler` - 独立采样器（默认 RandomSampler）
    pub fn new(
        seed: Option<u64>,
        direction: Option<StudyDirection>,
        n_startup_trials: Option<usize>,
        deterministic_objective: bool,
        constraints_func: Option<ConstraintsFn>,
        independent_sampler: Option<Arc<dyn Sampler>>,
    ) -> Self {
        let s = seed.unwrap_or_else(|| {
            // 对齐 Python LazyRandomState(seed=None): 使用随机种子
            use std::time::SystemTime;
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });
        Self {
            seed,
            direction: direction.unwrap_or(StudyDirection::Minimize),
            independent_sampler: independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(Some(s + 1)))),
            n_startup_trials: n_startup_trials.unwrap_or(10),
            n_preliminary_samples: 2048,
            n_local_search: 10,
            deterministic_objective: deterministic_objective,
            constraints_func,
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(s)),
            gprs_cache: Mutex::new(None),
            constraints_gprs_cache: Mutex::new(None),
        }
    }

    /// GP 核心采样实现
    fn sample_relative_impl(
        &self,
        completed_trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let param_names: Vec<String> = search_space.keys().cloned().collect();
        let n_params = param_names.len();

        // 判断分类参数
        let is_categorical: Vec<bool> = param_names.iter().map(|name| {
            matches!(search_space[name], Distribution::CategoricalDistribution(_))
        }).collect();

        // 构建归一化的训练数据
        let mut x_train = Vec::new();
        let mut y_train = Vec::new();

        for trial in completed_trials {
            if let Some(vals) = &trial.values {
                let mut row = Vec::with_capacity(n_params);
                let mut complete = true;
                for name in &param_names {
                    if let Some(pv) = trial.params.get(name) {
                        let dist = &search_space[name];
                        let internal = dist.to_internal_repr(pv)?;
                        row.push(normalize_param(internal, dist));
                    } else {
                        complete = false;
                        break;
                    }
                }
                if complete {
                    x_train.push(row);
                    // 对齐 Python: _sign = -1.0 if MINIMIZE else 1.0
                    // 统一为「越大越好」以便 logEI 正确工作
                    let sign = match self.direction {
                        StudyDirection::Minimize | StudyDirection::NotSet => -1.0,
                        StudyDirection::Maximize => 1.0,
                    };
                    let mut v = vals[0] * sign;
                    // 对齐 Python warn_and_convert_inf: 将 ±Inf 裁剪为 ±f64::MAX
                    if v.is_infinite() {
                        crate::optuna_warn!(
                            "Trial {} has infinite objective value; clipping to finite bound.",
                            trial.number
                        );
                        v = if v > 0.0 { f64::MAX } else { f64::MIN };
                    }
                    y_train.push(v);
                }
            }
        }

        if x_train.is_empty() {
            return Ok(HashMap::new());
        }

        // 标准化目标值
        let y_mean: f64 = y_train.iter().sum::<f64>() / y_train.len() as f64;
        let y_std: f64 = {
            let variance = y_train.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>()
                / y_train.len() as f64;
            variance.sqrt().max(EPS)
        };
        let standardized_y: Vec<f64> = y_train.iter()
            .map(|y| (y - y_mean) / y_std)
            .collect();

        // 拟合 GP 超参数 — 带缓存复用
        // 维度检查: 若搜索空间维度变化则清除缓存
        // 对应 Python: if self._gprs_cache_list is not None and len(...) != dim
        let seed = self.seed.unwrap_or(42);
        {
            let mut cache = self.gprs_cache.lock();
            if let Some(ref list) = *cache {
                if !list.is_empty()
                    && list[0].inverse_squared_lengthscales.len() != n_params
                {
                    *cache = None;
                    // 同时清除约束缓存
                    *self.constraints_gprs_cache.lock() = None;
                }
            }
        }

        // 从缓存取出目标 GP 初始参数（单目标只有 1 个）
        let obj_cache = {
            let cache = self.gprs_cache.lock();
            cache.as_ref().and_then(|list| list.first().cloned())
        };
        let gpr = fit_kernel_params(
            &x_train, &standardized_y, &is_categorical, seed,
            obj_cache.as_ref(),
            self.deterministic_objective,
        );

        // 存储本轮拟合的核参数到缓存
        {
            let new_cache = KernelParamsCache {
                inverse_squared_lengthscales: gpr.inverse_squared_lengthscales.clone(),
                kernel_scale: gpr.kernel_scale,
                noise_var: gpr.noise_var,
            };
            *self.gprs_cache.lock() = Some(vec![new_cache]);
        }

        // 约束处理: 获取每个试验的约束值和可行性 — 带缓存
        let constraint_gps: Vec<(GPRegressor, f64)> = if self.constraints_func.is_some() {
            self.build_constraint_gps_cached(completed_trials, &x_train, &is_categorical, seed)
        } else {
            Vec::new()
        };

        // 计算可行性
        let is_feasible: Vec<bool> = if self.constraints_func.is_some() {
            completed_trials.iter().map(|t| {
                t.system_attrs.get("constraints")
                    .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
                    .map(|c| c.iter().all(|cv| *cv <= 0.0))
                    .unwrap_or(false)
            }).collect()
        } else {
            vec![true; completed_trials.len()]
        };

        // 当前最佳标准化值（仅考虑可行试验）
        let f0 = standardized_y.iter().zip(is_feasible.iter())
            .filter(|(_, feas)| **feas)
            .map(|(y, _)| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        // 采集函数优化：随机候选 + 贪心选择
        let eval_acqf = |candidate: &[f64], gpr: &GPRegressor, constraint_gps: &[(GPRegressor, f64)]| -> f64 {
            let (mean, var) = gpr.posterior(candidate);
            let mut acqf = log_ei(mean, var, f0);
            // 加上约束可行性的对数概率
            for (c_gpr, c_threshold) in constraint_gps {
                let (c_mean, c_var) = c_gpr.posterior(candidate);
                let sigma = (c_var + EPS).sqrt();
                let z = (c_mean - c_threshold) / sigma;
                acqf += normal_cdf(z).max(1e-30).ln(); // logPI
            }
            acqf
        };

        let mut best_acqf = f64::NEG_INFINITY;
        let mut best_params = vec![0.5; n_params];

        let mut rng = self.rng.lock();

        // 1. 先考虑已有最佳点（仅可行的）
        if let Some(best_idx) = standardized_y.iter().enumerate()
            .zip(is_feasible.iter())
            .filter(|(_, feas)| **feas)
            .map(|((i, y), _)| (i, y))
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i) {
            let candidate = &x_train[best_idx];
            let acqf = eval_acqf(candidate, &gpr, &constraint_gps);
            if acqf > best_acqf {
                best_acqf = acqf;
                best_params = candidate.clone();
            }
        }

        // 2. 随机候选点评估
        for _ in 0..self.n_preliminary_samples {
            let candidate: Vec<f64> = (0..n_params).map(|d| {
                if is_categorical[d] {
                    match &search_space[&param_names[d]] {
                        Distribution::CategoricalDistribution(c) => {
                            (rng.gen_range(0.0..1.0) * c.choices.len() as f64).floor()
                        }
                        _ => rng.gen_range(0.0..1.0),
                    }
                } else {
                    rng.gen_range(0.0..1.0)
                }
            }).collect();

            let acqf = eval_acqf(&candidate, &gpr, &constraint_gps);
            if acqf > best_acqf {
                best_acqf = acqf;
                best_params = candidate;
            }
        }

        // 3. 局部搜索：在最佳点附近微调
        for _ in 0..self.n_local_search {
            let perturbed: Vec<f64> = (0..n_params).map(|d| {
                if is_categorical[d] {
                    best_params[d]
                } else {
                    let noise = (rng.gen_range(0.0..1.0) - 0.5) * 0.1;
                    (best_params[d] + noise).clamp(0.0, 1.0)
                }
            }).collect();

            let acqf = eval_acqf(&perturbed, &gpr, &constraint_gps);
            if acqf > best_acqf {
                best_acqf = acqf;
                best_params = perturbed;
            }
        }

        // 将归一化参数转回内部表示
        let mut result = HashMap::new();
        for (d, name) in param_names.iter().enumerate() {
            let dist = &search_space[name];
            let internal = unnormalize_param(best_params[d], dist);
            result.insert(name.clone(), internal);
        }

        Ok(result)
    }

    /// 构建约束 GP — 带缓存复用。对每个约束维度独立拟合 GP, 返回 (gpr, threshold) 对。
    /// 对应 Python `_get_constraints_acqf_args` 中的缓存逻辑。
    fn build_constraint_gps_cached(
        &self,
        trials: &[FrozenTrial],
        x_train: &[Vec<f64>],
        is_categorical: &[bool],
        seed: u64,
    ) -> Vec<(GPRegressor, f64)> {
        // 收集约束值: 翻转符号 (使得 >0 = 可行)
        let mut constraint_matrix: Vec<Vec<f64>> = Vec::new(); // [n_constraints][n_trials]

        for (i, trial) in trials.iter().enumerate() {
            let cvals = trial.system_attrs.get("constraints")
                .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
                .unwrap_or_default();
            if i == 0 {
                constraint_matrix.resize(cvals.len(), Vec::new());
            }
            for (j, cv) in cvals.iter().enumerate() {
                if j < constraint_matrix.len() {
                    constraint_matrix[j].push(-cv); // 翻转
                }
            }
        }

        // 取出约束缓存
        let cached = self.constraints_gprs_cache.lock().clone();

        let mut result = Vec::new();
        let mut new_caches = Vec::new();
        for (idx, constraint_vals) in constraint_matrix.iter().enumerate() {
            if constraint_vals.len() != x_train.len() {
                continue;
            }
            // 标准化
            let mean = constraint_vals.iter().sum::<f64>() / constraint_vals.len() as f64;
            let std = {
                let var = constraint_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / constraint_vals.len() as f64;
                var.sqrt().max(EPS)
            };
            let standardized: Vec<f64> = constraint_vals.iter()
                .map(|v| (v - mean) / std)
                .collect();
            let threshold = -mean / std; // 对应原始空间中的 0

            // 从缓存取第 idx 个约束 GP 的核参数
            let c_cache = cached.as_ref().and_then(|list| list.get(idx).cloned());
            let gpr = fit_kernel_params(
                x_train, &standardized, is_categorical, seed + 100 + idx as u64,
                c_cache.as_ref(),
                false, // constraints are not deterministic
            );

            // 存储核参数用于下次复用
            new_caches.push(KernelParamsCache {
                inverse_squared_lengthscales: gpr.inverse_squared_lengthscales.clone(),
                kernel_scale: gpr.kernel_scale,
                noise_var: gpr.noise_var,
            });

            result.push((gpr, threshold));
        }

        // 更新约束缓存
        *self.constraints_gprs_cache.lock() = Some(new_caches);

        result
    }
}

impl Sampler for GpSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        // 对齐 Python: 过滤掉 single() 分布（只有一个可能值的分布无需 GP 建模）
        self.search_space
            .lock()
            .calculate(trials)
            .into_iter()
            .filter(|(_, dist)| !dist.single())
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

        // 获取已完成的试验
        let completed: Vec<&FrozenTrial> = trials.iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        // 启动阶段使用独立采样器
        if completed.len() < self.n_startup_trials {
            return Ok(HashMap::new());
        }

        let completed_owned: Vec<FrozenTrial> = completed.into_iter().cloned().collect();
        self.sample_relative_impl(&completed_owned, search_space)
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // 对齐 Python: 当已有足够完成试验但仍回退到独立采样时发出警告
        let n_complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
        if n_complete >= self.n_startup_trials {
            crate::optuna_warn!(
                "GpSampler is falling back to independent sampling for param '{}' in trial {}.",
                param_name,
                trial.number
            );
        }
        self.independent_sampler.sample_independent(trials, trial, param_name, distribution)
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        _values: Option<&[f64]>,
    ) {
        // 对齐 Python _process_constraints_after_trial:
        // 计算约束值并存储到 trial.system_attrs["constraints"]
        if let Some(ref cf) = self.constraints_func {
            if state == TrialState::Complete || state == TrialState::Pruned {
                let constraints = cf(trial);
                // 约束值存储到 system_attrs（由 study.tell 在外层完成）
                // 此处将约束值暂存到 trial 的 system_attrs 中
                // 注意: 这需要在 tell_with_options 中通过 compute_constraints 实现
                // GP sampler 的约束处理与 TPE 共享 CONSTRAINTS_KEY 机制
                let _ = constraints; // 约束存储由 study.tell_with_options 中的 compute_constraints 统一处理
            }
        }
    }

    /// 对齐 Python `GPSampler.reseed_rng(seed)`: 重新设置随机种子。
    fn reseed_rng(&self, seed: u64) {
        *self.rng.lock() = ChaCha8Rng::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matern52_at_zero() {
        let k = matern52(0.0);
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matern52_decays() {
        let k1 = matern52(0.1);
        let k2 = matern52(1.0);
        let k3 = matern52(10.0);
        assert!(k1 > k2);
        assert!(k2 > k3);
        assert!(k3 > 0.0);
    }

    #[test]
    fn test_cholesky() {
        // 2x2 正定矩阵
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        assert!((l[0][0] - 2.0).abs() < 1e-10);
        assert!((l[1][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - (2.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((normal_cdf(f64::INFINITY) - 1.0).abs() < 1e-10);
        assert!(normal_cdf(f64::NEG_INFINITY) < 1e-10);
    }

    #[test]
    fn test_log_ei_positive() {
        // 当 mean > f0 时，EI 应该为正
        let lei = log_ei(1.0, 1.0, 0.0);
        assert!(lei > f64::NEG_INFINITY);
    }

    #[test]
    fn test_gp_posterior_at_training_point() {
        // GP 在训练点处的后验应接近训练值
        let x_train = vec![vec![0.0], vec![1.0]];
        let y_train = vec![0.0, 1.0];
        let is_cat = vec![false];
        let gpr = GPRegressor::new(
            x_train, y_train, is_cat,
            vec![1.0], 1.0, 1e-6,
        );
        let (mean, var) = gpr.posterior(&[0.0]);
        assert!((mean - 0.0).abs() < 0.1, "mean at x=0: {mean}");
        assert!(var < 0.1, "var at x=0 should be small: {var}");
    }

    #[test]
    fn test_gp_posterior_interpolation() {
        // GP 在训练点之间应该有较大的不确定性
        let x_train = vec![vec![0.0], vec![1.0]];
        let y_train = vec![0.0, 1.0];
        let is_cat = vec![false];
        let gpr = GPRegressor::new(
            x_train, y_train, is_cat,
            vec![1.0], 1.0, 1e-6,
        );
        let (mean, var) = gpr.posterior(&[0.5]);
        // 应在 0 和 1 之间
        assert!(mean > -0.5 && mean < 1.5, "mean at x=0.5: {mean}");
        // 在训练点之间不确定性较大
        let (_, var_at_train) = gpr.posterior(&[0.0]);
        assert!(var > var_at_train, "var between points should be larger");
    }

    #[test]
    fn test_gp_sampler_creation() {
        let _sampler = GpSampler::new(Some(42), None, None, false, None, None);
    }

    #[test]
    fn test_gp_sampler_optimize() {
        let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(Some(42), Some(StudyDirection::Minimize), Some(5), false, None, None));
        let study = crate::study::create_study(
            None, Some(sampler), None, None,
            Some(crate::study::StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(x * x)
        }, Some(30), None, None).unwrap();

        let best = study.best_value().unwrap();
        // GP 应该找到接近 0 的最小值（x=0 时 x^2=0）
        assert!(best < 5.0, "GP should find good solution, got {best}");
    }

    /// 测试 GP 参数缓存机制:
    /// 1. 第一轮调用后缓存应该被填充
    /// 2. 后续调用应复用缓存作为初始参数
    /// 对应 Python `test_gpsampler_cache` 逻辑
    #[test]
    fn test_gp_kernel_cache() {
        let sampler = GpSampler::new(Some(42), Some(StudyDirection::Minimize), Some(3), false, None, None);

        // 初始时缓存为空
        assert!(sampler.gprs_cache.lock().is_none());

        // 构造模拟试验数据
        let mut trials = Vec::new();
        for i in 0..5 {
            let x = i as f64 * 0.25;
            let y = x * x;
            let mut params = HashMap::new();
            params.insert("x".to_string(),
                crate::distributions::ParamValue::Float(x));
            let mut dists = HashMap::new();
            dists.insert("x".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution {
                        low: 0.0, high: 1.0, log: false, step: None,
                    },
                ));
            trials.push(FrozenTrial {
                number: i as i64,
                trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![y]),
                params,
                distributions: dists,
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            });
        }

        let mut search_space = IndexMap::new();
        search_space.insert("x".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution {
                    low: 0.0, high: 1.0, log: false, step: None,
                },
            ));

        // 第一次调用 — 应该填充缓存
        let _result = sampler.sample_relative_impl(&trials, &search_space).unwrap();
        {
            let cache = sampler.gprs_cache.lock();
            assert!(cache.is_some(), "缓存应该在第一次调用后被填充");
            let list = cache.as_ref().unwrap();
            assert_eq!(list.len(), 1, "单目标应有 1 个缓存条目");
            assert_eq!(list[0].inverse_squared_lengthscales.len(), 1,
                "单参数应有 1 个长度尺度");
        }

        // 记录第一次的缓存参数
        let first_cache = sampler.gprs_cache.lock().as_ref().unwrap()[0].clone();

        // 第二次调用 — 应该使用缓存作为初始值
        let _result2 = sampler.sample_relative_impl(&trials, &search_space).unwrap();
        {
            let cache = sampler.gprs_cache.lock();
            assert!(cache.is_some(), "缓存应该在第二次调用后仍存在");
        }

        // 测试维度变化时缓存失效
        // 添加新参数 y，此时搜索空间维度从 1 变为 2
        let mut search_space2 = search_space.clone();
        search_space2.insert("y".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution {
                    low: 0.0, high: 1.0, log: false, step: None,
                },
            ));
        // 给试验添加 y 参数
        for (i, trial) in trials.iter_mut().enumerate() {
            trial.params.insert("y".to_string(),
                crate::distributions::ParamValue::Float(i as f64 * 0.2));
            trial.distributions.insert("y".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution {
                        low: 0.0, high: 1.0, log: false, step: None,
                    },
                ));
        }

        let _result3 = sampler.sample_relative_impl(&trials, &search_space2).unwrap();
        {
            let cache = sampler.gprs_cache.lock();
            assert!(cache.is_some());
            let list = cache.as_ref().unwrap();
            // 新搜索空间有 2 个参数，缓存应更新
            assert_eq!(list[0].inverse_squared_lengthscales.len(), 2,
                "维度变化后缓存应更新为新维度");
        }
    }

    /// 测试 fit_kernel_params 的 gpr_cache 参数:
    /// 使用缓存初始值应该产生不差于无缓存的结果。
    #[test]
    fn test_fit_kernel_params_with_cache() {
        let x = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
        let y = vec![0.0, 0.0625, 0.25, 0.5625, 1.0];
        let is_cat = vec![false];

        // 无缓存拟合
        let gpr1 = fit_kernel_params(&x, &y, &is_cat, 42, None, false);
        let lml1 = gpr1.log_marginal_likelihood();

        // 用 gpr1 的参数作为缓存
        let cache = KernelParamsCache {
            inverse_squared_lengthscales: gpr1.inverse_squared_lengthscales.clone(),
            kernel_scale: gpr1.kernel_scale,
            noise_var: gpr1.noise_var,
        };
        let gpr2 = fit_kernel_params(&x, &y, &is_cat, 42, Some(&cache), false);
        let lml2 = gpr2.log_marginal_likelihood();

        // 有缓存的结果不应该更差（缓存提供了一个额外的候选初始点）
        assert!(lml2 >= lml1 - 1e-6,
            "使用缓存的 LML ({lml2}) 不应差于无缓存 ({lml1})");
    }
}
