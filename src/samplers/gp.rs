//! 高斯过程 (GP) 采样器模块 — Matern 5/2 核 + logEI/logEHVI 采集函数
//!
//! 对应 Python `optuna.samplers.GPSampler`。
//! 纯 Rust 实现，无需 PyTorch / scipy 依赖。
//!
//! ## 算法概述
//! 1. 使用 Matern 5/2 核的高斯过程回归拟合已完成试验
//! 2. 通过 L-BFGS-B 优化核超参数（最大化边际似然）
//! 3. 单目标: logEI（对数期望改善）采集函数
//! 4. 多目标: logEHVI（对数期望超体积改善）采集函数 + QMC 采样
//! 5. 支持自动相关性确定 (ARD) — 每个维度独立的长度尺度

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
/// 对齐 Python `_EPS = 1e-12` (stabilizing noise)。
/// Python 在 LogEI/LogPI/LogEHVI 中使用 `stabilizing_noise=1e-12` 加到方差上。
const STABILIZING_NOISE: f64 = 1e-12;
/// 对齐 Python `_SQRT_HALF = math.sqrt(0.5)`
const SQRT_HALF: f64 = 0.7071067811865475244;
/// 对齐 Python `_INV_SQRT_2PI = 1 / sqrt(2π)`
const INV_SQRT_2PI: f64 = 0.3989422804014326779;
/// 对齐 Python `_SQRT_HALF_PI = sqrt(π/2)`
const SQRT_HALF_PI: f64 = 1.2533141373155002512;
/// 对齐 Python `_LOG_SQRT_2PI = log(sqrt(2π))`
const LOG_SQRT_2PI: f64 = 0.9189385332046727418;

// ════════════════════════════════════════════════════════════════════════
// Matern 5/2 核函数
// ════════════════════════════════════════════════════════════════════════

/// Matern 5/2 核: k(d²) = exp(-√(5d²)) * (5/3 * d² + √(5d²) + 1)
pub fn matern52(squared_distance: f64) -> f64 {
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
pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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
pub fn solve_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
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
pub fn solve_upper(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
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
pub struct GPRegressor {
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
    pub fn new(
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
    pub fn train_kernel_matrix(&self) -> Vec<Vec<f64>> {
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
    pub fn posterior(&self, x: &[f64]) -> (f64, f64) {
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

    /// 追加 running trials 的假数据（constant liar 策略）。
    ///
    /// 对齐 Python `GPRegressor.append_running_data`:
    /// 将 running trials 的归一化参数作为额外训练点加入 GP，
    /// 使用当前 y_train.max() 作为假目标值（"最佳常量说谎者"策略），
    /// 然后重新计算 Cholesky 分解。
    ///
    /// 这使得 GP 在并行优化时不会重复探索 running trials 已经在探索的区域。
    pub(crate) fn append_running_data(
        &mut self,
        x_running: &[Vec<f64>],
        constant_liar_value: f64,
    ) {
        if x_running.is_empty() {
            return;
        }
        for xr in x_running {
            self.x_train.push(xr.clone());
            self.y_train.push(constant_liar_value);
        }
        // 重新计算 Cholesky 分解以包含新数据
        self.cache_cholesky();
    }

    /// 边际对数似然: log p(y | X, θ)
    pub fn log_marginal_likelihood(&self) -> f64 {
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
pub fn default_log_prior(
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
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// 标准正态 PDF: φ(z)
pub fn normal_pdf(z: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * z * z).exp()
}

/// Log Expected Improvement — 对齐 Python `standard_logei` + `logei`。
///
/// 完整对齐 Python 的两段式 logEI:
/// 1. 主分支 (z >= -25): log(z/2 * erfc(-z/√2) + exp(-z²/2)/√(2π))
/// 2. 尾部分支 (z < -25): -z²/2 - log(√(2π)) + log(1 + √(π/2) * z * erfcx(-z/√2))
///    使用 erfcx 避免精度丢失。
///
/// logEI = standard_logei(z) + log(σ), 其中 z = (μ - f₀) / σ。
pub fn log_ei(mean: f64, var: f64, f0: f64) -> f64 {
    if var < 1e-30 {
        // 方差为零时，EI 为 max(mean - f0, 0)
        return if mean > f0 { (mean - f0).ln() } else { f64::NEG_INFINITY };
    }

    let sigma = var.sqrt();
    let z = (mean - f0) / sigma;

    // 对齐 Python `standard_logei` 的两段式计算
    let standard_lei = if z >= -25.0 {
        // 主分支: log(z/2 * erfc(-√½ * z) + exp(-z²/2) / √(2π))
        let z_half = 0.5 * z;
        let cdf_term = z_half * libm::erfc(-SQRT_HALF * z);     // z * Φ(z)
        let pdf_term = (-z_half * z).exp() * INV_SQRT_2PI;       // φ(z)
        let ei = cdf_term + pdf_term;
        if ei > 0.0 { ei.ln() } else { f64::NEG_INFINITY }
    } else {
        // 尾部分支 (z < -25): 使用 erfcx 保持高精度
        // -z²/2 - log(√(2π)) + log(1 + √(π/2) * z * erfcx(-√½ * z))
        let erfcx_val = erfcx(-SQRT_HALF * z);
        -0.5 * z * z - LOG_SQRT_2PI
            + (1.0 + SQRT_HALF_PI * z * erfcx_val).ln()
    };

    standard_lei + sigma.ln()
}

/// erfcx(x) = exp(x²) * erfc(x) — 缩放互补误差函数。
///
/// 对齐 Python `torch.special.erfcx`。
/// 对大 x 值保持精度（erfc 趋近 0，但 erfcx 趋近 1/(x√π)）。
/// 使用 Abramowitz & Stegun 近似。
pub fn erfcx(x: f64) -> f64 {
    // 对于负数大值: erfcx(x) = exp(x²) * erfc(x) ≈ 2*exp(x²) 会爆炸
    // 但在我们的 logEI 尾部分支中 x > 0 (因为 -SQRT_HALF * z > 0 when z < -25)
    if x < 0.0 {
        // erfcx(x) = exp(x²) * erfc(x)
        // 对于 x << 0, erfc(x) ≈ 2, 所以 erfcx(x) ≈ 2*exp(x²) — 可能极大
        return (x * x).exp() * libm::erfc(x);
    }
    // 对于 x >= 0: 使用连分数展开近似
    // erfcx(x) ≈ 1/(√π * x) * (1 - 1/(2x²) + 3/(4x⁴) - ...) for large x
    if x > 26.0 {
        // 渐近展开: erfcx(x) ≈ 1/(√π * x) * Σ (-1)^n (2n-1)!! / (2x²)^n
        let inv_2x2 = 1.0 / (2.0 * x * x);
        let inv_sqrt_pi = 1.0 / PI.sqrt();
        return inv_sqrt_pi / x * (1.0 - inv_2x2 * (1.0 - 3.0 * inv_2x2));
    }
    // 中间范围: 直接计算 exp(x²) * erfc(x)
    (x * x).exp() * libm::erfc(x)
}

/// 对数标准正态 CDF: log Φ(z) — 高精度实现。
///
/// 对齐 Python `torch.special.log_ndtr`:
/// - z >= -5: 直接 log(0.5 * erfc(-z/√2))
/// - z < -5: 使用渐近展开避免 log(0) 下溢
///
/// 这比 `normal_cdf(z).max(1e-30).ln()` 精确得多。
pub fn log_ndtr(z: f64) -> f64 {
    if z > 6.0 {
        // Φ(z) ≈ 1 for z > 6, log Φ(z) ≈ 0
        // 更精确: log Φ(z) = log(1 - Q(z)) ≈ -Q(z) where Q(z) = 1-Φ(z)
        // Q(z) ≈ φ(z)/z 用 Mills ratio
        let log_q = -0.5 * z * z - LOG_SQRT_2PI - z.ln();
        return (-log_q.exp()).ln_1p(); // log(1 - Q(z))
    }
    if z >= -26.0 {
        // 直接计算: log(0.5 * erfc(-z/√2))
        // erfc 对 x ∈ [0, 18.4] (即 z ∈ [-26, ...]) 不会下溢到零
        let cdf = 0.5 * libm::erfc(-z * SQRT_HALF);
        return cdf.ln();
    }
    // 尾部渐近展开 (z < -26):
    // log Φ(z) = -z²/2 - log(√(2π)) - log|z| + log(series)
    // Φ(z) = φ(z)/|z| * Σ_{k=0}^∞ (-1)^k (2k-1)!! / z^{2k}
    // 使用 Horner 形式: 1 - 1/z²(1 - 3/z²(1 - 5/z²(1 - 7/z²(1 - 9/z²))))
    let z2 = z * z;
    let log_phi = -0.5 * z2 - LOG_SQRT_2PI;
    let abs_z = -z; // z < 0, abs_z > 0
    let inv_z2 = 1.0 / z2;
    log_phi - abs_z.ln()
        + (1.0 - inv_z2 * (1.0 - 3.0 * inv_z2 * (1.0 - 5.0 * inv_z2
            * (1.0 - 7.0 * inv_z2 * (1.0 - 9.0 * inv_z2))))).ln()
}

// ════════════════════════════════════════════════════════════════════════
// 搜索空间归一化
// ════════════════════════════════════════════════════════════════════════

/// 生成 Sobol 正态样本矩阵 [n_samples, dim]。
///
/// 对齐 Python `_sample_from_normal_sobol`:
/// 1. Sobol' QMC 生成 [0,1] 均匀样本
/// 2. 将 [0,1] 映射到 [-1,1]: x → 2(x - 0.5)
/// 3. 通过 erfinv 转化为标准正态: z = √2 * erfinv(2x - 1)
pub(crate) fn sample_from_normal_sobol(dim: usize, n_samples: usize, seed: u64) -> Vec<Vec<f64>> {
    use super::qmc::sobol_point_pub;
    let sqrt2 = std::f64::consts::SQRT_2;
    (0..n_samples)
        .map(|i| {
            let sobol = sobol_point_pub(i as u64 + 1, dim, true, seed); // skip index 0
            sobol
                .into_iter()
                .map(|u| {
                    // 映射到 [-1,1] 然后 erfinv → 标准正态
                    let mapped = 2.0 * (u - 0.5);
                    // 裁剪到 (-1+ε, 1-ε) 避免 erfinv 的无穷大
                    let clamped = mapped.clamp(-0.99999, 0.99999);
                    sqrt2 * erfinv(clamped)
                })
                .collect()
        })
        .collect()
}

/// 逆误差函数 erfinv(x) — 高精度实现。
///
/// 对齐 Python `torch.erfinv` / `scipy.special.erfinv`。
/// 使用 Winitzki 近似作为初始值，加 2 轮 Halley 迭代收敛到机器精度。
fn erfinv(x: f64) -> f64 {
    if x.abs() >= 1.0 {
        return if x > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if x.abs() < 1e-15 {
        return x; // erfinv(0) = 0
    }
    // Winitzki 初始近似 (max relative error ~0.2%)
    let a = 0.147;
    let ln_part = (1.0 - x * x).ln();
    let b = 2.0 / (PI * a) + 0.5 * ln_part;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let mut y = sign * (((b * b - ln_part / a).sqrt() - b).sqrt());

    // Halley 迭代精炼: 收敛到机器精度 (~1e-15)
    // erf'(y) = 2/√π * exp(-y²)
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..2 {
        let ey = libm::erf(y);
        let deriv = two_over_sqrt_pi * (-y * y).exp();
        if deriv.abs() < 1e-300 { break; } // 避免除零
        let correction = (ey - x) / deriv;
        // Halley: y_new = y - correction / (1 + y * correction)
        y -= correction / (1.0 + y * correction);
    }
    y
}

/// Log Expected Hypervolume Improvement (logEHVI) 计算。
///
/// 对齐 Python `acqf.logehvi`:
/// 给定后验样本 Y_post [n_qmc_samples, n_objectives]，
/// 和非支配盒分解 (lower_bounds, intervals)，
/// 计算 log(E[HVI])。
///
/// Y_post 中的值已是「越大越好」的标准化分数。
///
/// 对齐 Python: 使用 `diff.clamp_(min=EPS, max=interval)` 而非跳过整个盒。
/// 这保留了微小贡献，避免丢失信息。
fn log_ehvi(
    y_post: &[Vec<f64>],        // [n_qmc_samples, n_objectives]
    box_lower: &[Vec<f64>],     // [n_boxes, n_objectives]
    box_intervals: &[Vec<f64>], // [n_boxes, n_objectives]
) -> f64 {
    let n_qmc = y_post.len();
    if n_qmc == 0 {
        return f64::NEG_INFINITY;
    }
    let log_n_qmc = (n_qmc as f64).ln();

    // 对齐 Python: diff.clamp_(min=EPS, max=interval), 然后 log().sum()
    // 再对 (qmc_samples × boxes) 做 logsumexp
    let mut log_vals: Vec<f64> = Vec::new();
    for sample in y_post {
        for (lb, interval) in box_lower.iter().zip(box_intervals.iter()) {
            let mut log_prod = 0.0;
            for d in 0..sample.len() {
                // 对齐 Python: diff = (sample - lower).clamp(EPS, interval)
                let diff = (sample[d] - lb[d]).clamp(STABILIZING_NOISE, interval[d]);
                log_prod += diff.ln();
            }
            log_vals.push(log_prod);
        }
    }

    if log_vals.is_empty() {
        return f64::NEG_INFINITY;
    }

    // logsumexp
    let max_log = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_log == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = log_vals.iter().map(|&v| (v - max_log).exp()).sum();
    max_log + sum_exp.ln() - log_n_qmc
}

/// 检测 Pareto 前沿（loss/minimization 空间，值越小越好）。
///
/// 对齐 Python `_is_pareto_front(loss_vals)`。
/// 返回布尔向量: true = 该点在 Pareto 前沿上。
fn is_pareto_front_min(loss_values: &[Vec<f64>]) -> Vec<bool> {
    let n = loss_values.len();
    let mut on_front = vec![true; n];
    for i in 0..n {
        if !on_front[i] { continue; }
        for j in 0..n {
            if i == j || !on_front[j] { continue; }
            // j 支配 i ?（所有 <= 且至少一个 <）
            let mut all_le = true;
            let mut any_lt = false;
            for d in 0..loss_values[i].len() {
                if loss_values[j][d] > loss_values[i][d] {
                    all_le = false;
                    break;
                }
                if loss_values[j][d] < loss_values[i][d] {
                    any_lt = true;
                }
            }
            if all_le && any_lt {
                on_front[i] = false;
                break;
            }
        }
    }
    on_front
}

/// 归一化参数到 [0, 1]
pub fn normalize_param(
    value: f64,
    dist: &Distribution,
) -> f64 {
    match dist {
        Distribution::FloatDistribution(d) => {
            let step = d.step.unwrap_or(0.0);
            let (raw_low, raw_high) = if d.log {
                // 对齐 Python: log 空间下先扩展再取 log
                ((d.low - 0.5 * step).ln(), (d.high + 0.5 * step).ln())
            } else {
                (d.low - 0.5 * step, d.high + 0.5 * step)
            };
            let v = if d.log { value.ln() } else { value };
            let range = raw_high - raw_low;
            if range < 1e-14 { 0.5 } else { (v - raw_low) / range }
        }
        Distribution::IntDistribution(d) => {
            let step = d.step as f64;
            let (raw_low, raw_high) = if d.log {
                ((d.low as f64 - 0.5 * step).ln(), (d.high as f64 + 0.5 * step).ln())
            } else {
                (d.low as f64 - 0.5 * step, d.high as f64 + 0.5 * step)
            };
            let v = if d.log { (value).ln() } else { value };
            let range = raw_high - raw_low;
            if range < 1e-14 { 0.5 } else { (v - raw_low) / range }
        }
        Distribution::CategoricalDistribution(_) => value,
    }
}

/// 反归一化参数
///
/// 对齐 Python `_unnormalize_one_param`:
/// 使用 step-adjusted bounds: `[low - 0.5*step, high + 0.5*step]`
pub fn unnormalize_param(
    value: f64,
    dist: &Distribution,
) -> f64 {
    match dist {
        Distribution::FloatDistribution(d) => {
            let step = d.step.unwrap_or(0.0);
            let (raw_low, raw_high) = if d.log {
                ((d.low - 0.5 * step).ln(), (d.high + 0.5 * step).ln())
            } else {
                (d.low - 0.5 * step, d.high + 0.5 * step)
            };
            let v = value * (raw_high - raw_low) + raw_low;
            let v = if d.log { v.exp() } else { v };
            v.clamp(d.low, d.high)
        }
        Distribution::IntDistribution(d) => {
            let step = d.step as f64;
            let (raw_low, raw_high) = if d.log {
                ((d.low as f64 - 0.5 * step).ln(), (d.high as f64 + 0.5 * step).ln())
            } else {
                (d.low as f64 - 0.5 * step, d.high as f64 + 0.5 * step)
            };
            let v = value * (raw_high - raw_low) + raw_low;
            let v = if d.log { crate::search_space::round_ties_even(v.exp()) } else { crate::search_space::round_ties_even(v) };
            v.clamp(d.low as f64, d.high as f64)
        }
        Distribution::CategoricalDistribution(_) => value,
    }
}

// ════════════════════════════════════════════════════════════════════════
// GPSampler — 高斯过程采样器
// ════════════════════════════════════════════════════════════════════════

/// 对齐 Python `gp.warn_and_convert_inf`:
/// 将非有限值（±inf）裁剪到该列已有有限值的 [min, max] 范围。
/// 若某列全部非有限，则裁剪到 0.0。这保持了数值稳定性，
/// 避免 GP 拟合时因极端值导致不稳定。
///
/// Python 原始实现:
/// ```python
/// is_any_finite = np.any(is_values_finite, axis=0)
/// np.clip(values,
///     np.where(is_any_finite, np.min(np.where(finite, values, inf), axis=0), 0.0),
///     np.where(is_any_finite, np.max(np.where(finite, values, -inf), axis=0), 0.0))
/// ```
fn warn_and_convert_inf(score_vals: &mut [Vec<f64>], n_objectives: usize) {
    // 检查是否有非有限值
    let has_nonfinite = score_vals.iter().any(|row| row.iter().any(|v| !v.is_finite()));
    if !has_nonfinite {
        return;
    }

    crate::optuna_warn!("Clip non-finite values to the min/max finite values for GP fittings.");

    // 逐列计算有限值的 min/max
    for obj in 0..n_objectives {
        let finite_vals: Vec<f64> = score_vals.iter()
            .map(|row| row[obj])
            .filter(|v| v.is_finite())
            .collect();

        let (clip_min, clip_max) = if finite_vals.is_empty() {
            // 该列全部非有限 → 设为 0.0
            (0.0, 0.0)
        } else {
            let min_val = finite_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = finite_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_val, max_val)
        };

        // 裁剪该列的非有限值
        for row in score_vals.iter_mut() {
            if !row[obj].is_finite() {
                row[obj] = row[obj].clamp(clip_min, clip_max);
            }
        }
    }
}

/// 约束函数类型: 接收 FrozenTrial，返回约束值向量。所有值 ≤ 0 表示可行。
pub type ConstraintsFn = Arc<dyn Fn(&FrozenTrial) -> Vec<f64> + Send + Sync>;

/// 高斯过程 (GP) 采样器。
///
/// 对应 Python `optuna.samplers.GPSampler`。
/// 单目标: Matern 5/2 核 + logEI 采集函数 + ARD。
/// 多目标: Matern 5/2 核 + logEHVI 采集函数（QMC 采样）。
pub struct GpSampler {
    /// 随机种子
    seed: Option<u64>,
    /// 优化方向（对齐 Python study.directions）
    /// 单目标时 len==1, 多目标时 len>1。
    directions: Vec<StudyDirection>,
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
    /// QMC 采样数量（用于 LogEHVI 多目标采集函数）
    /// 对应 Python `n_qmc_samples=128`。
    n_qmc_samples: usize,
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
    /// * `direction` - 单目标优化方向（与 `directions` 互斥）
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
        let dirs = vec![direction.unwrap_or(StudyDirection::Minimize)];
        Self::with_directions(seed, dirs, n_startup_trials, deterministic_objective, constraints_func, independent_sampler)
    }

    /// 创建支持多目标的 GP 采样器。
    ///
    /// 对齐 Python `GPSampler` 通过 `study.directions` 获取多目标信息。
    /// Rust 版本在构造时传入 directions。
    pub fn with_directions(
        seed: Option<u64>,
        directions: Vec<StudyDirection>,
        n_startup_trials: Option<usize>,
        deterministic_objective: bool,
        constraints_func: Option<ConstraintsFn>,
        independent_sampler: Option<Arc<dyn Sampler>>,
    ) -> Self {
        let s = seed.unwrap_or_else(|| {
            use std::time::SystemTime;
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });
        Self {
            seed,
            directions,
            independent_sampler: independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(Some(s + 1)))),
            n_startup_trials: n_startup_trials.unwrap_or(10),
            n_preliminary_samples: 2048,
            n_local_search: 10,
            deterministic_objective,
            constraints_func,
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
            rng: Mutex::new(ChaCha8Rng::seed_from_u64(s)),
            n_qmc_samples: 128,
            gprs_cache: Mutex::new(None),
            constraints_gprs_cache: Mutex::new(None),
        }
    }

    /// GP 核心采样实现（支持单目标 + 多目标）。
    ///
    /// 单目标: logEI 采集函数（对齐 Python `acqf.LogEI`）。
    /// 多目标: logEHVI 采集函数（对齐 Python `acqf.LogEHVI`），
    ///         使用 QMC Sobol 采样近似后验超体积改善。
    fn sample_relative_impl(
        &self,
        completed_trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
        running_trials: &[FrozenTrial],
    ) -> Result<HashMap<String, f64>> {
        let param_names: Vec<String> = search_space.keys().cloned().collect();
        let n_params = param_names.len();
        let n_objectives = self.directions.len();

        // 判断分类参数
        let is_categorical: Vec<bool> = param_names.iter().map(|name| {
            matches!(search_space[name], Distribution::CategoricalDistribution(_))
        }).collect();

        // 构建归一化的训练数据
        let mut x_train: Vec<Vec<f64>> = Vec::new();
        // 多目标: score_vals[trial_idx][obj_idx]（带符号翻转，越大越好）
        let mut score_vals: Vec<Vec<f64>> = Vec::new();

        // 对齐 Python: _sign = -1.0 if MINIMIZE else 1.0
        let signs: Vec<f64> = self.directions.iter().map(|d| {
            match d {
                StudyDirection::Minimize | StudyDirection::NotSet => -1.0,
                StudyDirection::Maximize => 1.0,
            }
        }).collect();

        for trial in completed_trials {
            if let Some(vals) = &trial.values {
                if vals.len() != n_objectives { continue; }
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
                    let signed_vals: Vec<f64> = vals.iter().enumerate().map(|(j, &v)| {
                        v * signs[j]
                    }).collect();
                    score_vals.push(signed_vals);
                }
            }
        }

        if x_train.is_empty() {
            return Ok(HashMap::new());
        }

        // 对齐 Python `gp.warn_and_convert_inf`:
        // 将非有限值裁剪到该列有限值的 [min, max] 范围，而非 f64::MAX/MIN。
        // 若某列全部非有限，则设为 0.0。
        warn_and_convert_inf(&mut score_vals, n_objectives);

        let n_trials = x_train.len();

        // 标准化每个目标的值
        // standardized_score_vals[obj][trial]
        let mut standardized_by_obj: Vec<Vec<f64>> = Vec::with_capacity(n_objectives);
        let mut means: Vec<f64> = Vec::with_capacity(n_objectives);
        let mut stds: Vec<f64> = Vec::with_capacity(n_objectives);
        for obj in 0..n_objectives {
            let col: Vec<f64> = score_vals.iter().map(|sv| sv[obj]).collect();
            let mean = col.iter().sum::<f64>() / n_trials as f64;
            let std_val = {
                let variance = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_trials as f64;
                variance.sqrt().max(EPS)
            };
            let standardized: Vec<f64> = col.iter().map(|v| (v - mean) / std_val).collect();
            standardized_by_obj.push(standardized);
            means.push(mean);
            stds.push(std_val);
        }

        // standardized_score_vals[trial][obj]
        let standardized_score_vals: Vec<Vec<f64>> = (0..n_trials).map(|t| {
            (0..n_objectives).map(|o| standardized_by_obj[o][t]).collect()
        }).collect();

        // 拟合 GP 超参数 — 带缓存复用
        let seed = self.seed.unwrap_or(42);
        {
            let mut cache = self.gprs_cache.lock();
            if let Some(ref list) = *cache {
                if !list.is_empty()
                    && list[0].inverse_squared_lengthscales.len() != n_params
                {
                    *cache = None;
                    *self.constraints_gprs_cache.lock() = None;
                }
            }
        }

        // 对齐 Python: 每个目标独立拟合一个 GP
        let mut gprs_list: Vec<GPRegressor> = Vec::with_capacity(n_objectives);
        let mut new_caches: Vec<KernelParamsCache> = Vec::with_capacity(n_objectives);
        for obj in 0..n_objectives {
            let cache_entry = {
                let cache = self.gprs_cache.lock();
                cache.as_ref().and_then(|list| list.get(obj).cloned())
            };
            let gpr = fit_kernel_params(
                &x_train, &standardized_by_obj[obj], &is_categorical, seed,
                cache_entry.as_ref(),
                self.deterministic_objective,
            );
            new_caches.push(KernelParamsCache {
                inverse_squared_lengthscales: gpr.inverse_squared_lengthscales.clone(),
                kernel_scale: gpr.kernel_scale,
                noise_var: gpr.noise_var,
            });
            gprs_list.push(gpr);
        }
        *self.gprs_cache.lock() = Some(new_caches);

        // 约束处理
        let constraint_gps: Vec<(GPRegressor, f64)> = if self.constraints_func.is_some() {
            self.build_constraint_gps_cached(completed_trials, &x_train, &is_categorical, seed)
        } else {
            Vec::new()
        };

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

        // ═══ 构建采集函数 + optimize_acqf_mixed ═══
        let mut rng = self.rng.lock();

        // 构建搜索空间信息（连续/离散分类）
        let ss_info = super::gp_optim_mixed::build_search_space_info(search_space, &param_names);

        if n_objectives == 1 {
            // ── 单目标: logEI ──
            // 对齐 Python: threshold = 可行试验中 standardized_score 的最大值
            let f0 = standardized_by_obj[0].iter().zip(is_feasible.iter())
                .filter(|(_, feas)| **feas)
                .map(|(y, _)| *y)
                .fold(f64::NEG_INFINITY, f64::max);

            // 对齐 Python: 当 f0 == -inf（全部不可行）时，logEI 返回 0
            // (对齐 Python `LogEI.eval_acqf` 中的 `if np.isneginf(threshold)`)

            // 对齐 Python Constant Liar 策略:
            // 单目标无约束时，将 running trials 的归一化参数加入 GP，
            // 使用 y_train.max() 作为假目标值。
            let mut gpr = std::mem::replace(&mut gprs_list[0], GPRegressor::new(
                vec![], vec![], vec![], vec![], 1.0, DEFAULT_MINIMUM_NOISE_VAR,
            ));
            if !running_trials.is_empty() && constraint_gps.is_empty() {
                // 收集 running trials 的归一化参数
                let mut x_running: Vec<Vec<f64>> = Vec::new();
                for rt in running_trials {
                    let mut row = Vec::with_capacity(n_params);
                    let mut complete = true;
                    for name in &param_names {
                        if let Some(pv) = rt.params.get(name) {
                            let dist = &search_space[name];
                            if let Ok(internal) = dist.to_internal_repr(pv) {
                                row.push(normalize_param(internal, dist));
                            } else {
                                complete = false;
                                break;
                            }
                        } else {
                            complete = false;
                            break;
                        }
                    }
                    if complete {
                        x_running.push(row);
                    }
                }
                if !x_running.is_empty() {
                    // 对齐 Python: constant_liar_value = gpr._y_train.max()
                    let constant_liar_value = gpr.y_train.iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    gpr.append_running_data(&x_running, constant_liar_value);
                }
            }

            let eval_acqf = |candidate: &[f64]| -> f64 {
                // 对齐 Python: 若 threshold 为 -inf（全部不可行），返回 0（不优化 EI）
                if f0 == f64::NEG_INFINITY {
                    // 仅返回约束概率部分
                    let mut acqf = 0.0_f64;
                    for (c_gpr, c_threshold) in &constraint_gps {
                        let (c_mean, c_var) = c_gpr.posterior(candidate);
                        let sigma = (c_var + STABILIZING_NOISE).sqrt();
                        let z = (c_mean - c_threshold) / sigma;
                        acqf += log_ndtr(z);
                    }
                    return acqf;
                }
                let (mean, var) = gpr.posterior(candidate);
                let mut acqf = log_ei(mean, var + STABILIZING_NOISE, f0);
                for (c_gpr, c_threshold) in &constraint_gps {
                    let (c_mean, c_var) = c_gpr.posterior(candidate);
                    let sigma = (c_var + STABILIZING_NOISE).sqrt();
                    let z = (c_mean - c_threshold) / sigma;
                    acqf += log_ndtr(z);
                }
                acqf
            };

            // 热启动: 最佳可行训练点
            let warmstart: Vec<Vec<f64>> = standardized_by_obj[0].iter().enumerate()
                .zip(is_feasible.iter())
                .filter(|(_, feas)| **feas)
                .map(|((i, y), _)| (i, *y))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| vec![x_train[i].clone()])
                .unwrap_or_default();

            // 提取 GP lengthscales 用于预条件
            let lengthscales: Vec<f64> = gpr.inverse_squared_lengthscales.iter()
                .map(|&isl| 1.0 / isl.sqrt().max(1e-6))
                .collect();

            let best_params = super::gp_optim_mixed::optimize_acqf_mixed(
                &eval_acqf,
                &ss_info,
                &warmstart,
                n_params,
                self.n_preliminary_samples,
                self.n_local_search,
                &lengthscales,
                &mut rng,
                seed,
            );

            self.unnormalize_result(&best_params, &param_names, search_space)
        } else {
            // ── 多目标: logEHVI ──
            // 对齐 Python `acqf.LogEHVI` / `acqf.ConstrainedLogEHVI`

            // 对齐 Python: 有约束时仅用可行试验构建 Pareto/EHVI;
            // 无约束时用全部试验。
            let has_constraints = !constraint_gps.is_empty();
            let is_all_infeasible = has_constraints && is_feasible.iter().all(|f| !f);

            // 用于 EHVI 盒分解的标准化 score 值（仅可行试验或全部）
            let ehvi_score_vals: Vec<Vec<f64>> = if has_constraints && !is_all_infeasible {
                // 对齐 Python: Y_feasible = standardized_score_vals[is_feasible]
                standardized_score_vals.iter().zip(is_feasible.iter())
                    .filter(|(_, feas)| **feas)
                    .map(|(sv, _)| sv.clone())
                    .collect()
            } else if is_all_infeasible {
                // 全部不可行时不构建 EHVI (Python: self._acqf = None)
                Vec::new()
            } else {
                standardized_score_vals.clone()
            };

            // 构建 EHVI 评估闭包所需的盒分解数据（可能为空）
            let ehvi_data: Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> =
                if !ehvi_score_vals.is_empty() {
                    // 1. 计算非支配盒分解（在 loss 空间: 取负 → 越小越好）
                    let loss_vals: Vec<Vec<f64>> = ehvi_score_vals.iter()
                        .map(|sv| sv.iter().map(|&v| -v).collect())
                        .collect();
                    let pareto_mask = is_pareto_front_min(&loss_vals);
                    let pareto_sols: Vec<Vec<f64>> = loss_vals.iter().zip(pareto_mask.iter())
                        .filter(|(_, on)| **on)
                        .map(|(v, _)| v.clone())
                        .collect();

                    // 参考点: 对齐 Python `np.nextafter(np.maximum(1.1*rp, 0.9*rp), inf)`
                    let ref_point: Vec<f64> = (0..n_objectives).map(|d| {
                        let max_val = loss_vals.iter().map(|v| v[d]).fold(f64::NEG_INFINITY, f64::max);
                        let rp = if max_val >= 0.0 { 1.1 * max_val } else { 0.9 * max_val };
                        // 对齐 Python nextafter: 精确偏移 1 ULP
                        f64::from_bits(rp.to_bits().wrapping_add(1))
                    }).collect();

                    let (box_lower_loss, box_upper_loss) = crate::multi_objective::get_non_dominated_box_bounds(
                        &pareto_sols, &ref_point,
                    );

                    // 对齐 Python: 翻转 loss→score 空间
                    let box_lower_score: Vec<Vec<f64>> = box_upper_loss.iter()
                        .map(|ub| ub.iter().map(|&v| -v).collect())
                        .collect();
                    let box_upper_score: Vec<Vec<f64>> = box_lower_loss.iter()
                        .map(|lb| lb.iter().map(|&v| -v).collect())
                        .collect();
                    let box_intervals: Vec<Vec<f64>> = box_lower_score.iter().zip(box_upper_score.iter())
                        .map(|(lb, ub)| lb.iter().zip(ub.iter()).map(|(&l, &u)| (u - l).max(STABILIZING_NOISE)).collect())
                        .collect();

                    Some((box_lower_score, box_intervals, Vec::new()))
                } else {
                    None
                };

            // 2. 生成 QMC 正态样本 [n_qmc_samples, n_objectives]
            let qmc_seed = rng.random_range(0u64..1u64 << 30);
            let fixed_samples = sample_from_normal_sobol(
                n_objectives, self.n_qmc_samples, qmc_seed,
            );

            // 3. 构建采集函数评估闭包
            let eval_acqf = |candidate: &[f64]| -> f64 {
                let mut acqf_val = 0.0_f64;

                // EHVI 部分（全部不可行时跳过）
                if let Some((ref box_lower_score, ref box_intervals, _)) = ehvi_data {
                    let mut y_post: Vec<Vec<f64>> = Vec::with_capacity(self.n_qmc_samples);
                    for s in 0..self.n_qmc_samples {
                        let mut sample = Vec::with_capacity(n_objectives);
                        for obj in 0..n_objectives {
                            let (mean, var) = gprs_list[obj].posterior(candidate);
                            let stdev = (var + STABILIZING_NOISE).sqrt();
                            sample.push(mean + stdev * fixed_samples[s][obj]);
                        }
                        y_post.push(sample);
                    }
                    acqf_val = log_ehvi(&y_post, box_lower_score, box_intervals);
                }

                // 约束: 加上 Σ log Φ(z)
                for (c_gpr, c_threshold) in &constraint_gps {
                    let (c_mean, c_var) = c_gpr.posterior(candidate);
                    let sigma = (c_var + STABILIZING_NOISE).sqrt();
                    let z = (c_mean - c_threshold) / sigma;
                    acqf_val += log_ndtr(z);
                }
                acqf_val
            };

            // 4. 热启动: 对齐 Python — 从 Pareto 前沿随机选取点
            let warmstart: Vec<Vec<f64>> = if has_constraints && !is_all_infeasible {
                // 仅在可行试验中找 Pareto 前沿
                let feasible_indices: Vec<usize> = is_feasible.iter().enumerate()
                    .filter(|(_, f)| **f)
                    .map(|(i, _)| i)
                    .collect();
                let feasible_loss: Vec<Vec<f64>> = feasible_indices.iter()
                    .map(|&i| standardized_score_vals[i].iter().map(|&v| -v).collect())
                    .collect();
                let pareto_mask = is_pareto_front_min(&feasible_loss);
                let pareto_indices: Vec<usize> = pareto_mask.iter().enumerate()
                    .filter(|(_, on)| **on)
                    .map(|(j, _)| feasible_indices[j])
                    .collect();
                let n_ws = (self.n_local_search / 2).min(pareto_indices.len());
                if n_ws > 0 && pareto_indices.len() > n_ws {
                    // 对齐 Python: 随机不重复选取
                    use rand::seq::SliceRandom;
                    let mut pi = pareto_indices.clone();
                    pi.partial_shuffle(&mut *rng, n_ws);
                    pi[..n_ws].iter().map(|&idx| x_train[idx].clone()).collect()
                } else {
                    pareto_indices.iter().take(n_ws).map(|&idx| x_train[idx].clone()).collect()
                }
            } else if !is_all_infeasible {
                // 无约束: 全部试验的 Pareto 前沿
                let loss_vals: Vec<Vec<f64>> = standardized_score_vals.iter()
                    .map(|sv| sv.iter().map(|&v| -v).collect())
                    .collect();
                let pareto_mask = is_pareto_front_min(&loss_vals);
                let pareto_indices: Vec<usize> = pareto_mask.iter().enumerate()
                    .filter(|(_, on)| **on)
                    .map(|(i, _)| i)
                    .collect();
                let n_ws = (self.n_local_search / 2).min(pareto_indices.len());
                if n_ws > 0 && pareto_indices.len() > n_ws {
                    use rand::seq::SliceRandom;
                    let mut pi = pareto_indices.clone();
                    pi.partial_shuffle(&mut *rng, n_ws);
                    pi[..n_ws].iter().map(|&idx| x_train[idx].clone()).collect()
                } else {
                    pareto_indices.iter().take(n_ws).map(|&idx| x_train[idx].clone()).collect()
                }
            } else {
                Vec::new()
            };

            // 提取 GP lengthscales (取各目标的平均)
            // 对齐 Python: np.mean([gpr.length_scales for gpr in gpr_list], axis=0)
            // 其中 length_scales = 1/sqrt(inverse_squared_lengthscales)
            // 正确: mean(1/sqrt(isl))，NOT 1/sqrt(mean(isl))
            let lengthscales: Vec<f64> = (0..n_params).map(|d| {
                let avg_ls: f64 = gprs_list.iter()
                    .map(|gpr| 1.0 / gpr.inverse_squared_lengthscales[d].sqrt().max(1e-6))
                    .sum::<f64>() / n_objectives as f64;
                avg_ls
            }).collect();

            // 5. 使用 optimize_acqf_mixed 优化
            let best_params = super::gp_optim_mixed::optimize_acqf_mixed(
                &eval_acqf,
                &ss_info,
                &warmstart,
                n_params,
                self.n_preliminary_samples,
                self.n_local_search,
                &lengthscales,
                &mut rng,
                seed,
            );

            self.unnormalize_result(&best_params, &param_names, search_space)
        }
    }

    /// 将归一化参数转回内部表示
    fn unnormalize_result(
        &self,
        best_params: &[f64],
        param_names: &[String],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
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

        // 对齐 Python: running trials 仅在单目标无约束时使用（constant liar 策略）。
        // 多目标或有约束时 use_cache = True（不传 running trials）。
        let running_trials: Vec<FrozenTrial> = if self.directions.len() == 1 && self.constraints_func.is_none() {
            trials.iter()
                .filter(|t| t.state == TrialState::Running)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };

        let completed_owned: Vec<FrozenTrial> = completed.into_iter().cloned().collect();
        self.sample_relative_impl(&completed_owned, search_space, &running_trials)
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
        let _result = sampler.sample_relative_impl(&trials, &search_space, &[]).unwrap();
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
        let _result2 = sampler.sample_relative_impl(&trials, &search_space, &[]).unwrap();
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

        let _result3 = sampler.sample_relative_impl(&trials, &search_space2, &[]).unwrap();
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

    #[test]
    fn test_erfinv_basic() {
        // erfinv(0) = 0
        assert!((erfinv(0.0)).abs() < 1e-10);
        // erfinv(erf(1)) ≈ 1
        let erf1 = libm::erf(1.0);
        assert!((erfinv(erf1) - 1.0).abs() < 0.01, "erfinv(erf(1))={}", erfinv(erf1));
        // 对称性: erfinv(-x) = -erfinv(x)
        assert!((erfinv(0.5) + erfinv(-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_sobol_normal_samples() {
        let samples = sample_from_normal_sobol(2, 100, 42);
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].len(), 2);
        // 所有值应有限
        for s in &samples {
            for &v in s {
                assert!(v.is_finite(), "Sobol normal sample should be finite: {v}");
            }
        }
        // 均值应接近 0
        let mean0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / 100.0;
        assert!(mean0.abs() < 0.5, "QMC normal mean should be near 0: {mean0}");
    }

    #[test]
    fn test_log_ehvi_basic() {
        // 简单 2D 测试: 1 个 box [0,0] → [1,1]，1 个后验样本 [0.5, 0.5]
        let y_post = vec![vec![0.5, 0.5]];
        let box_lower = vec![vec![0.0, 0.0]];
        let box_intervals = vec![vec![1.0, 1.0]];
        let lehvi = log_ehvi(&y_post, &box_lower, &box_intervals);
        // 改善 = 0.5 * 0.5 = 0.25, log(0.25) ≈ -1.386
        assert!((lehvi - 0.25_f64.ln()).abs() < 0.1, "logEHVI={lehvi}");
    }

    #[test]
    fn test_is_pareto_front_min() {
        let vals = vec![
            vec![1.0, 3.0],  // Pareto
            vec![2.0, 2.0],  // Pareto
            vec![3.0, 1.0],  // Pareto
            vec![2.0, 3.0],  // dominated by [1,3]
        ];
        let front = is_pareto_front_min(&vals);
        assert_eq!(front, vec![true, true, true, false]);
    }

    #[test]
    fn test_gp_multi_objective_optimize() {
        // 双目标优化: min(x^2, (x-1)^2) — Pareto 前沿在 [0, 1]
        let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::with_directions(
            Some(42),
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(5),
            false, None, None,
        ));
        let study = crate::study::create_study(
            None, Some(sampler), None, None,
            None, Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        ).unwrap();

        study.optimize_multi(|trial| {
            let x = trial.suggest_float("x", 0.0, 2.0, false, None)?;
            Ok(vec![x * x, (x - 1.0) * (x - 1.0)])
        }, Some(15), None, None).unwrap();

        let trials = study.trials().unwrap();
        let completed: Vec<_> = trials.iter().filter(|t| t.state == TrialState::Complete).collect();
        assert!(completed.len() >= 10, "Should have at least 10 completed trials");

        // 验证 GP 采样器确实在运行（不止是随机采样）
        // Pareto 前沿上的点 x ∈ [0,1]
        let pareto_trials: Vec<_> = completed.iter().filter(|t| {
            if let Some(vals) = &t.values {
                vals[0] <= 1.5 && vals[1] <= 1.5
            } else { false }
        }).collect();
        assert!(!pareto_trials.is_empty(), "Should have some near-Pareto trials");
    }

    #[test]
    fn test_gp_sampler_multi_obj_cache() {
        // 测试多目标 GP 缓存: 应该有 n_objectives 个缓存条目
        let sampler = GpSampler::with_directions(
            Some(42),
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(3),
            false, None, None,
        );

        let mut trials = Vec::new();
        for i in 0..5 {
            let x = i as f64 * 0.25;
            let mut params = HashMap::new();
            params.insert("x".to_string(), crate::distributions::ParamValue::Float(x));
            let mut dists = HashMap::new();
            dists.insert("x".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution {
                        low: 0.0, high: 1.0, log: false, step: None,
                    },
                ));
            trials.push(FrozenTrial {
                number: i as i64, trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![x * x, (x - 1.0).powi(2)]),
                params, distributions: dists,
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None, datetime_complete: None,
            });
        }

        let mut search_space = IndexMap::new();
        search_space.insert("x".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution {
                    low: 0.0, high: 1.0, log: false, step: None,
                },
            ));

        let _result = sampler.sample_relative_impl(&trials, &search_space, &[]).unwrap();
        {
            let cache = sampler.gprs_cache.lock();
            assert!(cache.is_some(), "缓存应该在第一次调用后被填充");
            let list = cache.as_ref().unwrap();
            assert_eq!(list.len(), 2, "双目标应有 2 个缓存条目");
        }
    }

    #[test]
    fn test_gp_sampler_with_int_params() {
        // 混合搜索空间: 1 个连续 + 1 个整数参数
        // f(x, n) = -(x - 0.5)^2 - (n - 3)^2
        let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
            Some(42), None, Some(5), false, None, None,
        ));
        let study = crate::study::create_study(
            None, Some(sampler), None, None, None, None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let n = trial.suggest_int_default("n", 1, 5)?;
            Ok(-(x - 0.5).powi(2) - (n as f64 - 3.0).powi(2))
        }, Some(20), None, None).unwrap();

        let trials = study.trials().unwrap();
        let completed: Vec<_> = trials.iter().filter(|t| t.state == TrialState::Complete).collect();
        assert!(completed.len() >= 15, "Got {} completed", completed.len());

        // 应该有扫描到最优附近的点
        let near_opt = completed.iter().any(|t| {
            if let Some(vals) = &t.values {
                vals[0] > -0.5 // 至少不比随机差太多
            } else { false }
        });
        assert!(near_opt, "Should find some good solutions");
    }

    #[test]
    fn test_gp_sampler_with_categorical() {
        // 混合搜索空间: 1 个连续 + 1 个分类参数
        let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
            Some(42), None, Some(5), false, None, None,
        ));
        let study = crate::study::create_study(
            None, Some(sampler), None, None, None, None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let cat = trial.suggest_categorical("cat", vec![
                crate::distributions::CategoricalChoice::Str("a".into()),
                crate::distributions::CategoricalChoice::Str("b".into()),
                crate::distributions::CategoricalChoice::Str("c".into()),
            ])?;
            let bonus = if cat == crate::distributions::CategoricalChoice::Str("b".into()) { 1.0 } else { 0.0 };
            Ok(-(x - 0.5).powi(2) + bonus)
        }, Some(20), None, None).unwrap();

        let trials = study.trials().unwrap();
        let completed: Vec<_> = trials.iter().filter(|t| t.state == TrialState::Complete).collect();
        assert!(completed.len() >= 15, "Got {} completed", completed.len());
    }

    // ═══════════════════════════════════════════════════════════════
    // 对齐修复项的测试
    // ═══════════════════════════════════════════════════════════════

    /// 测试 warn_and_convert_inf: 非有限值应被裁剪到该列有限值的 [min,max] 范围。
    /// 对齐 Python `gp.warn_and_convert_inf` 的逐列裁剪逻辑。
    #[test]
    fn test_warn_and_convert_inf() {
        // 情况 1: 全部有限 → 无变化
        let mut vals1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        warn_and_convert_inf(&mut vals1, 2);
        assert_eq!(vals1, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        // 情况 2: 有 inf → 裁剪到该列有限值范围
        let mut vals2 = vec![
            vec![1.0, f64::INFINITY],
            vec![3.0, 2.0],
            vec![f64::NEG_INFINITY, 5.0],
        ];
        warn_and_convert_inf(&mut vals2, 2);
        // 列0: 有限值 [1.0, 3.0] → -inf 裁剪到 1.0
        assert_eq!(vals2[2][0], 1.0);
        // 列1: 有限值 [2.0, 5.0] → inf 裁剪到 5.0
        assert_eq!(vals2[0][1], 5.0);
        // 有限值不变
        assert_eq!(vals2[0][0], 1.0);
        assert_eq!(vals2[1][0], 3.0);
        assert_eq!(vals2[1][1], 2.0);

        // 情况 3: 某列全部非有限 → 设为 0.0
        let mut vals3 = vec![
            vec![1.0, f64::INFINITY],
            vec![2.0, f64::NEG_INFINITY],
        ];
        warn_and_convert_inf(&mut vals3, 2);
        assert_eq!(vals3[0][1], 0.0);
        assert_eq!(vals3[1][1], 0.0);
        // 列0 有限值不变
        assert_eq!(vals3[0][0], 1.0);
        assert_eq!(vals3[1][0], 2.0);
    }

    /// 测试 log_ei 的尾部精度: z < -25 时使用 erfcx 分支。
    /// 对齐 Python `standard_logei` 的两段式计算。
    #[test]
    fn test_log_ei_tail_precision() {
        // z 在 -25 附近的边界
        let lei_24 = log_ei(0.0, 1.0, 24.0);  // z = -24
        let lei_26 = log_ei(0.0, 1.0, 26.0);  // z = -26 (进入尾部分支)
        let lei_40 = log_ei(0.0, 1.0, 40.0);  // z = -40 (深尾部)

        // 所有值应有限且单调递减
        assert!(lei_24.is_finite(), "logEI at z=-24 should be finite: {lei_24}");
        assert!(lei_26.is_finite(), "logEI at z=-26 should be finite: {lei_26}");
        assert!(lei_40.is_finite(), "logEI at z=-40 should be finite: {lei_40}");
        assert!(lei_24 > lei_26, "logEI should decrease: {lei_24} > {lei_26}");
        assert!(lei_26 > lei_40, "logEI should decrease: {lei_26} > {lei_40}");

        // z=-26 时参考值（Python 计算）: ≈ -338.5
        // 允许较大容差因不同的近似方法
        assert!(lei_26 < -300.0, "logEI at z=-26 should be very negative: {lei_26}");

        // z=0 时（高 EI），EI ≈ φ(0) = 1/√(2π) ≈ 0.3989
        let lei_0 = log_ei(0.0, 1.0, 0.0);
        assert!((lei_0 - 0.3989_f64.ln()).abs() < 0.1, "logEI at z=0: {lei_0}");
    }

    /// 测试 erfcx 函数的正确性。
    #[test]
    fn test_erfcx() {
        // erfcx(0) = erfc(0) = 1.0
        assert!((erfcx(0.0) - 1.0).abs() < 1e-10, "erfcx(0) = {}", erfcx(0.0));

        // erfcx(x) 对大正数应趋近 1/(√π * x)
        let x = 10.0;
        let expected = 1.0 / (PI.sqrt() * x);
        let actual = erfcx(x);
        assert!((actual - expected) / expected < 0.1,
            "erfcx({x}) = {actual}, expected ≈ {expected}");

        // erfcx(x) 对负数：erfcx(-1) = exp(1) * erfc(-1) ≈ exp(1) * 1.8427 ≈ 5.009
        let neg_val = erfcx(-1.0);
        let expected_neg = 1.0_f64.exp() * libm::erfc(-1.0);
        assert!((neg_val - expected_neg).abs() < 0.01,
            "erfcx(-1) = {neg_val}, expected {expected_neg}");
    }

    /// 测试 log_ndtr 高精度对数正态 CDF。
    /// 对齐 Python `torch.special.log_ndtr`。
    #[test]
    fn test_log_ndtr() {
        // log Φ(0) = log(0.5) ≈ -0.6931
        let ln0 = log_ndtr(0.0);
        assert!((ln0 - (-0.5_f64.ln().abs())).abs() < 1e-6,
            "log_ndtr(0) = {ln0}, expected {}", -0.5_f64.ln().abs());

        // log Φ(3) ≈ log(0.99865) ≈ -0.00135
        let ln3 = log_ndtr(3.0);
        assert!(ln3 > -0.01 && ln3 < 0.0, "log_ndtr(3) = {ln3}");

        // log Φ(-3) ≈ log(0.00135) ≈ -6.607
        let lnm3 = log_ndtr(-3.0);
        assert!((lnm3 - (-6.607)).abs() < 0.1, "log_ndtr(-3) = {lnm3}");

        // 深尾部: log Φ(-10) ≈ -51.12
        let lnm10 = log_ndtr(-10.0);
        assert!(lnm10.is_finite(), "log_ndtr(-10) should be finite: {lnm10}");
        assert!(lnm10 < -40.0, "log_ndtr(-10) should be very negative: {lnm10}");

        // 深尾部: log Φ(-20)
        let lnm20 = log_ndtr(-20.0);
        assert!(lnm20.is_finite(), "log_ndtr(-20) should be finite: {lnm20}");
        assert!(lnm20 < lnm10, "log_ndtr should be monotonically decreasing in tail");

        // 正大值: log Φ(10) ≈ 0
        let ln10 = log_ndtr(10.0);
        assert!(ln10 > -1e-6, "log_ndtr(10) should be ≈ 0: {ln10}");
    }

    /// 测试 log_ehvi 不再跳过微小贡献的盒。
    /// 对齐 Python `logehvi` 的 clamp 行为。
    #[test]
    fn test_log_ehvi_no_skip() {
        // 后验样本刚好在盒下界（微小改善）
        let y_post = vec![vec![1e-15, 1e-15]]; // 极小改善
        let box_lower = vec![vec![0.0, 0.0]];
        let box_intervals = vec![vec![1.0, 1.0]];
        let lehvi = log_ehvi(&y_post, &box_lower, &box_intervals);
        // 对齐 Python: diff.clamp(EPS, interval), 即使改善极小也应有贡献
        // 不应返回 NEG_INFINITY
        assert!(lehvi.is_finite(),
            "log_ehvi should not skip near-zero contributions: {lehvi}");
    }

    /// 测试 append_running_data (constant liar 支持)。
    #[test]
    fn test_gpr_append_running_data() {
        let x_train = vec![vec![0.0], vec![1.0]];
        let y_train = vec![0.0, 1.0];
        let is_cat = vec![false];
        let mut gpr = GPRegressor::new(
            x_train, y_train, is_cat,
            vec![1.0], 1.0, 1e-6,
        );
        assert_eq!(gpr.x_train.len(), 2);

        // 追加 running trial 数据
        let x_running = vec![vec![0.5]];
        let constant_liar_value = 1.0; // y_train.max()
        gpr.append_running_data(&x_running, constant_liar_value);

        assert_eq!(gpr.x_train.len(), 3, "应增加 1 个训练点");
        assert_eq!(gpr.y_train.len(), 3);
        assert_eq!(gpr.y_train[2], 1.0, "新点的 y 值应为 constant_liar_value");

        // Cholesky 应重新计算
        assert!(gpr.chol_l.is_some(), "Cholesky 应在追加后重新计算");
        assert!(gpr.alpha.is_some());

        // 新训练点处的后验不确定性应降低
        let (_, var_at_05) = gpr.posterior(&[0.5]);
        assert!(var_at_05 < 0.5, "追加点后该位置不确定性应降低: {var_at_05}");
    }

    /// 测试 GP 采样器在全部不可行试验时的行为。
    /// 对齐 Python: 当所有试验不可行时，logEI 返回 0，仅优化约束概率。
    #[test]
    fn test_gp_sampler_all_infeasible() {
        let sampler = GpSampler::new(
            Some(42), Some(StudyDirection::Minimize), Some(3), false,
            Some(Arc::new(|_trial: &FrozenTrial| vec![1.0])), // 所有试验约束违反 = 1.0
            None,
        );

        let mut trials = Vec::new();
        for i in 0..5 {
            let x = i as f64 * 0.25;
            let mut params = HashMap::new();
            params.insert("x".to_string(), crate::distributions::ParamValue::Float(x));
            let mut dists = HashMap::new();
            dists.insert("x".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution {
                        low: 0.0, high: 1.0, log: false, step: None,
                    },
                ));
            let mut sys_attrs = HashMap::new();
            // 所有试验约束违反 > 0（不可行）
            sys_attrs.insert("constraints".to_string(),
                serde_json::json!([1.0]));
            trials.push(FrozenTrial {
                number: i as i64, trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![x * x]),
                params, distributions: dists,
                user_attrs: HashMap::new(),
                system_attrs: sys_attrs,
                intermediate_values: HashMap::new(),
                datetime_start: None, datetime_complete: None,
            });
        }

        let mut search_space = IndexMap::new();
        search_space.insert("x".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution {
                    low: 0.0, high: 1.0, log: false, step: None,
                },
            ));

        // 应该成功运行而不 panic
        let result = sampler.sample_relative_impl(&trials, &search_space, &[]);
        assert!(result.is_ok(), "全部不可行时不应出错: {:?}", result.err());
    }
}
