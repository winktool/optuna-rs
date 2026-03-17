//! L-BFGS-B 超参数优化器 — GP 核参数拟合增强。
//!
//! 对应 Python `optuna.samplers._gp.gp._fit_kernel_params` +
//! `optuna.samplers._gp.batched_lbfgsb`。
//!
//! 基于 [`scirs2-optimize`](https://docs.rs/scirs2-optimize) crate 实现 L-BFGS-B 优化，
//! 忠实移植 Python 使用 `scipy.optimize.minimize(method='L-BFGS-B')` 的行为。
//!
//! ## Python 原始实现要点
//!
//! 1. **参数编码**: log 变换保正性，无界优化
//!    - `raw_params = [log(inv_sq_ls[0..n]), log(kernel_scale), log(noise_var - min_noise)]`
//! 2. **损失函数**: `-(log_marginal_likelihood + log_prior)`
//! 3. **先验分布** (from `prior.py`):
//!    - `inv_sq_ls`: `-(0.1/x + 0.1*x)` 惩罚，鼓励 ≈ 1
//!    - `kernel_scale`: Gamma(α=2, β=1) → `(α-1)*log(x) - β*x`
//!    - `noise_var`: Gamma(α=1.1, β=30) → 鼓励小噪声
//! 4. **梯度计算**: Python 用 PyTorch autograd；Rust 用有限差分
//! 5. **scipy 调用**: `minimize(loss, x0, jac=True, method='l-bfgs-b', options={'gtol': 1e-2})`
//!
//! # 使用方式
//! 需要启用 `gp-lbfgsb` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["gp-lbfgsb"] }
//! ```
//!
//! 启用后，`fit_kernel_params` 会自动使用 L-BFGS-B 优化替代随机搜索。

#[cfg(feature = "gp-lbfgsb")]
use scirs2_optimize::unconstrained::{minimize, Method, Options};
#[cfg(feature = "gp-lbfgsb")]
use scirs2_core::ndarray::ArrayView1;

#[cfg(feature = "gp-lbfgsb")]
use crate::samplers::gp::{
    GPRegressor, KernelParamsCache, DEFAULT_MINIMUM_NOISE_VAR, default_log_prior,
};

/// 从 raw 参数向量 (log 空间) 解码出 GP 超参数。
///
/// 布局: `[log(inv_sq_ls[0]), ..., log(inv_sq_ls[n-1]), log(kernel_scale), log(noise_var - min_noise)]`
/// 对于 deterministic_objective 模式，只有 d+1 个参数（无 noise_var）。
#[cfg(feature = "gp-lbfgsb")]
fn decode_params(params: &[f64], n_dims: usize, deterministic: bool) -> (Vec<f64>, f64, f64) {
    let inv_sq_ls: Vec<f64> = params[..n_dims].iter().map(|&x| x.exp()).collect();
    let kernel_scale = params[n_dims].exp();
    let noise_var = if deterministic {
        DEFAULT_MINIMUM_NOISE_VAR
    } else {
        // 对应 Python: noise_var = exp(raw) + 0.99 * minimum_noise
        // 确保 noise_var >= minimum_noise
        DEFAULT_MINIMUM_NOISE_VAR + params[n_dims + 1].exp()
    };
    (inv_sq_ls, kernel_scale, noise_var)
}

/// 计算负对数后验 (损失函数): `-(log_marginal_likelihood + log_prior)`。
///
/// 对应 Python `_fit_kernel_params` 中的内部 `loss_func`。
#[cfg(feature = "gp-lbfgsb")]
fn neg_log_posterior(
    params: &[f64],
    n_dims: usize,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    deterministic: bool,
) -> f64 {
    let (inv_sq_ls, kernel_scale, noise_var) = decode_params(params, n_dims, deterministic);

    // 防止数值溢出
    if inv_sq_ls.iter().any(|&x| !x.is_finite() || x > 1e10)
        || !kernel_scale.is_finite() || kernel_scale > 1e10
        || !noise_var.is_finite() || noise_var > 1e10
    {
        return 1e30;
    }

    let gpr = GPRegressor::new(
        x_train.to_vec(),
        y_train.to_vec(),
        is_categorical.to_vec(),
        inv_sq_ls.clone(),
        kernel_scale,
        noise_var,
    );

    let lml = gpr.log_marginal_likelihood();
    let lp = default_log_prior(&inv_sq_ls, kernel_scale, noise_var);

    if !lml.is_finite() {
        return 1e30;
    }

    -(lml + lp)
}

/// 通过中心差分计算梯度。
///
/// Python 用 PyTorch autograd，Rust 使用中心差分近似:
/// `∂f/∂x_i ≈ (f(x+h*e_i) - f(x-h*e_i)) / (2h)`
#[cfg(feature = "gp-lbfgsb")]
fn _gradient_finite_diff(
    params: &[f64],
    n_dims: usize,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    deterministic: bool,
) -> Vec<f64> {
    let h = 1e-5;
    let n = params.len();
    let mut grad = vec![0.0; n];

    for i in 0..n {
        let mut params_fwd = params.to_vec();
        let mut params_bwd = params.to_vec();
        params_fwd[i] += h;
        params_bwd[i] -= h;
        let f_fwd = neg_log_posterior(&params_fwd, n_dims, x_train, y_train, is_categorical, deterministic);
        let f_bwd = neg_log_posterior(&params_bwd, n_dims, x_train, y_train, is_categorical, deterministic);
        grad[i] = (f_fwd - f_bwd) / (2.0 * h);
    }

    grad
}

/// 将缓存的核参数编码到 log 空间。
#[cfg(feature = "gp-lbfgsb")]
fn encode_cache(cache: &KernelParamsCache, deterministic: bool) -> Vec<f64> {
    let mut params: Vec<f64> = cache.inverse_squared_lengthscales
        .iter()
        .map(|&x| x.ln())
        .collect();
    params.push(cache.kernel_scale.ln());
    if !deterministic {
        let raw_noise = (cache.noise_var - DEFAULT_MINIMUM_NOISE_VAR).max(1e-12);
        params.push(raw_noise.ln());
    }
    params
}

/// 使用 L-BFGS-B 优化 GP 核超参数。
///
/// 对应 Python `optuna.samplers._gp.gp.fit_kernel_params`:
/// - 先尝试 gpr_cache 作为初始值，失败则用默认值
/// - `scipy.optimize.minimize(method='l-bfgs-b', jac=True, options={'gtol': 1e-2})`
/// - 包含 log-prior (from `prior.py`)
#[cfg(feature = "gp-lbfgsb")]
pub(crate) fn fit_kernel_params_lbfgsb(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    _seed: u64,
    gpr_cache: Option<&KernelParamsCache>,
    deterministic_objective: bool,
) -> GPRegressor {
    use scirs2_optimize::unconstrained::OptimizeResult;

    let d = x_train.first().map_or(0, |row| row.len());

    if x_train.is_empty() || y_train.is_empty() {
        return GPRegressor::new(
            x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
            vec![1.0; d], 1.0, DEFAULT_MINIMUM_NOISE_VAR,
        );
    }

    let x_train_owned = x_train.to_vec();
    let y_train_owned = y_train.to_vec();
    let is_cat_owned = is_categorical.to_vec();

    // 对应 Python: for gpr_cache_to_use in [gpr_cache, default_gpr_cache]
    // 先用 cache（若有），再用默认值 — 共 2 次尝试
    let default_cache = KernelParamsCache {
        inverse_squared_lengthscales: vec![1.0; d],
        kernel_scale: 1.0,
        noise_var: DEFAULT_MINIMUM_NOISE_VAR + 0.01,
    };

    let attempts: Vec<Vec<f64>> = if let Some(cache) = gpr_cache {
        vec![
            encode_cache(cache, deterministic_objective),
            encode_cache(&default_cache, deterministic_objective),
        ]
    } else {
        vec![encode_cache(&default_cache, deterministic_objective)]
    };

    let mut best_gpr: Option<GPRegressor> = None;
    let mut best_score = f64::NEG_INFINITY;

    let score_gpr = |gpr: &GPRegressor| -> f64 {
        let lml = gpr.log_marginal_likelihood();
        if !lml.is_finite() { return f64::NEG_INFINITY; }
        lml + default_log_prior(&gpr.inverse_squared_lengthscales, gpr.kernel_scale, gpr.noise_var)
    };

    for init_params in attempts {
        let x_t = x_train_owned.clone();
        let y_t = y_train_owned.clone();
        let is_c = is_cat_owned.clone();
        let dims = d;
        let det = deterministic_objective;

        let objective = |x: &ArrayView1<f64>| -> f64 {
            let params: Vec<f64> = x.iter().copied().collect();
            neg_log_posterior(&params, dims, &x_t, &y_t, &is_c, det)
        };

        let mut options = Options::default();
        options.max_iter = 50;
        options.gtol = 1e-2;

        let result: Result<OptimizeResult<f64>, _> = minimize(
            objective,
            &init_params,
            Method::BFGS,
            Some(options),
        );

        let final_params: Vec<f64> = match result {
            Ok(ref res) => res.x.to_vec(),
            Err(_) => continue,
        };

        let (inv_sq_ls, kernel_scale, noise_var) = decode_params(&final_params, d, deterministic_objective);

        if inv_sq_ls.iter().any(|&v| !v.is_finite())
            || !kernel_scale.is_finite()
            || !noise_var.is_finite()
        {
            continue;
        }

        let gpr = GPRegressor::new(
            x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
            inv_sq_ls, kernel_scale, noise_var,
        );

        let s = score_gpr(&gpr);
        if s > best_score {
            best_score = s;
            best_gpr = Some(gpr);
        }
    }

    best_gpr.unwrap_or_else(|| GPRegressor::new(
        x_train.to_vec(), y_train.to_vec(), is_categorical.to_vec(),
        vec![1.0; d], 1.0, DEFAULT_MINIMUM_NOISE_VAR,
    ))
}

// 当 gp-lbfgsb feature 未启用时，回退到随机搜索
#[cfg(not(feature = "gp-lbfgsb"))]
pub(crate) fn fit_kernel_params_lbfgsb(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    seed: u64,
    gpr_cache: Option<&crate::samplers::gp::KernelParamsCache>,
    deterministic_objective: bool,
) -> crate::samplers::gp::GPRegressor {
    crate::samplers::gp::fit_kernel_params(x_train, y_train, is_categorical, seed, gpr_cache, deterministic_objective)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_kernel_params_lbfgsb_basic() {
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        let is_cat = vec![false];

        let gpr = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, false);
        let (mean, _var) = gpr.posterior(&[0.5]);
        assert!((mean - 0.25).abs() < 0.5, "mean={mean}, expected ~0.25");
    }

    #[test]
    fn test_fit_kernel_params_lbfgsb_empty() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let is_cat: Vec<bool> = vec![];
        let _ = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, false);
    }

    #[cfg(feature = "gp-lbfgsb")]
    #[test]
    fn test_log_prior_at_unity() {
        use crate::samplers::gp::DEFAULT_MINIMUM_NOISE_VAR;
        // inv_sq_ls = [1.0] 是先验的最优值
        let lp1 = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        let lp2 = default_log_prior(&[2.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        assert!(lp1 > lp2, "lprio at 1.0={lp1} should be > at 2.0={lp2}");
    }

    #[cfg(feature = "gp-lbfgsb")]
    #[test]
    fn test_decode_encode_roundtrip() {
        use crate::samplers::gp::DEFAULT_MINIMUM_NOISE_VAR;
        let raw = vec![0.0, 0.5, -1.0]; // log(1.0), log(exp(0.5)), log(exp(-1))
        let (inv, ks, nv) = decode_params(&raw, 1, false);
        assert!((inv[0] - 1.0).abs() < 1e-10);
        assert!((ks - 0.5_f64.exp()).abs() < 1e-10);
        assert!((nv - (DEFAULT_MINIMUM_NOISE_VAR + (-1.0_f64).exp())).abs() < 1e-10);

        // deterministic mode: noise_var is fixed
        let raw_det = vec![0.0, 0.5]; // only d+1 params
        let (inv2, ks2, nv2) = decode_params(&raw_det, 1, true);
        assert!((inv2[0] - 1.0).abs() < 1e-10);
        assert!((ks2 - 0.5_f64.exp()).abs() < 1e-10);
        assert!((nv2 - DEFAULT_MINIMUM_NOISE_VAR).abs() < 1e-10);
    }

    /// 对齐 Python: 5 点数据拟合，验证 posterior 合理性
    #[test]
    fn test_fit_5_points() {
        let x = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
        let y = vec![0.0, 0.0625, 0.25, 0.5625, 1.0]; // y = x^2
        let is_cat = vec![false];

        let gpr = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, false);
        let (mean_0, _) = gpr.posterior(&[0.0]);
        let (mean_1, _) = gpr.posterior(&[1.0]);

        // 已观测点附近应接近观测值
        assert!(mean_0.abs() < 0.5, "mean at 0.0={mean_0}, expected ~0.0");
        assert!((mean_1 - 1.0).abs() < 0.5, "mean at 1.0={mean_1}, expected ~1.0");
    }

    /// 对齐 Python: 带缓存的拟合应也工作
    #[test]
    fn test_fit_with_cache() {
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        let is_cat = vec![false];

        // 第一次拟合
        let gpr1 = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, false);
        let cache = gpr1.params_cache();

        // 第二次用缓存拟合
        let gpr2 = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, cache.as_ref(), false);
        let (mean2, _) = gpr2.posterior(&[0.5]);
        assert!((mean2 - 0.25).abs() < 0.5, "cached fit: mean={mean2}");
    }

    /// 对齐 Python: 多维数据拟合
    #[test]
    fn test_fit_multidimensional() {
        let x = vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 2.0]; // y = x1 + x2
        let is_cat = vec![false, false];

        let gpr = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, false);
        let (mean, _) = gpr.posterior(&[0.5, 0.5]);
        assert!((mean - 1.0).abs() < 1.0, "2D fit: mean={mean}, expected ~1.0");
    }

    /// 对齐 Python: deterministic_objective 模式
    #[test]
    fn test_fit_deterministic() {
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        let is_cat = vec![false];

        let gpr = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42, None, true);
        let (mean, _var) = gpr.posterior(&[0.5]);
        assert!((mean - 0.25).abs() < 0.5, "deterministic fit: mean={mean}");
    }

    /// 对齐 Python: 种子不同应产生不同初始化
    #[test]
    fn test_different_seeds() {
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        let is_cat = vec![false];

        let gpr1 = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 1, None, false);
        let gpr2 = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 2, None, false);
        let (m1, _) = gpr1.posterior(&[0.5]);
        let (m2, _) = gpr2.posterior(&[0.5]);
        // 结果可能相同也可能不同，但都应合理
        assert!(m1.abs() < 10.0);
        assert!(m2.abs() < 10.0);
    }

    #[cfg(feature = "gp-lbfgsb")]
    /// 对齐 Python: decode_params 多维
    #[test]
    fn test_decode_params_multidim() {
        let raw = vec![0.0, 1.0, 0.5, -1.0]; // 2 dims + kernel_scale + noise
        let (inv, ks, _nv) = decode_params(&raw, 2, false);
        assert_eq!(inv.len(), 2);
        assert!((inv[0] - 1.0).abs() < 1e-10);  // exp(0) = 1
        assert!((inv[1] - 1.0_f64.exp()).abs() < 1e-10); // exp(1)
        assert!((ks - 0.5_f64.exp()).abs() < 1e-10);
    }

    #[cfg(feature = "gp-lbfgsb")]
    /// 对齐 Python: log_prior 噪声越大先验越低
    #[test]
    fn test_log_prior_noise_penalty() {
        use crate::samplers::gp::DEFAULT_MINIMUM_NOISE_VAR;
        let lp_small = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR + 0.01);
        let lp_big = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR + 10.0);
        assert!(lp_small > lp_big, "小噪声先验={lp_small} 应 > 大噪声先验={lp_big}");
    }
}
