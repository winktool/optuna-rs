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
    GPRegressor, DEFAULT_MINIMUM_NOISE_VAR,
};

/// Python `optuna.samplers._gp.prior.default_log_prior` 的忠实移植。
///
/// 先验分布:
/// - `inverse_squared_lengthscales`: 自定义惩罚 `-(0.1/x + 0.1*x)`
/// - `kernel_scale`: Gamma(concentration=2, rate=1)
/// - `noise_var`: Gamma(concentration=1.1, rate=30)
#[cfg(feature = "gp-lbfgsb")]
fn default_log_prior(
    inverse_squared_lengthscales: &[f64],
    kernel_scale: f64,
    noise_var: f64,
) -> f64 {
    // inv_sq_ls 先验: -(0.1/x + 0.1*x) 对每个维度求和
    let ls_prior: f64 = inverse_squared_lengthscales.iter()
        .map(|&x| -(0.1 / x + 0.1 * x))
        .sum();

    // kernel_scale ~ Gamma(α=2, β=1): log_prior = (α-1)*log(x) - β*x
    let ks_prior = (2.0 - 1.0) * kernel_scale.ln() - 1.0 * kernel_scale;

    // noise_var ~ Gamma(α=1.1, β=30): 鼓励小噪声
    let nv_prior = (1.1 - 1.0) * noise_var.ln() - 30.0 * noise_var;

    ls_prior + ks_prior + nv_prior
}

/// 从 raw 参数向量 (log 空间) 解码出 GP 超参数。
///
/// 布局: `[log(inv_sq_ls[0]), ..., log(inv_sq_ls[n-1]), log(kernel_scale), log(noise_var - min_noise)]`
#[cfg(feature = "gp-lbfgsb")]
fn decode_params(params: &[f64], n_dims: usize) -> (Vec<f64>, f64, f64) {
    let inv_sq_ls: Vec<f64> = params[..n_dims].iter().map(|&x| x.exp()).collect();
    let kernel_scale = params[n_dims].exp();
    // noise_var = exp(raw) + minimum_noise，确保 >= minimum_noise
    let noise_var = DEFAULT_MINIMUM_NOISE_VAR + params[n_dims + 1].exp();
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
) -> f64 {
    let (inv_sq_ls, kernel_scale, noise_var) = decode_params(params, n_dims);

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

/// 通过有限差分计算梯度。
///
/// Python 用 PyTorch autograd，Rust 使用前向差分近似:
/// `∂f/∂x_i ≈ (f(x+h*e_i) - f(x)) / h`
#[cfg(feature = "gp-lbfgsb")]
fn gradient_finite_diff(
    params: &[f64],
    n_dims: usize,
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
) -> Vec<f64> {
    let h = 1e-5;
    let n = params.len();
    let f0 = neg_log_posterior(params, n_dims, x_train, y_train, is_categorical);
    let mut grad = vec![0.0; n];

    for i in 0..n {
        let mut params_h = params.to_vec();
        params_h[i] += h;
        let f_h = neg_log_posterior(&params_h, n_dims, x_train, y_train, is_categorical);
        grad[i] = (f_h - f0) / h;
    }

    grad
}

/// 使用 L-BFGS-B 优化 GP 核超参数。
///
/// 忠实对应 Python `optuna.samplers._gp.gp._fit_kernel_params`:
/// - `scipy.optimize.minimize(method='l-bfgs-b', jac=True, options={'gtol': 1e-2})`
/// - 包含 log-prior (from `prior.py`)
/// - 多次随机重启选最优
///
/// # 参数
/// * `x_train` - 训练输入 (N × D)
/// * `y_train` - 训练目标 (N,)
/// * `is_categorical` - 每个维度是否为分类参数
/// * `seed` - 随机种子（用于多次重启的初始化）
///
/// # 返回
/// 优化后的 GPRegressor 实例
#[cfg(feature = "gp-lbfgsb")]
pub(crate) fn fit_kernel_params_lbfgsb(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    seed: u64,
) -> GPRegressor {
    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;
    use scirs2_optimize::unconstrained::OptimizeResult;

    let d = x_train.first().map_or(0, |row| row.len());

    // 数据为空或太少时回退到默认参数
    if x_train.is_empty() || y_train.is_empty() {
        return GPRegressor::new(
            x_train.to_vec(),
            y_train.to_vec(),
            is_categorical.to_vec(),
            vec![1.0; d],
            1.0,
            DEFAULT_MINIMUM_NOISE_VAR,
        );
    }

    let n_params = d + 2; // d 个 inv_sq_ls + kernel_scale + noise_var
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // 克隆训练数据供闭包使用
    let x_train_owned = x_train.to_vec();
    let y_train_owned = y_train.to_vec();
    let is_cat_owned = is_categorical.to_vec();

    let mut best_gpr: Option<GPRegressor> = None;
    let mut best_lml = f64::NEG_INFINITY;

    // 多次随机重启 + L-BFGS-B 优化（匹配 Python 的多重启策略）
    let n_restarts = 10;
    for _ in 0..n_restarts {
        // 随机初始化（在 log 空间），匹配 Python 的 np.random.uniform(-1, 1) 范围
        let init_params: Vec<f64> = (0..n_params)
            .map(|_| rng.gen_range(-1.0_f64..1.0))
            .collect();

        // 构造目标函数闭包（scirs2-optimize API: fn(&ArrayView1<f64>) -> f64）
        let x_t = x_train_owned.clone();
        let y_t = y_train_owned.clone();
        let is_c = is_cat_owned.clone();
        let dims = d;

        let objective = |x: &ArrayView1<f64>| -> f64 {
            let params: Vec<f64> = x.iter().copied().collect();
            neg_log_posterior(&params, dims, &x_t, &y_t, &is_c)
        };

        // 配置优化器: gtol=1e-2 匹配 Python 的 options={'gtol': 1e-2}
        let mut options = Options::default();
        options.max_iter = 50;  // 核参数拟合通常 50 轮足够
        options.gtol = 1e-2;

        // 执行 BFGS 优化 (L-BFGS-B 无界等价于 BFGS)
        // Python: `scipy.optimize.minimize(method='l-bfgs-b')` 无 bounds
        let result: Result<OptimizeResult<f64>, _> = minimize(
            objective,
            &init_params,
            Method::BFGS,
            Some(options),
        );

        // 从优化结果构建 GPR
        let final_params: Vec<f64> = match result {
            Ok(ref res) if res.success => res.x.to_vec(),
            Ok(ref res) => res.x.to_vec(), // 即使未收敛也用最后的参数
            Err(_) => continue,             // 优化出错，跳过
        };

        let (inv_sq_ls, kernel_scale, noise_var) = decode_params(&final_params, d);

        // 检查参数有效性
        if inv_sq_ls.iter().any(|&v| !v.is_finite())
            || !kernel_scale.is_finite()
            || !noise_var.is_finite()
        {
            continue;
        }

        let gpr = GPRegressor::new(
            x_train.to_vec(),
            y_train.to_vec(),
            is_categorical.to_vec(),
            inv_sq_ls,
            kernel_scale,
            noise_var,
        );

        let lml = gpr.log_marginal_likelihood();
        if lml.is_finite() && lml > best_lml {
            best_lml = lml;
            best_gpr = Some(gpr);
        }
    }

    // 回退到默认参数
    best_gpr.unwrap_or_else(|| GPRegressor::new(
        x_train.to_vec(),
        y_train.to_vec(),
        is_categorical.to_vec(),
        vec![1.0; d],
        1.0,
        DEFAULT_MINIMUM_NOISE_VAR,
    ))
}

// 当 gp-lbfgsb feature 未启用时，回退到随机搜索
#[cfg(not(feature = "gp-lbfgsb"))]
pub(crate) fn fit_kernel_params_lbfgsb(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    seed: u64,
) -> crate::samplers::gp::GPRegressor {
    crate::samplers::gp::fit_kernel_params(x_train, y_train, is_categorical, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_kernel_params_lbfgsb_basic() {
        // 简单测试：确保不 panic 且后验合理
        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];
        let is_cat = vec![false];

        let gpr = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42);
        let (mean, _var) = gpr.posterior(&[0.5]);
        // 在训练点附近，后验均值应接近训练值
        assert!((mean - 0.25).abs() < 0.5, "mean={mean}, expected ~0.25");
    }

    #[test]
    fn test_fit_kernel_params_lbfgsb_empty() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let is_cat: Vec<bool> = vec![];
        let _ = fit_kernel_params_lbfgsb(&x, &y, &is_cat, 42);
    }

    #[cfg(feature = "gp-lbfgsb")]
    #[test]
    fn test_log_prior_at_unity() {
        // inv_sq_ls = [1.0] 是先验的最优值
        let lp1 = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        let lp2 = default_log_prior(&[2.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        // 在 inv_sq_ls=1 附近先验应更高
        assert!(lp1 > lp2, "lprio at 1.0={lp1} should be > at 2.0={lp2}");
    }

    #[cfg(feature = "gp-lbfgsb")]
    #[test]
    fn test_decode_encode_roundtrip() {
        let raw = vec![0.0, 0.5, -1.0]; // log(1.0), log(exp(0.5)), log(exp(-1))
        let (inv, ks, nv) = decode_params(&raw, 1);
        assert!((inv[0] - 1.0).abs() < 1e-10);
        assert!((ks - 0.5_f64.exp()).abs() < 1e-10);
        assert!((nv - (DEFAULT_MINIMUM_NOISE_VAR + (-1.0_f64).exp())).abs() < 1e-10);
    }
}
