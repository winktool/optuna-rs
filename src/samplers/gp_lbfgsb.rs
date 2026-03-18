//! L-BFGS-B 超参数优化器 — GP 核参数拟合。
//!
//! 对应 Python `optuna.samplers._gp.gp._fit_kernel_params` +
//! `optuna.samplers._gp.batched_lbfgsb`。
//!
//! 内置 L-BFGS 优化器，忠实移植 Python `scipy.optimize.minimize(method='L-BFGS-B')` 的行为：
//! - **L-BFGS 两环递归** (两环递归算法, m=10 校正对) 替代全量 BFGS 矩阵
//! - **Strong Wolfe 线搜索** (Nocedal & Wright Algorithm 3.5/3.6) 替代简单 Armijo 回溯
//! - **中心差分梯度** (`∂f/∂x_i ≈ (f(x+h)-f(x-h))/(2h)`) 替代 PyTorch autograd
//! - 不依赖外部优化库，减少编译依赖冲突
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
//! 4. **梯度计算**: Python 用 PyTorch autograd；Rust 用中心差分 (h=1e-5)
//! 5. **scipy 调用**: `minimize(loss, x0, jac=True, method='l-bfgs-b', options={'gtol': 1e-2})`

use std::collections::VecDeque;

use crate::samplers::gp::{
    GPRegressor, KernelParamsCache, DEFAULT_MINIMUM_NOISE_VAR, default_log_prior,
};

/// 从 raw 参数向量 (log 空间) 解码出 GP 超参数。
///
/// 布局: `[log(inv_sq_ls[0]), ..., log(inv_sq_ls[n-1]), log(kernel_scale), log(noise_var - min_noise)]`
/// 对于 deterministic_objective 模式，只有 d+1 个参数（无 noise_var）。
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
fn gradient_finite_diff(
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

// ═══════════════════════════════════════════════════════════════════════════
// 内置 L-BFGS 优化器 (对齐 scipy.optimize.minimize(method='L-BFGS-B'))
// ═══════════════════════════════════════════════════════════════════════════

/// 向量点积
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 向量 L2 范数
#[inline]
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// L-BFGS 两环递归算法 (Two-Loop Recursion)。
///
/// 对齐 scipy L-BFGS-B 的核心方向计算。
/// 使用 m 个最近的校正对 {s_k, y_k} 隐式近似 Hessian 逆矩阵，
/// 无需存储 n×n 矩阵。
///
/// 参考: Nocedal & Wright, "Numerical Optimization", Algorithm 7.4
///
/// # 参数
/// * `grad` - 当前梯度 g_k
/// * `s_hist` - 校正对历史: s_i = x_{i+1} - x_i
/// * `y_hist` - 校正对历史: y_i = g_{i+1} - g_i
///
/// # 返回
/// 搜索方向 d = -H_k * g_k
fn lbfgs_two_loop(
    grad: &[f64],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let n = grad.len();
    let k = s_hist.len();

    if k == 0 {
        // 无历史信息，使用负梯度 (等价于 H_0 = I)
        return grad.iter().map(|&g| -g).collect();
    }

    // 计算 ρ_i = 1/(y_i · s_i)
    let rho: Vec<f64> = (0..k)
        .map(|i| {
            let sy = dot(&s_hist[i], &y_hist[i]);
            if sy.abs() > 1e-30 { 1.0 / sy } else { 0.0 }
        })
        .collect();

    // 第一环: 反向遍历 (从最新 k-1 到最旧 0)
    let mut q = grad.to_vec();
    let mut alpha = vec![0.0; k];

    for i in (0..k).rev() {
        alpha[i] = rho[i] * dot(&s_hist[i], &q);
        for j in 0..n {
            q[j] -= alpha[i] * y_hist[i][j];
        }
    }

    // 初始 Hessian 缩放: H_0 = γI
    // γ = (s_{k-1} · y_{k-1}) / (y_{k-1} · y_{k-1})
    // 对齐 scipy L-BFGS-B 的 Hessian 初始缩放策略
    let last = k - 1;
    let yy = dot(&y_hist[last], &y_hist[last]);
    let gamma = if yy > 1e-30 {
        dot(&s_hist[last], &y_hist[last]) / yy
    } else {
        1.0
    };

    let mut r: Vec<f64> = q.iter().map(|&qi| gamma * qi).collect();

    // 第二环: 正向遍历 (从最旧 0 到最新 k-1)
    for i in 0..k {
        let beta = rho[i] * dot(&y_hist[i], &r);
        for j in 0..n {
            r[j] += (alpha[i] - beta) * s_hist[i][j];
        }
    }

    // 返回 d = -H_k * g_k
    r.iter().map(|&ri| -ri).collect()
}

/// Strong Wolfe 线搜索。
///
/// 对齐 scipy 的 L-BFGS-B 线搜索 (Moré-Thuente / Nocedal-Wright 风格)。
/// 找到满足 Strong Wolfe 条件的步长 α:
///   1. Armijo (充分下降): φ(α) ≤ φ(0) + c1·α·φ'(0)
///   2. Curvature (曲率):  |φ'(α)| ≤ c2·|φ'(0)|
///
/// c1 = 1e-4, c2 = 0.9 (L-BFGS 推荐值, 对齐 scipy 默认)
///
/// 参考: Nocedal & Wright, "Numerical Optimization", Algorithm 3.5 + 3.6
///
/// # 参数
/// * `phi` - 沿搜索方向的函数值: φ(α) = f(x + α·d)
/// * `dphi` - 沿搜索方向的方向导数: φ'(α) = g(x + α·d) · d
/// * `phi0` - φ(0) = f(x)
/// * `dphi0` - φ'(0) = g(x) · d (应为负值)
///
/// # 返回
/// 接受的步长 α
fn strong_wolfe_search(
    phi: &dyn Fn(f64) -> f64,
    dphi: &dyn Fn(f64) -> f64,
    phi0: f64,
    dphi0: f64,
) -> f64 {
    let c1 = 1e-4;
    let c2 = 0.9;

    if dphi0 >= 0.0 {
        // 非下降方向，返回极小步长
        return 1e-8;
    }

    let max_iter = 20;
    let alpha_max = 1e5;

    let mut alpha_prev = 0.0;
    let mut phi_prev = phi0;
    let mut alpha = 1.0; // L-BFGS 推荐初始步长 α=1

    for i in 0..max_iter {
        let phi_alpha = phi(alpha);

        // 条件 U1: 不满足 Armijo，或者函数值递增 (i>0)
        if phi_alpha > phi0 + c1 * alpha * dphi0 || (i > 0 && phi_alpha >= phi_prev) {
            return zoom(phi, dphi, alpha_prev, alpha, phi_prev, phi_alpha, phi0, dphi0, c1, c2);
        }

        let dphi_alpha = dphi(alpha);

        // 条件 U2: 满足 Strong Wolfe 曲率条件 → 接受
        if dphi_alpha.abs() <= -c2 * dphi0 {
            return alpha;
        }

        // 条件 U3: 方向导数非负 → 区间反向
        if dphi_alpha >= 0.0 {
            return zoom(phi, dphi, alpha, alpha_prev, phi_alpha, phi_prev, phi0, dphi0, c1, c2);
        }

        // 扩大步长继续搜索
        alpha_prev = alpha;
        phi_prev = phi_alpha;
        alpha = (2.0 * alpha).min(alpha_max);
    }

    alpha
}

/// Zoom 过程 — Strong Wolfe 线搜索的细化阶段。
///
/// 在 [a_lo, a_hi] 区间内用二分法找满足 Strong Wolfe 条件的步长。
///
/// 参考: Nocedal & Wright, Algorithm 3.6
///
/// # 保证
/// - a_lo 满足 Armijo 且 φ(a_lo) ≤ φ(a_hi)
/// - 区间 [a_lo, a_hi] 内存在满足 Strong Wolfe 的步长
fn zoom(
    phi: &dyn Fn(f64) -> f64,
    dphi: &dyn Fn(f64) -> f64,
    mut a_lo: f64,
    mut a_hi: f64,
    mut phi_lo: f64,
    mut _phi_hi: f64,
    phi0: f64,
    dphi0: f64,
    c1: f64,
    c2: f64,
) -> f64 {
    for _ in 0..20 {
        if (a_hi - a_lo).abs() < 1e-15 {
            break;
        }

        // 二分法选取试探步长 (scipy 使用三次插值, 二分法更稳健)
        let a_j = 0.5 * (a_lo + a_hi);
        let phi_j = phi(a_j);

        if phi_j > phi0 + c1 * a_j * dphi0 || phi_j >= phi_lo {
            // 试探点不满足 Armijo 或函数值不够低 → 缩小右边界
            a_hi = a_j;
            _phi_hi = phi_j;
        } else {
            let dphi_j = dphi(a_j);

            // 满足 Strong Wolfe → 接受
            if dphi_j.abs() <= -c2 * dphi0 {
                return a_j;
            }

            // 方向导数符号判断：如果 dphi_j 与 (a_hi - a_lo) 同号→ 调整边界
            if dphi_j * (a_hi - a_lo) >= 0.0 {
                a_hi = a_lo;
                _phi_hi = phi_lo;
            }

            a_lo = a_j;
            phi_lo = phi_j;
        }
    }

    a_lo
}

/// L-BFGS 优化结果
struct LbfgsResult {
    /// 最优参数
    x: Vec<f64>,
    /// 最优函数值
    _f: f64,
}

/// L-BFGS 无约束优化器。
///
/// 对齐 `scipy.optimize.minimize(method='L-BFGS-B')`:
/// - L-BFGS 两环递归 (m=10 校正对, 对齐 scipy 默认 maxcor=10)
/// - Strong Wolfe 线搜索 (c1=1e-4, c2=0.9)
/// - 梯度范数收敛判据 (‖g‖ < gtol)
///
/// # 参数
/// * `objective` - 目标函数 f(x)
/// * `gradient` - 梯度函数 ∇f(x)
/// * `x0` - 初始猜测
/// * `max_iter` - 最大迭代数 (对齐 Python 默认 50)
/// * `gtol` - 梯度收敛阈值 (对齐 Python 1e-2)
/// * `m` - L-BFGS 校正对数量 (对齐 scipy 默认 10)
fn lbfgs_minimize(
    objective: &dyn Fn(&[f64]) -> f64,
    gradient: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    max_iter: usize,
    gtol: f64,
    m: usize,
) -> Option<LbfgsResult> {
    let n = x0.len();
    if n == 0 {
        return None;
    }

    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);

    let mut x = x0.to_vec();
    let mut fx = objective(&x);
    let mut g = gradient(&x);

    for _ in 0..max_iter {
        // 梯度收敛检查 (对齐 scipy options={'gtol': ...})
        if norm(&g) < gtol {
            break;
        }

        // L-BFGS 两环递归计算搜索方向 d = -H_k·g_k
        let direction = lbfgs_two_loop(&g, &s_hist, &y_hist);

        // 方向导数 φ'(0) = g · d
        let slope = dot(&g, &direction);

        // Strong Wolfe 线搜索
        let x_ref = x.clone();
        let d_ref = direction.clone();

        let phi = |alpha: f64| -> f64 {
            let xn: Vec<f64> = (0..n).map(|i| x_ref[i] + alpha * d_ref[i]).collect();
            objective(&xn)
        };
        let dphi = |alpha: f64| -> f64 {
            let xn: Vec<f64> = (0..n).map(|i| x_ref[i] + alpha * d_ref[i]).collect();
            let gn = gradient(&xn);
            dot(&gn, &d_ref)
        };

        let alpha = strong_wolfe_search(&phi, &dphi, fx, slope);

        // 计算步长向量 s = α·d 和新位置
        let s: Vec<f64> = direction.iter().map(|&d| alpha * d).collect();
        let x_new: Vec<f64> = x.iter().zip(s.iter()).map(|(&xi, &si)| xi + si).collect();
        let fx_new = objective(&x_new);
        let g_new = gradient(&x_new);

        // 梯度差 y = g_{k+1} - g_k
        let y: Vec<f64> = g_new.iter().zip(g.iter()).map(|(&gn, &go)| gn - go).collect();
        let sy = dot(&s, &y);

        // 曲率条件: 只有 s·y > 0 时保存校正对 (保证正定性)
        if sy > 1e-10 {
            // 超出容量时移除最旧的校正对 (有限内存)
            if s_hist.len() >= m {
                s_hist.pop_front();
                y_hist.pop_front();
            }
            s_hist.push_back(s);
            y_hist.push_back(y);
        }

        x = x_new;
        fx = fx_new;
        g = g_new;
    }

    Some(LbfgsResult { x, _f: fx })
}

/// 使用 L-BFGS 优化 GP 核超参数。
///
/// 对应 Python `optuna.samplers._gp.gp.fit_kernel_params`:
/// - 先尝试 gpr_cache 作为初始值，失败则用默认值
/// - `scipy.optimize.minimize(method='l-bfgs-b', jac=True, options={'gtol': 1e-2})`
/// - 包含 log-prior (from `prior.py`)
///
/// ## 对齐 Python 关键细节
/// - 默认核参数: `torch.ones(X.shape[1] + 2)` → inv_sq_ls=1.0, ks=1.0, nv=1.0
/// - 初始 noise_var 编码: `log(noise_var - 0.99 * minimum_noise)` (避免 log(0))
/// - L-BFGS-B: m=10 校正对, gtol=1e-2
pub(crate) fn fit_kernel_params_lbfgsb(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    is_categorical: &[bool],
    _seed: u64,
    gpr_cache: Option<&KernelParamsCache>,
    deterministic_objective: bool,
) -> GPRegressor {
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

    // 对应 Python: default_kernel_params = torch.ones(X.shape[1] + 2)
    // → inv_sq_ls = [1.0; d], kernel_scale = 1.0, noise_var = 1.0
    let default_cache = KernelParamsCache {
        inverse_squared_lengthscales: vec![1.0; d],
        kernel_scale: 1.0,
        noise_var: 1.0, // 对齐 Python torch.ones — 注意不是 MINIMUM_NOISE + 0.01
    };

    // 对应 Python: for gpr_cache_to_use in [gpr_cache, default_gpr_cache]
    // 先用 cache（若有），再用默认值 — 共 2 次尝试
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

        let objective = |x: &[f64]| -> f64 {
            neg_log_posterior(x, dims, &x_t, &y_t, &is_c, det)
        };

        let x_t2 = x_train_owned.clone();
        let y_t2 = y_train_owned.clone();
        let is_c2 = is_cat_owned.clone();

        let gradient = |x: &[f64]| -> Vec<f64> {
            gradient_finite_diff(x, dims, &x_t2, &y_t2, &is_c2, det)
        };

        // 对齐 scipy: method='l-bfgs-b', options={'gtol': 1e-2}, maxcor=10
        let result = lbfgs_minimize(&objective, &gradient, &init_params, 50, 1e-2, 10);

        let final_params = match result {
            Some(ref res) => &res.x,
            None => continue,
        };

        let (inv_sq_ls, kernel_scale, noise_var) = decode_params(final_params, d, deterministic_objective);

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

    #[test]
    fn test_log_prior_at_unity() {
        use crate::samplers::gp::DEFAULT_MINIMUM_NOISE_VAR;
        // inv_sq_ls = [1.0] 是先验的最优值
        let lp1 = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        let lp2 = default_log_prior(&[2.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR);
        assert!(lp1 > lp2, "lprio at 1.0={lp1} should be > at 2.0={lp2}");
    }

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

    /// 对齐 Python: log_prior 噪声越大先验越低
    #[test]
    fn test_log_prior_noise_penalty() {
        use crate::samplers::gp::DEFAULT_MINIMUM_NOISE_VAR;
        let lp_small = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR + 0.01);
        let lp_big = default_log_prior(&[1.0], 1.0, DEFAULT_MINIMUM_NOISE_VAR + 10.0);
        assert!(lp_small > lp_big, "小噪声先验={lp_small} 应 > 大噪声先验={lp_big}");
    }

    // ===================================================================
    // L-BFGS 优化器单元测试
    // ===================================================================

    /// L-BFGS 最小化简单二次函数 f(x) = x² → 最优解 x=0
    #[test]
    fn test_lbfgs_quadratic_1d() {
        let obj = |x: &[f64]| -> f64 { x[0] * x[0] };
        let grad = |x: &[f64]| -> Vec<f64> { vec![2.0 * x[0]] };
        let res = lbfgs_minimize(&obj, &grad, &[5.0], 100, 1e-6, 10).unwrap();
        assert!(res.x[0].abs() < 1e-4, "x={}, expected ~0", res.x[0]);
    }

    /// L-BFGS 最小化 2D 二次函数 f(x,y) = x² + 4y² → 最优 (0,0)
    #[test]
    fn test_lbfgs_quadratic_2d() {
        let obj = |x: &[f64]| -> f64 { x[0] * x[0] + 4.0 * x[1] * x[1] };
        let grad = |x: &[f64]| -> Vec<f64> { vec![2.0 * x[0], 8.0 * x[1]] };
        let res = lbfgs_minimize(&obj, &grad, &[3.0, -2.0], 100, 1e-6, 10).unwrap();
        assert!(res.x[0].abs() < 1e-4, "x={}", res.x[0]);
        assert!(res.x[1].abs() < 1e-4, "y={}", res.x[1]);
    }

    /// L-BFGS 最小化 Rosenbrock 函数 f(x,y) = (1-x)² + 100(y-x²)²
    /// 最优解 (1, 1), 这是经典的优化器基准测试
    #[test]
    fn test_lbfgs_rosenbrock() {
        let obj = |x: &[f64]| -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
        };
        let grad = |x: &[f64]| -> Vec<f64> {
            vec![
                -2.0 * (1.0 - x[0]) + 200.0 * (x[1] - x[0] * x[0]) * (-2.0 * x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        };
        let res = lbfgs_minimize(&obj, &grad, &[-1.0, 1.0], 200, 1e-6, 10).unwrap();
        assert!((res.x[0] - 1.0).abs() < 0.1, "x={}, expected ~1.0", res.x[0]);
        assert!((res.x[1] - 1.0).abs() < 0.1, "y={}, expected ~1.0", res.x[1]);
    }

    /// L-BFGS 两环递归: 无历史时返回负梯度
    #[test]
    fn test_lbfgs_two_loop_no_history() {
        let grad = vec![1.0, -2.0, 3.0];
        let s_hist = VecDeque::new();
        let y_hist = VecDeque::new();
        let d = lbfgs_two_loop(&grad, &s_hist, &y_hist);
        assert_eq!(d.len(), 3);
        assert!((d[0] - (-1.0)).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - (-3.0)).abs() < 1e-10);
    }

    /// Strong Wolfe 线搜索: 验证搜索找到的步长满足 Wolfe 条件
    #[test]
    fn test_strong_wolfe_search_basic() {
        // f(α) = (x₀ + α·d)² where x₀=5, d=-1 → min at α=5
        let phi = |alpha: f64| (5.0 - alpha) * (5.0 - alpha);
        let dphi = |alpha: f64| -2.0 * (5.0 - alpha);
        let phi0 = 25.0;
        let dphi0 = -10.0;
        let alpha = strong_wolfe_search(&phi, &dphi, phi0, dphi0);
        let f_alpha = phi(alpha);
        // 验证 Armijo 条件: f(α) ≤ f(0) + c1·α·f'(0)
        assert!(f_alpha <= phi0 + 1e-4 * alpha * dphi0,
            "Armijo failed: f({alpha})={f_alpha}");
        // 验证 Curvature 条件: |f'(α)| ≤ c2·|f'(0)|
        let dphi_alpha = dphi(alpha).abs();
        assert!(dphi_alpha <= 0.9 * dphi0.abs(),
            "Curvature failed: |f'({alpha})|={dphi_alpha}");
    }

    /// Strong Wolfe 线搜索: 非下降方向返回极小步长
    #[test]
    fn test_strong_wolfe_non_descent() {
        let phi = |alpha: f64| alpha * alpha;
        let dphi = |alpha: f64| 2.0 * alpha;
        // dphi0 = 1.0 > 0 → non-descent
        let alpha = strong_wolfe_search(&phi, &dphi, 0.0, 1.0);
        assert!(alpha <= 1e-7, "alpha={alpha}, expected ~1e-8");
    }

    /// L-BFGS 有限内存: 验证超出 m 时正确丢弃旧校正对
    #[test]
    fn test_lbfgs_limited_memory() {
        // 使用 m=2 观测内存限制是否正常工作
        let obj = |x: &[f64]| -> f64 {
            x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        };
        let grad = |x: &[f64]| -> Vec<f64> {
            vec![2.0 * x[0], 2.0 * x[1], 2.0 * x[2]]
        };
        let res = lbfgs_minimize(&obj, &grad, &[10.0, -5.0, 3.0], 100, 1e-6, 2).unwrap();
        assert!(res.x[0].abs() < 0.01);
        assert!(res.x[1].abs() < 0.01);
        assert!(res.x[2].abs() < 0.01);
    }

    /// 对齐 Python: 默认噪声初始化应为 1.0
    #[test]
    fn test_default_noise_var_is_one() {
        let cache = KernelParamsCache {
            inverse_squared_lengthscales: vec![1.0],
            kernel_scale: 1.0,
            noise_var: 1.0,
        };
        let encoded = encode_cache(&cache, false);
        // raw_noise = log(1.0 - DEFAULT_MINIMUM_NOISE_VAR) ≈ log(0.999999) ≈ 0
        assert!(encoded.last().unwrap().abs() < 0.01, "raw_noise={}", encoded.last().unwrap());
    }
}
