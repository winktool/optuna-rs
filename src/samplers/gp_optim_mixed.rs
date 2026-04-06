//! GP 采集函数混合优化 — 对齐 Python `optuna._gp.optim_mixed`
//!
//! 实现 `optimize_acqf_mixed`:
//! 1. QMC Sobol 初始候选采样
//! 2. 轮盘赌选择初始点
//! 3. 交替连续/离散局部搜索:
//!    - 连续: 预条件 L-BFGS-B（有限差分梯度）
//!    - 离散: 穷举搜索（分类/小离散）或 Brent 线搜索（大离散）

use std::collections::VecDeque;

use super::qmc::sobol_point_pub;

// ════════════════════════════════════════════════════════════════════════
// 搜索空间信息
// ════════════════════════════════════════════════════════════════════════

/// 搜索空间结构化信息 — 区分连续/离散参数。
///
/// 对齐 Python `SearchSpace.continuous_indices` / `discrete_indices`。
pub struct SearchSpaceInfo {
    /// 连续参数索引 (FloatDistribution 无 step)
    pub continuous_indices: Vec<usize>,
    /// 离散参数索引 (Int / Categorical / Float+step)
    pub discrete_indices: Vec<usize>,
    /// 每个离散参数的归一化选择列表
    pub discrete_choices: Vec<Vec<f64>>,
    /// 是否为分类参数 (对应 discrete_indices)
    pub is_categorical_discrete: Vec<bool>,
    /// 离散搜索容差 (对应 discrete_indices)
    pub discrete_xtols: Vec<f64>,
}

/// 从搜索空间构建 SearchSpaceInfo。
pub fn build_search_space_info(
    search_space: &indexmap::IndexMap<String, crate::distributions::Distribution>,
    param_names: &[String],
) -> SearchSpaceInfo {
    use crate::distributions::Distribution;
    let mut continuous_indices = Vec::new();
    let mut discrete_indices = Vec::new();
    let mut discrete_choices = Vec::new();
    let mut is_categorical_discrete = Vec::new();
    let mut discrete_xtols = Vec::new();

    for (d, name) in param_names.iter().enumerate() {
        let dist = &search_space[name];
        match dist {
            Distribution::FloatDistribution(fd) => {
                if fd.step.is_some() {
                    // Float with step → discrete
                    let step = fd.step.unwrap();
                    let (low, high) = if fd.log {
                        (fd.low.ln(), fd.high.ln())
                    } else {
                        (fd.low, fd.high)
                    };
                    let range = high - low;
                    if range < 1e-14 {
                        continuous_indices.push(d); // degenerate
                        continue;
                    }
                    // 生成网格点（归一化空间）
                    let actual_step = if fd.log {
                        // log 空间中 step 对应原始空间
                        step
                    } else {
                        step
                    };
                    let mut choices = Vec::new();
                    let mut v = fd.low;
                    while v <= fd.high + 1e-10 * actual_step {
                        let norm_v = super::gp::normalize_param(v, dist);
                        choices.push(norm_v);
                        v += actual_step;
                    }
                    if choices.is_empty() {
                        choices.push(0.5);
                    }
                    let xtol = if choices.len() > 1 {
                        let min_diff = choices.windows(2)
                            .map(|w| (w[1] - w[0]).abs())
                            .fold(f64::MAX, f64::min);
                        min_diff / 4.0
                    } else {
                        0.01
                    };
                    discrete_indices.push(d);
                    discrete_choices.push(choices);
                    is_categorical_discrete.push(false);
                    discrete_xtols.push(xtol);
                } else {
                    // Float without step → continuous
                    continuous_indices.push(d);
                }
            }
            Distribution::IntDistribution(id) => {
                let (low, high) = if id.log {
                    ((id.low as f64).ln(), (id.high as f64).ln())
                } else {
                    (id.low as f64, id.high as f64)
                };
                let range = high - low;
                if range < 1e-14 {
                    continuous_indices.push(d); // degenerate single value
                    continue;
                }
                let step = id.step as f64;
                let mut choices = Vec::new();
                let mut v = id.low as f64;
                while v <= id.high as f64 + 0.5 * step {
                    let norm_v = super::gp::normalize_param(v, dist);
                    choices.push(norm_v);
                    v += step;
                }
                if choices.is_empty() {
                    choices.push(0.5);
                }
                let xtol = if choices.len() > 1 {
                    let min_diff = choices.windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .fold(f64::MAX, f64::min);
                    min_diff / 4.0
                } else {
                    0.01
                };
                discrete_indices.push(d);
                discrete_choices.push(choices);
                is_categorical_discrete.push(false);
                discrete_xtols.push(xtol);
            }
            Distribution::CategoricalDistribution(cd) => {
                let n = cd.choices.len();
                let choices: Vec<f64> = (0..n).map(|i| i as f64).collect();
                discrete_indices.push(d);
                discrete_choices.push(choices);
                is_categorical_discrete.push(true);
                discrete_xtols.push(0.25);
            }
        }
    }
    SearchSpaceInfo {
        continuous_indices,
        discrete_indices,
        discrete_choices,
        is_categorical_discrete,
        discrete_xtols,
    }
}

// ════════════════════════════════════════════════════════════════════════
// QMC Sobol 候选采样
// ════════════════════════════════════════════════════════════════════════

/// 使用 Sobol 序列采样归一化候选点。
///
/// 对齐 Python `search_space.sample_normalized_params`。
/// 分类参数: `floor(val * n_choices)`
/// 离散数值参数: snap 到最近的网格点
pub(crate) fn sample_normalized_sobol(
    n: usize,
    n_params: usize,
    ss_info: &SearchSpaceInfo,
    seed: u64,
) -> Vec<Vec<f64>> {
    (0..n).map(|i| {
        let sobol = sobol_point_pub(i as u64 + 1, n_params, true, seed);
        let mut point = sobol;
        // 对离散参数 snap 到合法值
        for (di, &dim_idx) in ss_info.discrete_indices.iter().enumerate() {
            if dim_idx < point.len() {
                let choices = &ss_info.discrete_choices[di];
                if ss_info.is_categorical_discrete[di] {
                    // 分类: floor(val * n_choices)
                    let n_choices = choices.len() as f64;
                    point[dim_idx] = (point[dim_idx] * n_choices).floor().min(n_choices - 1.0).max(0.0);
                } else {
                    // 离散数值: snap 到最近网格点
                    point[dim_idx] = snap_to_nearest(point[dim_idx], choices);
                }
            }
        }
        point
    }).collect()
}

/// Snap 到最近网格点
fn snap_to_nearest(val: f64, choices: &[f64]) -> f64 {
    let mut best = choices[0];
    let mut best_dist = (val - best).abs();
    for &c in &choices[1..] {
        let d = (val - c).abs();
        if d < best_dist {
            best_dist = d;
            best = c;
        }
    }
    best
}

// ════════════════════════════════════════════════════════════════════════
// 轮盘赌选择
// ════════════════════════════════════════════════════════════════════════

/// 轮盘赌选择: 根据 acqf 值概率性地选择初始点。
///
/// 对齐 Python `optimize_acqf_mixed` 中的 roulette selection:
/// `probs = exp(fvals - fvals[max_i])`, 排除 max_i, 归一化后随机选择。
fn roulette_select(
    fvals: &[f64],
    max_idx: usize,
    n_select: usize,
    rng: &mut rand_chacha::ChaCha8Rng,
) -> Vec<usize> {
    use rand::RngExt;
    let max_fval = fvals[max_idx];
    let mut probs: Vec<f64> = fvals.iter().map(|&f| (f - max_fval).exp()).collect();
    probs[max_idx] = 0.0;

    let sum: f64 = probs.iter().sum();
    if sum < 1e-30 {
        // 所有点几乎一样好，随机选
        let mut indices: Vec<usize> = (0..fvals.len()).filter(|&i| i != max_idx).collect();
        indices.truncate(n_select);
        return indices;
    }

    // 归一化
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 不放回采样 (简单实现)
    let mut selected = Vec::with_capacity(n_select);
    let mut used = vec![false; fvals.len()];
    used[max_idx] = true;

    for _ in 0..n_select {
        let r: f64 = rng.random_range(0.0..1.0);
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &p) in probs.iter().enumerate() {
            if used[i] { continue; }
            cumsum += p;
            if r <= cumsum {
                chosen = i;
                break;
            }
            chosen = i; // fallback to last
        }
        if !used[chosen] {
            selected.push(chosen);
            used[chosen] = true;
            // 重新归一化剩余概率
            let remaining_sum: f64 = probs.iter().enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(_, p)| p)
                .sum();
            if remaining_sum > 1e-30 {
                for (i, p) in probs.iter_mut().enumerate() {
                    if used[i] { *p = 0.0; } else { *p /= remaining_sum; }
                }
            }
        }
    }

    selected
}

// ════════════════════════════════════════════════════════════════════════
// L-BFGS-B 梯度上升 (连续参数)
// ════════════════════════════════════════════════════════════════════════

/// 对连续参数做预条件 L-BFGS-B 梯度上升。
///
/// 对齐 Python `_gradient_ascent_batched`:
/// - 预条件: z = x / lengthscale, 优化 z ∈ [0, 1/l]
/// - L-BFGS-B (m=10): 有限差分梯度 + Strong Wolfe 线搜索
/// - 投影到 bound 约束
///
/// 返回 (是否更新)。直接修改 `x` 和 `fval`。
fn gradient_ascent_continuous(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    x: &mut Vec<f64>,
    fval: &mut f64,
    continuous_indices: &[usize],
    lengthscales: &[f64],
    tol: f64,
) -> bool {
    if continuous_indices.is_empty() {
        return false;
    }

    let n_cont = continuous_indices.len();
    let pgtol = tol.sqrt();
    let max_iter = 200;
    let m = 10; // L-BFGS memory
    let h = 1e-5; // finite diff step

    // 预提取连续参数的 lengthscale — 对齐 Python:
    // `lengthscales = acqf.length_scales[continuous_indices]`
    let cont_ls: Vec<f64> = continuous_indices.iter().map(|&idx| lengthscales[idx]).collect();

    // 提取连续参数并预条件化: z = x / l
    let mut z: Vec<f64> = (0..n_cont).map(|ci| {
        x[continuous_indices[ci]] / cont_ls[ci]
    }).collect();
    let bounds: Vec<(f64, f64)> = (0..n_cont).map(|ci| {
        (0.0, 1.0 / cont_ls[ci])
    }).collect();

    // 负采集函数 (因为 L-BFGS 做最小化)
    let neg_acqf = |z_vals: &[f64]| -> f64 {
        let mut x_copy = x.clone();
        for (ci, &zi) in z_vals.iter().enumerate() {
            x_copy[continuous_indices[ci]] = zi * cont_ls[ci];
        }
        -eval_acqf(&x_copy)
    };

    // 有限差分梯度 (中心差分)
    let gradient = |z_vals: &[f64]| -> Vec<f64> {
        let f0 = neg_acqf(z_vals);
        let mut grad = vec![0.0; n_cont];
        for i in 0..n_cont {
            let mut z_fwd = z_vals.to_vec();
            let mut z_bwd = z_vals.to_vec();
            z_fwd[i] += h;
            z_bwd[i] -= h;
            // 投影到 bounds
            z_fwd[i] = z_fwd[i].clamp(bounds[i].0, bounds[i].1);
            z_bwd[i] = z_bwd[i].clamp(bounds[i].0, bounds[i].1);
            let actual_h = z_fwd[i] - z_bwd[i];
            if actual_h.abs() > 1e-15 {
                grad[i] = (neg_acqf(&z_fwd) - neg_acqf(&z_bwd)) / actual_h;
            }
        }
        let _ = f0;
        grad
    };

    // 投影到 bounds
    let project = |z_vals: &mut Vec<f64>| {
        for (i, z) in z_vals.iter_mut().enumerate() {
            *z = z.clamp(bounds[i].0, bounds[i].1);
        }
    };

    // L-BFGS-B 优化
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m);
    let mut fz = neg_acqf(&z);
    let mut g = gradient(&z);
    let mut n_iterations = 0;

    for _ in 0..max_iter {
        // 投影梯度收敛检查
        let pg_norm = projected_gradient_norm(&z, &g, &bounds);
        if pg_norm < pgtol {
            break;
        }

        // L-BFGS 两环递归
        let direction = lbfgs_two_loop_b(&g, &s_hist, &y_hist);

        // 方向导数
        let slope: f64 = g.iter().zip(direction.iter()).map(|(gi, di)| gi * di).sum();
        if slope >= 0.0 {
            break; // 非下降方向
        }

        // Armijo 回溯线搜索 (with projection)
        let c1 = 1e-4;
        let mut step = 1.0;
        let z_save = z.clone();
        let mut z_new;
        let mut fz_new;

        loop {
            z_new = z_save.iter().zip(direction.iter())
                .map(|(&zi, &di)| zi + step * di).collect();
            project(&mut z_new);
            fz_new = neg_acqf(&z_new);
            if fz_new <= fz + c1 * step * slope || step < 1e-10 {
                break;
            }
            step *= 0.5;
        }

        if !fz_new.is_finite() || (fz_new - fz).abs() < 1e-15 {
            break;
        }

        let g_new = gradient(&z_new);

        // 更新 L-BFGS 历史
        let s_k: Vec<f64> = z_new.iter().zip(z.iter())
            .map(|(zn, zo)| zn - zo).collect();
        let y_k: Vec<f64> = g_new.iter().zip(g.iter())
            .map(|(gn, go)| gn - go).collect();
        let sy: f64 = s_k.iter().zip(y_k.iter()).map(|(s, y)| s * y).sum();
        if sy > 1e-10 {
            if s_hist.len() >= m {
                s_hist.pop_front();
                y_hist.pop_front();
            }
            s_hist.push_back(s_k);
            y_hist.push_back(y_k);
        }

        z = z_new;
        fz = fz_new;
        g = g_new;
        n_iterations += 1;
    }

    if n_iterations == 0 {
        return false;
    }

    // 反变换回 x 空间
    let mut x_new = x.clone();
    for (ci, &zi) in z.iter().enumerate() {
        x_new[continuous_indices[ci]] = (zi * cont_ls[ci]).clamp(0.0, 1.0);
    }
    let fval_new = eval_acqf(&x_new);

    if fval_new > *fval {
        *x = x_new;
        *fval = fval_new;
        true
    } else {
        false
    }
}

/// 投影梯度范数: max |clamp(x - g, lo, hi) - x|
fn projected_gradient_norm(x: &[f64], g: &[f64], bounds: &[(f64, f64)]) -> f64 {
    x.iter().zip(g.iter()).zip(bounds.iter())
        .map(|((&xi, &gi), &(lo, hi))| {
            let projected = (xi - gi).clamp(lo, hi);
            (projected - xi).abs()
        })
        .fold(0.0_f64, f64::max)
}

/// L-BFGS 两环递归 (同 gp_lbfgsb.rs 但独立实现以避免循环依赖)
fn lbfgs_two_loop_b(
    grad: &[f64],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
) -> Vec<f64> {
    let n = grad.len();
    let k = s_hist.len();
    if k == 0 {
        return grad.iter().map(|&g| -g).collect();
    }

    let rho: Vec<f64> = (0..k).map(|i| {
        let sy: f64 = s_hist[i].iter().zip(y_hist[i].iter()).map(|(s, y)| s * y).sum();
        if sy.abs() > 1e-30 { 1.0 / sy } else { 0.0 }
    }).collect();

    let mut q = grad.to_vec();
    let mut alpha = vec![0.0; k];

    for i in (0..k).rev() {
        alpha[i] = rho[i] * s_hist[i].iter().zip(q.iter()).map(|(s, qi)| s * qi).sum::<f64>();
        for j in 0..n { q[j] -= alpha[i] * y_hist[i][j]; }
    }

    let last = k - 1;
    let yy: f64 = y_hist[last].iter().map(|y| y * y).sum();
    let gamma = if yy > 1e-30 {
        s_hist[last].iter().zip(y_hist[last].iter()).map(|(s, y)| s * y).sum::<f64>() / yy
    } else { 1.0 };

    let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();
    for i in 0..k {
        let beta: f64 = rho[i] * y_hist[i].iter().zip(r.iter()).map(|(y, ri)| y * ri).sum::<f64>();
        for j in 0..n { r[j] += (alpha[i] - beta) * s_hist[i][j]; }
    }

    r.iter().map(|ri| -ri).collect()
}

// ════════════════════════════════════════════════════════════════════════
// 离散参数搜索
// ════════════════════════════════════════════════════════════════════════

const MAX_INT_EXHAUSTIVE_SEARCH_PARAMS: usize = 16;

/// 穷举搜索: 遍历所有可能的选择值。
///
/// 对齐 Python `_exhaustive_search`。用于分类参数或选择数 ≤ 16。
fn exhaustive_search(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    x: &mut Vec<f64>,
    fval: &mut f64,
    param_idx: usize,
    choices: &[f64],
) -> bool {
    if choices.len() <= 1 {
        return false;
    }
    let current_val = x[param_idx];
    let mut best_val = current_val;
    let mut best_f = *fval;

    for &c in choices {
        if (c - current_val).abs() < 1e-12 {
            continue; // 跳过当前值
        }
        x[param_idx] = c;
        let f = eval_acqf(x);
        if f > best_f {
            best_f = f;
            best_val = c;
        }
    }

    x[param_idx] = best_val;
    if best_f > *fval {
        *fval = best_f;
        true
    } else {
        x[param_idx] = current_val;
        false
    }
}

/// Brent 风格的离散线搜索。
///
/// 对齐 Python `_discrete_line_search`:
/// 在网格点间做线性插值, 用 Brent 方法找极小值, 再 snap 到最近网格点。
fn discrete_line_search(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    x: &mut Vec<f64>,
    fval: &mut f64,
    param_idx: usize,
    grids: &[f64], // 已排序
    xtol: f64,
) -> bool {
    if grids.len() <= 1 {
        return false;
    }

    let current_val = x[param_idx];

    // 找当前值在 grid 中最近的索引
    let current_idx = find_nearest_index(current_val, grids);

    // 在网格点上缓存 acqf 值
    let mut cache: Vec<Option<f64>> = vec![None; grids.len()];
    let eval_at_grid = |x: &mut Vec<f64>, grid_idx: usize, cache: &mut Vec<Option<f64>>| -> f64 {
        if let Some(f) = cache[grid_idx] {
            return f;
        }
        x[param_idx] = grids[grid_idx];
        let f = eval_acqf(x);
        cache[grid_idx] = Some(f);
        f
    };

    // 评估当前点
    cache[current_idx] = Some(*fval);

    // 简化版: 评估当前点邻域 + 两端点, 选最优
    // (比完整 Brent 插值更简单但对我们的场景足够)
    let mut best_idx = current_idx;
    let mut best_f = *fval;

    // 评估所有网格点（对于 >16 的，只评估邻域 + 等间隔采样）
    let n_grids = grids.len();
    if n_grids <= 64 {
        // 全部评估
        for gi in 0..n_grids {
            let f = eval_at_grid(x, gi, &mut cache);
            if f > best_f {
                best_f = f;
                best_idx = gi;
            }
        }
    } else {
        // 评估邻域 (±3) + 等间隔 16 个点
        let _eval_range = |start: usize, end: usize| {
            for gi in start..end.min(n_grids) {
                let f = eval_at_grid(x, gi, &mut cache);
                if f > best_f {
                    best_f = f;
                    best_idx = gi;
                }
            }
        };
        let lo = current_idx.saturating_sub(3);
        let hi = (current_idx + 4).min(n_grids);
        // Evaluate neighborhood
        for gi in lo..hi {
            let f = eval_at_grid(x, gi, &mut cache);
            if f > best_f { best_f = f; best_idx = gi; }
        }
        // Evaluate evenly spaced
        let step = n_grids / 16;
        if step > 0 {
            let mut gi = 0;
            while gi < n_grids {
                let f = eval_at_grid(x, gi, &mut cache);
                if f > best_f { best_f = f; best_idx = gi; }
                gi += step;
            }
        }
        // Evaluate last
        let f = eval_at_grid(x, n_grids - 1, &mut cache);
        if f > best_f { best_f = f; best_idx = n_grids - 1; }

        // Second pass: neighborhood around best
        let lo2 = best_idx.saturating_sub(3);
        let hi2 = (best_idx + 4).min(n_grids);
        for gi in lo2..hi2 {
            let f = eval_at_grid(x, gi, &mut cache);
            if f > best_f { best_f = f; best_idx = gi; }
        }
    }

    let _ = xtol;
    x[param_idx] = grids[best_idx];
    if best_f > *fval {
        *fval = best_f;
        true
    } else {
        x[param_idx] = current_val;
        false
    }
}

/// 在排序数组中找最近元素的索引
fn find_nearest_index(val: f64, sorted: &[f64]) -> usize {
    if sorted.is_empty() { return 0; }
    match sorted.binary_search_by(|v| v.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => {
            if i == 0 { 0 }
            else if i >= sorted.len() { sorted.len() - 1 }
            else if (sorted[i] - val).abs() < (sorted[i - 1] - val).abs() { i }
            else { i - 1 }
        }
    }
}

/// 离散搜索路由: 分类/小选择 → 穷举; 大选择 → 线搜索。
///
/// 对齐 Python `_local_search_discrete`。
fn local_search_discrete(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    x: &mut Vec<f64>,
    fval: &mut f64,
    param_idx: usize,
    choices: &[f64],
    is_categorical: bool,
    xtol: f64,
) -> bool {
    if is_categorical || choices.len() <= MAX_INT_EXHAUSTIVE_SEARCH_PARAMS {
        exhaustive_search(eval_acqf, x, fval, param_idx, choices)
    } else {
        discrete_line_search(eval_acqf, x, fval, param_idx, choices, xtol)
    }
}

// ════════════════════════════════════════════════════════════════════════
// 交替优化主循环
// ════════════════════════════════════════════════════════════════════════

const CONTINUOUS_DIM: i32 = -1;

/// 交替连续/离散局部搜索。
///
/// 对齐 Python `local_search_mixed_batched`:
/// 1. 连续参数: 预条件 L-BFGS-B
/// 2. 离散参数: 逐维穷举/线搜索
/// 3. 收敛判断: last_changed_dims 追踪
///
/// 注: Rust 不做 batch，顺序处理每个起点（对齐 Python TODO 注释）。
fn local_search_mixed(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    xs0: Vec<Vec<f64>>,
    ss_info: &SearchSpaceInfo,
    lengthscales: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_starts = xs0.len();
    let mut xs = xs0;
    let mut fvals: Vec<f64> = xs.iter().map(|x| eval_acqf(x)).collect();
    let mut last_changed: Vec<i32> = vec![CONTINUOUS_DIM; n_starts];
    let mut converged = vec![false; n_starts];

    for _ in 0..max_iter {
        let mut all_converged = true;

        for b in 0..n_starts {
            if converged[b] { continue; }
            all_converged = false;

            // 1. 连续梯度上升
            let updated = gradient_ascent_continuous(
                eval_acqf,
                &mut xs[b],
                &mut fvals[b],
                &ss_info.continuous_indices,
                lengthscales,
                tol,
            );
            if updated {
                last_changed[b] = CONTINUOUS_DIM;
            }

            // 2. 逐离散维度搜索
            for (di, &dim_idx) in ss_info.discrete_indices.iter().enumerate() {
                // 收敛检查: 如果上次改变的就是这个维度 → 收敛
                if last_changed[b] == di as i32 {
                    converged[b] = true;
                    break;
                }

                let updated = local_search_discrete(
                    eval_acqf,
                    &mut xs[b],
                    &mut fvals[b],
                    dim_idx,
                    &ss_info.discrete_choices[di],
                    ss_info.is_categorical_discrete[di],
                    ss_info.discrete_xtols[di],
                );
                if updated {
                    last_changed[b] = di as i32;
                }
            }

            // 尾部收敛: 如果遍历完所有离散维度后 last_changed 仍然是 CONTINUOUS
            if !converged[b] && last_changed[b] == CONTINUOUS_DIM {
                converged[b] = true;
            }
        }

        if all_converged || converged.iter().all(|&c| c) {
            break;
        }
    }

    (xs, fvals)
}

// ════════════════════════════════════════════════════════════════════════
// 顶层入口: optimize_acqf_mixed
// ════════════════════════════════════════════════════════════════════════

/// 混合搜索空间中优化采集函数。
///
/// 对齐 Python `optuna._gp.optim_mixed.optimize_acqf_mixed`:
/// 1. QMC Sobol 采样 2048 个初始候选
/// 2. 批量评估 → 轮盘赌选择 n_local_search 个起点
/// 3. 交替连续/离散局部搜索
/// 4. 返回最优候选
pub fn optimize_acqf_mixed(
    eval_acqf: &dyn Fn(&[f64]) -> f64,
    ss_info: &SearchSpaceInfo,
    warmstart: &[Vec<f64>],
    n_params: usize,
    n_preliminary: usize,
    n_local_search: usize,
    lengthscales: &[f64],
    rng: &mut rand_chacha::ChaCha8Rng,
    _seed: u64,
) -> Vec<f64> {
    use rand::RngExt;

    // 1. QMC Sobol 采样
    let qmc_seed = rng.random_range(0u64..1u64 << 30);
    let sampled = sample_normalized_sobol(n_preliminary, n_params, ss_info, qmc_seed);

    // 2. 批量评估
    let fvals: Vec<f64> = sampled.iter().map(|x| eval_acqf(x)).collect();

    // 3. 找最优 + 轮盘选择
    let max_idx = fvals.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let n_warmstart = warmstart.len();
    let n_additional = n_local_search.saturating_sub(1 + n_warmstart);
    let n_additional = n_additional.min(fvals.len().saturating_sub(1));

    let additional_indices = roulette_select(&fvals, max_idx, n_additional, rng);

    // 4. 组装起点: best + roulette + warmstart
    let mut x_starts: Vec<Vec<f64>> = Vec::with_capacity(1 + additional_indices.len() + n_warmstart);
    x_starts.push(sampled[max_idx].clone());
    for &idx in &additional_indices {
        x_starts.push(sampled[idx].clone());
    }
    for ws in warmstart {
        x_starts.push(ws.clone());
    }

    // 5. 局部搜索
    let (xs_opt, fvals_opt) = local_search_mixed(
        eval_acqf, x_starts, ss_info, lengthscales, 1e-4, 100,
    );

    // 6. 返回最优
    let best_idx = fvals_opt.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    xs_opt[best_idx].clone()
}

// ════════════════════════════════════════════════════════════════════════
// 测试
// ════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_to_nearest() {
        let grids = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(snap_to_nearest(0.1, &grids), 0.0);
        assert_eq!(snap_to_nearest(0.3, &grids), 0.25);
        assert_eq!(snap_to_nearest(0.6, &grids), 0.5);
        assert_eq!(snap_to_nearest(0.9, &grids), 1.0);
    }

    #[test]
    fn test_find_nearest_index() {
        let grids = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(find_nearest_index(0.1, &grids), 0);
        assert_eq!(find_nearest_index(0.3, &grids), 1);
        assert_eq!(find_nearest_index(0.74, &grids), 3);
        assert_eq!(find_nearest_index(1.0, &grids), 4);
    }

    #[test]
    fn test_exhaustive_search_finds_best() {
        // f(x) = -(x - 0.75)^2 → 最优 x=0.75
        let eval = |x: &[f64]| -> f64 {
            -(x[0] - 0.75) * (x[0] - 0.75)
        };
        let mut x = vec![0.0];
        let mut fval = eval(&x);
        let choices = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let updated = exhaustive_search(&eval, &mut x, &mut fval, 0, &choices);
        assert!(updated);
        assert!((x[0] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_ascent_quadratic() {
        // f(x, y) = -(x-0.3)^2 - (y-0.7)^2
        // 最优: (0.3, 0.7)
        let eval = |x: &[f64]| -> f64 {
            -(x[0] - 0.3) * (x[0] - 0.3) - (x[1] - 0.7) * (x[1] - 0.7)
        };
        let mut x = vec![0.1, 0.1];
        let mut fval = eval(&x);
        let cont_indices = vec![0, 1];
        let lengthscales = vec![1.0, 1.0];

        let updated = gradient_ascent_continuous(
            &eval, &mut x, &mut fval, &cont_indices, &lengthscales, 1e-4,
        );
        assert!(updated);
        assert!((x[0] - 0.3).abs() < 0.05, "x[0]={}", x[0]);
        assert!((x[1] - 0.7).abs() < 0.05, "x[1]={}", x[1]);
    }

    #[test]
    fn test_roulette_select_basic() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let fvals = vec![1.0, 2.0, 5.0, 3.0, 0.5];
        let selected = roulette_select(&fvals, 2, 2, &mut rng);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|&i| i != 2), "Should not select max_idx");
    }

    #[test]
    fn test_optimize_acqf_mixed_1d_continuous() {
        // f(x) = -(x - 0.6)^2: best at x=0.6
        use rand::SeedableRng;
        let eval = |x: &[f64]| -> f64 {
            -(x[0] - 0.6) * (x[0] - 0.6)
        };
        let ss_info = SearchSpaceInfo {
            continuous_indices: vec![0],
            discrete_indices: vec![],
            discrete_choices: vec![],
            is_categorical_discrete: vec![],
            discrete_xtols: vec![],
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let result = optimize_acqf_mixed(
            &eval, &ss_info, &[], 1, 256, 5, &[1.0], &mut rng, 42,
        );
        assert!((result[0] - 0.6).abs() < 0.05, "result={:?}", result);
    }

    #[test]
    fn test_optimize_acqf_mixed_with_categorical() {
        // f(x, cat) = -(x - 0.5)^2 + (cat == 2) * 1.0
        // best: x=0.5, cat=2
        use rand::SeedableRng;
        let eval = |x: &[f64]| -> f64 {
            let cont = -(x[0] - 0.5) * (x[0] - 0.5);
            let cat_bonus = if (x[1] - 2.0).abs() < 0.5 { 1.0 } else { 0.0 };
            cont + cat_bonus
        };
        let ss_info = SearchSpaceInfo {
            continuous_indices: vec![0],
            discrete_indices: vec![1],
            discrete_choices: vec![vec![0.0, 1.0, 2.0]],
            is_categorical_discrete: vec![true],
            discrete_xtols: vec![0.25],
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let result = optimize_acqf_mixed(
            &eval, &ss_info, &[], 2, 256, 5, &[1.0], &mut rng, 42,
        );
        assert!((result[0] - 0.5).abs() < 0.1, "x={}", result[0]);
        assert!((result[1] - 2.0).abs() < 0.5, "cat={}", result[1]);
    }
}
