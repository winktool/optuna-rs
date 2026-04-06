use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// 约束值的系统属性键名。
///
/// 对应 Python `optuna.samplers._base._CONSTRAINTS_KEY`。
pub const CONSTRAINTS_KEY: &str = "constraints";

/// 检查试验是否满足约束条件（可行解）。
///
/// 对应 Python `_constrained_optimization._get_feasible_trials()` 的单元素版本。
/// 约束值 <= 0 视为满足。无约束值的试验视为不可行。
pub fn is_feasible(trial: &FrozenTrial) -> bool {
    match trial.system_attrs.get(CONSTRAINTS_KEY) {
        Some(serde_json::Value::Array(constraints)) => {
            constraints.iter().all(|v| v.as_f64().map_or(false, |c| c <= 0.0))
        }
        _ => false, // 无约束值 → 不可行（与 Python 一致）
    }
}

/// 获取可行试验列表。
///
/// 对应 Python `_get_feasible_trials()`。
pub fn get_feasible_trials(trials: &[FrozenTrial]) -> Vec<&FrozenTrial> {
    trials.iter().filter(|t| is_feasible(t)).collect()
}

/// 获取试验的总约束违反量。
///
/// 对应 Python `_evaluate_penalty()` 的单元素版本。
/// 返回 sum(max(0, c_i))。无约束值返回 NaN。
pub fn constraint_violation(trial: &FrozenTrial) -> f64 {
    match trial.system_attrs.get(CONSTRAINTS_KEY) {
        Some(serde_json::Value::Array(constraints)) => {
            constraints.iter()
                .filter_map(|v| v.as_f64())
                .map(|c| c.max(0.0))
                .sum()
        }
        _ => f64::NAN,
    }
}

/// 约束支配关系判断。
///
/// 对应 Python `_constrained_dominates()`。
///
/// 试验 a 约束支配试验 b 当且仅当：
/// 1. a 可行而 b 不可行
/// 2. a 和 b 都不可行，但 a 的总约束违反量更小
/// 3. a 和 b 都可行，且 a Pareto 支配 b
pub fn constrained_dominates(
    a: &FrozenTrial,
    b: &FrozenTrial,
    directions: &[StudyDirection],
) -> bool {
    let feasible_a = is_feasible(a);
    let feasible_b = is_feasible(b);

    // 检查约束值是否存在
    let has_constraints_a = a.system_attrs.contains_key(CONSTRAINTS_KEY);
    let has_constraints_b = b.system_attrs.contains_key(CONSTRAINTS_KEY);

    if !has_constraints_a && !has_constraints_b {
        // 都无约束值：退化为普通支配
        return match (&a.values, &b.values) {
            (Some(va), Some(vb)) => dominates(va, vb, directions),
            _ => false,
        };
    }

    if has_constraints_a && !has_constraints_b {
        return true; // a 有约束值，b 没有 → a 支配
    }
    if !has_constraints_a && has_constraints_b {
        return false; // a 没有约束值，b 有 → a 被支配
    }

    // 两者都有约束值
    if a.state != TrialState::Complete { return false; }
    if b.state != TrialState::Complete { return true; }

    if feasible_a && feasible_b {
        // 都可行 → 普通 Pareto 支配
        match (&a.values, &b.values) {
            (Some(va), Some(vb)) => dominates(va, vb, directions),
            _ => false,
        }
    } else if feasible_a {
        true // a 可行，b 不可行
    } else if feasible_b {
        false // a 不可行，b 可行
    } else {
        // 都不可行 → 比较约束违反量
        constraint_violation(a) < constraint_violation(b)
    }
}

/// 带约束的快速非支配排序。
///
/// 对应 Python NSGA-II 中使用 `_constrained_dominates` 的排序。
pub fn constrained_fast_non_dominated_sort(
    trials: &[&FrozenTrial],
    directions: &[StudyDirection],
) -> Vec<Vec<usize>> {
    let n = trials.len();
    if n == 0 { return vec![]; }

    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if constrained_dominates(trials[i], trials[j], directions) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if constrained_dominates(trials[j], trials[i], directions) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Returns true if solution `a` Pareto-dominates solution `b`.
///
/// A dominates B when A is at least as good in all objectives and strictly
/// better in at least one.
pub fn dominates(a: &[f64], b: &[f64], directions: &[StudyDirection]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), directions.len());

    // 防御式处理：若方向未设置（NotSet），不应让调用栈 panic 崩溃。
    // Python 对应场景会在更上层做方向有效性校验；这里返回 false 以保持运行稳定性。
    if directions.iter().any(|d| matches!(d, StudyDirection::NotSet)) {
        return false;
    }

    let mut dominated_in_any = false;
    for i in 0..a.len() {
        let cmp = match directions[i] {
            StudyDirection::Minimize => a[i].partial_cmp(&b[i]),
            StudyDirection::Maximize => b[i].partial_cmp(&a[i]),
            StudyDirection::NotSet => return false,
        };
        match cmp {
            Some(std::cmp::Ordering::Greater) => return false, // a is worse in this objective
            Some(std::cmp::Ordering::Less) => dominated_in_any = true,
            _ => {}
        }
    }
    dominated_in_any
}

/// Fast non-dominated sorting (NSGA-II style).
///
/// Returns rank-ordered fronts as vectors of indices into the input slice.
/// Front 0 is the Pareto front, front 1 is the second-best, etc.
pub fn fast_non_dominated_sort(
    trials: &[&FrozenTrial],
    directions: &[StudyDirection],
) -> Vec<Vec<usize>> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }

    let values: Vec<&[f64]> = trials
        .iter()
        .map(|t| t.values.as_deref().unwrap_or(&[]))
        .collect();

    // domination_count[i] = how many solutions dominate i
    let mut domination_count = vec![0usize; n];
    // dominated_set[i] = set of solutions that i dominates
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(values[i], values[j], directions) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(values[j], values[i], directions) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Compute crowding distance for each trial.
///
/// Returns a vector of crowding distances, one per input trial.
/// Boundary solutions get `f64::INFINITY`.
pub fn crowding_distance(trials: &[&FrozenTrial], directions: &[StudyDirection]) -> Vec<f64> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }

    let n_objectives = directions.len();
    let mut distances = vec![0.0_f64; n];

    for m in 0..n_objectives {
        // Sort indices by objective m
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let va = trials[a].values.as_ref().map(|v| v[m]).unwrap_or(f64::NAN);
            let vb = trials[b].values.as_ref().map(|v| v[m]).unwrap_or(f64::NAN);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build padded values array: [-inf, v_sorted[0], ..., v_sorted[n-1], +inf]
        // Matching Python's sentinel padding approach.
        let mut vs: Vec<f64> = Vec::with_capacity(n + 2);
        vs.push(f64::NEG_INFINITY);
        for &idx in &indices {
            vs.push(trials[idx].values.as_ref().map(|v| v[m]).unwrap_or(0.0));
        }
        vs.push(f64::INFINITY);

        // If all values are the same, skip this dimension (matching Python).
        if vs[1] == vs[n] {
            continue;
        }

        // Smallest finite value and largest finite value (matching Python).
        let v_min = vs.iter().copied().find(|&x| x != f64::NEG_INFINITY).unwrap_or(0.0);
        let v_max = vs.iter().rev().copied().find(|&x| x != f64::INFINITY).unwrap_or(0.0);
        let width = v_max - v_min;
        let width = if width <= 0.0 { 1.0 } else { width };

        for j in 0..n {
            // Matching Python: gap = 0 if vs[j] == vs[j+2] else vs[j+2] - vs[j]
            // This handles inf - inf = 0 and -inf - (-inf) = 0 correctly.
            let gap = if vs[j] == vs[j + 2] {
                0.0
            } else {
                vs[j + 2] - vs[j]
            };
            distances[indices[j]] += gap / width;
        }
    }

    distances
}

/// Mark which trials are on the Pareto front.
pub fn is_pareto_front(trials: &[&FrozenTrial], directions: &[StudyDirection]) -> Vec<bool> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }

    let fronts = fast_non_dominated_sort(trials, directions);
    let mut result = vec![false; n];
    if let Some(front_0) = fronts.first() {
        for &i in front_0 {
            result[i] = true;
        }
    }
    result
}

/// Compute 2D hypervolume indicator.
///
/// `points` are objective value pairs; `reference` is the reference point.
/// Points that are dominated by the reference point contribute positive volume.
pub fn hypervolume_2d(points: &[[f64; 2]], reference: [f64; 2]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Filter to points that are not dominated by the reference point
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1])
        .copied()
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort by first objective ascending
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut volume = 0.0;
    let mut y_bound = reference[1];

    for p in &pts {
        if p[1] < y_bound {
            volume += (reference[0] - p[0]) * (y_bound - p[1]);
            y_bound = p[1];
        }
    }

    volume
}

/// 计算 N 维超体积指标（通用版本）。
///
/// 对应 Python `optuna._hypervolume` 包的功能。
/// - 2D: O(N log N) 扫描线算法
/// - 3D: O(N²) 坐标压缩 + 累积最大值（对齐 Python `_compute_3d`）
/// - ≥4D: WFG 包含-排除递归算法，含 Pareto 剪枝（对齐 Python `_compute_hv`）
///
/// # 参数
/// * `points` - 目标值向量列表（最小化方向）
/// * `reference` - 参考点（被所有 Pareto 前沿点支配）
///
/// # 返回
/// 超体积指标值（非负浮点数）
pub fn hypervolume(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    if points.is_empty() || reference.is_empty() {
        return 0.0;
    }

    let n_objectives = reference.len();

    // 筛选被参考点支配的点（在所有维度上都小于参考点）
    let feasible: Vec<Vec<f64>> = points
        .iter()
        .filter(|p| {
            p.len() == n_objectives && p.iter().zip(reference.iter()).all(|(pi, ri)| *pi < *ri)
        })
        .cloned()
        .collect();

    if feasible.is_empty() {
        return 0.0;
    }

    // 2D 特化路径 — O(N log N)
    if n_objectives == 2 {
        let pts_2d: Vec<[f64; 2]> = feasible.iter().map(|p| [p[0], p[1]]).collect();
        return hypervolume_2d(&pts_2d, [reference[0], reference[1]]);
    }

    // 3D 特化路径 — O(N²)，对齐 Python _compute_3d
    if n_objectives == 3 {
        return hypervolume_3d(&feasible, reference);
    }

    // ≥4D: WFG 包含-排除算法
    hypervolume_wfg(&feasible, reference)
}

/// 3D 超体积特化算法 —— O(N²) 坐标压缩 + 累积最大值。
///
/// 对应 Python `optuna._hypervolume.wfg._compute_3d()`。
///
/// ## 算法原理
/// 1. 按 X 维升序排列
/// 2. 对 Y 维独立排序得到 y_order
/// 3. 构造 N×N 矩阵 z_delta: 在 (x-rank, y-rank) 位置放置 ref_z - z 值
/// 4. 对矩阵做两次累积最大值（行→列），传播每个网格单元的最大 Z 高度
/// 5. 最终体积 = z_delta @ y_delta @ x_delta（矩阵-向量-向量乘积）
fn hypervolume_3d(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let n = points.len();
    if n == 0 {
        return 0.0;
    }

    // 1. 按 X 维（第 0 维）升序排列
    let mut sorted: Vec<&Vec<f64>> = points.iter().collect();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    // 2. 对 Y 维（第 1 维）独立排序，得到排列索引
    let mut y_order: Vec<usize> = (0..n).collect();
    y_order.sort_by(|&a, &b| {
        sorted[a][1]
            .partial_cmp(&sorted[b][1])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. 构造 N×N 矩阵 z_delta
    //    对齐 Python: z_delta[y_order, np.arange(n)] = ref[2] - sorted[y_order, 2]
    //    即 z_delta[y_order[i]][i] = ref[2] - sorted[y_order[i]][2]
    //    y_order[i] 是 y 排名第 i 的点在 x-sorted 数组中的索引 (x_rank)
    //    所以 row = x_rank, col = y_rank_loop_idx, val = 该点的 z 差值
    let mut z_delta = vec![vec![0.0_f64; n]; n];
    for (i, &x_rank_of_point) in y_order.iter().enumerate() {
        // i = y 维循环变量 (y-rank), x_rank_of_point = y_order[i] = 该点的 x 排名
        // 位置: (x_rank_of_point, i), 值: 该点自身的 z 差值
        z_delta[x_rank_of_point][i] = reference[2] - sorted[x_rank_of_point][2];
    }

    // 4a. 累积最大值: 沿行（axis=0，即从上到下）
    for col in 0..n {
        for row in 1..n {
            if z_delta[row - 1][col] > z_delta[row][col] {
                z_delta[row][col] = z_delta[row - 1][col];
            }
        }
    }

    // 4b. 累积最大值: 沿列（axis=1，即从左到右）
    for row in 0..n {
        for col in 1..n {
            if z_delta[row][col - 1] > z_delta[row][col] {
                z_delta[row][col] = z_delta[row][col - 1];
            }
        }
    }

    // 5. 计算 x_delta 和 y_delta 间隔
    let mut x_delta = vec![0.0_f64; n];
    for i in 0..n - 1 {
        x_delta[i] = sorted[i + 1][0] - sorted[i][0];
    }
    x_delta[n - 1] = reference[0] - sorted[n - 1][0];

    let mut y_delta = vec![0.0_f64; n];
    for i in 0..n - 1 {
        y_delta[i] = sorted[y_order[i + 1]][1] - sorted[y_order[i]][1];
    }
    y_delta[n - 1] = reference[1] - sorted[y_order[n - 1]][1];

    // 6. 矩阵-向量-向量乘积: volume = np.dot(np.dot(z_delta, y_delta), x_delta)
    //    对齐 Python: result[j] = sum_i(z_delta[j][i] * y_delta[i])
    //                 volume = sum_j(result[j] * x_delta[j])
    let mut volume = 0.0;
    for j in 0..n {
        let mut row_sum = 0.0;
        for i in 0..n {
            row_sum += z_delta[j][i] * y_delta[i];
        }
        volume += row_sum * x_delta[j];
    }

    volume
}

/// N 维超体积 WFG 算法（≥4D）。
///
/// 对应 Python `optuna._hypervolume.wfg._compute_hv()`。
/// 使用包含-排除原理递归计算，含 Pareto 前沿剪枝加速。
///
/// ## 算法
/// HV({p1,...,pN}) = Σ inclusive_hv(pi) - Σ exclusive_hv(pi)
/// 其中 inclusive_hv = Π(ref[j] - p[j])
/// exclusive_hv 通过限制（limit）其他点到 p 的范围后递归计算
fn hypervolume_wfg(points: &[Vec<f64>], reference: &[f64]) -> f64 {
    let n = points.len();
    let dim = reference.len();

    if n == 0 {
        return 0.0;
    }

    // 基础情况：单个点
    if n == 1 {
        return points[0]
            .iter()
            .zip(reference.iter())
            .map(|(p, r)| (r - p).max(0.0))
            .product();
    }

    // 基础情况：两个点 → S(A) + S(B) - S(A∩B)
    if n == 2 {
        let vol_a: f64 = points[0]
            .iter()
            .zip(reference.iter())
            .map(|(p, r)| (r - p).max(0.0))
            .product();
        let vol_b: f64 = points[1]
            .iter()
            .zip(reference.iter())
            .map(|(p, r)| (r - p).max(0.0))
            .product();
        // 交集体积
        let vol_ab: f64 = (0..dim)
            .map(|j| (reference[j] - points[0][j].max(points[1][j])).max(0.0))
            .product();
        return vol_a + vol_b - vol_ab;
    }

    // 维度降至 2 或 3 时使用特化路径
    if dim == 2 {
        let pts_2d: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
        return hypervolume_2d(&pts_2d, [reference[0], reference[1]]);
    }
    if dim == 3 {
        return hypervolume_3d(points, reference);
    }

    // 按第 0 维排序（升序）
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    // WFG 包含-排除递归
    let mut volume = 0.0;
    for i in 0..n {
        // inclusive_hv(p_i) = Π(ref[j] - p_i[j])
        let inclusive: f64 = sorted[i]
            .iter()
            .zip(reference.iter())
            .map(|(p, r)| (r - p).max(0.0))
            .product();

        // exclusive_hv: 限制 i+1..n 中的点到 p_i 的范围，递归计算
        let exclusive = compute_exclusive_hv(&sorted, i, reference);

        volume += inclusive - exclusive;
    }

    volume
}

/// 计算点 i 的排他超体积。
///
/// 对应 Python `_compute_exclusive_hv()`。
/// 将 i+1..n 中的点限制（limit）到点 i 的范围，然后递归计算 WFG。
fn compute_exclusive_hv(sorted_points: &[Vec<f64>], idx: usize, reference: &[f64]) -> f64 {
    let n = sorted_points.len();
    let dim = reference.len();

    if idx >= n - 1 {
        return 0.0; // 最后一个点没有排他体积
    }

    let p = &sorted_points[idx];

    // 限制后续点: limited[j] = max(q[j], p[j])
    let mut limited_points: Vec<Vec<f64>> = Vec::new();
    for k in (idx + 1)..n {
        let limited: Vec<f64> = (0..dim)
            .map(|j| sorted_points[k][j].max(p[j]))
            .collect();
        // 过滤: 如果限制后的点在所有维度上都小于参考点，才保留
        if limited.iter().zip(reference.iter()).all(|(l, r)| *l < *r) {
            limited_points.push(limited);
        }
    }

    if limited_points.is_empty() {
        return 0.0;
    }

    // Pareto 剪枝: 只保留非支配点（对齐 Python 的 _is_pareto_front 调用）
    let limited_points = pareto_filter_minimize(&limited_points);

    // 递归调用 WFG
    hypervolume_wfg(&limited_points, reference)
}

/// 过滤只保留 Pareto 前沿点（所有目标最小化方向）。
///
/// 对应 Python `_is_pareto_front()` 用于超体积计算中的剪枝。
fn pareto_filter_minimize(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    if n <= 1 {
        return points.to_vec();
    }

    let mut is_pareto = vec![true; n];
    for i in 0..n {
        if !is_pareto[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !is_pareto[j] {
                continue;
            }
            // j 在所有维度上 <= i，且至少一个 < i → i 被支配
            let all_le = points[j].iter().zip(points[i].iter()).all(|(a, b)| a <= b);
            let any_lt = points[j].iter().zip(points[i].iter()).any(|(a, b)| a < b);
            if all_le && any_lt {
                is_pareto[i] = false;
                break;
            }
        }
    }

    points
        .iter()
        .zip(is_pareto.iter())
        .filter(|(_, p)| **p)
        .map(|(pt, _)| pt.clone())
        .collect()
}

// ── HSSP: 超体积子集选择问题 ──

/// 解决超体积子集选择问题 (HSSP) 的 2D 特化版本。
///
/// 对应 Python `optuna._hypervolume.hssp._solve_hssp_2d()`。
/// 贪心策略: 每次选择对超体积贡献最大的点。
///
/// # 参数
/// * `loss_vals` - 损失值矩阵 (N × 2)
/// * `indices`   - 对应的全局索引
/// * `subset_size` - 选择数量
/// * `ref_point` - 参考点 [r0, r1]
///
/// # 返回
/// 选中的全局索引
fn solve_hssp_2d(
    loss_vals: &[[f64; 2]],
    indices: &[usize],
    subset_size: usize,
    ref_point: [f64; 2],
) -> Vec<usize> {
    let n = loss_vals.len();
    assert!(subset_size <= n);

    // 工作副本：按 dim0 排序后的索引 + 损失值 + 矩形对角线
    let mut sorted_idx: Vec<usize> = (0..n).collect();
    sorted_idx.sort_by(|&a, &b| {
        loss_vals[a][0].partial_cmp(&loss_vals[b][0]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_vals: Vec<[f64; 2]> = sorted_idx.iter().map(|&i| loss_vals[i]).collect();
    let mut rect_diags: Vec<[f64; 2]> = vec![ref_point; n];
    let mut local_idx: Vec<usize> = sorted_idx.clone();
    let mut selected = Vec::with_capacity(subset_size);

    for _ in 0..subset_size {
        // 计算每个点的超体积贡献
        // 使用 fold 模拟 np.argmax 的行为：相等时取第一个（即 >，而非 >=）。
        let (max_pos, _) = sorted_vals
            .iter()
            .zip(rect_diags.iter())
            .enumerate()
            .map(|(i, (v, r))| (i, (r[0] - v[0]) * (r[1] - v[1])))
            .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            });

        selected.push(indices[local_idx[max_pos]]);
        let chosen_val = sorted_vals[max_pos];

        // 移除选中点并更新矩形对角线
        let remaining = sorted_vals.len();
        // 更新 max_pos 左侧点的 dim0 对角线
        for i in 0..max_pos {
            rect_diags[i][0] = rect_diags[i][0].min(chosen_val[0]);
        }
        // 更新 max_pos 右侧点的 dim1 对角线
        for i in (max_pos + 1)..remaining {
            rect_diags[i][1] = rect_diags[i][1].min(chosen_val[1]);
        }

        // 删除选中点
        sorted_vals.remove(max_pos);
        rect_diags.remove(max_pos);
        local_idx.remove(max_pos);
    }

    selected
}

/// 解决通用 HSSP（超体积子集选择问题）。
///
/// 对应 Python `optuna._hypervolume.hssp._solve_hssp()`。
/// 使用贪心算法，保证 (1 - 1/e) 近似比。
///
/// # 参数
/// * `loss_vals` - 损失值矩阵 (N × M)
/// * `indices`   - 对应的全局索引
/// * `subset_size` - 选择数量
/// * `ref_point` - 参考点
///
/// # 返回
/// 选中的全局索引
pub fn solve_hssp(
    loss_vals: &[Vec<f64>],
    indices: &[usize],
    subset_size: usize,
    ref_point: &[f64],
) -> Vec<usize> {
    let n = indices.len();
    if subset_size >= n {
        return indices.to_vec();
    }

    // 处理重复点: 先去重
    let n_obj = ref_point.len();
    let mut unique_map: Vec<(Vec<f64>, usize)> = Vec::new();
    for (i, vals) in loss_vals.iter().enumerate() {
        if !unique_map.iter().any(|(v, _)| v == vals) {
            unique_map.push((vals.clone(), i));
        }
    }
    let n_unique = unique_map.len();

    if n_unique < subset_size {
        // 唯一点不够，补充重复点
        let mut chosen = vec![false; n];
        for (_, orig_idx) in &unique_map {
            chosen[*orig_idx] = true;
        }
        let mut result: Vec<usize> = unique_map.iter().map(|(_, i)| indices[*i]).collect();
        for i in 0..n {
            if result.len() >= subset_size {
                break;
            }
            if !chosen[i] {
                result.push(indices[i]);
                chosen[i] = true;
            }
        }
        return result;
    }

    // 参考点含 infinity 的特殊情况
    if !ref_point.iter().all(|r| r.is_finite()) {
        return indices[..subset_size].to_vec();
    }

    // 2D 特化
    if n_obj == 2 {
        let vals_2d: Vec<[f64; 2]> = unique_map.iter().map(|(v, _)| [v[0], v[1]]).collect();
        let unique_indices: Vec<usize> = unique_map.iter().map(|(_, i)| *i).collect();
        let selected = solve_hssp_2d(&vals_2d, &unique_indices, subset_size, [ref_point[0], ref_point[1]]);
        return selected.iter().map(|&i| indices[i]).collect();
    }

    // 通用贪心算法
    let unique_vals: Vec<Vec<f64>> = unique_map.iter().map(|(v, _)| v.clone()).collect();
    let unique_idx: Vec<usize> = unique_map.iter().map(|(_, i)| *i).collect();

    // 初始贡献 = 包含超体积 (每个点独立的 HV)
    let mut contribs: Vec<f64> = unique_vals
        .iter()
        .map(|v| {
            v.iter()
                .zip(ref_point.iter())
                .map(|(vi, ri)| (ri - vi).max(0.0))
                .product()
        })
        .collect();

    let mut selected_indices = Vec::with_capacity(subset_size);
    let mut selected_vecs: Vec<Vec<f64>> = Vec::with_capacity(subset_size);
    let mut active: Vec<bool> = vec![true; n_unique];

    for _ in 0..subset_size {
        // 选择贡献最大的点（对齐 Python np.argmax: 相等时取第一个）
        let best = (0..n_unique)
            .filter(|&i| active[i])
            .fold(None::<(usize, f64)>, |acc, i| {
                let v = contribs[i];
                match acc {
                    None => Some((i, v)),
                    Some((_, bv)) if v > bv => Some((i, v)),
                    _ => acc,
                }
            })
            .unwrap()
            .0;

        selected_indices.push(unique_idx[best]);
        selected_vecs.push(unique_vals[best].clone());
        active[best] = false;

        // 使用子模性质更新贡献上界
        if selected_vecs.len() < subset_size {
            for i in 0..n_unique {
                if !active[i] {
                    continue;
                }
                // 简单上界: inclusive_hv - (已选中点的交集体积)
                let inclusive_hv: f64 = unique_vals[i]
                    .iter()
                    .zip(ref_point.iter())
                    .map(|(vi, ri)| (ri - vi).max(0.0))
                    .product();

                // 与最新选中点的交集上界
                let intersect_val: f64 = unique_vals[i]
                    .iter()
                    .zip(selected_vecs.last().unwrap().iter())
                    .zip(ref_point.iter())
                    .map(|((vi, si), ri)| (ri - vi.max(*si)).max(0.0))
                    .product();

                let new_bound = inclusive_hv - intersect_val;
                contribs[i] = contribs[i].min(new_bound);
            }
        }
    }

    selected_indices.iter().map(|&i| indices[i]).collect()
}

/// 计算非支配空间的盒子分解边界。
///
/// 精确对齐 Python `optuna._hypervolume.box_decomposition.get_non_dominated_box_bounds()`。
/// 使用 Lacour17 算法 (Algorithm 2) 对任意维度进行正确的非重叠盒子分解。
///
/// # 参数
/// * `loss_vals` - Pareto 前沿点的损失值 (N × M)
/// * `ref_point` - 参考点
///
/// # 返回
/// `(lower_bounds, upper_bounds)` — 各盒子的下界和上界
pub fn get_non_dominated_box_bounds(
    loss_vals: &[Vec<f64>],
    ref_point: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = loss_vals.len();
    let m = ref_point.len();

    if n == 0 || m == 0 {
        return (vec![], vec![]);
    }

    // 去重并按字典序排序（对齐 Python np.unique）
    let mut unique_sorted = loss_vals.to_vec();
    unique_sorted.sort_by(|a, b| {
        for d in 0..m {
            match a[d].partial_cmp(&b[d]) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => return std::cmp::Ordering::Equal,
            }
        }
        std::cmp::Ordering::Equal
    });
    unique_sorted.dedup();

    // 取 Pareto 前沿
    let pareto_mask = is_pareto_front_vals(&unique_sorted);
    let sorted_pareto_sols: Vec<Vec<f64>> = unique_sorted
        .into_iter()
        .zip(pareto_mask.iter())
        .filter(|&(_, on)| *on)
        .map(|(v, _)| v)
        .collect();

    lacour17_non_dominated_box_bounds(&sorted_pareto_sols, ref_point)
}

/// Pareto 前沿检测（最小化方向，纯数值版本）。
fn is_pareto_front_vals(vals: &[Vec<f64>]) -> Vec<bool> {
    let n = vals.len();
    let mut on_front = vec![true; n];
    for i in 0..n {
        if !on_front[i] { continue; }
        for j in 0..n {
            if i == j || !on_front[j] { continue; }
            let mut all_le = true;
            let mut any_lt = false;
            for d in 0..vals[i].len() {
                if vals[j][d] > vals[i][d] { all_le = false; break; }
                if vals[j][d] < vals[i][d] { any_lt = true; }
            }
            if all_le && any_lt { on_front[i] = false; break; }
        }
    }
    on_front
}

/// Lacour17 Algorithm 2: 计算上界集 U(N) 及其定义点。
///
/// 对齐 Python `_get_upper_bound_set()`。
/// `sorted_pareto_sols` 必须按第一个目标排好序。
///
/// 返回 (upper_bound_set, def_points)：
///   upper_bound_set: Vec<Vec<f64>> (n_bounds × m)
///   def_points: Vec<Vec<Vec<f64>>> (n_bounds × m × m)
fn lacour17_upper_bound_set(
    sorted_pareto_sols: &[Vec<f64>],
    ref_point: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
    let m = ref_point.len();

    // skip_ineq_judge[j][k] = true iff (j==k || k==0)
    // 对齐 Python: skip = eye(m) | col_0_all_true
    let mut skip = vec![vec![false; m]; m];
    for j in 0..m {
        skip[j][j] = true; // diagonal
        skip[j][0] = true; // column 0
    }

    // 初始化: ubs = [ref_point], dps = full(-inf) with diagonal = ref_point
    let mut ubs = vec![ref_point.to_vec()];
    let mut dps = vec![vec![vec![f64::NEG_INFINITY; m]; m]];
    for j in 0..m {
        dps[0][j][j] = ref_point[j];
    }

    for sol in sorted_pareto_sols {
        // is_dominated[i] = all(sol[d] < ubs[i][d], for d in 0..m)
        let n_ubs = ubs.len();
        let is_dominated: Vec<bool> = (0..n_ubs)
            .map(|i| (0..m).all(|d| sol[d] < ubs[i][d]))
            .collect();

        if !is_dominated.iter().any(|&x| x) {
            continue;
        }

        let mut new_ubs = Vec::new();
        let mut new_dps = Vec::new();

        // 保留未被支配的 ubs
        for i in 0..n_ubs {
            if !is_dominated[i] {
                new_ubs.push(ubs[i].clone());
                new_dps.push(dps[i].clone());
            }
        }

        // 对每个被支配的 ubs[i]，更新维度
        for i in 0..n_ubs {
            if !is_dominated[i] { continue; }

            // update[j] = sol[j] >= max_{k: !skip[j][k]} dps[i][k][j]
            for j in 0..m {
                let max_non_skip = (0..m)
                    .filter(|&k| !skip[j][k])
                    .map(|k| dps[i][k][j])
                    .fold(f64::NEG_INFINITY, f64::max);

                if sol[j] >= max_non_skip {
                    // 生成新的 ubs 和 dps
                    let mut new_u = ubs[i].clone();
                    new_u[j] = sol[j];

                    let mut new_d = dps[i].clone();
                    new_d[j] = sol.clone();

                    new_ubs.push(new_u);
                    new_dps.push(new_d);
                }
            }
        }

        ubs = new_ubs;
        dps = new_dps;
    }

    (ubs, dps)
}

/// Lacour17 Eq. (2): 从上界集和定义点计算盒子边界。
///
/// 对齐 Python `_get_box_bounds()`。
/// 返回 (lower_bounds, upper_bounds)。
fn lacour17_box_bounds(
    ubs: &[Vec<f64>],
    dps: &[Vec<Vec<f64>>],
    ref_point: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_bounds = ubs.len();
    let m = ref_point.len();
    if n_bounds == 0 || m < 2 {
        return (vec![], vec![]);
    }

    let mut lower = vec![vec![0.0; m]; n_bounds];
    let mut upper = vec![vec![0.0; m]; n_bounds];

    for i in 0..n_bounds {
        // 维度 0: lower = dps[i][0][0], upper = ref_point[0]
        lower[i][0] = dps[i][0][0];
        upper[i][0] = ref_point[0];

        // 维度 1..m: maximum.accumulate 沿 axis=-2，取 diag 偏移
        // bounds[0, :, j] = max(dps[i][0..j][j]) for j >= 1
        // bounds[1, :, j] = ubs[i][j]
        for j in 1..m {
            // np.maximum.accumulate(dps, axis=-2)[:, j-1, j]
            // = max(dps[i][0][j], dps[i][1][j], ..., dps[i][j-1][j])
            let mut acc_max = f64::NEG_INFINITY;
            for k in 0..j {
                acc_max = acc_max.max(dps[i][k][j]);
            }
            lower[i][j] = acc_max;
            upper[i][j] = ubs[i][j];
        }
    }

    // 移除空盒子（upper[d] <= lower[d] 的任何维度）
    let mut result_lower = Vec::new();
    let mut result_upper = Vec::new();
    for i in 0..n_bounds {
        let empty = (0..m).any(|d| upper[i][d] <= lower[i][d]);
        if !empty {
            result_lower.push(lower[i].clone());
            result_upper.push(upper[i].clone());
        }
    }

    (result_lower, result_upper)
}

/// 完整的 Lacour17 非支配盒子分解。
///
/// 精确对齐 Python `_get_non_dominated_box_bounds()`。
fn lacour17_non_dominated_box_bounds(
    sorted_pareto_sols: &[Vec<f64>],
    ref_point: &[f64],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = ref_point.len();
    if sorted_pareto_sols.is_empty() || m == 0 {
        return (vec![], vec![]);
    }

    // Step 1: 对原始 Pareto 解计算上界集，然后取负
    let (ubs, _) = lacour17_upper_bound_set(sorted_pareto_sols, ref_point);
    let neg_ubs: Vec<Vec<f64>> = ubs.iter()
        .map(|u| u.iter().map(|&v| -v).collect())
        .collect();

    // Step 2: 对取负后的上界集去重排序，并取 Pareto 前沿
    let mut sorted_neg_ubs = neg_ubs;
    sorted_neg_ubs.sort_by(|a, b| {
        for d in 0..m {
            match a[d].partial_cmp(&b[d]) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ord) => return ord,
                None => return std::cmp::Ordering::Equal,
            }
        }
        std::cmp::Ordering::Equal
    });
    sorted_neg_ubs.dedup();

    let pareto_mask = is_pareto_front_vals(&sorted_neg_ubs);
    let neg_pareto: Vec<Vec<f64>> = sorted_neg_ubs
        .into_iter()
        .zip(pareto_mask.iter())
        .filter(|&(_, on)| *on)
        .map(|(v, _)| v)
        .collect();

    // Step 3: 用取负后的 Pareto 点作为输入，以 +∞ 为参考点，再次计算上界集
    let inf_ref: Vec<f64> = vec![f64::INFINITY; m];
    let (neg_lower_set, neg_def_points) = lacour17_upper_bound_set(&neg_pareto, &inf_ref);

    // Step 4: 计算盒子边界并取负翻转
    let (neg_box_lower, neg_box_upper) = lacour17_box_bounds(
        &neg_lower_set, &neg_def_points, &inf_ref,
    );

    // 取负翻转: box_upper = -neg_box_lower, box_lower = -neg_box_upper
    let box_lower: Vec<Vec<f64>> = neg_box_upper.iter()
        .map(|v| v.iter().map(|&x| -x).collect())
        .collect();
    let box_upper: Vec<Vec<f64>> = neg_box_lower.iter()
        .map(|v| v.iter().map(|&x| -x).collect())
        .collect();

    (box_lower, box_upper)
}

/// Get the Pareto-optimal trials from a list of complete trials.
///
/// 对齐 Python `get_pareto_front_trials(study, consider_constraint=False)`。
/// 当 `consider_constraint=true` 时，只考虑可行解（约束值全 ≤ 0）。
pub fn get_pareto_front_trials(
    trials: &[FrozenTrial],
    directions: &[StudyDirection],
) -> Vec<FrozenTrial> {
    get_pareto_front_trials_with_constraint(trials, directions, false)
}

/// 带约束考虑的 Pareto 前沿获取。
///
/// 对齐 Python `get_pareto_front_trials(study, consider_constraint=True)`。
pub fn get_pareto_front_trials_with_constraint(
    trials: &[FrozenTrial],
    directions: &[StudyDirection],
    consider_constraint: bool,
) -> Vec<FrozenTrial> {
    let complete: Vec<&FrozenTrial> = trials
        .iter()
        .filter(|t| {
            if t.state != TrialState::Complete || t.values.is_none() {
                return false;
            }
            if consider_constraint {
                // 对齐 Python: 只保留可行解（约束值全 ≤ 0）
                is_feasible(t)
            } else {
                true
            }
        })
        .collect();

    if complete.is_empty() {
        return vec![];
    }

    let on_front = is_pareto_front(&complete, directions);
    complete
        .into_iter()
        .zip(on_front)
        .filter(|(_, is_front)| *is_front)
        .map(|(t, _)| t.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialState;
    use std::collections::HashMap;

    fn make_trial(number: i64, values: Vec<f64>) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(values),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_dominates_minimize() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        // Not dominating if equal
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
        // Not dominating if better in one but worse in other
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
    }

    #[test]
    fn test_dominates_maximize() {
        let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
        assert!(dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        assert!(!dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
    }

    #[test]
    fn test_dominates_mixed() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Maximize];
        // a=[1, 3] vs b=[2, 2]: a is better (lower) in obj0, better (higher) in obj1
        assert!(dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0], &dirs));
    }

    #[test]
    fn test_dominates_notset_direction_no_panic_and_false() {
        let dirs = vec![StudyDirection::NotSet, StudyDirection::Minimize];
        assert!(!dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
    }

    #[test]
    fn test_fast_non_dominated_sort_simple() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![3.0, 2.0]);
        let t3 = make_trial(3, vec![4.0, 4.0]); // dominated by all front-0

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
        let fronts = fast_non_dominated_sort(&trials, &dirs);

        assert_eq!(fronts.len(), 2);
        // Front 0 should have t0, t1, t2
        let mut front0 = fronts[0].clone();
        front0.sort();
        assert_eq!(front0, vec![0, 1, 2]);
        // Front 1 should have t3
        assert_eq!(fronts[1], vec![3]);
    }

    #[test]
    fn test_fast_non_dominated_sort_empty() {
        let dirs = vec![StudyDirection::Minimize];
        let fronts = fast_non_dominated_sort(&[], &dirs);
        assert!(fronts.is_empty());
    }

    #[test]
    fn test_crowding_distance_basic() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![3.0, 2.0]);
        let t3 = make_trial(3, vec![4.0, 1.0]);

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
        let dists = crowding_distance(&trials, &dirs);

        // Boundary solutions (min/max in each objective) get infinity
        assert!(dists[0].is_infinite());
        assert!(dists[3].is_infinite());
        // Interior solutions get finite positive distances
        assert!(dists[1] > 0.0 && dists[1].is_finite());
        assert!(dists[2] > 0.0 && dists[2].is_finite());
    }

    #[test]
    fn test_crowding_distance_two_points() {
        let dirs = vec![StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0]);
        let t1 = make_trial(1, vec![2.0]);
        let trials: Vec<&FrozenTrial> = vec![&t0, &t1];
        let dists = crowding_distance(&trials, &dirs);
        assert!(dists[0].is_infinite());
        assert!(dists[1].is_infinite());
    }

    #[test]
    fn test_is_pareto_front() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![5.0, 5.0]); // dominated

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2];
        let front = is_pareto_front(&trials, &dirs);
        assert_eq!(front, vec![true, true, false]);
    }

    #[test]
    fn test_hypervolume_2d_simple() {
        // Simple case: one point at (1,1), reference at (3,3)
        let vol = hypervolume_2d(&[[1.0, 1.0]], [3.0, 3.0]);
        assert!((vol - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        let vol = hypervolume_2d(&[[1.0, 3.0], [3.0, 1.0]], [4.0, 4.0]);
        // Area: (4-1)*(4-3) + (4-3)*(3-1) = 3 + 2 = 5
        assert!((vol - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_empty() {
        assert_eq!(hypervolume_2d(&[], [1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_hypervolume_2d_dominated_by_reference() {
        // Point is worse than reference
        let vol = hypervolume_2d(&[[5.0, 5.0]], [3.0, 3.0]);
        assert_eq!(vol, 0.0);
    }

    // ── 3D hypervolume tests (O(N²) _compute_3d path) ──

    #[test]
    fn test_hypervolume_3d_single_point() {
        // (1,1,1) ref (3,3,3) → 2*2*2 = 8
        let vol = hypervolume(
            &[vec![1.0, 1.0, 1.0]],
            &[3.0, 3.0, 3.0],
        );
        assert!((vol - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_3d_two_points() {
        // Python: 10.0
        let vol = hypervolume(
            &[vec![1.0, 3.0, 2.0], vec![2.0, 1.0, 3.0]],
            &[4.0, 4.0, 4.0],
        );
        assert!((vol - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_3d_three_points() {
        // Python: 46.0
        let vol = hypervolume(
            &[
                vec![1.0, 2.0, 3.0],
                vec![2.0, 1.0, 2.0],
                vec![3.0, 3.0, 1.0],
            ],
            &[5.0, 5.0, 5.0],
        );
        assert!((vol - 46.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_3d_empty() {
        let vol = hypervolume(&[], &[5.0, 5.0, 5.0]);
        assert_eq!(vol, 0.0);
    }

    // ── 4D+ hypervolume tests (WFG path) ──

    #[test]
    fn test_hypervolume_4d_two_points() {
        // Python: 81.0
        let vol = hypervolume(
            &[vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]],
            &[4.0, 4.0, 4.0, 4.0],
        );
        assert!((vol - 81.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_4d_three_points() {
        // Python: 160.0
        let vol = hypervolume(
            &[
                vec![1.0, 2.0, 3.0, 1.0],
                vec![2.0, 1.0, 1.0, 3.0],
                vec![3.0, 3.0, 2.0, 2.0],
            ],
            &[5.0, 5.0, 5.0, 5.0],
        );
        assert!((vol - 160.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_pareto_front_trials() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![5.0, 5.0]);
        let trials = vec![t0, t1, t2];

        let front = get_pareto_front_trials(&trials, &dirs);
        assert_eq!(front.len(), 2);
        let numbers: Vec<i64> = front.iter().map(|t| t.number).collect();
        assert!(numbers.contains(&0));
        assert!(numbers.contains(&1));
    }

    // ── 约束支持测试 ──

    fn make_constrained_trial(number: i64, values: Vec<f64>, constraints: Vec<f64>) -> FrozenTrial {
        let mut t = make_trial(number, values);
        t.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!(constraints),
        );
        t
    }

    #[test]
    fn test_is_feasible() {
        // 约束 <= 0 → 可行
        let t1 = make_constrained_trial(0, vec![1.0], vec![-1.0, -0.5]);
        assert!(is_feasible(&t1));

        // 有违反 → 不可行
        let t2 = make_constrained_trial(1, vec![1.0], vec![-1.0, 0.5]);
        assert!(!is_feasible(&t2));

        // 无约束值 → 不可行
        let t3 = make_trial(2, vec![1.0]);
        assert!(!is_feasible(&t3));
    }

    #[test]
    fn test_constraint_violation() {
        let t1 = make_constrained_trial(0, vec![1.0], vec![-1.0, 0.5, 0.3]);
        assert!((constraint_violation(&t1) - 0.8).abs() < 1e-10); // max(0,-1)+max(0,0.5)+max(0,0.3)

        let t2 = make_constrained_trial(1, vec![1.0], vec![-1.0, -2.0]);
        assert!((constraint_violation(&t2) - 0.0).abs() < 1e-10);

        let t3 = make_trial(2, vec![1.0]);
        assert!(constraint_violation(&t3).is_nan());
    }

    #[test]
    fn test_constrained_dominates_feasible_vs_infeasible() {
        let dirs = vec![StudyDirection::Minimize];
        // a 可行，b 不可行 → a 支配
        let a = make_constrained_trial(0, vec![10.0], vec![-1.0]);
        let b = make_constrained_trial(1, vec![1.0], vec![0.5]);
        assert!(constrained_dominates(&a, &b, &dirs));
        assert!(!constrained_dominates(&b, &a, &dirs));
    }

    #[test]
    fn test_constrained_dominates_both_infeasible() {
        let dirs = vec![StudyDirection::Minimize];
        // 都不可行 → 比较约束违反量
        let a = make_constrained_trial(0, vec![1.0], vec![0.1]); // 违反 0.1
        let b = make_constrained_trial(1, vec![1.0], vec![0.5]); // 违反 0.5
        assert!(constrained_dominates(&a, &b, &dirs));
        assert!(!constrained_dominates(&b, &a, &dirs));
    }

    #[test]
    fn test_constrained_dominates_both_feasible() {
        let dirs = vec![StudyDirection::Minimize];
        // 都可行 → 普通 Pareto 支配
        let a = make_constrained_trial(0, vec![1.0], vec![-1.0]);
        let b = make_constrained_trial(1, vec![2.0], vec![-0.5]);
        assert!(constrained_dominates(&a, &b, &dirs));
        assert!(!constrained_dominates(&b, &a, &dirs));
    }

    #[test]
    fn test_get_feasible_trials() {
        let t1 = make_constrained_trial(0, vec![1.0], vec![-1.0]);
        let t2 = make_constrained_trial(1, vec![2.0], vec![0.5]);
        let t3 = make_constrained_trial(2, vec![3.0], vec![-0.1, -0.2]);
        let trials = vec![t1, t2, t3];
        let feasible = get_feasible_trials(&trials);
        assert_eq!(feasible.len(), 2);
        assert_eq!(feasible[0].number, 0);
        assert_eq!(feasible[1].number, 2);
    }

    #[test]
    fn test_constrained_non_dominated_sort() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0]); // 可行
        let t1 = make_constrained_trial(1, vec![0.5, 0.5], vec![0.5]); // 不可行
        let t2 = make_constrained_trial(2, vec![2.0, 3.0], vec![-0.5]); // 可行
        let refs: Vec<&FrozenTrial> = vec![&t0, &t1, &t2];
        let fronts = constrained_fast_non_dominated_sort(&refs, &dirs);
        // 可行解 (t0, t2) 应在第一个前沿，不可行 t1 在第二个
        assert!(fronts.len() >= 2);
        assert!(fronts[0].contains(&0) && fronts[0].contains(&2));
    }

    #[test]
    fn test_constrained_dominates_when_only_one_has_constraints() {
        let dirs = vec![StudyDirection::Minimize];
        let mut a = make_trial(0, vec![1.0]);
        a.system_attrs
            .insert(CONSTRAINTS_KEY.to_string(), serde_json::json!([-1.0]));
        let b = make_trial(1, vec![0.5]);

        assert!(constrained_dominates(&a, &b, &dirs));
        assert!(!constrained_dominates(&b, &a, &dirs));
    }

    #[test]
    fn test_constrained_dominates_state_gating() {
        let dirs = vec![StudyDirection::Minimize];
        let mut a = make_constrained_trial(0, vec![1.0], vec![-1.0]);
        let b = make_constrained_trial(1, vec![2.0], vec![-1.0]);

        // a 非完成态时不支配
        a.state = TrialState::Running;
        assert!(!constrained_dominates(&a, &b, &dirs));

        // b 非完成态时 a 直接支配
        let mut b2 = b.clone();
        b2.state = TrialState::Running;
        let a2 = make_constrained_trial(2, vec![3.0], vec![-1.0]);
        assert!(constrained_dominates(&a2, &b2, &dirs));
    }

    #[test]
    fn test_get_pareto_front_trials_filters_incomplete_and_none_values() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 1.0]);
        let mut t1 = make_trial(1, vec![2.0, 2.0]);
        let mut t2 = make_trial(2, vec![0.5, 3.0]);

        // 应被过滤掉: 非 COMPLETE
        t1.state = TrialState::Running;
        // 应被过滤掉: values=None
        t2.values = None;

        let trials = vec![t0.clone(), t1, t2];
        let front = get_pareto_front_trials(&trials, &dirs);
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].number, t0.number);
    }

    // ── HSSP 测试 ──

    #[test]
    fn test_solve_hssp_basic_2d() {
        // 3 个点，选择 2 个，2D 情况
        let vals = vec![
            vec![1.0, 5.0],
            vec![3.0, 3.0],
            vec![5.0, 1.0],
        ];
        let indices: Vec<usize> = vec![0, 1, 2];
        let ref_point = vec![10.0, 10.0];

        let selected = solve_hssp(&vals, &indices, 2, &ref_point);
        assert_eq!(selected.len(), 2);
        // 贪心应选择超体积贡献最大的两个
    }

    #[test]
    fn test_solve_hssp_all_selected() {
        // subset_size >= N，全部返回
        let vals = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let indices: Vec<usize> = vec![0, 1];
        let ref_point = vec![10.0, 10.0];
        let selected = solve_hssp(&vals, &indices, 5, &ref_point);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_solve_hssp_3d() {
        // 3D HSSP
        let vals = vec![
            vec![1.0, 5.0, 3.0],
            vec![3.0, 1.0, 5.0],
            vec![5.0, 3.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ];
        let indices: Vec<usize> = vec![0, 1, 2, 3];
        let ref_point = vec![10.0, 10.0, 10.0];
        let selected = solve_hssp(&vals, &indices, 2, &ref_point);
        assert_eq!(selected.len(), 2);
    }

    // ── Box Decomposition 测试 ──

    #[test]
    fn test_box_decomp_2d() {
        let vals = vec![
            vec![1.0, 4.0],
            vec![3.0, 2.0],
        ];
        let ref_point = vec![5.0, 5.0];
        let (lower, upper) = get_non_dominated_box_bounds(&vals, &ref_point);
        // 2D: 应产生 3 个盒子（n+1）
        assert_eq!(lower.len(), 3);
        assert_eq!(upper.len(), 3);
    }

    #[test]
    fn test_box_decomp_empty() {
        let vals: Vec<Vec<f64>> = vec![];
        let ref_point = vec![5.0, 5.0];
        let (lower, upper) = get_non_dominated_box_bounds(&vals, &ref_point);
        assert!(lower.is_empty());
        assert!(upper.is_empty());
    }

    #[test]
    fn test_box_decomp_3d() {
        let vals = vec![
            vec![1.0, 3.0, 2.0],
            vec![2.0, 1.0, 3.0],
        ];
        let ref_point = vec![5.0, 5.0, 5.0];
        let (lower, upper) = get_non_dominated_box_bounds(&vals, &ref_point);
        assert!(!lower.is_empty());
        assert_eq!(lower.len(), upper.len());
    }

    // ========================================================================
    // Python 交叉验证测试
    // ========================================================================

    /// Python 交叉验证: compute_hypervolume([[1,4],[2,3],[3,2],[4,1]], [5,5]) = 10.0
    #[test]
    fn test_python_cross_hypervolume_2d() {
        let pts = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let ref_point = vec![5.0, 5.0];
        let vol = hypervolume(&pts, &ref_point);
        assert!((vol - 10.0).abs() < 1e-8, "Python: hypervolume=10.0, got {vol}");
    }

    /// Python 交叉验证: NotSet 方向不 panic（已修复为返回 false）
    #[test]
    fn test_python_cross_dominates_notset() {
        let dirs = vec![StudyDirection::NotSet, StudyDirection::Minimize];
        assert!(!dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
    }

    /// Python 交叉验证: 3D 超体积
    /// 点 [1,1,1] 支配体积 = (3-1)^3 = 8; [2,2,2] 完全被包含，总超体积 = 8
    #[test]
    fn test_python_cross_hypervolume_3d() {
        let pts = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv = hypervolume(&pts, &ref_point);
        assert!((hv - 8.0).abs() < 1e-12, "Python: hv=8.0, got={hv}");
    }

    /// Python 交叉验证: 3D 超体积 — 4 个非支配点
    #[test]
    fn test_python_cross_hypervolume_3d_four_points() {
        let pts = vec![
            vec![1.0, 4.0, 3.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 1.0, 4.0],
            vec![4.0, 3.0, 1.0],
        ];
        let ref_point = vec![5.0, 5.0, 5.0];
        let hv = hypervolume(&pts, &ref_point);
        assert!((hv - 33.0).abs() < 1e-10, "Python: hv=33.0, got={hv}");
    }

    /// Python 交叉验证: 3D 超体积 — 含被支配的点
    #[test]
    fn test_python_cross_hypervolume_3d_with_dominated() {
        let pts = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 0.5],
        ];
        let ref_point = vec![4.0, 4.0, 4.0];
        let hv = hypervolume(&pts, &ref_point);
        assert!((hv - 27.5).abs() < 1e-10, "Python: hv=27.5, got={hv}");
    }

    /// Python 交叉验证: 3D 超体积 — 单点
    #[test]
    fn test_python_cross_hypervolume_3d_single() {
        let pts = vec![vec![2.0, 3.0, 1.0]];
        let ref_point = vec![5.0, 5.0, 5.0];
        let hv = hypervolume(&pts, &ref_point);
        assert!((hv - 24.0).abs() < 1e-10, "Python: hv=24.0, got={hv}");
    }

    /// Python 交叉验证: 单点 2D 超体积
    /// pts=[[2,3]], ref=[5,5] → hv = (5-2)*(5-3) = 6
    #[test]
    fn test_python_cross_hypervolume_single() {
        let pts = vec![vec![2.0, 3.0]];
        let ref_point = vec![5.0, 5.0];
        let hv = hypervolume(&pts, &ref_point);
        assert!((hv - 6.0).abs() < 1e-12, "Python: hv=6.0, got={hv}");
    }

    /// 对齐 Python: get_pareto_front_trials_with_constraint(consider_constraint=true)
    /// 只保留可行解（约束值全 ≤ 0）
    #[test]
    fn test_pareto_front_with_constraint() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];

        // 可行试验: 约束 [-1.0]（满足 ≤ 0）
        let mut t1 = make_trial(0, vec![1.0, 2.0]);
        t1.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-1.0]),
        );

        // 不可行试验: 约束 [1.0]（违反 > 0）— 即使 Pareto 更优也应被排除
        let mut t2 = make_trial(1, vec![0.5, 0.5]);
        t2.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!([1.0]),
        );

        // 可行试验: 约束 [0.0]（等于 0，满足 ≤ 0）
        let mut t3 = make_trial(2, vec![1.5, 1.5]);
        t3.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!([0.0]),
        );

        let trials = vec![t1, t2, t3];

        // consider_constraint=false → 包含不可行的 t2
        let front_no_constraint = get_pareto_front_trials(&trials, &dirs);
        assert!(front_no_constraint.iter().any(|t| t.number == 1),
            "无约束时 t2 应在 Pareto 前沿");

        // consider_constraint=true → 排除 t2
        let front_with_constraint = get_pareto_front_trials_with_constraint(
            &trials, &dirs, true
        );
        assert!(!front_with_constraint.iter().any(|t| t.number == 1),
            "有约束时 t2 应被排除");
        assert!(front_with_constraint.iter().any(|t| t.number == 0),
            "可行的 t1 应在前沿");
    }

    /// 对齐 Python: is_feasible 语义
    #[test]
    fn test_is_feasible_semantics() {
        let mut t = make_trial(0, vec![1.0]);
        // 无约束信息 → 不可行
        assert!(!is_feasible(&t), "无约束信息应视为不可行");

        // 约束全 ≤ 0 → 可行
        t.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-1.0, 0.0]),
        );
        assert!(is_feasible(&t), "约束全 ≤ 0 应视为可行");

        // 约束有 > 0 → 不可行
        t.system_attrs.insert(
            CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-1.0, 0.1]),
        );
        assert!(!is_feasible(&t), "约束有 > 0 应视为不可行");
    }
}
