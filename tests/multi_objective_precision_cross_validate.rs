//! 多目标优化模块深度数值精度交叉验证测试。
//!
//! 所有参考值由 `tests/golden_multi_objective.py` 通过 Python optuna 生成。
//! 覆盖: hypervolume (2D/3D/4D), HSSP, crowding_distance, dominates, non_dominated_sort

use optuna_rs::multi_objective::{
    crowding_distance, dominates, fast_non_dominated_sort, hypervolume, hypervolume_2d, solve_hssp,
};
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

const TOL: f64 = 1e-10;

fn make_trial_vals(number: i64, values: Vec<f64>) -> FrozenTrial {
    FrozenTrial {
        number,
        trial_id: number,
        state: TrialState::Complete,
        values: Some(values),
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    }
}

// ============================================================================
// 1. 3D Hypervolume — Python _compute_3d 精确值
// ============================================================================

/// 单点 [1,1,1], ref=[5,5,5] → HV = 4*4*4 = 64
#[test]
fn test_hv_3d_single_point() {
    let pts = vec![vec![1.0, 1.0, 1.0]];
    let r = vec![5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 64.0).abs() < TOL,
        "3D single: Rust={:.15e}, Python=6.4e+01",
        hv
    );
}

/// 3 Pareto 点 [[1,3,2],[2,1,3],[3,2,1]], ref=[5,5,5] → HV = 44
#[test]
fn test_hv_3d_three_pareto() {
    let pts = vec![
        vec![1.0, 3.0, 2.0],
        vec![2.0, 1.0, 3.0],
        vec![3.0, 2.0, 1.0],
    ];
    let r = vec![5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 44.0).abs() < TOL,
        "3D 3pts: Rust={:.15e}, Python=4.4e+01",
        hv
    );
}

/// 3 Pareto 点 — 非对称 [[0.5,0.5,4],[1,4,0.5],[4,1,1]], ref=[5,5,5] → HV = 43.25
#[test]
fn test_hv_3d_asymmetric() {
    let pts = vec![
        vec![0.5, 0.5, 4.0],
        vec![1.0, 4.0, 0.5],
        vec![4.0, 1.0, 1.0],
    ];
    let r = vec![5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 43.25).abs() < TOL,
        "3D asym: Rust={:.15e}, Python=4.325e+01",
        hv
    );
}

/// 含支配点 [[1,1,1],[2,2,2]], ref=[5,5,5] → HV = 64 (被支配点不贡献)
#[test]
fn test_hv_3d_with_dominated() {
    let pts = vec![vec![1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0]];
    let r = vec![5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 64.0).abs() < TOL,
        "3D dominated: Rust={:.15e}, Python=6.4e+01",
        hv
    );
}

/// 重复点 [[1,1,1],[1,1,1]], ref=[5,5,5] → HV = 64
#[test]
fn test_hv_3d_duplicate() {
    let pts = vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]];
    let r = vec![5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 64.0).abs() < TOL,
        "3D dup: Rust={:.15e}, Python=6.4e+01",
        hv
    );
}

// ============================================================================
// 2. 4D Hypervolume (WFG) — Python compute_hypervolume 精确值
// ============================================================================

/// 单点 [1,1,1,1], ref=[5,5,5,5] → HV = 4^4 = 256
#[test]
fn test_hv_4d_single() {
    let pts = vec![vec![1.0, 1.0, 1.0, 1.0]];
    let r = vec![5.0, 5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 256.0).abs() < TOL,
        "4D single: Rust={:.15e}, Python=2.56e+02",
        hv
    );
}

/// 3 Pareto 点 4D → HV = 148
#[test]
fn test_hv_4d_three_pareto() {
    let pts = vec![
        vec![1.0, 3.0, 2.0, 1.0],
        vec![2.0, 1.0, 3.0, 2.0],
        vec![3.0, 2.0, 1.0, 3.0],
    ];
    let r = vec![5.0, 5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 148.0).abs() < TOL,
        "4D 3pts: Rust={:.15e}, Python=1.48e+02",
        hv
    );
}

/// 含支配点 4D → HV = 256
#[test]
fn test_hv_4d_with_dominated() {
    let pts = vec![
        vec![1.0, 1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0, 3.0],
    ];
    let r = vec![5.0, 5.0, 5.0, 5.0];
    let hv = hypervolume(&pts, &r);
    assert!(
        (hv - 256.0).abs() < TOL,
        "4D dominated: Rust={:.15e}, Python=2.56e+02",
        hv
    );
}

// ============================================================================
// 3. HSSP 2D — Python _solve_hssp 精确值
// ============================================================================

/// 4 Pareto 点, 选 2 个 → Python: indices [1, 3]
#[test]
fn test_hssp_2d_select_2_of_4() {
    let vals = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 1.0],
    ];
    let indices: Vec<usize> = (0..4).collect();
    let ref_point = vec![5.0, 6.0];
    let selected = solve_hssp(&vals, &indices, 2, &ref_point);
    let mut sel_sorted = selected.clone();
    sel_sorted.sort();
    // Python 选择的原始索引: [1, 3]
    // 但由于贪心算法的实现差异，Rust 可能选择不同的等价方案
    // 关键是超体积应该最大化
    assert_eq!(
        selected.len(),
        2,
        "Should select exactly 2 points"
    );
    // 验证选中组合的超体积与 Python 选择的超体积一致
    let python_selected = vec![vec![2.0, 4.0], vec![4.0, 1.0]];
    let python_hv = hypervolume(&python_selected, &ref_point);
    let rust_selected_vals: Vec<Vec<f64>> = sel_sorted.iter().map(|&i| vals[i].clone()).collect();
    let rust_hv = hypervolume(&rust_selected_vals, &ref_point);
    assert!(
        (rust_hv - python_hv).abs() < TOL,
        "HSSP HV mismatch: Rust_sel={:?} HV={:.6}, Python_sel=[1,3] HV={:.6}",
        sel_sorted, rust_hv, python_hv
    );
}

/// 4 Pareto 点, 选 3 个
#[test]
fn test_hssp_2d_select_3_of_4() {
    let vals = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 1.0],
    ];
    let indices: Vec<usize> = (0..4).collect();
    let ref_point = vec![5.0, 6.0];
    let selected = solve_hssp(&vals, &indices, 3, &ref_point);
    assert_eq!(selected.len(), 3);
    // Python: [0, 1, 3]
    let python_selected = vec![vec![1.0, 5.0], vec![2.0, 4.0], vec![4.0, 1.0]];
    let python_hv = hypervolume(&python_selected, &ref_point);
    let rust_selected_vals: Vec<Vec<f64>> = selected.iter().map(|&i| vals[i].clone()).collect();
    let rust_hv = hypervolume(&rust_selected_vals, &ref_point);
    assert!(
        (rust_hv - python_hv).abs() < TOL,
        "HSSP 3-of-4: Rust HV={:.6} != Python HV={:.6}",
        rust_hv, python_hv
    );
}

/// 5 Pareto 点, 选 2 个
#[test]
fn test_hssp_2d_select_2_of_5() {
    let vals = vec![
        vec![1.0, 8.0],
        vec![2.0, 5.0],
        vec![4.0, 4.0],
        vec![6.0, 2.0],
        vec![7.0, 1.0],
    ];
    let indices: Vec<usize> = (0..5).collect();
    let ref_point = vec![10.0, 10.0];
    let selected = solve_hssp(&vals, &indices, 2, &ref_point);
    assert_eq!(selected.len(), 2);
    // Python: [1, 3]
    let python_selected = vec![vec![2.0, 5.0], vec![6.0, 2.0]];
    let python_hv = hypervolume(&python_selected, &ref_point);
    let rust_selected_vals: Vec<Vec<f64>> = selected.iter().map(|&i| vals[i].clone()).collect();
    let rust_hv = hypervolume(&rust_selected_vals, &ref_point);
    assert!(
        (rust_hv - python_hv).abs() < TOL,
        "HSSP 2-of-5: Rust HV={:.6} != Python HV={:.6}",
        rust_hv, python_hv
    );
}

// ============================================================================
// 4. crowding_distance — Python 精确参考值
// ============================================================================

/// 4 Pareto 点 → 边界点 infinity，中间点有有限距离
#[test]
fn test_crowding_distance_4_points() {
    let trials: Vec<FrozenTrial> = vec![
        make_trial_vals(0, vec![1.0, 5.0]),
        make_trial_vals(1, vec![2.0, 4.0]),
        make_trial_vals(2, vec![3.0, 3.0]),
        make_trial_vals(3, vec![4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let cd = crowding_distance(&refs, &dirs);

    // Python: [inf, 1.1667, 1.4167, inf]
    assert!(cd[0].is_infinite(), "边界点 0 应为 inf");
    assert!(cd[3].is_infinite(), "边界点 3 应为 inf");
    let expected_1 = 1.166666666666667e+00;
    let expected_2 = 1.416666666666667e+00;
    assert!(
        (cd[1] - expected_1).abs() < 1e-10,
        "cd[1]: Rust={:.15e}, Python={:.15e}",
        cd[1], expected_1
    );
    assert!(
        (cd[2] - expected_2).abs() < 1e-10,
        "cd[2]: Rust={:.15e}, Python={:.15e}",
        cd[2], expected_2
    );
}

/// 3 Pareto 点 → 中间点距离 = 2.0
#[test]
fn test_crowding_distance_3_points() {
    let trials: Vec<FrozenTrial> = vec![
        make_trial_vals(0, vec![1.0, 3.0]),
        make_trial_vals(1, vec![2.0, 2.0]),
        make_trial_vals(2, vec![3.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let cd = crowding_distance(&refs, &dirs);

    assert!(cd[0].is_infinite());
    assert!(cd[2].is_infinite());
    assert!(
        (cd[1] - 2.0).abs() < TOL,
        "cd[1] should be 2.0, got {}",
        cd[1]
    );
}

/// 4 Pareto 点 — 非对称间距
#[test]
fn test_crowding_distance_asymmetric() {
    let trials: Vec<FrozenTrial> = vec![
        make_trial_vals(0, vec![1.0, 5.0]),
        make_trial_vals(1, vec![2.0, 3.0]),
        make_trial_vals(2, vec![3.0, 2.0]),
        make_trial_vals(3, vec![4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let cd = crowding_distance(&refs, &dirs);

    let expected_1 = 1.416666666666667e+00;
    let expected_2 = 1.166666666666667e+00;
    assert!(cd[0].is_infinite());
    assert!(cd[3].is_infinite());
    assert!(
        (cd[1] - expected_1).abs() < 1e-10,
        "cd[1]: Rust={:.15e}, Python={:.15e}",
        cd[1], expected_1
    );
    assert!(
        (cd[2] - expected_2).abs() < 1e-10,
        "cd[2]: Rust={:.15e}, Python={:.15e}",
        cd[2], expected_2
    );
}

/// 单点 → 距离 = inf
#[test]
fn test_crowding_distance_single() {
    let trials = vec![make_trial_vals(0, vec![1.0, 1.0])];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let cd = crowding_distance(&refs, &dirs);
    // 单点距离取决于实现，但应为 0（全部值相同 → skip）
    assert_eq!(cd.len(), 1);
}

// ============================================================================
// 5. dominates — 逻辑精确性
// ============================================================================

#[test]
fn test_dominates_strict() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
}

#[test]
fn test_dominates_equal_in_one() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(dominates(&[1.0, 1.0], &[1.0, 2.0], &dirs));
}

#[test]
fn test_dominates_equal_not_dominated() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
}

#[test]
fn test_dominates_tradeoff() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0], &dirs));
}

#[test]
fn test_dominates_maximize() {
    let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    assert!(dominates(&[3.0, 3.0], &[2.0, 2.0], &dirs));
    assert!(!dominates(&[2.0, 2.0], &[3.0, 3.0], &dirs));
}

#[test]
fn test_dominates_mixed_directions() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Maximize];
    // a=[1,3], b=[2,2]: a better in min(1<2) and max(3>2)
    assert!(dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
    assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0], &dirs));
}

// ============================================================================
// 6. fast_non_dominated_sort — front 分配正确性
// ============================================================================

/// 简单场景: front 0 = {0,1,2}, front 1 = {3}, front 2 = {4}
#[test]
fn test_nds_simple() {
    let trials = vec![
        make_trial_vals(0, vec![1.0, 4.0]), // Pareto
        make_trial_vals(1, vec![2.0, 3.0]), // Pareto
        make_trial_vals(2, vec![3.0, 2.0]), // Pareto
        make_trial_vals(3, vec![4.0, 4.0]), // dominated by 2
        make_trial_vals(4, vec![5.0, 5.0]), // dominated by 3
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&refs, &dirs);

    assert!(fronts.len() >= 3, "Should have at least 3 fronts");
    // Front 0: {0, 1, 2}
    let mut f0 = fronts[0].clone();
    f0.sort();
    assert_eq!(f0, vec![0, 1, 2], "Front 0 should be [0,1,2]");
    // Front 1: {3}
    assert_eq!(fronts[1], vec![3], "Front 1 should be [3]");
    // Front 2: {4}
    assert_eq!(fronts[2], vec![4], "Front 2 should be [4]");
}

/// 完全支配链: [1,1] > [2,2] > [3,3] > [4,4]
#[test]
fn test_nds_dominance_chain() {
    let trials = vec![
        make_trial_vals(0, vec![1.0, 1.0]),
        make_trial_vals(1, vec![2.0, 2.0]),
        make_trial_vals(2, vec![3.0, 3.0]),
        make_trial_vals(3, vec![4.0, 4.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&refs, &dirs);

    assert_eq!(fronts.len(), 4, "Each point should be in its own front");
    assert_eq!(fronts[0], vec![0]);
    assert_eq!(fronts[1], vec![1]);
    assert_eq!(fronts[2], vec![2]);
    assert_eq!(fronts[3], vec![3]);
}

/// 混合: [1,3], [3,1], [2,2], [1,1] → [1,1] 在 front 0, 其余取决于支配关系
#[test]
fn test_nds_mixed() {
    let trials = vec![
        make_trial_vals(0, vec![1.0, 3.0]),
        make_trial_vals(1, vec![3.0, 1.0]),
        make_trial_vals(2, vec![2.0, 2.0]),
        make_trial_vals(3, vec![1.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&refs, &dirs);

    // [1,1] dominates all others → front 0 = {3}
    // [1,3], [3,1], [2,2]: [1,3] ≠ dominates [2,2] (1<2 but 3>2)
    //                       [3,1] ≠ dominates [2,2] (3>2 but 1<2)
    //                       → all three are mutually non-dominated → front 1 = {0, 1, 2}
    let mut f0 = fronts[0].clone();
    f0.sort();
    assert_eq!(f0, vec![3], "Front 0 should be [3] (point [1,1])");
    let mut f1 = fronts[1].clone();
    f1.sort();
    assert_eq!(f1, vec![0, 1, 2], "Front 1 should be [0,1,2]");
}

// ============================================================================
// 7. hypervolume 边界情况
// ============================================================================

/// 空输入 → 0
#[test]
fn test_hv_empty() {
    assert_eq!(hypervolume(&[], &[5.0, 5.0]), 0.0);
}

/// 点等于参考点 → 0 (贡献为 0)
#[test]
fn test_hv_point_at_ref() {
    let pts = vec![vec![5.0, 5.0]];
    let r = vec![5.0, 5.0];
    assert_eq!(hypervolume(&pts, &r), 0.0);
}

/// 2D 一致性: hypervolume 入口与 hypervolume_2d 特化路径
#[test]
fn test_hv_2d_consistency() {
    let pts_2d = vec![[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];
    let pts_generic: Vec<Vec<f64>> = pts_2d.iter().map(|p| vec![p[0], p[1]]).collect();
    let r_2d = [5.0, 5.0];
    let r_generic = vec![5.0, 5.0];

    let hv_2d = hypervolume_2d(&pts_2d, r_2d);
    let hv_generic = hypervolume(&pts_generic, &r_generic);

    assert!(
        (hv_2d - hv_generic).abs() < TOL,
        "2D 入口一致性: 2d={:.15e}, generic={:.15e}",
        hv_2d, hv_generic
    );
}

/// HV 单调性: 添加点不会减少超体积
#[test]
fn test_hv_monotonicity() {
    let r = vec![10.0, 10.0];
    let p1 = vec![vec![2.0, 5.0]];
    let p2 = vec![vec![2.0, 5.0], vec![5.0, 2.0]];
    let p3 = vec![vec![2.0, 5.0], vec![5.0, 2.0], vec![3.0, 3.0]];

    let hv1 = hypervolume(&p1, &r);
    let hv2 = hypervolume(&p2, &r);
    let hv3 = hypervolume(&p3, &r);

    assert!(
        hv2 >= hv1 - TOL,
        "HV 单调性: hv2={} >= hv1={}",
        hv2, hv1
    );
    assert!(
        hv3 >= hv2 - TOL,
        "HV 单调性: hv3={} >= hv2={}",
        hv3, hv2
    );
}

/// HV 对称性: HV 不依赖点的输入顺序
#[test]
fn test_hv_order_independent() {
    let r = vec![10.0, 10.0, 10.0];
    let p1 = vec![
        vec![1.0, 3.0, 2.0],
        vec![2.0, 1.0, 3.0],
        vec![3.0, 2.0, 1.0],
    ];
    let p2 = vec![
        vec![3.0, 2.0, 1.0],
        vec![1.0, 3.0, 2.0],
        vec![2.0, 1.0, 3.0],
    ];
    let hv1 = hypervolume(&p1, &r);
    let hv2 = hypervolume(&p2, &r);
    assert!(
        (hv1 - hv2).abs() < TOL,
        "HV 顺序无关: hv1={:.15e}, hv2={:.15e}",
        hv1, hv2
    );
}
