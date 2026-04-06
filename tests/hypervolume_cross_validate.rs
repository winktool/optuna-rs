// ═══════════════════════════════════════════════════════════════════════════════
// Deep cross-validation of hypervolume and HSSP against Python Optuna baselines.
//
// All expected values verified against Python Optuna `compute_hypervolume`
// and `_solve_hssp`.
//
// Covers:
//  1. hypervolume_2d —— 多点/含支配点/单点/边界
//  2. hypervolume_3d  (hypervolume 3D 特化路径)
//  3. hypervolume_wfg (4D+)
//  4. hypervolume: 所有维度统一入口
//  5. solve_hssp —— 2D/3D子集选择
//  6. 边界：空输入/全相同/参考点边界
// ═══════════════════════════════════════════════════════════════════════════════

use optuna_rs::multi_objective::{hypervolume, hypervolume_2d, solve_hssp};

const TOL: f64 = 1e-9;

fn assert_close(got: f64, exp: f64, tol: f64, label: &str) {
    let diff = (got - exp).abs();
    let denom = exp.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{label}: expected {exp}, got {got}, rel_err={:.2e}",
        diff / denom
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. hypervolume_2d: 4 non-dominated points
//    Python: compute_hypervolume([[1,5],[2,4],[3,3],[4,1]], [5,6]) = 11.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_2d_4points() {
    let pts = vec![[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 1.0]];
    let r = [5.0, 6.0];
    assert_close(hypervolume_2d(&pts, r), 11.0, TOL, "hv_2d_4pts");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. hypervolume_2d: with a dominated point
//    Python: 9.0 (dominated point [2.5,4.5] doesn't contribute)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_2d_with_dominated() {
    let pts = vec![[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [2.5, 4.5]];
    let r = [5.0, 6.0];
    assert_close(hypervolume_2d(&pts, r), 9.0, TOL, "hv_2d_dominated");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. hypervolume: single point 2D
//    Python: compute_hypervolume([[2,3]], [5,5]) = 6.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_single_point() {
    let pts = vec![vec![2.0, 3.0]];
    let r = vec![5.0, 5.0];
    assert_close(hypervolume(&pts, &r), 6.0, TOL, "hv_single");
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. hypervolume: hand-computed [1,1] with ref [3,3] = 4.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_hand_computed() {
    let pts = vec![vec![1.0, 1.0]];
    let r = vec![3.0, 3.0];
    assert_close(hypervolume(&pts, &r), 4.0, TOL, "hv_hand");
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. hypervolume: points on reference boundary → 0
//    Python: compute_hypervolume([[5,3],[2,5]], [5,5]) = 0.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_boundary_zero() {
    let pts = vec![vec![5.0, 3.0], vec![2.0, 5.0]];
    let r = vec![5.0, 5.0];
    assert_close(hypervolume(&pts, &r), 0.0, TOL, "hv_boundary");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. hypervolume: all identical points
//    Python: compute_hypervolume([[2,3],[2,3],[2,3]], [5,5]) = 6.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_identical_points() {
    let pts = vec![vec![2.0, 3.0], vec![2.0, 3.0], vec![2.0, 3.0]];
    let r = vec![5.0, 5.0];
    assert_close(hypervolume(&pts, &r), 6.0, TOL, "hv_identical");
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. hypervolume 3D: 3 points
//    Python: compute_hypervolume([[1,5,3],[2,4,2],[3,3,1]], [5,6,5]) = 32.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_3d_3points() {
    let pts = vec![
        vec![1.0, 5.0, 3.0],
        vec![2.0, 4.0, 2.0],
        vec![3.0, 3.0, 1.0],
    ];
    let r = vec![5.0, 6.0, 5.0];
    assert_close(hypervolume(&pts, &r), 32.0, TOL, "hv_3d_3pts");
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. hypervolume 3D: single point
//    Python: compute_hypervolume([[1,1,1]], [3,3,3]) = 8.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_3d_single_point() {
    let pts = vec![vec![1.0, 1.0, 1.0]];
    let r = vec![3.0, 3.0, 3.0];
    assert_close(hypervolume(&pts, &r), 8.0, TOL, "hv_3d_single");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. hypervolume 3D: another case
//    Python: compute_hypervolume([[1,1,4],[2,3,1],[4,2,2]], [5,5,5]) = 36.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_3d_case2() {
    let pts = vec![
        vec![1.0, 1.0, 4.0],
        vec![2.0, 3.0, 1.0],
        vec![4.0, 2.0, 2.0],
    ];
    let r = vec![5.0, 5.0, 5.0];
    assert_close(hypervolume(&pts, &r), 36.0, TOL, "hv_3d_case2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. hypervolume 4D (WFG path)
//     Python: compute_hypervolume([[1,1,1,1],[2,2,2,2]], [3,3,3,3]) = 16.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_4d() {
    let pts = vec![vec![1.0, 1.0, 1.0, 1.0], vec![2.0, 2.0, 2.0, 2.0]];
    let r = vec![3.0, 3.0, 3.0, 3.0];
    assert_close(hypervolume(&pts, &r), 16.0, TOL, "hv_4d");
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. hypervolume 5D (WFG path)
//     Python: compute_hypervolume([[1,1,1,1,1],[2,2,2,2,2]], [3,3,3,3,3]) = 32.0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_5d() {
    let pts = vec![
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0, 2.0],
    ];
    let r = vec![3.0, 3.0, 3.0, 3.0, 3.0];
    assert_close(hypervolume(&pts, &r), 32.0, TOL, "hv_5d");
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. hypervolume: empty input → 0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_empty() {
    let pts: Vec<Vec<f64>> = vec![];
    let r = vec![5.0, 5.0];
    assert_close(hypervolume(&pts, &r), 0.0, TOL, "hv_empty");
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. hypervolume 2d: empty input → 0
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_2d_empty() {
    let pts: Vec<[f64; 2]> = vec![];
    let r = [5.0, 5.0];
    assert_close(hypervolume_2d(&pts, r), 0.0, TOL, "hv_2d_empty");
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. solve_hssp 2D: select 2 from 4
//     Python: [1, 3] (points [2,4] and [4,1])
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hssp_2d_select2() {
    let vals = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 1.0],
    ];
    let indices: Vec<usize> = (0..4).collect();
    let ref_point = vec![5.0, 6.0];
    let selected = solve_hssp(&vals, &indices, 2, &ref_point);

    let mut sel = selected.clone();
    sel.sort();
    assert_eq!(sel, vec![1, 3], "hssp 2d select 2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. solve_hssp 2D: select 3 from 4
//     Python: [0, 1, 3]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hssp_2d_select3() {
    let vals = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 1.0],
    ];
    let indices: Vec<usize> = (0..4).collect();
    let ref_point = vec![5.0, 6.0];
    let selected = solve_hssp(&vals, &indices, 3, &ref_point);

    let mut sel = selected.clone();
    sel.sort();
    assert_eq!(sel, vec![0, 1, 3], "hssp 2d select 3");
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. solve_hssp: select all → return all
//     Python: [0, 1, 2, 3]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hssp_select_all() {
    let vals = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![4.0, 1.0],
    ];
    let indices: Vec<usize> = (0..4).collect();
    let ref_point = vec![5.0, 6.0];
    let selected = solve_hssp(&vals, &indices, 4, &ref_point);

    let mut sel = selected.clone();
    sel.sort();
    assert_eq!(sel, vec![0, 1, 2, 3], "hssp select all");
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. solve_hssp 3D: select 2 from 3
//     Python: [1, 2]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hssp_3d_select2() {
    let vals = vec![
        vec![1.0, 5.0, 3.0],
        vec![2.0, 4.0, 2.0],
        vec![3.0, 3.0, 1.0],
    ];
    let indices: Vec<usize> = (0..3).collect();
    let ref_point = vec![5.0, 6.0, 5.0];
    let selected = solve_hssp(&vals, &indices, 2, &ref_point);

    let mut sel = selected.clone();
    sel.sort();
    assert_eq!(sel, vec![1, 2], "hssp 3d select 2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. hypervolume monotonicity: adding a non-dominated point increases HV
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_monotonicity() {
    let r = vec![10.0, 10.0];

    let pts1 = vec![vec![1.0, 5.0], vec![5.0, 1.0]];
    let hv1 = hypervolume(&pts1, &r);

    let pts2 = vec![vec![1.0, 5.0], vec![5.0, 1.0], vec![3.0, 3.0]];
    let hv2 = hypervolume(&pts2, &r);

    assert!(
        hv2 >= hv1,
        "Adding non-dominated point should not decrease HV: {hv1} > {hv2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. hypervolume: adding dominated point doesn't change HV
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_dominated_no_change() {
    let r = vec![5.0, 6.0];

    let pts1 = vec![vec![1.0, 5.0], vec![2.0, 4.0], vec![3.0, 3.0]];
    let hv1 = hypervolume(&pts1, &r);

    // Add dominated point [2.5, 4.5] (dominated by [2.0, 4.0])
    let pts2 = vec![
        vec![1.0, 5.0],
        vec![2.0, 4.0],
        vec![3.0, 3.0],
        vec![2.5, 4.5],
    ];
    let hv2 = hypervolume(&pts2, &r);

    assert_close(hv1, hv2, TOL, "dominated point should not change HV");
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. hypervolume non-negative
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_hv_non_negative() {
    let test_cases: Vec<(Vec<Vec<f64>>, Vec<f64>)> = vec![
        (vec![vec![1.0, 1.0]], vec![3.0, 3.0]),
        (vec![vec![0.0, 0.0]], vec![1.0, 1.0]),
        (vec![vec![2.99, 2.99]], vec![3.0, 3.0]),
        (vec![], vec![5.0, 5.0]),
        (vec![vec![5.0, 5.0]], vec![5.0, 5.0]),  // on boundary
    ];

    for (i, (pts, r)) in test_cases.iter().enumerate() {
        let hv = hypervolume(pts, r);
        assert!(hv >= 0.0, "case {i}: HV should be non-negative, got {hv}");
    }
}
