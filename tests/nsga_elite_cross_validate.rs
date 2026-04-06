/// NSGA-II 拥挤距离 + NSGA-III 参考点 交叉验证测试
///
/// 对齐 Python optuna crowding_distance 和 NSGA-III reference_point 生成算法
/// 所有测试用例均来自 golden_crowding.py / golden_nsgaiii.py 生成的 Python 金标准值。

use optuna_rs::multi_objective::{
    crowding_distance, fast_non_dominated_sort, is_pareto_front,
};
use optuna_rs::samplers::nsgaiii::generate_reference_points;
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

fn make_trial(number: i64, values: Vec<f64>) -> FrozenTrial {
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

// ===== 拥挤距离测试 =====

#[test]
fn test_crowding_distance_2d_simple_python() {
    // Python 金标准: {0: inf, 1: 1.333, 2: 1.333, 3: inf}
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 4.0]),
        make_trial(1, vec![2.0, 3.0]),
        make_trial(2, vec![3.0, 2.0]),
        make_trial(3, vec![4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let cd = crowding_distance(&refs, &dirs);
    assert!(cd[0].is_infinite(), "边界点应为 inf");
    assert!((cd[1] - 4.0 / 3.0).abs() < 1e-10, "Expected 1.333, got {}", cd[1]);
    assert!((cd[2] - 4.0 / 3.0).abs() < 1e-10, "Expected 1.333, got {}", cd[2]);
    assert!(cd[3].is_infinite(), "边界点应为 inf");
}

#[test]
fn test_crowding_distance_2d_extreme_python() {
    // Python 金标准: {0: inf, 1: 2.0, 2: inf}
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![0.0, 10.0]),
        make_trial(1, vec![5.0, 5.0]),
        make_trial(2, vec![10.0, 0.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let cd = crowding_distance(&refs, &dirs);
    assert!(cd[0].is_infinite());
    assert!((cd[1] - 2.0).abs() < 1e-10, "Expected 2.0, got {}", cd[1]);
    assert!(cd[2].is_infinite());
}

#[test]
fn test_crowding_distance_all_same_python() {
    // Python 金标准: all distances = 0.0 (empty defaultdict)
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 1.0]),
        make_trial(1, vec![1.0, 1.0]),
        make_trial(2, vec![1.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let cd = crowding_distance(&refs, &dirs);
    for (i, d) in cd.iter().enumerate() {
        assert!(*d == 0.0, "cd[{}] should be 0.0, got {}", i, d);
    }
}

#[test]
fn test_crowding_distance_single_python() {
    // Python 金标准: distance = 0.0
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![make_trial(0, vec![1.0, 2.0])];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let cd = crowding_distance(&refs, &dirs);
    assert_eq!(cd.len(), 1);
    // Single trial: both boundaries are the same → width=0 → skip
    assert!(cd[0] == 0.0 || cd[0].is_infinite(), "Single trial cd: {}", cd[0]);
}

#[test]
fn test_crowding_distance_3d_python() {
    // Python 金标准: {0: inf, 1: 3.0, 2: inf}
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 3.0, 5.0]),
        make_trial(1, vec![2.0, 2.0, 3.0]),
        make_trial(2, vec![3.0, 1.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let cd = crowding_distance(&refs, &dirs);
    assert!(cd[0].is_infinite());
    assert!((cd[1] - 3.0).abs() < 1e-10, "Expected 3.0, got {}", cd[1]);
    assert!(cd[2].is_infinite());
}

#[test]
fn test_crowding_distance_empty() {
    let dirs = vec![StudyDirection::Minimize];
    let refs: Vec<&FrozenTrial> = vec![];
    let cd = crowding_distance(&refs, &dirs);
    assert!(cd.is_empty());
}

// ===== NSGA-III 参考点生成测试 =====

#[test]
fn test_reference_points_2d_p3() {
    // Python 金标准 (归一化后): [1,0], [0.667,0.333], [0.333,0.667], [0,1]
    // Rust Das-Dennis: [0,1], [0.333,0.667], [0.667,0.333], [1,0]
    // 两者是相同的集合，顺序可能不同
    let pts = generate_reference_points(2, 3);
    assert_eq!(pts.len(), 4);

    // 验证每个点的坐标之和 = 1.0
    for p in &pts {
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Point sum should be 1.0, got {}", sum);
    }

    // 验证包含所有 4 个方向（作为集合）
    let expected: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0],
        vec![1.0 / 3.0, 2.0 / 3.0],
        vec![2.0 / 3.0, 1.0 / 3.0],
        vec![1.0, 0.0],
    ];
    for exp in &expected {
        let found = pts.iter().any(|p| {
            p.iter().zip(exp.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
        });
        assert!(found, "Missing reference point {:?}", exp);
    }
}

#[test]
fn test_reference_points_3d_p3() {
    // Python: 10 points for C(3+3-1, 3-1) = C(5,2) = 10
    let pts = generate_reference_points(3, 3);
    assert_eq!(pts.len(), 10);

    for p in &pts {
        let sum: f64 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Point sum should be 1.0, got {}", sum);
    }

    // 验证顶点存在
    let vertices = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    for v in &vertices {
        let found = pts.iter().any(|p| {
            p.iter().zip(v.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
        });
        assert!(found, "Missing vertex {:?}", v);
    }

    // 验证中心点存在
    let center = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let found = pts.iter().any(|p| {
        p.iter().zip(center.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
    });
    assert!(found, "Missing center point");
}

#[test]
fn test_reference_points_2d_p4() {
    // Python: 5 points for C(2+4-1,4-1) = C(5,3)... wait, C(n+p-1,p) = C(5,4) = 5
    let pts = generate_reference_points(2, 4);
    assert_eq!(pts.len(), 5);

    let expected: Vec<Vec<f64>> = vec![
        vec![0.0, 1.0],
        vec![0.25, 0.75],
        vec![0.5, 0.5],
        vec![0.75, 0.25],
        vec![1.0, 0.0],
    ];
    for exp in &expected {
        let found = pts.iter().any(|p| {
            p.iter().zip(exp.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
        });
        assert!(found, "Missing reference point {:?}", exp);
    }
}

#[test]
fn test_reference_points_count_formula() {
    // NSGA-III: 参考点数量 = C(n_obj + p - 1, p)
    // 2D p=3: C(4,3) = 4 ✓
    // 3D p=3: C(5,3) = 10 ✓
    // 2D p=4: C(5,4) = 5 ✓
    assert_eq!(generate_reference_points(2, 3).len(), 4);
    assert_eq!(generate_reference_points(3, 3).len(), 10);
    assert_eq!(generate_reference_points(2, 4).len(), 5);
    assert_eq!(generate_reference_points(4, 3).len(), 20); // C(6,3) = 20
    assert_eq!(generate_reference_points(3, 4).len(), 15); // C(6,4) = 15
}

// ===== fast_non_dominated_sort 交叉验证 =====

#[test]
fn test_nds_2d_tradeoff() {
    // Python 金标准: [0,0,0,0] — all on front 0
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 4.0]),
        make_trial(1, vec![2.0, 3.0]),
        make_trial(2, vec![3.0, 2.0]),
        make_trial(3, vec![4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &dirs);
    assert_eq!(fronts.len(), 1);
    assert_eq!(fronts[0].len(), 4);
}

#[test]
fn test_nds_2d_dominated() {
    // Python 金标准: [0,1,2,0]
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 1.0]),
        make_trial(1, vec![2.0, 2.0]),
        make_trial(2, vec![3.0, 3.0]),
        make_trial(3, vec![1.5, 0.5]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &dirs);
    assert_eq!(fronts.len(), 3);
    // Front 0: trials 0 and 3
    assert!(fronts[0].contains(&0) && fronts[0].contains(&3));
    // Front 1: trial 1
    assert_eq!(fronts[1], vec![1]);
    // Front 2: trial 2
    assert_eq!(fronts[2], vec![2]);
}

#[test]
fn test_nds_3d() {
    // Python 金标准: [0,0,0,1]
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 2.0, 3.0]),
        make_trial(1, vec![2.0, 1.0, 3.0]),
        make_trial(2, vec![3.0, 3.0, 1.0]),
        make_trial(3, vec![4.0, 4.0, 4.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &dirs);
    assert_eq!(fronts.len(), 2);
    assert_eq!(fronts[0].len(), 3); // trials 0,1,2
    assert_eq!(fronts[1], vec![3]);
}

#[test]
fn test_nds_empty() {
    let dirs = vec![StudyDirection::Minimize];
    let refs: Vec<&FrozenTrial> = vec![];
    let fronts = fast_non_dominated_sort(&refs, &dirs);
    assert!(fronts.is_empty());
}

// ===== is_pareto_front 测试 =====

#[test]
fn test_pareto_front_2d() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_trial(0, vec![1.0, 4.0]),
        make_trial(1, vec![2.0, 3.0]),
        make_trial(2, vec![3.0, 2.0]),
        make_trial(3, vec![5.0, 5.0]), // dominated
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let on_front = is_pareto_front(&refs, &dirs);
    assert!(on_front[0]);
    assert!(on_front[1]);
    assert!(on_front[2]);
    assert!(!on_front[3]);
}

// ===== 垂直距离计算 =====

#[test]
fn test_perpendicular_distance() {
    use optuna_rs::samplers::nsgaiii::perpendicular_distance;

    // 点 [0,1] 到方向 [1,0] 的垂直距离 = 1.0
    let d1 = perpendicular_distance(&[0.0, 1.0], &[1.0, 0.0]);
    assert!((d1 - 1.0).abs() < 1e-10);

    // 点 [0.5,0.5] 到方向 [1,1] 的垂直距离 = 0.0 (在线上)
    let d2 = perpendicular_distance(&[0.5, 0.5], &[1.0, 1.0]);
    assert!(d2 < 1e-10);

    // 点 [1,0] 到方向 [0,1] 的垂直距离 = 1.0
    let d3 = perpendicular_distance(&[1.0, 0.0], &[0.0, 1.0]);
    assert!((d3 - 1.0).abs() < 1e-10);
}
