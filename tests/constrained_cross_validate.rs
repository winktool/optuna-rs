/// 约束支配 (Constrained Domination) 交叉验证测试
///
/// 对齐 Python optuna.samplers.nsgaii._constraints_evaluation 模块
/// 验证 constrained_dominates / constraint_violation / is_feasible / constrained_fast_non_dominated_sort
/// 所有测试用例均来自 golden_constrained.py 生成的 Python 金标准值。

use optuna_rs::multi_objective::{
    constrained_dominates, constrained_fast_non_dominated_sort,
    constraint_violation, get_feasible_trials, is_feasible, CONSTRAINTS_KEY,
};
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

/// 构造带约束值的完成态 FrozenTrial
fn make_constrained_trial(number: i64, values: Vec<f64>, constraints: Vec<f64>) -> FrozenTrial {
    let mut system_attrs = HashMap::new();
    system_attrs.insert(
        CONSTRAINTS_KEY.to_string(),
        serde_json::json!(constraints),
    );
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
        system_attrs,
        intermediate_values: HashMap::new(),
    }
}

/// 构造无约束值的完成态 FrozenTrial
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

// ===== is_feasible 测试 =====

#[test]
fn test_is_feasible_all_negative() {
    // Python: all(v <= 0 for v in [-1.0, -1.0]) → True
    let t = make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -1.0]);
    assert!(is_feasible(&t));
}

#[test]
fn test_is_feasible_with_zero() {
    // Python: all(v <= 0 for v in [-1.0, 0.0]) → True (0 满足约束)
    let t = make_constrained_trial(0, vec![1.0], vec![-1.0, 0.0]);
    assert!(is_feasible(&t));
}

#[test]
fn test_is_feasible_has_positive() {
    // Python: all(v <= 0 for v in [1.0, 0.5]) → False
    let t = make_constrained_trial(0, vec![1.0], vec![1.0, 0.5]);
    assert!(!is_feasible(&t));
}

#[test]
fn test_is_feasible_no_constraints() {
    // Python: 无约束值 → 视为不可行
    let t = make_trial(0, vec![1.0]);
    assert!(!is_feasible(&t));
}

#[test]
fn test_is_feasible_mixed() {
    // Python: all(v <= 0 for v in [-0.3, 0.7, -0.1, 0.2]) → False (0.7 > 0)
    let t = make_constrained_trial(0, vec![1.0], vec![-0.3, 0.7, -0.1, 0.2]);
    assert!(!is_feasible(&t));
}

// ===== constraint_violation 测试 =====

#[test]
fn test_constraint_violation_feasible() {
    // Python: sum(v for v in [-1.0, -1.0] if v > 0) = 0.0
    let t = make_constrained_trial(0, vec![1.0, 1.0], vec![-1.0, -1.0]);
    assert!((constraint_violation(&t) - 0.0).abs() < 1e-15);
}

#[test]
fn test_constraint_violation_infeasible() {
    // Python: sum(v for v in [0.5, 1.5] if v > 0) = 2.0
    let t = make_constrained_trial(1, vec![2.0, 2.0], vec![0.5, 1.5]);
    assert!((constraint_violation(&t) - 2.0).abs() < 1e-15);
}

#[test]
fn test_constraint_violation_no_constraints() {
    // Python: penalty = nan
    let t = make_trial(2, vec![3.0, 3.0]);
    assert!(constraint_violation(&t).is_nan());
}

#[test]
fn test_constraint_violation_mixed_positive_negative() {
    // Python: sum(v for v in [-0.3, 0.7, -0.1, 0.2] if v > 0) = 0.9
    let t = make_constrained_trial(3, vec![4.0, 4.0], vec![-0.3, 0.7, -0.1, 0.2]);
    let viol = constraint_violation(&t);
    assert!((viol - 0.9).abs() < 1e-14, "Expected 0.9, got {}", viol);
}

#[test]
fn test_constraint_violation_single_large() {
    // Python: sum(v for v in [100.0] if v > 0) = 100.0
    let t = make_constrained_trial(0, vec![1.0], vec![100.0]);
    assert!((constraint_violation(&t) - 100.0).abs() < 1e-12);
}

// ===== constrained_dominates 测试 — 精确对齐 Python 金标准 =====

#[test]
fn test_cd_both_feasible_a_dominates_python() {
    // Python 金标准: _constrained_dominates(
    //   trial(vals=[1,2], c=[-1,-1]),
    //   trial(vals=[3,4], c=[-0.5,-0.5]),
    //   [MIN, MIN]) = True
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -1.0]);
    let b = make_constrained_trial(1, vec![3.0, 4.0], vec![-0.5, -0.5]);
    assert!(constrained_dominates(&a, &b, &dirs));
    assert!(!constrained_dominates(&b, &a, &dirs));
}

#[test]
fn test_cd_feasible_vs_infeasible_python() {
    // Python 金标准: a feasible(vals=[5,5], c=[-1,-1]) dom b infeasible(vals=[1,1], c=[1,0.5])
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![5.0, 5.0], vec![-1.0, -1.0]);
    let b = make_constrained_trial(1, vec![1.0, 1.0], vec![1.0, 0.5]);
    assert!(constrained_dominates(&a, &b, &dirs)); // True
    assert!(!constrained_dominates(&b, &a, &dirs)); // False
}

#[test]
fn test_cd_both_infeasible_smaller_violation_python() {
    // Python 金标准: a infeasible(c=[0.5,0.3]) violation=0.8
    //               b infeasible(c=[1.0,2.0]) violation=3.0
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![5.0, 5.0], vec![0.5, 0.3]);
    let b = make_constrained_trial(1, vec![1.0, 1.0], vec![1.0, 2.0]);
    assert!(constrained_dominates(&a, &b, &dirs)); // True
    assert!(!constrained_dominates(&b, &a, &dirs)); // False
}

#[test]
fn test_cd_both_feasible_tradeoff_python() {
    // Python 金标准: a=[1,4], b=[3,2] → 无支配关系 (tradeoff)
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0, -1.0]);
    let b = make_constrained_trial(1, vec![3.0, 2.0], vec![-0.5, -0.5]);
    assert!(!constrained_dominates(&a, &b, &dirs)); // False
    assert!(!constrained_dominates(&b, &a, &dirs)); // False
}

#[test]
fn test_cd_one_has_constraints_python() {
    // Python 金标准: a 有约束(-1,-1), b 无约束 → a 支配 b
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![3.0, 3.0], vec![-1.0, -1.0]);
    let b = make_trial(1, vec![1.0, 1.0]);
    assert!(constrained_dominates(&a, &b, &dirs)); // True
    assert!(!constrained_dominates(&b, &a, &dirs)); // False
}

#[test]
fn test_cd_state_gating_python() {
    // Python 金标准: a Running + b Complete → a 不支配 b
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let mut a = make_constrained_trial(0, vec![1.0, 1.0], vec![-1.0, -1.0]);
    a.state = TrialState::Running;
    let b = make_constrained_trial(1, vec![5.0, 5.0], vec![-1.0, -1.0]);
    assert!(!constrained_dominates(&a, &b, &dirs)); // False

    // b Complete, a Running → b 支配 a
    assert!(constrained_dominates(&b, &a, &dirs)); // True
}

// ===== 边界情况 =====

#[test]
fn test_cd_both_no_constraints_regular_domination() {
    // 都无约束 → 退化为普通 Pareto 支配
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_trial(0, vec![1.0, 2.0]);
    let b = make_trial(1, vec![3.0, 4.0]);
    assert!(constrained_dominates(&a, &b, &dirs));
    assert!(!constrained_dominates(&b, &a, &dirs));
}

#[test]
fn test_cd_both_infeasible_equal_violation() {
    // 相同违反量 → 都不支配
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![1.0, 1.0], vec![0.5, 0.5]); // violation 1.0
    let b = make_constrained_trial(1, vec![2.0, 2.0], vec![0.3, 0.7]); // violation 1.0
    assert!(!constrained_dominates(&a, &b, &dirs));
    assert!(!constrained_dominates(&b, &a, &dirs));
}

#[test]
fn test_cd_maximize_direction() {
    // Maximize: a=[3,4] 支配 b=[1,2]
    let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    let a = make_constrained_trial(0, vec![3.0, 4.0], vec![-1.0, -1.0]);
    let b = make_constrained_trial(1, vec![1.0, 2.0], vec![-0.5, -0.5]);
    assert!(constrained_dominates(&a, &b, &dirs));
    assert!(!constrained_dominates(&b, &a, &dirs));
}

#[test]
fn test_cd_single_objective() {
    // 单目标约束支配
    let dirs = vec![StudyDirection::Minimize];
    let a = make_constrained_trial(0, vec![1.0], vec![-1.0]);
    let b = make_constrained_trial(1, vec![5.0], vec![-0.5]);
    assert!(constrained_dominates(&a, &b, &dirs));
    assert!(!constrained_dominates(&b, &a, &dirs));
}

// ===== constrained_fast_non_dominated_sort 测试 =====

#[test]
fn test_constrained_sort_feasible_vs_infeasible() {
    // 可行解在第一前沿，不可行解在后面
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t0 = make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0]);
    let t1 = make_constrained_trial(1, vec![0.5, 0.5], vec![0.5]);
    let t2 = make_constrained_trial(2, vec![2.0, 3.0], vec![-0.5]);
    let refs: Vec<&FrozenTrial> = vec![&t0, &t1, &t2];
    let fronts = constrained_fast_non_dominated_sort(&refs, &dirs);
    assert!(fronts.len() >= 2);
    // 可行解 t0, t2 在第一个前沿
    assert!(fronts[0].contains(&0) && fronts[0].contains(&2));
    // 不可行 t1 在第二个前沿
    assert!(fronts[1].contains(&1));
}

#[test]
fn test_constrained_sort_multiple_infeasible() {
    // 多个不可行解按违反量排序
    let dirs = vec![StudyDirection::Minimize];
    let t0 = make_constrained_trial(0, vec![1.0], vec![-1.0]); // 可行
    let t1 = make_constrained_trial(1, vec![2.0], vec![0.1]);  // 违反 0.1
    let t2 = make_constrained_trial(2, vec![3.0], vec![0.5]);  // 违反 0.5
    let t3 = make_constrained_trial(3, vec![4.0], vec![1.0]);  // 违反 1.0
    let refs: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
    let fronts = constrained_fast_non_dominated_sort(&refs, &dirs);
    // t0 (可行) 在第一前沿
    assert_eq!(fronts[0], vec![0]);
    // 不可行按违反量链式支配: t1 → t2 → t3
    assert_eq!(fronts[1], vec![1]);
    assert_eq!(fronts[2], vec![2]);
    assert_eq!(fronts[3], vec![3]);
}

#[test]
fn test_constrained_sort_empty() {
    let dirs = vec![StudyDirection::Minimize];
    let refs: Vec<&FrozenTrial> = vec![];
    let fronts = constrained_fast_non_dominated_sort(&refs, &dirs);
    assert!(fronts.is_empty());
}

#[test]
fn test_constrained_sort_all_feasible_tradeoff() {
    // 所有可行 + tradeoff → 同一前沿
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t0 = make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0]);
    let t1 = make_constrained_trial(1, vec![2.0, 3.0], vec![-1.0]);
    let t2 = make_constrained_trial(2, vec![3.0, 2.0], vec![-1.0]);
    let t3 = make_constrained_trial(3, vec![4.0, 1.0], vec![-1.0]);
    let refs: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
    let fronts = constrained_fast_non_dominated_sort(&refs, &dirs);
    // 所有 tradeoff，都在第一前沿
    assert_eq!(fronts.len(), 1);
    assert_eq!(fronts[0].len(), 4);
}

// ===== get_feasible_trials 测试 =====

#[test]
fn test_get_feasible_trials_filters_correctly() {
    let t0 = make_constrained_trial(0, vec![1.0], vec![-1.0]); // 可行
    let t1 = make_constrained_trial(1, vec![2.0], vec![0.5]);  // 不可行
    let t2 = make_constrained_trial(2, vec![3.0], vec![-0.1, -0.2]); // 可行
    let t3 = make_trial(3, vec![4.0]); // 无约束 → 不可行
    let trials = vec![t0, t1, t2, t3];
    let feasible = get_feasible_trials(&trials);
    assert_eq!(feasible.len(), 2);
    assert_eq!(feasible[0].number, 0);
    assert_eq!(feasible[1].number, 2);
}

#[test]
fn test_get_feasible_trials_boundary_zero() {
    // 约束恰好为 0 → 可行
    let t0 = make_constrained_trial(0, vec![1.0], vec![0.0, 0.0]);
    let trials = vec![t0];
    let feasible = get_feasible_trials(&trials);
    assert_eq!(feasible.len(), 1);
}

// ===== _evaluate_penalty 批量验证 =====

#[test]
fn test_evaluate_penalty_batch_python() {
    // Python 金标准：penalties = [0.0, 2.0, nan, 0.9]
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 1.0], vec![-1.0, -1.0]),
        make_constrained_trial(1, vec![2.0, 2.0], vec![0.5, 1.5]),
        make_trial(2, vec![3.0, 3.0]),
        make_constrained_trial(3, vec![4.0, 4.0], vec![-0.3, 0.7, -0.1, 0.2]),
    ];

    let penalties: Vec<f64> = trials.iter().map(|t| constraint_violation(t)).collect();
    assert!((penalties[0] - 0.0).abs() < 1e-15, "Expected 0.0, got {}", penalties[0]);
    assert!((penalties[1] - 2.0).abs() < 1e-15, "Expected 2.0, got {}", penalties[1]);
    assert!(penalties[2].is_nan(), "Expected NaN, got {}", penalties[2]);
    assert!((penalties[3] - 0.9).abs() < 1e-14, "Expected 0.9, got {}", penalties[3]);
}
