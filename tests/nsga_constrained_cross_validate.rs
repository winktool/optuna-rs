// ═══════════════════════════════════════════════════════════════════════════════
// Deep cross-validation of NSGA-II constrained sorting, rank_population
// equivalence, and additional edge cases against Python Optuna baselines.
//
// All expected values were generated from Python Optuna (latest) and verified
// interactively before being hardcoded here.
//
// Covers:
//  1. constrained_dominates() — feasible×feasible, feasible×infeasible,
//     infeasible×infeasible, equal violations, non-dominating feasible pairs
//  2. constraint_violation() / is_feasible() — penalty computation
//  3. constrained_fast_non_dominated_sort() — mixed feasible/infeasible ranking
//  4. rank_population equivalence — unconstrained multi-front ranking
//  5. Single-objective ranking
//  6. MAXIMIZE direction ranking
//  7. Mixed MINIMIZE/MAXIMIZE ranking
//  8. 5-objective ranking
//  9. All-identical trials
// 10. Single trial / empty input
// 11. Edge case: crowding distance with fronts from rank_population
// 12. Penalty (constraint_violation) cross-validation
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use optuna_rs::multi_objective::{
    constrained_dominates, constrained_fast_non_dominated_sort,
    constraint_violation, crowding_distance, dominates,
    fast_non_dominated_sort, is_feasible, is_pareto_front,
    CONSTRAINTS_KEY,
};
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};

const TOL: f64 = 1e-9;

fn make_trial(number: i64, values: Vec<f64>) -> FrozenTrial {
    FrozenTrial::new(
        number,
        TrialState::Complete,
        None,
        Some(values),
        Some(chrono::Utc::now()),
        Some(chrono::Utc::now()),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        number,
    )
    .unwrap()
}

fn make_constrained_trial(number: i64, values: Vec<f64>, constraints: Vec<f64>) -> FrozenTrial {
    let mut t = make_trial(number, values);
    t.system_attrs
        .insert(CONSTRAINTS_KEY.to_string(), serde_json::json!(constraints));
    t
}

fn fronts_to_ranks(fronts: &[Vec<usize>], n: usize) -> Vec<i64> {
    let mut ranks = vec![-1i64; n];
    for (rank, front) in fronts.iter().enumerate() {
        for &idx in front {
            ranks[idx] = rank as i64;
        }
    }
    ranks
}

fn assert_close(got: f64, exp: f64, tol: f64, label: &str) {
    if exp.is_infinite() {
        assert!(
            got.is_infinite() && got.signum() == exp.signum(),
            "{label}: expected {exp}, got {got}"
        );
    } else {
        let diff = (got - exp).abs();
        let denom = exp.abs().max(1.0);
        assert!(
            diff / denom < tol,
            "{label}: expected {exp}, got {got}, rel_err={:.2e}",
            diff / denom
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. constrained_dominates — feasible vs feasible
//    Python: _constrained_dominates(t1, t2, d) == True
//            _constrained_dominates(t2, t1, d) == False
//    t1 = values=[1,2], constraints=[-1,-0.5]  (feasible, dominates)
//    t2 = values=[2,3], constraints=[-0.3,-0.1] (feasible, dominated)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_feasible_vs_feasible() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t1 = make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -0.5]);
    let t2 = make_constrained_trial(1, vec![2.0, 3.0], vec![-0.3, -0.1]);

    // Python: ff_12 = True
    assert!(constrained_dominates(&t1, &t2, &d));
    // Python: ff_21 = False
    assert!(!constrained_dominates(&t2, &t1, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. constrained_dominates — feasible vs infeasible
//    Python: feasible always dominates infeasible regardless of objective values
//    t3 = values=[10,10], constraints=[-1,-0.2]  (feasible, bad objectives)
//    t4 = values=[0.1,0.1], constraints=[0.5,-0.1] (infeasible, good objectives)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_feasible_vs_infeasible() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t3 = make_constrained_trial(2, vec![10.0, 10.0], vec![-1.0, -0.2]);
    let t4 = make_constrained_trial(3, vec![0.1, 0.1], vec![0.5, -0.1]);

    // Python: fi_34 = True (feasible wins)
    assert!(constrained_dominates(&t3, &t4, &d));
    // Python: if_43 = False
    assert!(!constrained_dominates(&t4, &t3, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. constrained_dominates — infeasible vs infeasible (different violations)
//    t5: constraints=[0.1,0.2] → violation=0.3
//    t6: constraints=[0.5,0.5] → violation=1.0
//    Python: ii_56 = True (smaller violation wins)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_infeasible_vs_infeasible() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t5 = make_constrained_trial(4, vec![1.0, 1.0], vec![0.1, 0.2]);
    let t6 = make_constrained_trial(5, vec![1.0, 1.0], vec![0.5, 0.5]);

    // Python: ii_56 = True
    assert!(constrained_dominates(&t5, &t6, &d));
    // Python: ii_65 = False
    assert!(!constrained_dominates(&t6, &t5, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. constrained_dominates — equal violation amounts
//    t7: constraints=[0.3,0.2] → violation=0.5
//    t8: constraints=[0.3,0.2] → violation=0.5
//    Python: ii_equal_78 = False, ii_equal_87 = False (neither dominates)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_equal_violation() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t7 = make_constrained_trial(6, vec![1.0, 2.0], vec![0.3, 0.2]);
    let t8 = make_constrained_trial(7, vec![3.0, 4.0], vec![0.3, 0.2]);

    // Python: ii_equal_78 = False
    assert!(!constrained_dominates(&t7, &t8, &d));
    // Python: ii_equal_87 = False
    assert!(!constrained_dominates(&t8, &t7, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. constrained_dominates — non-dominating feasible pair
//    t9: values=[1,3], t10: values=[2,1] — trade-off, neither dominates
//    Python: ff_ndom_910 = False, ff_ndom_109 = False
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_non_dominating_feasible() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let t9 = make_constrained_trial(8, vec![1.0, 3.0], vec![-1.0, -0.5]);
    let t10 = make_constrained_trial(9, vec![2.0, 1.0], vec![-0.3, -0.1]);

    // Python: ff_ndom_910 = False
    assert!(!constrained_dominates(&t9, &t10, &d));
    // Python: ff_ndom_109 = False
    assert!(!constrained_dominates(&t10, &t9, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. constraint_violation — penalty values matching Python _evaluate_penalty
//    Python penalties: [0.0, 0.0, 0.0, 0.5, 0.3, 1.0, 0.5, 0.5, 0.0, 0.0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constraint_violation_penalties() {
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -0.5]),  // 0.0
        make_constrained_trial(1, vec![2.0, 3.0], vec![-0.3, -0.1]),  // 0.0
        make_constrained_trial(2, vec![10.0, 10.0], vec![-1.0, -0.2]),// 0.0
        make_constrained_trial(3, vec![0.1, 0.1], vec![0.5, -0.1]),   // 0.5
        make_constrained_trial(4, vec![1.0, 1.0], vec![0.1, 0.2]),    // 0.3
        make_constrained_trial(5, vec![1.0, 1.0], vec![0.5, 0.5]),    // 1.0
        make_constrained_trial(6, vec![1.0, 2.0], vec![0.3, 0.2]),    // 0.5
        make_constrained_trial(7, vec![3.0, 4.0], vec![0.3, 0.2]),    // 0.5
        make_constrained_trial(8, vec![1.0, 3.0], vec![-1.0, -0.5]),  // 0.0
        make_constrained_trial(9, vec![2.0, 1.0], vec![-0.3, -0.1]),  // 0.0
    ];

    let expected = [0.0, 0.0, 0.0, 0.5, 0.3, 1.0, 0.5, 0.5, 0.0, 0.0];
    for (i, trial) in trials.iter().enumerate() {
        let cv = constraint_violation(trial);
        assert_close(cv, expected[i], TOL, &format!("penalty[{i}]"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. is_feasible cross-validation with penalties
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_is_feasible_matches_penalty() {
    let trials = vec![
        make_constrained_trial(0, vec![1.0], vec![-1.0, -0.5]),   // feasible
        make_constrained_trial(1, vec![1.0], vec![0.0, -0.1]),    // feasible (0 is <=0)
        make_constrained_trial(2, vec![1.0], vec![0.1, -0.2]),    // infeasible
        make_constrained_trial(3, vec![1.0], vec![-0.1, -0.2]),   // feasible
        make_trial(4, vec![1.0]),                                   // no constraints → infeasible
    ];

    assert!(is_feasible(&trials[0]));
    assert!(is_feasible(&trials[1]));
    assert!(!is_feasible(&trials[2]));
    assert!(is_feasible(&trials[3]));
    assert!(!is_feasible(&trials[4]));
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. constrained_fast_non_dominated_sort — mixed feasible/infeasible
//    Python _rank_population(pop, d, is_constrained=True):
//      rank 0: [0, 1, 4]  (feasible front)
//      rank 1: [5]         (feasible, dominated by 0,1,4)
//      rank 2: [2]         (infeasible, violation=0.1)
//      rank 3: [3]         (infeasible, violation=0.5)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_fns_mixed_population() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0, -0.5]),  // feasible
        make_constrained_trial(1, vec![2.0, 3.0], vec![-0.5, -0.1]),  // feasible
        make_constrained_trial(2, vec![1.5, 3.5], vec![0.1, 0.0]),    // infeasible (0.1)
        make_constrained_trial(3, vec![0.5, 0.5], vec![0.2, 0.3]),    // infeasible (0.5)
        make_constrained_trial(4, vec![3.0, 2.0], vec![-0.1, -0.2]),  // feasible
        make_constrained_trial(5, vec![5.0, 5.0], vec![-1.0, -0.3]),  // feasible, dominated
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: rank 0 = {0,1,4}, rank 1 = {5}, rank 2 = {2}, rank 3 = {3}
    assert_eq!(ranks[0], 0, "trial 0 should be rank 0");
    assert_eq!(ranks[1], 0, "trial 1 should be rank 0");
    assert_eq!(ranks[4], 0, "trial 4 should be rank 0");
    assert_eq!(ranks[5], 1, "trial 5 should be rank 1");
    assert_eq!(ranks[2], 2, "trial 2 should be rank 2");
    assert_eq!(ranks[3], 3, "trial 3 should be rank 3");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Unconstrained rank_population equivalence (4 fronts)
//    Python _rank_population(pop, d):
//      rank 0: [0,1,2], rank 1: [3,4,5], rank 2: [6,7], rank 3: [8]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_unconstrained_rank_population_4fronts() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 6.0], vec![2.0, 5.0], vec![3.0, 4.0], // front 0
        vec![2.0, 6.0], vec![3.0, 5.0], vec![4.0, 4.0], // front 1
        vec![5.0, 6.0], vec![6.0, 5.0],                  // front 2
        vec![7.0, 7.0],                                   // front 3
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    let expected = vec![0, 0, 0, 1, 1, 1, 2, 2, 3];
    assert_eq!(ranks, expected);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Single-objective ranking
//     Python _fast_non_domination_rank([[5],[3],[7],[1],[3]]) = [2,1,3,0,1]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_single_objective_ranks() {
    let d = vec![StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = [5.0, 3.0, 7.0, 1.0, 3.0]
        .iter()
        .enumerate()
        .map(|(i, &v)| make_trial(i as i64, vec![v]))
        .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [2, 1, 3, 0, 1]
    assert_eq!(ranks, vec![2, 1, 3, 0, 1]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. MAXIMIZE direction (negate then compare)
//     Python: multiply by -1 for MAXIMIZE, then rank
//     Input: [[1,2],[2,3],[3,1]] with MAX×MAX
//     Python ranks after negation: [1, 0, 0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_maximize_direction_ranks() {
    let d = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 1.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [1, 0, 0]
    assert_eq!(ranks, vec![1, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Mixed MINIMIZE/MAXIMIZE direction
//     Input: [[1,1],[2,3],[3,2],[1,3]] with MIN×MAX
//     Python ranks: [1, 1, 2, 0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_mixed_direction_ranks() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Maximize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 2.0],
        vec![1.0, 3.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [1, 1, 2, 0]
    assert_eq!(ranks, vec![1, 1, 2, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. 5-objective ranking
//     Input: [[1,2,3,4,5],[5,4,3,2,1],[2,2,2,2,2],[1,1,1,1,1]]
//     Python ranks: [1, 1, 1, 0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_five_objective_ranks() {
    let d = vec![StudyDirection::Minimize; 5];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![5.0, 4.0, 3.0, 2.0, 1.0],
        vec![2.0, 2.0, 2.0, 2.0, 2.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [1, 1, 1, 0]
    assert_eq!(ranks, vec![1, 1, 1, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. 5-objective with penalty
//     penalty = [0.0, 0.5, 0.0, 1.0]
//     Python: [0, 1, 0, 2]
//     Note: Rust constrained_fast_non_dominated_sort uses constraint_violation
//           which matches Python's _evaluate_penalty
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_five_objective_with_penalty() {
    let d = vec![StudyDirection::Minimize; 5];
    // penalties: [0.0, 0.5, 0.0, 1.0]
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![-1.0]),  // feasible
        make_constrained_trial(1, vec![5.0, 4.0, 3.0, 2.0, 1.0], vec![0.5]),   // infeasible
        make_constrained_trial(2, vec![2.0, 2.0, 2.0, 2.0, 2.0], vec![-0.5]),  // feasible
        make_constrained_trial(3, vec![1.0, 1.0, 1.0, 1.0, 1.0], vec![1.0]),   // infeasible
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [0, 1, 0, 2]
    // rank 0 = feasible trials {0,2}; rank 1 = infeasible 0.5; rank 2 = infeasible 1.0
    assert_eq!(ranks[0], 0, "trial 0 feasible → rank 0");
    assert_eq!(ranks[2], 0, "trial 2 feasible → rank 0");
    assert_eq!(ranks[1], 1, "trial 1 infeasible(0.5) → rank 1");
    assert_eq!(ranks[3], 2, "trial 3 infeasible(1.0) → rank 2");
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. All identical trials → all on rank 0
//     Python: [0, 0, 0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_all_identical_rank0() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = (0..3)
        .map(|i| make_trial(i, vec![1.0, 2.0]))
        .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Python: [0, 0, 0]
    assert_eq!(ranks, vec![0, 0, 0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 16. Single trial → rank 0
//     Python: [0]
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_single_trial_rank0() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![make_trial(0, vec![3.0, 4.0])];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    assert_eq!(ranks, vec![0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 17. Empty input → empty output
//     Python: []
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_empty_input() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = vec![];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();

    let fronts = fast_non_dominated_sort(&refs, &d);
    assert!(fronts.is_empty());

    let cd = crowding_distance(&refs, &d);
    assert!(cd.is_empty());

    let pf = is_pareto_front(&refs, &d);
    assert!(pf.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// 18. Crowding distance for 3-trial front (Python baseline)
//     front0 = [[1,6],[2,5],[3,4]] → cd = {0:inf, 1:2.0, 2:inf}
//     front1 = [[2,6],[3,5],[4,4]] → cd = {3:inf, 4:2.0, 5:inf}
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_crowding_distance_fronts() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];

    // front 0
    let f0: Vec<FrozenTrial> = [
        vec![1.0, 6.0], vec![2.0, 5.0], vec![3.0, 4.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();
    let f0_refs: Vec<&FrozenTrial> = f0.iter().collect();
    let cd0 = crowding_distance(&f0_refs, &d);
    assert!(cd0[0].is_infinite(), "cd front0[0] should be inf");
    assert_close(cd0[1], 2.0, TOL, "cd front0[1]");
    assert!(cd0[2].is_infinite(), "cd front0[2] should be inf");

    // front 1
    let f1: Vec<FrozenTrial> = [
        vec![2.0, 6.0], vec![3.0, 5.0], vec![4.0, 4.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();
    let f1_refs: Vec<&FrozenTrial> = f1.iter().collect();
    let cd1 = crowding_distance(&f1_refs, &d);
    assert!(cd1[0].is_infinite(), "cd front1[0] should be inf");
    assert_close(cd1[1], 2.0, TOL, "cd front1[1]");
    assert!(cd1[2].is_infinite(), "cd front1[2] should be inf");
}

// ═══════════════════════════════════════════════════════════════════════════
// 19. Constrained dominates: antisymmetry invariant
//     For any a, b: if constrained_dominates(a, b) then !constrained_dominates(b, a)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_antisymmetry() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -0.5]),
        make_constrained_trial(1, vec![2.0, 3.0], vec![-0.3, -0.1]),
        make_constrained_trial(2, vec![0.5, 0.5], vec![0.3, 0.2]),
        make_constrained_trial(3, vec![3.0, 1.0], vec![0.1, 0.4]),
        make_constrained_trial(4, vec![1.5, 1.5], vec![-0.5, -0.5]),
    ];

    for i in 0..trials.len() {
        for j in 0..trials.len() {
            if constrained_dominates(&trials[i], &trials[j], &d) {
                assert!(
                    !constrained_dominates(&trials[j], &trials[i], &d),
                    "Antisymmetry violated: ({i},{j})"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 20. Constrained dominates: irreflexivity
//     constrained_dominates(a, a) should be false for both feasible and infeasible
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_irreflexivity() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let feasible = make_constrained_trial(0, vec![1.0, 2.0], vec![-1.0, -0.5]);
    let infeasible = make_constrained_trial(1, vec![1.0, 2.0], vec![0.3, 0.2]);

    assert!(!constrained_dominates(&feasible, &feasible, &d));
    // Note: for infeasible, same violation → not < → false
    assert!(!constrained_dominates(&infeasible, &infeasible, &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 21. Constrained FNS: complete partition invariant
//     All trials should be assigned to exactly one front
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_fns_complete_partition() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0, -0.5]),
        make_constrained_trial(1, vec![2.0, 3.0], vec![-0.5, -0.1]),
        make_constrained_trial(2, vec![1.5, 3.5], vec![0.1, 0.0]),
        make_constrained_trial(3, vec![0.5, 0.5], vec![0.2, 0.3]),
        make_constrained_trial(4, vec![3.0, 2.0], vec![-0.1, -0.2]),
        make_constrained_trial(5, vec![5.0, 5.0], vec![-1.0, -0.3]),
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);

    // Exactly n trials assigned
    let total: usize = fronts.iter().map(|f| f.len()).sum();
    assert_eq!(total, trials.len(), "complete partition");

    // No duplicates
    let mut seen = vec![false; trials.len()];
    for front in &fronts {
        for &idx in front {
            assert!(!seen[idx], "duplicate idx {idx}");
            seen[idx] = true;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 22. Constrained FNS: feasible trials always ranked before infeasible
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_feasible_ranked_first() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0, 4.0], vec![-1.0, -0.5]),  // feasible
        make_constrained_trial(1, vec![2.0, 3.0], vec![-0.5, -0.1]),  // feasible
        make_constrained_trial(2, vec![1.5, 3.5], vec![0.1, 0.0]),    // infeasible
        make_constrained_trial(3, vec![0.5, 0.5], vec![0.2, 0.3]),    // infeasible
        make_constrained_trial(4, vec![3.0, 2.0], vec![-0.1, -0.2]),  // feasible
        make_constrained_trial(5, vec![5.0, 5.0], vec![-1.0, -0.3]),  // feasible
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    let feasible_indices = [0, 1, 4, 5];
    let infeasible_indices = [2, 3];

    let max_feasible_rank = feasible_indices.iter().map(|&i| ranks[i]).max().unwrap();
    let min_infeasible_rank = infeasible_indices.iter().map(|&i| ranks[i]).min().unwrap();

    assert!(
        max_feasible_rank < min_infeasible_rank,
        "All feasible ranks ({max_feasible_rank}) must be < all infeasible ranks ({min_infeasible_rank})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 23. Constrained FNS: infeasible trials ordered by violation
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_constrained_infeasible_ordered_by_violation() {
    let d = vec![StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0], vec![-1.0]),  // feasible
        make_constrained_trial(1, vec![1.0], vec![0.1]),   // infeasible, violation=0.1
        make_constrained_trial(2, vec![1.0], vec![0.5]),   // infeasible, violation=0.5
        make_constrained_trial(3, vec![1.0], vec![1.0]),   // infeasible, violation=1.0
        make_constrained_trial(4, vec![1.0], vec![0.3]),   // infeasible, violation=0.3
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // feasible first
    assert_eq!(ranks[0], 0);
    // infeasible ordered: 0.1 < 0.3 < 0.5 < 1.0
    assert!(ranks[1] < ranks[4], "0.1 before 0.3");
    assert!(ranks[4] < ranks[2], "0.3 before 0.5");
    assert!(ranks[2] < ranks[3], "0.5 before 1.0");
}

// ═══════════════════════════════════════════════════════════════════════════
// 24. Pareto front correctness with MAXIMIZE
//     [[1,2],[2,3],[3,1]] with MAX×MAX → front 0 = {1,2}
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_pareto_front_maximize() {
    let d = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 1.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let pf = is_pareto_front(&refs, &d);

    assert!(!pf[0], "trial 0 dominated by trial 1");
    assert!(pf[1], "trial 1 on Pareto front");
    assert!(pf[2], "trial 2 on Pareto front");
}

// ═══════════════════════════════════════════════════════════════════════════
// 25. Dominates: MAXIMIZE direction reversal
//     With MAXIMIZE, larger values are better
//     dominates([3,3],[1,1], MAX×MAX) = true
//     dominates([1,1],[3,3], MAX×MAX) = false
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_dominates_maximize() {
    let d = vec![StudyDirection::Maximize, StudyDirection::Maximize];

    assert!(dominates(&[3.0, 3.0], &[1.0, 1.0], &d));
    assert!(!dominates(&[1.0, 1.0], &[3.0, 3.0], &d));
    // Trade-off: neither dominates
    assert!(!dominates(&[3.0, 1.0], &[1.0, 3.0], &d));
    assert!(!dominates(&[1.0, 3.0], &[3.0, 1.0], &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 26. Dominates: mixed direction
//     MIN×MAX: [1,3] dominates [2,2] (better in both after direction)
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_dominates_mixed_direction() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Maximize];

    // [1,3] vs [2,2]: 1<2 (better for MIN), 3>2 (better for MAX) → dominates
    assert!(dominates(&[1.0, 3.0], &[2.0, 2.0], &d));
    assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0], &d));

    // [1,1] vs [2,3]: trade-off (1<2 better MIN, 1<3 worse MAX)
    assert!(!dominates(&[1.0, 1.0], &[2.0, 3.0], &d));
    assert!(!dominates(&[2.0, 3.0], &[1.0, 1.0], &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 27. Full pipeline: FNS → per-front CD → elite selection truncation
//     Simulates the NSGA-II elite_select method flow
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_full_elite_select_pipeline() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 6.0], vec![2.0, 5.0], vec![3.0, 4.0], // front 0
        vec![2.0, 6.0], vec![3.0, 5.0], vec![4.0, 4.0], // front 1
        vec![5.0, 6.0], vec![6.0, 5.0],                  // front 2
        vec![7.0, 7.0],                                   // front 3
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();

    // Step 1: FNS
    let fronts = fast_non_dominated_sort(&refs, &d);
    assert_eq!(fronts.len(), 4);
    assert_eq!(fronts[0].len(), 3); // front 0 = {0,1,2}
    assert_eq!(fronts[1].len(), 3); // front 1 = {3,4,5}
    assert_eq!(fronts[2].len(), 2); // front 2 = {6,7}
    assert_eq!(fronts[3].len(), 1); // front 3 = {8}

    // Step 2: Select top-5 (front0=3 + front1 needs truncation from 3→2)
    let n_select = 5;
    let mut selected = Vec::new();

    for front in &fronts {
        if selected.len() + front.len() <= n_select {
            selected.extend(front.iter());
        } else {
            // Truncation front: compute CD, sort descending, take remainder
            let front_trials: Vec<&FrozenTrial> = front.iter().map(|&i| &trials[i]).collect();
            let cd = crowding_distance(&front_trials, &d);

            let mut cd_indexed: Vec<(usize, f64)> =
                front.iter().zip(cd.iter()).map(|(&i, &c)| (i, c)).collect();
            cd_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let remain = n_select - selected.len();
            for &(idx, _) in cd_indexed.iter().take(remain) {
                selected.push(idx);
            }
            break;
        }
    }

    selected.sort();
    // front0 = {0,1,2}, then from front1, boundary trials (highest CD) selected
    // front1 = {3,4,5}, CD = {3:inf, 4:2.0, 5:inf} → pick 3 and 5
    assert_eq!(selected.len(), 5);
    assert!(selected.contains(&0));
    assert!(selected.contains(&1));
    assert!(selected.contains(&2));
    // From front1: 3(inf) and 5(inf) should be selected over 4(2.0)
    assert!(selected.contains(&3));
    assert!(selected.contains(&5));
    assert!(!selected.contains(&4));
}

// ═══════════════════════════════════════════════════════════════════════════
// 28. NotSet direction safety — should not panic, returns false
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_notset_direction_safety() {
    let d = vec![StudyDirection::NotSet, StudyDirection::Minimize];

    assert!(!dominates(&[1.0, 2.0], &[2.0, 3.0], &d));
    assert!(!dominates(&[2.0, 3.0], &[1.0, 2.0], &d));
}

// ═══════════════════════════════════════════════════════════════════════════
// 29. Constrained: all feasible reduces to unconstrained FNS
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_all_feasible_equals_unconstrained() {
    let d = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let trials: Vec<FrozenTrial> = [
        vec![1.0, 6.0], vec![2.0, 5.0], vec![3.0, 4.0],
        vec![2.0, 6.0], vec![3.0, 5.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| {
        make_constrained_trial(i as i64, v.clone(), vec![-1.0, -0.5])
    })
    .collect();

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let c_fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let c_ranks = fronts_to_ranks(&c_fronts, trials.len());

    // Re-create without constraints
    let plain: Vec<FrozenTrial> = [
        vec![1.0, 6.0], vec![2.0, 5.0], vec![3.0, 4.0],
        vec![2.0, 6.0], vec![3.0, 5.0],
    ]
    .iter()
    .enumerate()
    .map(|(i, v)| make_trial(i as i64, v.clone()))
    .collect();
    let p_refs: Vec<&FrozenTrial> = plain.iter().collect();
    let p_fronts = fast_non_dominated_sort(&p_refs, &d);
    let p_ranks = fronts_to_ranks(&p_fronts, plain.len());

    assert_eq!(c_ranks, p_ranks, "all-feasible constrained should equal unconstrained");
}

// ═══════════════════════════════════════════════════════════════════════════
// 30. Constrained: all infeasible — sorted purely by violation
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_all_infeasible_by_violation() {
    let d = vec![StudyDirection::Minimize];
    let trials = vec![
        make_constrained_trial(0, vec![1.0], vec![0.5, 0.3]),   // violation=0.8
        make_constrained_trial(1, vec![1.0], vec![0.1]),         // violation=0.1
        make_constrained_trial(2, vec![1.0], vec![0.3, 0.2]),   // violation=0.5
        make_constrained_trial(3, vec![1.0], vec![1.0]),         // violation=1.0
    ];

    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let fronts = constrained_fast_non_dominated_sort(&refs, &d);
    let ranks = fronts_to_ranks(&fronts, trials.len());

    // Order: 0.1(t1) < 0.5(t2) < 0.8(t0) < 1.0(t3)
    assert!(ranks[1] < ranks[2], "0.1 < 0.5");
    assert!(ranks[2] < ranks[0], "0.5 < 0.8");
    assert!(ranks[0] < ranks[3], "0.8 < 1.0");
}
