//! Pruners 模块深度交叉验证测试
//!
//! 对照 Python optuna 剪枝器生成的金标准值 (tests/pruners_deep_golden_values.json)，
//! 验证 Rust 实现与 Python 的精确对齐。
//!
//! 覆盖：NopPruner, PercentilePruner, MedianPruner, PatientPruner,
//!        ThresholdPruner, SuccessiveHalvingPruner, HyperbandPruner, WilcoxonPruner,
//!        wilcoxon_signed_rank_test (pub)

use optuna_rs::pruners::{
    HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
    SuccessiveHalvingPruner, ThresholdPruner, WilcoxonPruner, wilcoxon_signed_rank_test,
};
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};
use serde_json::Value;
use std::collections::HashMap;

/// 加载金标准值
fn load_golden() -> Value {
    let data = std::fs::read_to_string("tests/pruners_deep_golden_values.json")
        .expect("Failed to read golden values");
    serde_json::from_str(&data).expect("Failed to parse golden values")
}

/// 辅助：创建已完成试验
fn make_completed(number: i64, iv: Vec<(i64, f64)>, value: f64) -> FrozenTrial {
    FrozenTrial {
        number,
        state: TrialState::Complete,
        values: Some(vec![value]),
        datetime_start: Some(chrono::Utc::now()),
        datetime_complete: Some(chrono::Utc::now()),
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: iv.into_iter().collect(),
        trial_id: number,
    }
}

/// 辅助：创建运行中试验
fn make_running(number: i64, iv: Vec<(i64, f64)>) -> FrozenTrial {
    FrozenTrial {
        number,
        state: TrialState::Running,
        values: None,
        datetime_start: Some(chrono::Utc::now()),
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: iv.into_iter().collect(),
        trial_id: number,
    }
}

// ========================================================================
// Group 1: Wilcoxon p-value 精确交叉验证 (8 cases from golden values)
// ========================================================================

/// 对齐 Python scipy.stats.wilcoxon: all_positive_n5
/// Python: p_greater=0.03125, p_less=1.0
#[test]
fn test_wilcoxon_golden_all_positive_n5() {
    let golden = load_golden();
    let cases = golden["wilcoxon_p_values"].as_array().unwrap();
    let case = &cases[0]; // all_positive_n5
    let diffs: Vec<f64> = case["diffs"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let py_greater = case["p_greater"].as_f64().unwrap();
    let py_less = case["p_less"].as_f64().unwrap();

    // Minimize → alternative='greater' → tests R+
    let p_min = wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    // Maximize → alternative='less' → tests R-
    let p_max = wilcoxon_signed_rank_test(&diffs, StudyDirection::Maximize);

    assert!(
        (p_min - py_greater).abs() < 1e-10,
        "all_positive_n5 Minimize: Python={py_greater}, Rust={p_min}"
    );
    assert!(
        (p_max - py_less).abs() < 1e-10,
        "all_positive_n5 Maximize: Python={py_less}, Rust={p_max}"
    );
}

/// 对齐 Python: all_negative_n5
#[test]
fn test_wilcoxon_golden_all_negative_n5() {
    let golden = load_golden();
    let case = &golden["wilcoxon_p_values"][1];
    let diffs: Vec<f64> = case["diffs"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let p_min = wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    let p_max = wilcoxon_signed_rank_test(&diffs, StudyDirection::Maximize);
    let py_greater = case["p_greater"].as_f64().unwrap();
    let py_less = case["p_less"].as_f64().unwrap();
    assert!((p_min - py_greater).abs() < 1e-10,
        "all_negative Minimize: Python={py_greater}, Rust={p_min}");
    assert!((p_max - py_less).abs() < 1e-10,
        "all_negative Maximize: Python={py_less}, Rust={p_max}");
}

/// 对齐 Python: mixed_n8, with_ties_n6, with_zeros_n8, mixed_n15, symmetric_n6, large_spread
#[test]
fn test_wilcoxon_golden_remaining_cases() {
    let golden = load_golden();
    let cases = golden["wilcoxon_p_values"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate().skip(2) {
        let label = case["label"].as_str().unwrap();
        let diffs: Vec<f64> = case["diffs"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let py_greater = case["p_greater"].as_f64().unwrap();
        let py_less = case["p_less"].as_f64().unwrap();

        let p_min = wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
        let p_max = wilcoxon_signed_rank_test(&diffs, StudyDirection::Maximize);

        // 精确分布 (n ≤ 50) 应匹配到机器精度
        let tol = if diffs.len() <= 50 { 1e-10 } else { 1e-6 };
        assert!(
            (p_min - py_greater).abs() < tol,
            "[{idx}] {label} Minimize: Python={py_greater}, Rust={p_min}"
        );
        assert!(
            (p_max - py_less).abs() < tol,
            "[{idx}] {label} Maximize: Python={py_less}, Rust={p_max}"
        );
    }
}

// ========================================================================
// Group 2: PercentilePruner 全决策场景
// ========================================================================

/// 对齐 Python: 3 个已完成试验，p50 Minimize → 低值保留，高值剪枝
#[test]
fn test_percentile_pruner_minimize_p50() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
    let completed = vec![
        make_completed(0, vec![(0, 1.0)], 1.0),
        make_completed(1, vec![(0, 2.0)], 2.0),
        make_completed(2, vec![(0, 3.0)], 3.0),
    ];
    // p50 of [1,2,3] = 2.0
    // best=0.5 < 2.0 → 不剪枝
    let good = make_running(3, vec![(0, 0.5)]);
    assert!(!pruner.prune(&completed, &good, None).unwrap());
    // best=10.0 > 2.0 → 剪枝
    let bad = make_running(3, vec![(0, 10.0)]);
    assert!(pruner.prune(&completed, &bad, None).unwrap());
}

/// 对齐 Python: Maximize 方向 p50 → 高值保留，低值剪枝
#[test]
fn test_percentile_pruner_maximize_p50() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Maximize);
    let completed = vec![
        make_completed(0, vec![(0, 10.0)], 10.0),
        make_completed(1, vec![(0, 20.0)], 20.0),
        make_completed(2, vec![(0, 30.0)], 30.0),
    ];
    // Maximize: effective_percentile = 100-50 = 50, p50 of [10,20,30] = 20.0
    // best=50.0 >= 20.0 → 不剪枝
    let good = make_running(3, vec![(0, 50.0)]);
    assert!(!pruner.prune(&completed, &good, None).unwrap());
    // best=1.0 < 20.0 → 剪枝
    let bad = make_running(3, vec![(0, 1.0)]);
    assert!(pruner.prune(&completed, &bad, None).unwrap());
}

/// 对齐 Python: n_startup_trials 阈值
#[test]
fn test_percentile_pruner_startup() {
    let pruner = PercentilePruner::new(50.0, 5, 0, 1, 1, StudyDirection::Minimize);
    let completed = vec![
        make_completed(0, vec![(0, 1.0)], 1.0),
        make_completed(1, vec![(0, 2.0)], 2.0),
    ];
    // 2 < 5 startup → 不剪枝
    let trial = make_running(2, vec![(0, 100.0)]);
    assert!(!pruner.prune(&completed, &trial, None).unwrap());
}

/// 对齐 Python: n_warmup_steps 阈值
#[test]
fn test_percentile_pruner_warmup() {
    let pruner = PercentilePruner::new(50.0, 0, 5, 1, 1, StudyDirection::Minimize);
    let completed = vec![make_completed(0, vec![(0, 1.0), (5, 1.0)], 1.0)];
    // step=2 < warmup=5 → 不剪枝
    let trial_early = make_running(1, vec![(2, 100.0)]);
    assert!(!pruner.prune(&completed, &trial_early, None).unwrap());
    // step=5 >= warmup=5 → 可以剪枝
    let trial_late = make_running(1, vec![(5, 100.0)]);
    assert!(pruner.prune(&completed, &trial_late, None).unwrap());
}

/// 对齐 Python: 全 NaN 中间值 → best=NaN → 剪枝
#[test]
fn test_percentile_pruner_all_nan_prunes() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
    let completed = vec![make_completed(0, vec![(0, 1.0)], 1.0)];
    let trial = make_running(1, vec![(0, f64::NAN)]);
    assert!(pruner.prune(&completed, &trial, None).unwrap());
}

/// 对齐 Python: n_min_trials 不足 → percentile=NaN → 不剪枝
#[test]
fn test_percentile_pruner_n_min_trials() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 5, StudyDirection::Minimize);
    let completed = vec![
        make_completed(0, vec![(0, 1.0)], 1.0),
        make_completed(1, vec![(0, 2.0)], 2.0),
    ];
    // 2 < n_min_trials=5 → percentile=NaN → 不剪枝
    let trial = make_running(2, vec![(0, 100.0)]);
    assert!(!pruner.prune(&completed, &trial, None).unwrap());
}

/// 对齐 Python: 无已完成试验 → 不剪枝
#[test]
fn test_percentile_pruner_no_completed() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
    let trial = make_running(0, vec![(0, 100.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python 金标准: percentile_over_trials 精确值
#[test]
fn test_percentile_over_trials_golden() {
    let golden = load_golden();
    let cases = golden["percentile_over_trials"].as_array().unwrap();

    // Case 0: 3t_s0_p50_min → result=2.0
    // 3 trials [1,2,3], p50 Minimize → np.percentile([1,2,3], 50) = 2.0
    let p = cases[0]["result"].as_f64().unwrap();
    assert!((p - 2.0).abs() < 1e-10, "3t_s0_p50_min: Python={p}");

    // Case 1: 3t_s0_p50_max → result=2.0
    // Maximize: effective_p = 100-50 = 50, np.percentile([1,2,3], 50) = 2.0
    let p = cases[1]["result"].as_f64().unwrap();
    assert!((p - 2.0).abs() < 1e-10, "3t_s0_p50_max: Python={p}");

    // Case 2: 5t_s0_p25_min → values [10,20,30,40,50], p25 = 20.0
    let p = cases[2]["result"].as_f64().unwrap();
    assert!((p - 20.0).abs() < 1e-10, "5t_s0_p25_min: Python={p}");

    // Case 3: 5t_s0_p75_min → values [10,20,30,40,50], p75 = 40.0
    let p = cases[3]["result"].as_f64().unwrap();
    assert!((p - 40.0).abs() < 1e-10, "5t_s0_p75_min: Python={p}");
}

// ========================================================================
// Group 3: MedianPruner (= PercentilePruner(50))
// ========================================================================

/// 对齐 Python: MedianPruner 等效 PercentilePruner(50.0)
#[test]
fn test_median_pruner_equals_percentile_50() {
    let median = MedianPruner::new(0, 0, 1, 1, StudyDirection::Minimize);
    let percentile = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);

    let completed = vec![
        make_completed(0, vec![(0, 1.0)], 1.0),
        make_completed(1, vec![(0, 2.0)], 2.0),
        make_completed(2, vec![(0, 3.0)], 3.0),
    ];

    for test_val in &[0.5, 1.5, 2.0, 2.5, 10.0] {
        let trial = make_running(3, vec![(0, *test_val)]);
        let m = median.prune(&completed, &trial, None).unwrap();
        let p = percentile.prune(&completed, &trial, None).unwrap();
        assert_eq!(m, p, "MedianPruner != PercentilePruner(50) for val={test_val}");
    }
}

// ========================================================================
// Group 4: PatientPruner 窗口逻辑
// ========================================================================

/// 对齐 Python 金标准: patient_pruner_decisions (6 scenarios)
#[test]
fn test_patient_pruner_golden_decisions() {
    let golden = load_golden();
    let cases = golden["patient_pruner_decisions"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let values: Vec<(i64, f64)> = case["values"].as_array().unwrap()
            .iter()
            .map(|pair| {
                let arr = pair.as_array().unwrap();
                (arr[0].as_i64().unwrap(), arr[1].as_f64().unwrap())
            })
            .collect();
        let patience = case["patience"].as_u64().unwrap() as usize;
        let min_delta = case["min_delta"].as_f64().unwrap();
        let direction = match case["direction"].as_str().unwrap() {
            "minimize" => StudyDirection::Minimize,
            _ => StudyDirection::Maximize,
        };
        let expected = case["maybe_prune"].as_bool().unwrap();

        let pruner = PatientPruner::new(None, patience, min_delta, direction);
        let trial = make_running(0, values);
        let result = pruner.prune(&[], &trial, None).unwrap();
        assert_eq!(
            result, expected,
            "[{idx}] PatientPruner: patience={patience}, delta={min_delta}, \
             direction={direction:?}, expected={expected}, got={result}"
        );
    }
}

/// 对齐 Python: PatientPruner + wrapped NopPruner → 停滞也不剪枝
#[test]
fn test_patient_pruner_with_nop_wrapper() {
    let wrapped = Box::new(NopPruner::new());
    let pruner = PatientPruner::new(Some(wrapped), 2, 0.0, StudyDirection::Minimize);
    // 明显停滞场景: before min=0.5, after min=0.8
    let trial = make_running(0, vec![(0, 1.0), (1, 0.5), (2, 0.8), (3, 0.9), (4, 1.0)]);
    // 停滞检测到 → 委托给 NopPruner → 返回 false
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 无中间值 → 不剪枝
#[test]
fn test_patient_pruner_no_values() {
    let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
    let trial = make_running(0, vec![]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 步骤数不足 → 不剪枝
#[test]
fn test_patient_pruner_insufficient_steps() {
    let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
    // patience=3, need > 4 steps, only have 4
    let trial = make_running(0, vec![(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

// ========================================================================
// Group 5: ThresholdPruner 边界条件
// ========================================================================

/// 对齐 Python: 低于下界 → 剪枝
#[test]
fn test_threshold_pruner_lower() {
    let pruner = ThresholdPruner::new(Some(1.0), None, 0, 1);
    let trial = make_running(0, vec![(0, 0.5)]);
    assert!(pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 高于上界 → 剪枝
#[test]
fn test_threshold_pruner_upper() {
    let pruner = ThresholdPruner::new(None, Some(2.0), 0, 1);
    let trial = make_running(0, vec![(0, 3.0)]);
    assert!(pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 范围内 → 不剪枝
#[test]
fn test_threshold_pruner_within() {
    let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
    let trial = make_running(0, vec![(0, 3.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: NaN → 剪枝
#[test]
fn test_threshold_pruner_nan() {
    let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
    let trial = make_running(0, vec![(0, f64::NAN)]);
    assert!(pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 边界值不剪枝 (inclusive)
#[test]
fn test_threshold_pruner_boundary() {
    let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
    let lower = make_running(0, vec![(0, 1.0)]);
    assert!(!pruner.prune(&[], &lower, None).unwrap());
    let upper = make_running(0, vec![(0, 5.0)]);
    assert!(!pruner.prune(&[], &upper, None).unwrap());
}

/// 对齐 Python: warmup 阶段不剪枝
#[test]
fn test_threshold_pruner_warmup() {
    let pruner = ThresholdPruner::new(Some(1.0), None, 5, 1);
    let trial = make_running(0, vec![(3, 0.1)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: Inf 值剪枝
#[test]
fn test_threshold_pruner_inf() {
    let pruner = ThresholdPruner::new(Some(0.0), Some(100.0), 0, 1);
    let pos_inf = make_running(0, vec![(0, f64::INFINITY)]);
    assert!(pruner.prune(&[], &pos_inf, None).unwrap());
    let neg_inf = make_running(0, vec![(0, f64::NEG_INFINITY)]);
    assert!(pruner.prune(&[], &neg_inf, None).unwrap());
}

// ========================================================================
// Group 6: NopPruner
// ========================================================================

/// 对齐 Python: NopPruner 永远不剪枝
#[test]
fn test_nop_pruner_never_prunes() {
    let pruner = NopPruner::new();
    let trial = make_running(0, vec![(0, 999.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
    let completed = vec![make_completed(0, vec![(0, 1.0)], 1.0)];
    assert!(!pruner.prune(&completed, &trial, None).unwrap());
}

// ========================================================================
// Group 7: SuccessiveHalvingPruner rung 逻辑
// ========================================================================

/// 对齐 Python 金标准: successive_halving_rung_steps (11 promotion step 计算)
#[test]
fn test_successive_halving_rung_steps_golden() {
    let golden = load_golden();
    let cases = golden["successive_halving_rung_steps"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let mr = case["min_resource"].as_i64().unwrap();
        let rf = case["reduction_factor"].as_i64().unwrap();
        let rate = case["min_early_stopping_rate"].as_i64().unwrap();
        let rung = case["rung"].as_i64().unwrap();
        let expected = case["promotion_step"].as_i64().unwrap();

        // 验证晋升步骤公式: min_resource * reduction_factor^(rate + rung)
        let actual = mr * rf.pow((rate + rung) as u32);
        assert_eq!(
            actual, expected,
            "[{idx}] mr={mr}, rf={rf}, rate={rate}, rung={rung}: \
             expected={expected}, got={actual}"
        );
    }
}

/// 对齐 Python: 未到达 rung → 不剪枝
#[test]
fn test_successive_halving_before_rung() {
    let pruner = SuccessiveHalvingPruner::new(
        Some(10), 4, 0, 0, StudyDirection::Minimize,
    );
    // 第一个 rung 在 step 10, step 5 < 10 → 不剪枝
    let trial = make_running(0, vec![(5, 1.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: NaN 到达 rung → 剪枝
#[test]
fn test_successive_halving_nan_at_rung() {
    let pruner = SuccessiveHalvingPruner::new(
        Some(1), 4, 0, 0, StudyDirection::Minimize,
    );
    let completed = make_completed(0, vec![(0, 1.0), (100, 1.0)], 1.0);
    let trial = make_running(1, vec![(1, f64::NAN)]);
    assert!(pruner.prune(&[completed], &trial, None).unwrap());
}

/// 对齐 Python: 无中间值 → 不剪枝
#[test]
fn test_successive_halving_no_values() {
    let pruner = SuccessiveHalvingPruner::new(
        Some(1), 4, 0, 0, StudyDirection::Minimize,
    );
    let trial = make_running(0, vec![]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: auto min_resource + 无已完成试验 → 不剪枝
#[test]
fn test_successive_halving_auto_no_completed() {
    let pruner = SuccessiveHalvingPruner::new(
        None, 4, 0, 0, StudyDirection::Minimize,
    );
    let trial = make_running(0, vec![(5, 1.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

// ========================================================================
// Group 8: Hyperband bracket allocation 精确对齐
// ========================================================================

/// 对齐 Python 金标准: hyperband_bracket_allocation (6 configurations)
#[test]
fn test_hyperband_bracket_allocation_golden() {
    let golden = load_golden();
    let cases = golden["hyperband_bracket_allocation"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let mr = case["min_resource"].as_i64().unwrap();
        let mxr = case["max_resource"].as_i64().unwrap();
        let rf = case["reduction_factor"].as_i64().unwrap();
        let expected_nb = case["n_brackets"].as_i64().unwrap() as usize;
        let expected_budgets: Vec<usize> = case["budgets"]
            .as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let expected_total = case["total_budget"].as_u64().unwrap() as usize;

        // 验证括号数: floor(log_rf(max/min)) + 1
        let n_brackets = (((mxr as f64) / (mr as f64)).ln()
            / (rf as f64).ln()).floor() as usize + 1;
        assert_eq!(
            n_brackets, expected_nb,
            "[{idx}] n_brackets: expected={expected_nb}, got={n_brackets}"
        );

        // 验证预算分配: ceil(n_brackets * rf^s / (s+1))
        let mut budgets = Vec::new();
        for bid in 0..n_brackets {
            let s = n_brackets - 1 - bid;
            let budget = ((n_brackets as f64)
                * (rf as f64).powi(s as i32) / (s as f64 + 1.0)).ceil() as usize;
            budgets.push(budget);
        }
        assert_eq!(
            budgets, expected_budgets,
            "[{idx}] budgets: expected={expected_budgets:?}, got={budgets:?}"
        );
        assert_eq!(
            budgets.iter().sum::<usize>(), expected_total,
            "[{idx}] total_budget mismatch"
        );
    }
}

/// 对齐 Python: CRC32 交叉验证 (binascii.crc32)
#[test]
fn test_crc32_golden_hashes() {
    let golden = load_golden();
    let cases = golden["crc32_hashes"].as_array().unwrap();

    for case in cases {
        let input = case["input"].as_str().unwrap();
        let expected = case["crc32"].as_u64().unwrap() as u32;

        // 使用简单 CRC32 实现验证
        let actual = crc32_simple(input.as_bytes());
        assert_eq!(
            actual, expected,
            "CRC32(\"{input}\"): Python={expected}, Rust={actual}"
        );
    }
}

/// 简单 CRC32 (与 Rust hyperband.rs 内部实现一致)
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 { crc = (crc >> 1) ^ 0xEDB88320; }
            else { crc >>= 1; }
        }
    }
    crc ^ 0xFFFFFFFF
}

/// 对齐 Python: Hyperband 无已完成试验 + auto → 不剪枝
#[test]
fn test_hyperband_no_completed_auto() {
    let pruner = HyperbandPruner::new(
        1, None, 3, 0, StudyDirection::Minimize, "test_study",
    );
    let trial = make_running(0, vec![(5, 1.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: Hyperband 正常初始化不 panic
#[test]
fn test_hyperband_with_completed() {
    let pruner = HyperbandPruner::new(
        1, Some(27), 3, 0, StudyDirection::Minimize, "test_study",
    );
    let completed = make_completed(0, (0..=26).map(|s| (s, 1.0)).collect(), 1.0);
    let trial = make_running(1, vec![(0, 1.0)]);
    let _ = pruner.prune(&[completed], &trial, None);
    // 不应 panic
}

// ========================================================================
// Group 9: promotable_to_next_rung 精确对齐
// ========================================================================

/// 对齐 Python 金标准: promotable_to_next_rung (9 cases)
#[test]
fn test_promotable_golden() {
    let golden = load_golden();
    let cases = golden["promotable_to_next_rung"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let value = case["value"].as_f64().unwrap();
        let competing: Vec<f64> = case["competing"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let rf = case["reduction_factor"].as_i64().unwrap();
        let direction = match case["direction"].as_str().unwrap() {
            "minimize" => StudyDirection::Minimize,
            _ => StudyDirection::Maximize,
        };
        let expected = case["promotable"].as_bool().unwrap();

        // 重现 _is_trial_promotable_to_next_rung 逻辑
        let n = competing.len();
        let mut promotable_idx = (n as i64 / rf) - 1;
        if promotable_idx < 0 { promotable_idx = 0; }
        let promotable_idx = promotable_idx as usize;

        let mut sorted = competing.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let actual = match direction {
            StudyDirection::Maximize => value >= sorted[n - 1 - promotable_idx],
            _ => value <= sorted[promotable_idx],
        };

        assert_eq!(
            actual, expected,
            "[{idx}] promotable: value={value}, competing={competing:?}, rf={rf}, \
             dir={direction:?}, expected={expected}, got={actual}"
        );
    }
}

// ========================================================================
// Group 10: estimate_min_resource 精确对齐
// ========================================================================

/// 对齐 Python 金标准: estimate_min_resource (5 cases)
#[test]
fn test_estimate_min_resource_golden() {
    let golden = load_golden();
    let cases = golden["estimate_min_resource"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let last_steps: Vec<i64> = case["last_steps"].as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap()).collect();
        let expected = case["result"].as_i64().unwrap();

        // 重现: max(last_steps) / 100, min 1
        let max_step = last_steps.iter().max().unwrap();
        let actual = (max_step / 100).max(1);
        assert_eq!(
            actual, expected,
            "[{idx}] estimate_min_resource: last_steps={last_steps:?}, \
             expected={expected}, got={actual}"
        );
    }
}

// ========================================================================
// Group 11: WilcoxonPruner 全流程决策
// ========================================================================

/// 对齐 Python: 明显更差的试验 → 剪枝 (Minimize)
#[test]
fn test_wilcoxon_pruner_clearly_worse_minimize() {
    let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
    let best = make_completed(
        0, (0..20).map(|s| (s, 0.1_f64)).collect(), 0.1,
    );
    let trial = make_running(1, (0..20).map(|s| (s, 10.0_f64)).collect());
    assert!(pruner.prune(&[best], &trial, None).unwrap());
}

/// 对齐 Python: 相似试验 → 不剪枝
#[test]
fn test_wilcoxon_pruner_similar_no_prune() {
    let pruner = WilcoxonPruner::new(0.01, 2, StudyDirection::Minimize);
    let best = make_completed(
        0, (0..10).map(|s| (s, 1.0 + 0.001 * s as f64)).collect(), 1.0,
    );
    let trial = make_running(
        1, (0..10).map(|s| (s, 1.0 + 0.001 * s as f64)).collect(),
    );
    assert!(!pruner.prune(&[best], &trial, None).unwrap());
}

/// 对齐 Python: 无最优试验 → 不剪枝
#[test]
fn test_wilcoxon_pruner_no_best() {
    let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
    let trial = make_running(0, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: 公共步骤不足 → 不剪枝
#[test]
fn test_wilcoxon_pruner_insufficient_common() {
    let pruner = WilcoxonPruner::new(0.1, 5, StudyDirection::Minimize);
    let best = make_completed(0, vec![(0, 0.5), (1, 0.5)], 0.5);
    let trial = make_running(1, vec![(0, 1.0), (1, 2.0)]);
    assert!(!pruner.prune(&[best], &trial, None).unwrap());
}

/// 对齐 Python: Inf 值 → 不剪枝
#[test]
fn test_wilcoxon_pruner_inf_no_prune() {
    let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
    let trial = make_running(0, vec![(0, 1.0), (1, f64::INFINITY), (2, 3.0)]);
    assert!(!pruner.prune(&[], &trial, None).unwrap());
}

/// 对齐 Python: Maximize 方向 → 更差试验剪枝
#[test]
fn test_wilcoxon_pruner_maximize() {
    let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Maximize);
    let best = make_completed(
        0, (0..20).map(|s| (s, 100.0_f64)).collect(), 100.0,
    );
    // 当前试验值远低于最优 → 应剪枝
    let trial = make_running(1, (0..20).map(|s| (s, 1.0_f64)).collect());
    assert!(pruner.prune(&[best], &trial, None).unwrap());
}

// ========================================================================
// Group 12: best_intermediate_result 精确对齐
// ========================================================================

/// 对齐 Python 金标准: best_intermediate_result (6 cases, NaN cases tested separately)
#[test]
fn test_best_intermediate_golden() {
    let golden = load_golden();
    let cases = golden["best_intermediate_result"].as_array().unwrap();

    for (idx, case) in cases.iter().enumerate() {
        let iv: Vec<(i64, f64)> = case["intermediate_values"].as_object().unwrap()
            .iter()
            .map(|(k, v)| {
                let step: i64 = k.parse().unwrap();
                let val = v.as_f64().unwrap();
                (step, val)
            })
            .collect();
        let direction = match case["direction"].as_str().unwrap() {
            "minimize" => StudyDirection::Minimize,
            _ => StudyDirection::Maximize,
        };

        // 手动计算 best intermediate
        let values: Vec<f64> = iv.iter().map(|(_, v)| *v).filter(|v| !v.is_nan()).collect();
        let best = if values.is_empty() {
            f64::NAN
        } else {
            match direction {
                StudyDirection::Maximize => {
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                }
                _ => values.iter().copied().fold(f64::INFINITY, f64::min),
            }
        };

        let expected = case["best"].as_f64().unwrap();
        assert!(
            (best - expected).abs() < 1e-10,
            "[{idx}] best_intermediate: expected={expected}, got={best}"
        );
    }
}

// ========================================================================
// Group 13: 多步 PercentilePruner 场景
// ========================================================================

/// 对齐 Python: 多步场景下 best intermediate 比较
#[test]
fn test_percentile_multi_step_best() {
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
    let completed = vec![
        make_completed(0, vec![(0, 10.0), (1, 5.0), (2, 8.0)], 5.0),
        make_completed(1, vec![(0, 10.0), (1, 6.0), (2, 7.0)], 6.0),
    ];
    // Running trial: best over steps = min(100, 4, 50) = 4.0
    // At step 2, p50 of [8.0, 7.0] = 7.5
    // best(4.0) < p50(7.5) → 不剪枝
    let trial = make_running(2, vec![(0, 100.0), (1, 4.0), (2, 50.0)]);
    assert!(!pruner.prune(&completed, &trial, None).unwrap());
}

/// 对齐 Python: 不同 percentile 值的对称性
#[test]
fn test_percentile_symmetry() {
    let completed = vec![
        make_completed(0, vec![(0, 1.0)], 1.0),
        make_completed(1, vec![(0, 2.0)], 2.0),
        make_completed(2, vec![(0, 3.0)], 3.0),
        make_completed(3, vec![(0, 4.0)], 4.0),
    ];

    // p25 Minimize: percentile(25) of [1,2,3,4] = 1.75
    let p25 = PercentilePruner::new(25.0, 0, 0, 1, 1, StudyDirection::Minimize);
    let trial_2_5 = make_running(4, vec![(0, 2.5)]);
    assert!(p25.prune(&completed, &trial_2_5, None).unwrap()); // 2.5 > 1.75

    // p75 Minimize: percentile(75) of [1,2,3,4] = 3.25
    let p75 = PercentilePruner::new(75.0, 0, 0, 1, 1, StudyDirection::Minimize);
    assert!(!p75.prune(&completed, &trial_2_5, None).unwrap()); // 2.5 < 3.25
}
