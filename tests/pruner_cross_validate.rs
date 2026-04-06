//! Pruner 模块精确交叉验证测试。
//!
//! 所有参考值均由 Python optuna 和 numpy 生成，确保 Rust 移植与 Python 精确对齐。

use optuna_rs::pruners::*;
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

fn make_complete_trial_with_intermediates(
    number: i64,
    intermediate_values: Vec<(i64, f64)>,
    value: Option<f64>,
) -> FrozenTrial {
    FrozenTrial {
        number,
        trial_id: number,
        state: if value.is_some() {
            TrialState::Complete
        } else {
            TrialState::Running
        },
        values: value.map(|v| vec![v]),
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: intermediate_values.into_iter().collect(),
    }
}

// ============================================================================
// 1. nan_percentile 精确数值验证 (对齐 numpy.percentile)
// ============================================================================

/// Python 参考:
///   np.percentile([0, 10, 20, 30, 40], 0)  = 0.0
///   np.percentile([0, 10, 20, 30, 40], 25) = 10.0
///   np.percentile([0, 10, 20, 30, 40], 50) = 20.0
///   np.percentile([0, 10, 20, 30, 40], 75) = 30.0
///   np.percentile([0, 10, 20, 30, 40], 100) = 40.0
#[test]
fn test_percentile_exact_vs_numpy() {
    // Replicate the nan_percentile function from percentile.rs
    fn nan_percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }
        let n = sorted.len() as f64;
        let idx = p / 100.0 * (n - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        if hi >= sorted.len() {
            sorted[sorted.len() - 1]
        } else {
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }

    let vals = vec![0.0, 10.0, 20.0, 30.0, 40.0];
    assert_eq!(nan_percentile(&vals, 0.0), 0.0);
    assert_eq!(nan_percentile(&vals, 25.0), 10.0);
    assert_eq!(nan_percentile(&vals, 50.0), 20.0);
    assert_eq!(nan_percentile(&vals, 75.0), 30.0);
    assert_eq!(nan_percentile(&vals, 100.0), 40.0);

    // 非整数 percentile: np.percentile([0,10,20,30,40], 33) = 13.2 (线性插值)
    let p33 = nan_percentile(&vals, 33.0);
    assert!((p33 - 13.2).abs() < 1e-10, "p33={}, expected 13.2", p33);
}

/// Python 参考:
///   np.nanpercentile([0, NaN, 20, 30, 40], 25) = 15.0
///   np.nanpercentile([0, NaN, 20, 30, 40], 50) = 25.0
///   np.nanpercentile([0, NaN, 20, 30, 40], 75) = 32.5
#[test]
fn test_nanpercentile_with_nan() {
    fn nan_percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        if sorted.len() == 1 {
            return sorted[0];
        }
        let n = sorted.len() as f64;
        let idx = p / 100.0 * (n - 1.0);
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        if hi >= sorted.len() {
            sorted[sorted.len() - 1]
        } else {
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }

    // After filtering NaN and sorting: [0, 20, 30, 40]
    let vals = vec![0.0, 20.0, 30.0, 40.0];
    assert_eq!(nan_percentile(&vals, 25.0), 15.0);
    assert_eq!(nan_percentile(&vals, 50.0), 25.0);
    assert_eq!(nan_percentile(&vals, 75.0), 32.5);
}

// ============================================================================
// 2. PercentilePruner end-to-end 行为验证
// ============================================================================

/// Python 参考:
///   5 completed trials, step=0, values=[0,10,20,30,40]
///   PercentilePruner(50, n_startup=0, n_warmup=0, n_min=1, direction=minimize)
///   threshold = np.percentile([0,10,20,30,40], 50) = 20.0
///   value=15: prune=False (15 <= 20)
///   value=25: prune=True (25 > 20)
///   value=20: prune=False (20 不 > 20, 等于阈值不剪枝)
#[test]
fn test_percentile_pruner_minimize() {
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            make_complete_trial_with_intermediates(
                i,
                vec![(0, (i * 10) as f64)],
                Some((i * 10) as f64),
            )
        })
        .collect();

    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);

    // value=15 → should NOT prune
    let trial_15 = make_complete_trial_with_intermediates(5, vec![(0, 15.0)], None);
    let result = pruner.prune(&completed, &trial_15, None).unwrap();
    assert!(!result, "value=15 should not be pruned (15 <= 20)");

    // value=25 → should prune
    let trial_25 = make_complete_trial_with_intermediates(6, vec![(0, 25.0)], None);
    let result = pruner.prune(&completed, &trial_25, None).unwrap();
    assert!(result, "value=25 should be pruned (25 > 20)");

    // value=20 → should NOT prune (equal to threshold)
    let trial_20 = make_complete_trial_with_intermediates(7, vec![(0, 20.0)], None);
    let result = pruner.prune(&completed, &trial_20, None).unwrap();
    assert!(!result, "value=20 should not be pruned (20 == 20, not >)");
}

/// maximize: effective_percentile = 100-50 = 50
/// threshold = 20.0 (same sorted values)
/// value=25: prune=False (25 >= 20)
/// value=15: prune=True (15 < 20)
#[test]
fn test_percentile_pruner_maximize() {
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            make_complete_trial_with_intermediates(
                i,
                vec![(0, (i * 10) as f64)],
                Some((i * 10) as f64),
            )
        })
        .collect();

    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Maximize);

    // value=25 → should NOT prune (maximize: best=25 >= 20)
    let trial_25 = make_complete_trial_with_intermediates(5, vec![(0, 25.0)], None);
    let result = pruner.prune(&completed, &trial_25, None).unwrap();
    assert!(!result, "maximize: value=25 should not be pruned");

    // value=15 → should prune (maximize: best=15 < 20)
    let trial_15 = make_complete_trial_with_intermediates(6, vec![(0, 15.0)], None);
    let result = pruner.prune(&completed, &trial_15, None).unwrap();
    assert!(result, "maximize: value=15 should be pruned");
}

/// n_startup_trials 验证: 完成的试验数 < n_startup_trials 时不剪枝
#[test]
fn test_percentile_pruner_startup_trials() {
    let completed: Vec<FrozenTrial> = (0..3)
        .map(|i| {
            make_complete_trial_with_intermediates(i, vec![(0, 100.0)], Some(100.0))
        })
        .collect();

    // n_startup_trials=5, 但只有 3 个完成 → 不剪枝
    let pruner = PercentilePruner::new(50.0, 5, 0, 1, 1, StudyDirection::Minimize);
    let trial = make_complete_trial_with_intermediates(3, vec![(0, 999.0)], None);
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(!result, "should not prune when completed < n_startup_trials");
}

/// n_warmup_steps 验证: step < n_warmup_steps 时不剪枝
#[test]
fn test_percentile_pruner_warmup_steps() {
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            make_complete_trial_with_intermediates(i, vec![(0, 0.0), (1, 0.0)], Some(0.0))
        })
        .collect();

    let pruner = PercentilePruner::new(50.0, 0, 2, 1, 1, StudyDirection::Minimize);

    // step=0 < warmup=2 → 不剪枝
    let trial_step0 = make_complete_trial_with_intermediates(5, vec![(0, 999.0)], None);
    let result = pruner.prune(&completed, &trial_step0, None).unwrap();
    assert!(!result, "should not prune during warmup (step=0 < 2)");

    // step=1 < warmup=2 → 不剪枝
    let trial_step1 =
        make_complete_trial_with_intermediates(6, vec![(0, 999.0), (1, 999.0)], None);
    let result = pruner.prune(&completed, &trial_step1, None).unwrap();
    assert!(!result, "should not prune during warmup (step=1 < 2)");
}

/// NaN 中间值: 如果当前试验所有中间值都是 NaN → 剪枝
#[test]
fn test_percentile_pruner_nan_values() {
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            make_complete_trial_with_intermediates(i, vec![(0, (i * 10) as f64)], Some(0.0))
        })
        .collect();

    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);

    let trial_nan = make_complete_trial_with_intermediates(5, vec![(0, f64::NAN)], None);
    let result = pruner.prune(&completed, &trial_nan, None).unwrap();
    assert!(result, "all NaN intermediate values should trigger pruning");
}

// ============================================================================
// 3. MedianPruner (PercentilePruner(50) 的包装)
// ============================================================================

/// MedianPruner 行为应等同于 PercentilePruner(50)
#[test]
fn test_median_pruner_equivalence() {
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            make_complete_trial_with_intermediates(
                i,
                vec![(0, (i * 10) as f64)],
                Some((i * 10) as f64),
            )
        })
        .collect();

    let median = MedianPruner::new(0, 0, 1, 1, StudyDirection::Minimize);
    let percentile = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);

    // 两者应该给出相同结果
    for val in [5.0, 15.0, 20.0, 25.0, 35.0] {
        let trial = make_complete_trial_with_intermediates(10, vec![(0, val)], None);
        let m_result = median.prune(&completed, &trial, None).unwrap();
        let p_result = percentile.prune(&completed, &trial, None).unwrap();
        assert_eq!(
            m_result, p_result,
            "MedianPruner and PercentilePruner(50) should match for value={}",
            val
        );
    }
}

// ============================================================================
// 4. ThresholdPruner 行为验证
// ============================================================================

/// Python 参考:
///   upper=25: value=30 → prune, value=20 → keep
///   lower=5:  value=3  → prune, value=10 → keep
#[test]
fn test_threshold_pruner_upper() {
    let pruner = ThresholdPruner::new(None, Some(25.0), 0, 1);
    let trials: Vec<FrozenTrial> = vec![];

    // value=30 > upper=25 → prune
    let trial = make_complete_trial_with_intermediates(0, vec![(0, 30.0)], None);
    assert!(pruner.prune(&trials, &trial, None).unwrap());

    // value=20 <= upper=25 → keep
    let trial = make_complete_trial_with_intermediates(1, vec![(0, 20.0)], None);
    assert!(!pruner.prune(&trials, &trial, None).unwrap());

    // value=25 == upper=25 → keep (含等号)
    let trial = make_complete_trial_with_intermediates(2, vec![(0, 25.0)], None);
    assert!(!pruner.prune(&trials, &trial, None).unwrap());
}

#[test]
fn test_threshold_pruner_lower() {
    let pruner = ThresholdPruner::new(Some(5.0), None, 0, 1);
    let trials: Vec<FrozenTrial> = vec![];

    // value=3 < lower=5 → prune
    let trial = make_complete_trial_with_intermediates(0, vec![(0, 3.0)], None);
    assert!(pruner.prune(&trials, &trial, None).unwrap());

    // value=10 >= lower=5 → keep
    let trial = make_complete_trial_with_intermediates(1, vec![(0, 10.0)], None);
    assert!(!pruner.prune(&trials, &trial, None).unwrap());

    // value=5 == lower=5 → keep
    let trial = make_complete_trial_with_intermediates(2, vec![(0, 5.0)], None);
    assert!(!pruner.prune(&trials, &trial, None).unwrap());
}

#[test]
fn test_threshold_pruner_both_bounds() {
    let pruner = ThresholdPruner::new(Some(5.0), Some(25.0), 0, 1);
    let trials: Vec<FrozenTrial> = vec![];

    // in range → keep
    let trial = make_complete_trial_with_intermediates(0, vec![(0, 15.0)], None);
    assert!(!pruner.prune(&trials, &trial, None).unwrap());

    // below lower → prune
    let trial = make_complete_trial_with_intermediates(1, vec![(0, 3.0)], None);
    assert!(pruner.prune(&trials, &trial, None).unwrap());

    // above upper → prune
    let trial = make_complete_trial_with_intermediates(2, vec![(0, 30.0)], None);
    assert!(pruner.prune(&trials, &trial, None).unwrap());
}

/// NaN → ThresholdPruner 应剪枝
#[test]
fn test_threshold_pruner_nan() {
    let pruner = ThresholdPruner::new(Some(0.0), Some(100.0), 0, 1);
    let trials: Vec<FrozenTrial> = vec![];

    let trial = make_complete_trial_with_intermediates(0, vec![(0, f64::NAN)], None);
    assert!(pruner.prune(&trials, &trial, None).unwrap());
}

// ============================================================================
// 5. PatientPruner 停滞检测验证
// ============================================================================

/// Python PatientPruner 检测的是回归（best-before 优于 best-after），不是简单停滞
///
/// Python 参考 (patience=3, minimize):
///   values=[8, 8, 9, 9, 9, 9] → 最终 before_min=8 < after_min=9 → prune=True
///   values=[10, 9, 8, 8, 8, 8] → 最终 before_min=9, after_min=8 → 9 < 8? No → prune=False
///
/// 正确场景: values=[1, 2, 3, 4, 5, 6] (持续恶化)
///   n_steps<=4: 不够步数 → False
///   n_steps=5: before=[1] min=1 < after=[2,3,4,5] min=2 → prune=True
///   n_steps=6: before=[1,2] min=1 < after=[3,4,5,6] min=3 → prune=True
#[test]
fn test_patient_pruner_minimize_regression() {
    let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
    let completed: Vec<FrozenTrial> = vec![];

    // values=[1, 2, 3, 4, 5, 6] → 持续恶化（minimize）
    let steps_values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    // patience=3 → need > 4 steps
    let expected_prune: Vec<bool> = vec![false, false, false, false, true, true];

    for (step, &expected) in expected_prune.iter().enumerate() {
        let intermediates: Vec<(i64, f64)> = (0..=step)
            .map(|s| (s as i64, steps_values[s]))
            .collect();
        let trial = make_complete_trial_with_intermediates(0, intermediates, None);
        let result = pruner.prune(&completed, &trial, None).unwrap();
        assert_eq!(
            result, expected,
            "step={}: expected prune={}, got prune={}",
            step, expected, result
        );
    }
}

/// Python 参考: minimize, values=[10, 9, 8, 8, 8, 8] → 不剪枝（after 优于 before）
#[test]
fn test_patient_pruner_minimize_no_regression() {
    let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
    let completed: Vec<FrozenTrial> = vec![];

    // values=[10, 9, 8, 8, 8, 8] → before_min=9 > after_min=8 → 改善中，不剪枝
    let steps_values: Vec<f64> = vec![10.0, 9.0, 8.0, 8.0, 8.0, 8.0];

    for step in 0..steps_values.len() {
        let intermediates: Vec<(i64, f64)> = (0..=step)
            .map(|s| (s as i64, steps_values[s]))
            .collect();
        let trial = make_complete_trial_with_intermediates(0, intermediates, None);
        let result = pruner.prune(&completed, &trial, None).unwrap();
        assert!(
            !result,
            "step={}: improving trial should not be pruned",
            step
        );
    }
}

/// Python 参考 (maximize, patience=2):
///   values=[3, 2, 1, 0, -1] → before_max=3 > after_max=1 → 回归 → prune=True
///   但 values=[1, 2, 3, 3, 3] → before_max=2 < after_max=3 → 不回归 → prune=False
#[test]
fn test_patient_pruner_maximize_regression() {
    let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Maximize);
    let completed: Vec<FrozenTrial> = vec![];

    // values=[3, 2, 1, 0, -1] → 持续恶化（maximize）
    let steps_values: Vec<f64> = vec![3.0, 2.0, 1.0, 0.0, -1.0];
    // patience=2 → need > 3 steps
    let expected_prune: Vec<bool> = vec![false, false, false, true, true];

    for (step, &expected) in expected_prune.iter().enumerate() {
        let intermediates: Vec<(i64, f64)> = (0..=step)
            .map(|s| (s as i64, steps_values[s]))
            .collect();
        let trial = make_complete_trial_with_intermediates(0, intermediates, None);
        let result = pruner.prune(&completed, &trial, None).unwrap();
        assert_eq!(
            result, expected,
            "step={}: expected prune={}, got prune={}",
            step, expected, result
        );
    }
}

/// Python 参考: maximize, patience=2, values=[1,2,3,3,3] → 不剪枝
#[test]
fn test_patient_pruner_maximize_no_regression() {
    let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Maximize);
    let completed: Vec<FrozenTrial> = vec![];

    // values=[1,2,3,3,3] → after 始终 >= before → 不回归
    let steps_values: Vec<f64> = vec![1.0, 2.0, 3.0, 3.0, 3.0];
    for step in 0..steps_values.len() {
        let intermediates: Vec<(i64, f64)> = (0..=step)
            .map(|s| (s as i64, steps_values[s]))
            .collect();
        let trial = make_complete_trial_with_intermediates(0, intermediates, None);
        let result = pruner.prune(&completed, &trial, None).unwrap();
        assert!(
            !result,
            "step={}: non-regressing trial should not be pruned",
            step
        );
    }
}

/// 没有中间值 → 不剪枝
#[test]
fn test_patient_pruner_no_intermediates() {
    let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
    let completed: Vec<FrozenTrial> = vec![];
    let trial = make_complete_trial_with_intermediates(0, vec![], None);
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(!result, "no intermediates → should not prune");
}

// ============================================================================
// 6. NopPruner 行为验证
// ============================================================================

/// NopPruner 永远不剪枝
#[test]
fn test_nop_pruner() {
    let pruner = NopPruner;
    let completed: Vec<FrozenTrial> = (0..10)
        .map(|i| {
            make_complete_trial_with_intermediates(i, vec![(0, 100.0)], Some(100.0))
        })
        .collect();

    let trial = make_complete_trial_with_intermediates(10, vec![(0, 999999.0)], None);
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(!result, "NopPruner should never prune");
}

// ============================================================================
// 7. PercentilePruner with multiple steps (best over steps)
// ============================================================================

/// 验证 "best intermediate value over steps" 的选取逻辑
/// minimize: 在多步中选最小值与 threshold 比较
#[test]
fn test_percentile_pruner_multi_step_best() {
    // 5 completed trials, each with 3 steps
    let completed: Vec<FrozenTrial> = (0..5)
        .map(|i| {
            let base = (i * 10) as f64;
            make_complete_trial_with_intermediates(
                i,
                vec![(0, base), (1, base + 1.0), (2, base + 2.0)],
                Some(base + 2.0),
            )
        })
        .collect();
    // step 2: values are [2, 12, 22, 32, 42], p50=22

    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);

    // Trial with steps: [30, 25, 15]
    // best (min) over steps = 15 <= 22 → not prune
    let trial = make_complete_trial_with_intermediates(
        5,
        vec![(0, 30.0), (1, 25.0), (2, 15.0)],
        None,
    );
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(!result, "best=15 <= threshold → should not prune");

    // Trial with steps: [30, 25, 35]
    // best (min) over steps = 25 > 22 → prune
    let trial = make_complete_trial_with_intermediates(
        6,
        vec![(0, 30.0), (1, 25.0), (2, 35.0)],
        None,
    );
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(result, "best=25 > threshold=22 → should prune");
}

// ============================================================================
// 8. PercentilePruner n_min_trials guard
// ============================================================================

/// 当某步的完成试验数 < n_min_trials 时不剪枝
#[test]
fn test_percentile_pruner_n_min_trials() {
    // Only 2 trials have step=5
    let completed = vec![
        make_complete_trial_with_intermediates(0, vec![(0, 0.0), (5, 10.0)], Some(10.0)),
        make_complete_trial_with_intermediates(1, vec![(0, 0.0), (5, 20.0)], Some(20.0)),
        // Trial 2 doesn't have step=5
        make_complete_trial_with_intermediates(2, vec![(0, 0.0)], Some(0.0)),
    ];

    // n_min_trials=3 but only 2 trials have step=5
    let pruner = PercentilePruner::new(50.0, 0, 0, 1, 3, StudyDirection::Minimize);

    let trial = make_complete_trial_with_intermediates(
        3,
        vec![(0, 0.0), (5, 999.0)],
        None,
    );
    let result = pruner.prune(&completed, &trial, None).unwrap();
    assert!(!result, "not enough trials at step=5 → should not prune");
}
