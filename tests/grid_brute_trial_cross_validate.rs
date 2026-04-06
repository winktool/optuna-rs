//! GridSampler + BruteForceSampler + Trial API 交叉验证测试。
//!
//! 验证 GridSampler 完整覆盖、BruteForceSampler 穷举行为、
//! Trial suggest API 的边界条件。

use std::collections::HashSet;
use std::sync::Arc;
use optuna_rs::distributions::{
    CategoricalChoice, CategoricalDistribution, FloatDistribution, IntDistribution, ParamValue,
};
use optuna_rs::samplers::{BruteForceSampler, GridSampler, RandomSampler, Sampler};
use optuna_rs::study::{create_study, StudyDirection};
use optuna_rs::trial::TrialState;

// ============================================================================
// 1. GridSampler 完整覆盖验证
// ============================================================================

/// GridSampler 3×2 = 6 个 grid 点，6 个 trial 应该覆盖全部
#[test]
fn test_grid_sampler_full_coverage() {
    use std::collections::HashMap;

    let mut search_space = HashMap::new();
    search_space.insert("x".to_string(), vec![0.0, 1.0, 2.0]);
    search_space.insert("y".to_string(), vec![10.0, 20.0]);

    let sampler = GridSampler::new(search_space, None);
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 2.0, false, None)?;
                let y = trial.suggest_float("y", 10.0, 20.0, false, None)?;
                Ok(x + y)
            },
            Some(6),
            None,
            None,
        )
        .unwrap();

    let trials = study.get_trials(None).unwrap();
    assert_eq!(trials.len(), 6);

    // Collect all (x, y) pairs
    let mut pairs: HashSet<(i64, i64)> = HashSet::new();
    for t in &trials {
        let x = match t.params.get("x").unwrap() {
            ParamValue::Float(v) => *v as i64,
            ParamValue::Int(v) => *v,
            _ => panic!("unexpected x type"),
        };
        let y = match t.params.get("y").unwrap() {
            ParamValue::Float(v) => *v as i64,
            ParamValue::Int(v) => *v,
            _ => panic!("unexpected y type"),
        };
        pairs.insert((x, y));
    }

    // Should have all 6 combinations
    assert_eq!(pairs.len(), 6, "should cover all 6 grid points, got {:?}", pairs);
    assert!(pairs.contains(&(0, 10)));
    assert!(pairs.contains(&(0, 20)));
    assert!(pairs.contains(&(1, 10)));
    assert!(pairs.contains(&(1, 20)));
    assert!(pairs.contains(&(2, 10)));
    assert!(pairs.contains(&(2, 20)));
}

/// GridSampler best_trial 选择最小值
#[test]
fn test_grid_sampler_best_trial() {
    use std::collections::HashMap;

    let mut search_space = HashMap::new();
    search_space.insert("x".to_string(), vec![1.0, 2.0, 3.0]);

    let sampler = GridSampler::new(search_space, None);
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 1.0, 3.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            None,
        )
        .unwrap();

    let best = study.best_trial().unwrap();
    assert_eq!(best.values.as_ref().unwrap()[0], 1.0); // x=1 → x²=1
}

// ============================================================================
// 2. BruteForceSampler 穷举行为
// ============================================================================

/// BruteForceSampler 穷举所有 categorical 组合后停止
#[test]
fn test_brute_force_exhaustive() {
    let sampler = BruteForceSampler::new(None, false);
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    // Request 10 trials, but should stop after 6 (3 * 2)
    study
        .optimize(
            |trial| {
                let x = trial.suggest_categorical(
                    "x",
                    vec![
                        CategoricalChoice::Int(1),
                        CategoricalChoice::Int(2),
                        CategoricalChoice::Int(3),
                    ],
                )?;
                let y = trial.suggest_categorical(
                    "y",
                    vec![
                        CategoricalChoice::Int(10),
                        CategoricalChoice::Int(20),
                    ],
                )?;
                let x_val = match x {
                    CategoricalChoice::Int(v) => v as f64,
                    _ => unreachable!(),
                };
                let y_val = match y {
                    CategoricalChoice::Int(v) => v as f64,
                    _ => unreachable!(),
                };
                Ok(x_val + y_val)
            },
            Some(10),
            None,
            None,
        )
        .unwrap();

    let trials = study.get_trials(Some(&[TrialState::Complete])).unwrap();
    assert_eq!(trials.len(), 6, "BruteForceSampler should stop after 6 unique combos");

    // Verify all 6 unique combinations present
    let mut combos: HashSet<(i64, i64)> = HashSet::new();
    for t in &trials {
        let x = match t.params.get("x").unwrap() {
            ParamValue::Int(v) => *v,
            ParamValue::Float(v) => *v as i64,
            ParamValue::Categorical(CategoricalChoice::Int(v)) => *v,
            _ => panic!("unexpected param type"),
        };
        let y = match t.params.get("y").unwrap() {
            ParamValue::Int(v) => *v,
            ParamValue::Float(v) => *v as i64,
            ParamValue::Categorical(CategoricalChoice::Int(v)) => *v,
            _ => panic!("unexpected param type"),
        };
        combos.insert((x, y));
    }
    assert_eq!(combos.len(), 6);
}

// ============================================================================
// 3. Trial suggest API 边界验证
// ============================================================================

/// suggest_float: 范围边界 low == high 时应返回该值
#[test]
fn test_suggest_float_single_value() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 5.0, 5.0, false, None)?;
                assert_eq!(x, 5.0);
                Ok(x)
            },
            Some(3),
            None,
            None,
        )
        .unwrap();
}

/// suggest_int: 基本范围
#[test]
fn test_suggest_int_range() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let n = trial.suggest_int("n", 1, 10, false, 1)?;
                assert!(n >= 1 && n <= 10, "n={n} out of range [1,10]");
                Ok(n as f64)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
}

/// suggest_int: step > 1
#[test]
fn test_suggest_int_with_step() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let n = trial.suggest_int("n", 0, 10, false, 2)?;
                assert!(n % 2 == 0, "n={n} should be even with step=2");
                assert!(n >= 0 && n <= 10);
                Ok(n as f64)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
}

/// suggest_float with step
#[test]
fn test_suggest_float_with_step() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, Some(0.25))?;
                // Valid values: 0.0, 0.25, 0.5, 0.75, 1.0
                let valid = [0.0, 0.25, 0.5, 0.75, 1.0];
                assert!(
                    valid.iter().any(|&v| (x - v).abs() < 1e-12),
                    "x={x} should be one of {valid:?}"
                );
                Ok(x)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
}

/// suggest_float log scale
#[test]
fn test_suggest_float_log_scale() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let lr = trial.suggest_float("lr", 1e-5, 1e-1, true, None)?;
                assert!(lr >= 1e-5 && lr <= 1e-1, "lr={lr} out of log range");
                Ok(lr)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
}

/// suggest_categorical: 返回值应在选择列表中
#[test]
fn test_suggest_categorical_values() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let c = trial.suggest_categorical(
                    "opt",
                    vec![
                        CategoricalChoice::Str("adam".to_string()),
                        CategoricalChoice::Str("sgd".to_string()),
                        CategoricalChoice::Str("rmsprop".to_string()),
                    ],
                )?;
                let name = match &c {
                    CategoricalChoice::Str(s) => s.clone(),
                    _ => panic!("expected string"),
                };
                assert!(
                    ["adam", "sgd", "rmsprop"].contains(&name.as_str()),
                    "got unexpected: {name}"
                );
                Ok(match name.as_str() {
                    "adam" => 1.0,
                    "sgd" => 2.0,
                    "rmsprop" => 3.0,
                    _ => 99.0,
                })
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
}

// ============================================================================
// 4. Trial report + should_prune 集成
// ============================================================================

/// report 中间值后 should_prune 调用不应 panic
#[test]
fn test_trial_report_and_should_prune() {
    use optuna_rs::pruners::NopPruner;

    let sampler = RandomSampler::new(Some(42));
    let pruner = NopPruner::new();
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        Some(Arc::new(pruner)),
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                for step in 0..5 {
                    let val = (step as f64) * 0.1;
                    trial.report(val, step)?;
                    let _should = trial.should_prune()?;
                }
                Ok(1.0)
            },
            Some(5),
            None,
            None,
        )
        .unwrap();

    let trials = study.get_trials(Some(&[TrialState::Complete])).unwrap();
    assert_eq!(trials.len(), 5);
}

/// 同一参数名重复 suggest 应返回相同值 (define-by-run 一致性)
#[test]
fn test_suggest_same_param_returns_consistent() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x1 = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let x2 = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                assert_eq!(x1, x2, "same param name should return same value");
                Ok(x1)
            },
            Some(5),
            None,
            None,
        )
        .unwrap();
}

// ============================================================================
// 5. Distribution 内部表示精度
// ============================================================================

/// FloatDistribution 内部表示 = 值本身
#[test]
fn test_float_distribution_internal_repr() {
    let dist = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert_eq!(dist.to_internal_repr(0.5).unwrap(), 0.5);
    assert_eq!(dist.to_external_repr(0.5), 0.5);

    // Log scale
    let dist_log = FloatDistribution::new(1e-3, 1.0, true, None).unwrap();
    assert_eq!(dist_log.to_internal_repr(0.1).unwrap(), 0.1);
    assert_eq!(dist_log.to_external_repr(0.1), 0.1);
}

/// IntDistribution 内部表示 = 值本身 (as f64)
#[test]
fn test_int_distribution_internal_repr() {
    let dist = IntDistribution::new(0, 10, false, 1).unwrap();
    assert_eq!(dist.to_internal_repr(5).unwrap(), 5.0);
    assert_eq!(dist.to_external_repr(5.0).unwrap(), 5);

    // With step
    let dist_step = IntDistribution::new(0, 10, false, 2).unwrap();
    assert_eq!(dist_step.to_internal_repr(4).unwrap(), 4.0);
    assert_eq!(dist_step.to_external_repr(4.0).unwrap(), 4);
}

/// CategoricalDistribution 内部表示 = 索引
#[test]
fn test_categorical_distribution_internal_repr() {
    let dist = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".to_string()),
        CategoricalChoice::Str("b".to_string()),
        CategoricalChoice::Str("c".to_string()),
    ])
    .unwrap();

    // "a" → index 0, "b" → index 1, "c" → index 2
    assert_eq!(
        dist.to_internal_repr(&CategoricalChoice::Str("a".to_string())).unwrap(),
        0.0
    );
    assert_eq!(
        dist.to_internal_repr(&CategoricalChoice::Str("b".to_string())).unwrap(),
        1.0
    );
    assert_eq!(
        dist.to_internal_repr(&CategoricalChoice::Str("c".to_string())).unwrap(),
        2.0
    );

    // Reverse: index → choice
    assert_eq!(
        dist.to_external_repr(0.0).unwrap(),
        CategoricalChoice::Str("a".to_string())
    );
    assert_eq!(
        dist.to_external_repr(1.0).unwrap(),
        CategoricalChoice::Str("b".to_string())
    );
}
