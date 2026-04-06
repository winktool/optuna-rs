//! PED-ANOVA 交叉验证测试。
//!
//! 验证 PedAnovaEvaluator 的排名正确性、属性不变式和边界行为。
//! Python 参考值由 optuna v4.x PedAnovaImportanceEvaluator 生成。

use std::sync::Arc;
use optuna_rs::importance::{get_param_importances, PedAnovaEvaluator};
use optuna_rs::samplers::RandomSampler;
use optuna_rs::study::{create_study, StudyDirection};

// ============================================================================
// 1. 排名正确性 — dominant 参数排在前面
// ============================================================================

/// 目标 = x (y 无关) → x 重要性远大于 y
/// Python 参考: x=0.0916, y=0.0009 (raw), x > 100 * y
#[test]
fn test_pedanova_single_dominant_ranking() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let _y = trial.suggest_float("y", 0.0, 10.0, false, None)?;
                Ok(x) // Only x matters
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    // Normalized: x should be dominant (>90%)
    let x_imp = *imp.get("x").unwrap();
    let y_imp = *imp.get("y").unwrap();
    assert!(
        x_imp > 0.9,
        "x should be dominant param (>90%), got {x_imp}"
    );
    assert!(
        x_imp > y_imp,
        "x should rank higher than y: x={x_imp}, y={y_imp}"
    );
}

/// 目标 = 10x + 0.1y + 0.5z → x >> z > y 或 x >> y ≈ z
/// Python 参考 (seed=42, 60 trials): x=0.949, y=0.030, z=0.021
#[test]
fn test_pedanova_three_param_ranking() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(10.0 * x + 0.1 * y + 0.5 * z)
            },
            Some(60),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    // x should be the most important (>90% normalized)
    assert!(
        x_imp > 0.85,
        "x should dominate, got {x_imp}"
    );
    // x should be first key in sorted order
    let first_key = imp.keys().next().unwrap();
    assert_eq!(first_key, "x", "x should rank first");
}

/// 二次主导: x² + 0.01*y² → x 远比 y 重要
/// Python 参考 (seed=42, 60 trials): x=0.982, y=0.018
#[test]
fn test_pedanova_quadratic_dominance() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.01 * y * y)
            },
            Some(60),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    assert!(
        x_imp > 0.9,
        "x should dominate in quadratic, got {x_imp}"
    );
}

// ============================================================================
// 2. 属性不变式
// ============================================================================

/// 归一化后所有值非负且 sum = 1
#[test]
fn test_pedanova_normalization_properties() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(x + y + z)
            },
            Some(60),
            None,
            None,
        )
        .unwrap();

    // Normalized
    let imp = get_param_importances(&study, Some(&PedAnovaEvaluator::default()), None, None, true)
        .unwrap();

    let sum: f64 = imp.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "normalized importances should sum to 1.0, got {sum}"
    );
    for (name, &val) in &imp {
        assert!(
            val >= 0.0,
            "importance of {name} should be >= 0, got {val}"
        );
    }
}

/// 未归一化值也应非负
#[test]
fn test_pedanova_raw_nonnegative() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();

    let imp =
        get_param_importances(&study, Some(&PedAnovaEvaluator::default()), None, None, false)
            .unwrap();

    for (name, &val) in &imp {
        assert!(
            val >= 0.0,
            "raw importance of {name} should be >= 0, got {val}"
        );
    }
}

// ============================================================================
// 3. Maximize 方向
// ============================================================================

/// maximize 方向: 目标 = 10x + 0.1y → x 主导
/// Python 参考 (seed=42, 60 trials): x=0.965, y=0.035
#[test]
fn test_pedanova_maximize_direction() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
        None,
        None,
        Some(StudyDirection::Maximize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(10.0 * x + 0.1 * y)
            },
            Some(60),
            None,
            None,
        )
        .unwrap();

    let mut eval = PedAnovaEvaluator::default();
    eval.is_lower_better = false; // maximize
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    assert!(
        x_imp > 0.9,
        "x should dominate in maximize, got {x_imp}"
    );
}

// ============================================================================
// 4. 边界情况
// ============================================================================

/// 试验太少 (≤ min_n_top_trials=2) → 返回全 0
#[test]
fn test_pedanova_few_trials_returns_zero() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                Ok(x)
            },
            Some(2),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), None, None, false).unwrap();

    for (name, &val) in &imp {
        assert!(
            val.abs() < 1e-10,
            "with only 2 trials, {name} importance should be ~0, got {val}"
        );
    }
}

/// target_quantile=0.2 也能工作
#[test]
fn test_pedanova_custom_target_quantile() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(10.0 * x + 0.1 * y)
            },
            Some(60),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::new(Some(0.2), Some(1.0), Some(true));
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    assert!(
        x_imp > 0.85,
        "x should dominate with target_quantile=0.2, got {x_imp}"
    );
}

/// region_quantile < 1.0 + evaluate_on_local=true
#[test]
fn test_pedanova_region_quantile() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let _y = trial.suggest_float("y", 0.0, 10.0, false, None)?;
                Ok(x)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    // Using region_quantile=0.5
    let eval = PedAnovaEvaluator::new(Some(0.1), Some(0.5), Some(true));
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    assert!(
        x_imp > 0.8,
        "x should still dominate with region_quantile=0.5, got {x_imp}"
    );
}

/// evaluate_on_local=false 使用均匀基准分布
#[test]
fn test_pedanova_global_evaluation() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(10.0 * x + 0.1 * y)
            },
            Some(80),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::new(Some(0.1), Some(1.0), Some(false));
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let x_imp = *imp.get("x").unwrap();
    assert!(
        x_imp > 0.85,
        "x should dominate even in global mode, got {x_imp}"
    );
}

// ============================================================================
// 5. Trait 对象接口
// ============================================================================

/// 通过 ImportanceEvaluator trait 使用
#[test]
fn test_pedanova_trait_interface() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 10.0, false, None)?;
                Ok(x + 0.01 * y)
            },
            Some(40),
            None,
            None,
        )
        .unwrap();

    // Use through trait object
    let eval: Box<dyn optuna_rs::importance::ImportanceEvaluator> =
        Box::new(PedAnovaEvaluator::default());
    let imp = get_param_importances(&study, Some(&*eval), None, None, true).unwrap();

    assert!(imp.contains_key("x"));
    assert!(imp.contains_key("y"));
    let total: f64 = imp.values().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

/// 参数子集过滤
#[test]
fn test_pedanova_param_subset() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn optuna_rs::samplers::Sampler>),
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
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let _z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(x + y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), Some(&["x", "y"]), None, true).unwrap();

    assert_eq!(imp.len(), 2);
    assert!(imp.contains_key("x"));
    assert!(imp.contains_key("y"));
    assert!(!imp.contains_key("z"));
}

// ============================================================================
// 6. 内部组件验证
// ============================================================================

/// PedAnovaEvaluator::new 参数验证
#[test]
fn test_pedanova_creation_valid() {
    let _e = PedAnovaEvaluator::default();
    let _e2 = PedAnovaEvaluator::new(Some(0.2), Some(0.8), Some(false));
    let _e3 = PedAnovaEvaluator::new(Some(0.05), Some(1.0), Some(true));
}

/// PedAnovaEvaluator::new 不合法参数应 panic
#[test]
#[should_panic]
fn test_pedanova_invalid_quantile() {
    // target_quantile >= region_quantile should fail
    let _e = PedAnovaEvaluator::new(Some(0.5), Some(0.5), None);
}

/// target_quantile <= 0 should fail
#[test]
#[should_panic]
fn test_pedanova_zero_target_quantile() {
    let _e = PedAnovaEvaluator::new(Some(0.0), Some(1.0), None);
}
