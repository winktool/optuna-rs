//! Importance 模块精确交叉验证测试。
//!
//! 验证 fANOVA、MDI、PED-ANOVA 评估器的排名正确性和数值属性。
//! 由于随机森林训练依赖 RNG 实现细节，重点验证统计属性而非精确值。

use std::sync::Arc;
use optuna_rs::importance::{
    FanovaEvaluator, MeanDecreaseImpurityEvaluator, ImportanceEvaluator,
    get_param_importances,
};
use optuna_rs::study::{create_study, StudyDirection};
use optuna_rs::samplers::{RandomSampler, Sampler};
use optuna_rs::Study;

// ============================================================================
// 辅助函数
// ============================================================================

fn create_study_with_linear_objective(n_trials: usize, seed: u64) -> Study {
    let sampler = RandomSampler::new(Some(seed));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Minimize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                let z = trial.suggest_float("z", 0.0, 5.0, false, None)?;
                Ok(10.0 * x + 0.1 * y + 0.5 * z)
            },
            Some(n_trials),
            None,
            None,
        )
        .unwrap();
    study
}

// ============================================================================
// 1. fANOVA 排名验证
// ============================================================================

/// Python 参考 (seed=42, 60 trials, 10*x + 0.1*y + 0.5*z):
///   fANOVA ranking: x > z > y
///   x: 9.99265e-01, z: 4.29e-04, y: 3.06e-04
///
/// 排名必须正确，精确值允许差异（不同 RNG 和 bootstrap）
#[test]
fn test_fanova_ranking_linear_objective() {
    let study = create_study_with_linear_objective(60, 42);
    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let ix = importances.get("x").copied().unwrap_or(0.0);
    let iy = importances.get("y").copied().unwrap_or(0.0);
    let iz = importances.get("z").copied().unwrap_or(0.0);

    // 排名: x >> z > y (系数 10.0 vs 0.5 vs 0.1)
    assert!(ix > iz, "x importance ({ix}) should > z importance ({iz})");
    assert!(ix > iy, "x importance ({ix}) should > y importance ({iy})");
    // x 在线性目标中应占主导 (系数 10.0, 范围 0-10 → 贡献100)
    assert!(ix > 0.9, "x importance should be > 0.9, got {ix}");
}

/// fANOVA 归一化属性: 所有值 ≥ 0，sum = 1.0
#[test]
fn test_fanova_normalization_properties() {
    let study = create_study_with_linear_objective(40, 123);
    let evaluator = FanovaEvaluator::new(64, 64, Some(123));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let sum: f64 = importances.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "normalized importances should sum to 1.0, got {sum}"
    );

    for (name, &val) in &importances {
        assert!(val >= 0.0, "importance of {name} should be >= 0, got {val}");
    }
}

/// fANOVA 不归一化: sum 可以 > 1 或 < 1
#[test]
fn test_fanova_unnormalized() {
    let study = create_study_with_linear_objective(40, 99);
    let evaluator = FanovaEvaluator::new(64, 64, Some(99));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, false).unwrap();

    for (name, &val) in &importances {
        assert!(val >= 0.0, "raw importance of {name} should be >= 0, got {val}");
    }
    // 不归一化时仍然 x 最重要
    let ix = importances.get("x").copied().unwrap_or(0.0);
    let iy = importances.get("y").copied().unwrap_or(0.0);
    assert!(ix > iy, "x > y even unnormalized");
}

// ============================================================================
// 2. MDI 排名验证
// ============================================================================

/// Python 参考 (seed=42, 60 trials):
///   MDI ranking: x > z > y
///   x: 9.957e-01
#[test]
fn test_mdi_ranking_linear_objective() {
    let study = create_study_with_linear_objective(60, 42);
    let evaluator = MeanDecreaseImpurityEvaluator::new(64, 64, Some(42));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let ix = importances.get("x").copied().unwrap_or(0.0);
    let iy = importances.get("y").copied().unwrap_or(0.0);
    let iz = importances.get("z").copied().unwrap_or(0.0);

    assert!(ix > iz, "MDI: x ({ix}) > z ({iz})");
    assert!(ix > iy, "MDI: x ({ix}) > y ({iy})");
    assert!(ix > 0.8, "MDI: x should be > 0.8, got {ix}");
}

/// MDI 归一化属性
#[test]
fn test_mdi_normalization_properties() {
    let study = create_study_with_linear_objective(40, 77);
    let evaluator = MeanDecreaseImpurityEvaluator::new(64, 64, Some(77));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let sum: f64 = importances.values().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "MDI normalized sum should be 1.0, got {sum}"
    );

    for (_, &val) in &importances {
        assert!(val >= 0.0);
    }
}

// ============================================================================
// 3. 边界条件验证
// ============================================================================

/// 单个试验应报错（对齐 Python: ValueError）
#[test]
fn test_importance_single_trial_error() {
    let sampler = RandomSampler::new(Some(0));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Minimize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(1),
            None,
            None,
        )
        .unwrap();

    let result = get_param_importances(&study, None, None, None, true);
    assert!(result.is_err(), "single trial should produce an error");
}

/// 两个试验应返回有效结果
#[test]
fn test_importance_two_trials_ok() {
    let sampler = RandomSampler::new(Some(0));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Minimize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(2),
            None,
            None,
        )
        .unwrap();

    let result = get_param_importances(&study, None, None, None, true);
    assert!(result.is_ok(), "two trials should work: {:?}", result.err());
    let imp = result.unwrap();
    assert!(imp.contains_key("x"));
}

// ============================================================================
// 4. x^2 + noise 目标函数 — x 几乎完全主导
// ============================================================================

/// Python 参考 (seed=0, 50 trials):
///   x: 9.957e-01, noise: 4.34e-03
#[test]
fn test_fanova_quadratic_dominance() {
    let sampler = RandomSampler::new(Some(0));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Minimize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let noise = trial.suggest_float("noise", 0.0, 0.001, false, None)?;
                Ok(x * x + noise)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(0));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let ix = importances.get("x").copied().unwrap_or(0.0);
    let in_ = importances.get("noise").copied().unwrap_or(0.0);

    assert!(ix > 0.95, "fANOVA: x should dominate in x^2+noise, got {ix}");
    assert!(in_ < 0.05, "fANOVA: noise should be < 0.05, got {in_}");
    assert!(ix > in_, "x > noise");
}

// ============================================================================
// 5. 最大化方向验证
// ============================================================================

/// 最大化目标: maximize 10*x + 0.1*y → x 仍然最重要
#[test]
fn test_fanova_maximize_direction() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Maximize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                Ok(10.0 * x + 0.1 * y)
            },
            Some(40),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let ix = importances.get("x").copied().unwrap_or(0.0);
    let iy = importances.get("y").copied().unwrap_or(0.0);
    assert!(ix > iy, "maximize: x ({ix}) > y ({iy})");
    assert!(ix > 0.9, "maximize: x should dominate, got {ix}");
}

// ============================================================================
// 6. 参数子集过滤
// ============================================================================

/// 指定 params 子集时只返回指定参数的重要性
#[test]
fn test_importance_param_subset() {
    let study = create_study_with_linear_objective(30, 55);
    let evaluator = FanovaEvaluator::new(64, 64, Some(55));
    let importances = get_param_importances(
        &study,
        Some(&evaluator),
        Some(&["x", "y"]),
        None,
        true,
    )
    .unwrap();

    assert!(importances.contains_key("x"));
    assert!(importances.contains_key("y"));
    assert!(!importances.contains_key("z"), "z should not appear when not in params subset");
    assert_eq!(importances.len(), 2);

    let sum: f64 = importances.values().sum();
    assert!((sum - 1.0).abs() < 1e-10, "subset sum should be 1.0, got {sum}");
}

// ============================================================================
// 7. TraitObject 接口验证
// ============================================================================

/// 通过 trait object 接口（dyn ImportanceEvaluator）调用
#[test]
fn test_evaluator_trait_object() {
    let study = create_study_with_linear_objective(30, 88);
    let evaluator: Box<dyn ImportanceEvaluator> = Box::new(FanovaEvaluator::new(32, 32, Some(88)));
    let importances = get_param_importances(
        &study,
        Some(evaluator.as_ref()),
        None,
        None,
        true,
    )
    .unwrap();
    assert!(!importances.is_empty());
    let ix = importances.get("x").copied().unwrap_or(0.0);
    assert!(ix > 0.5, "trait object: x importance = {ix}");
}

// ============================================================================
// 8. 相同系数 → 相近重要性
// ============================================================================

/// 当两个参数有相同系数和相同范围时，重要性应接近
#[test]
fn test_equal_coefficient_similar_importance() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(None, Some(Arc::new(sampler) as Arc<dyn Sampler>), None, None, Some(StudyDirection::Minimize), None, false).unwrap();
    study
        .optimize(
            |trial| {
                let a = trial.suggest_float("a", 0.0, 10.0, false, None)?;
                let b = trial.suggest_float("b", 0.0, 10.0, false, None)?;
                Ok(a + b)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances = get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let ia = importances.get("a").copied().unwrap_or(0.0);
    let ib = importances.get("b").copied().unwrap_or(0.0);

    // 两者应在 0.3~0.7 范围内（理想 0.5 各占一半）
    assert!(
        (ia - ib).abs() < 0.3,
        "equal coefficients: a={ia}, b={ib} should be similar (diff < 0.3)"
    );
    assert!(ia > 0.2 && ia < 0.8, "a should be ~0.5, got {ia}");
    assert!(ib > 0.2 && ib < 0.8, "b should be ~0.5, got {ib}");
}
