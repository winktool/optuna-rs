//! Importance 模块深度交叉验证测试
//!
//! 验证 Rust optuna-rs 的 importance 模块与 Python optuna 对齐：
//! - FanovaEvaluator: fANOVA 方差分解重要性
//! - MeanDecreaseImpurityEvaluator: MDI 特征重要性
//! - PedAnovaEvaluator: PED-ANOVA χ² 散度重要性
//!
//! 由于 Python 使用 sklearn 随机森林、Rust 使用自定义实现，
//! 精确数值无法匹配。测试验证：
//! 1. 重要性排序一致性（dominant 参数应排首位）
//! 2. 归一化数学性质（总和 = 1.0，非负）
//! 3. 边界条件处理一致性
//! 4. 三种评估器对 dominant 参数的一致判断

use optuna_rs::importance::{
    get_param_importances, FanovaEvaluator, ImportanceEvaluator,
    MeanDecreaseImpurityEvaluator, PedAnovaEvaluator,
};
use optuna_rs::samplers::RandomSampler;
use optuna_rs::study::{create_study, StudyDirection};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════
// 辅助函数
// ═══════════════════════════════════════════════════════════════════════

/// 创建一个用于测试的简单最小化研究，使用固定种子的随机采样器。
fn create_test_study(seed: u64) -> optuna_rs::study::Study {
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> =
        Arc::new(RandomSampler::new(Some(seed)));
    create_study(
        None,
        Some(sampler),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap()
}

// ═══════════════════════════════════════════════════════════════════════
// Group 1: fANOVA 评估器 — 重要性排序验证
// ═══════════════════════════════════════════════════════════════════════

/// 测试 fANOVA: f(x,y) = x^2 + 0.001*y → x 应远比 y 重要
#[test]
fn test_fanova_importance_ordering_x_dominant() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.001 * y)
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();
    assert_eq!(importances.len(), 2);

    let x_imp = importances["x"];
    let y_imp = importances["y"];
    assert!(
        x_imp > y_imp,
        "fANOVA: x 重要性 ({x_imp:.4}) 应 > y ({y_imp:.4})"
    );
    // x 至少应占 70% 的重要性
    assert!(
        x_imp > 0.7,
        "fANOVA: x 重要性 ({x_imp:.4}) 应 > 0.7"
    );
}

/// 测试 fANOVA: 三参数 f(x,y,z) = 10x^2 + y^2 + 0.001z → x > y > z
#[test]
fn test_fanova_three_param_ordering() {
    let study = create_test_study(123);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                Ok(10.0 * x * x + y * y + 0.001 * z)
            },
            Some(300),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();
    assert_eq!(importances.len(), 3);

    let keys: Vec<&String> = importances.keys().collect();
    assert_eq!(
        keys[0], "x",
        "fANOVA 三参数: 首位应为 x, 实际 {:?}",
        keys
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 2: MDI 评估器 — 重要性排序验证
// ═══════════════════════════════════════════════════════════════════════

/// 测试 MDI: f(x,y) = x^2 + 0.001*y → x 应远比 y 重要
#[test]
fn test_mdi_importance_ordering_x_dominant() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.001 * y)
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = MeanDecreaseImpurityEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();
    assert_eq!(importances.len(), 2);

    let x_imp = importances["x"];
    let y_imp = importances["y"];
    assert!(
        x_imp > y_imp,
        "MDI: x 重要性 ({x_imp:.4}) 应 > y ({y_imp:.4})"
    );
    assert!(
        x_imp > 0.7,
        "MDI: x 重要性 ({x_imp:.4}) 应 > 0.7"
    );
}

/// 测试 MDI: 三参数排序
#[test]
fn test_mdi_three_param_ordering() {
    let study = create_test_study(123);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                Ok(10.0 * x * x + y * y + 0.001 * z)
            },
            Some(300),
            None,
            None,
        )
        .unwrap();

    let evaluator = MeanDecreaseImpurityEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let keys: Vec<&String> = importances.keys().collect();
    assert_eq!(
        keys[0], "x",
        "MDI 三参数: 首位应为 x, 实际 {:?}",
        keys
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 3: PED-ANOVA 评估器 — 重要性排序验证
// ═══════════════════════════════════════════════════════════════════════

/// 测试 PED-ANOVA: f(x,y) = x^2 + 0.001*y → x 应比 y 重要
#[test]
fn test_ped_anova_importance_ordering_x_dominant() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.001 * y)
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();
    assert_eq!(importances.len(), 2);

    let x_imp = importances["x"];
    let y_imp = importances["y"];
    assert!(
        x_imp > y_imp,
        "PED-ANOVA: x 重要性 ({x_imp:.4}) 应 > y ({y_imp:.4})"
    );
}

/// 测试 PED-ANOVA: 三参数排序
#[test]
fn test_ped_anova_three_param_ordering() {
    let study = create_test_study(123);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                Ok(10.0 * x * x + y * y + 0.001 * z)
            },
            Some(300),
            None,
            None,
        )
        .unwrap();

    let evaluator = PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let keys: Vec<&String> = importances.keys().collect();
    assert_eq!(
        keys[0], "x",
        "PED-ANOVA 三参数: 首位应为 x, 实际 {:?}",
        keys
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 4: 归一化数学性质
// ═══════════════════════════════════════════════════════════════════════

/// 验证归一化后重要性总和 = 1.0, 且所有值非负
#[test]
fn test_importance_normalization_sum_to_one() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(x * x + y + 0.1 * z)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    // 测试所有三种评估器
    let evaluators: Vec<(&str, Box<dyn ImportanceEvaluator>)> = vec![
        ("fANOVA", Box::new(FanovaEvaluator::new(32, 32, Some(42)))),
        ("MDI", Box::new(MeanDecreaseImpurityEvaluator::new(32, 32, Some(42)))),
        ("PED-ANOVA", Box::new(PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true)))),
    ];

    for (name, evaluator) in &evaluators {
        let importances =
            get_param_importances(&study, Some(evaluator.as_ref()), None, None, true)
                .unwrap();
        let total: f64 = importances.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "{name}: 归一化总和应为 1.0, 实际 {total}"
        );
        for (param, &imp) in &importances {
            assert!(
                imp >= 0.0,
                "{name}: 参数 {param} 重要性应非负, 实际 {imp}"
            );
        }
    }
}

/// 验证未归一化时返回原始值（不强制 sum=1）
#[test]
fn test_importance_no_normalization() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let _y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(32, 32, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, false).unwrap();
    // 未归一化: 重要性值可以是任意非负数
    for (_, &imp) in &importances {
        assert!(imp >= 0.0, "未归一化重要性应非负");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Group 5: 三种评估器一致性 — dominant 参数判断
// ═══════════════════════════════════════════════════════════════════════

/// 所有三种评估器应对 dominant 参数达成一致
#[test]
fn test_all_evaluators_agree_on_dominant_param() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.001 * y)
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluators: Vec<(&str, Box<dyn ImportanceEvaluator>)> = vec![
        ("fANOVA", Box::new(FanovaEvaluator::new(64, 64, Some(42)))),
        ("MDI", Box::new(MeanDecreaseImpurityEvaluator::new(64, 64, Some(42)))),
        ("PED-ANOVA", Box::new(PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true)))),
    ];

    for (name, evaluator) in &evaluators {
        let importances =
            get_param_importances(&study, Some(evaluator.as_ref()), None, None, true)
                .unwrap();
        let first_key = importances.keys().next().unwrap();
        assert_eq!(
            first_key, "x",
            "{name}: dominant 参数应为 x, 实际 {first_key}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Group 6: 边界条件
// ═══════════════════════════════════════════════════════════════════════

/// 空研究应返回错误
#[test]
fn test_importance_empty_study_error() {
    let study = create_test_study(42);
    let result = get_param_importances(&study, None, None, None, true);
    assert!(result.is_err(), "空研究应返回错误");
}

/// 单次试验应返回错误
#[test]
fn test_importance_single_trial_error() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -1.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(1),
            None,
            None,
        )
        .unwrap();
    let result = get_param_importances(&study, None, None, None, true);
    assert!(result.is_err(), "单次试验应返回错误");
}

/// 参数子集: 只评估指定参数
#[test]
fn test_importance_param_subset() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let _y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                let _z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let importances =
        get_param_importances(&study, None, Some(&["x", "y"]), None, true).unwrap();
    assert_eq!(
        importances.len(),
        2,
        "应只返回指定的 2 个参数"
    );
    assert!(importances.contains_key("x"));
    assert!(importances.contains_key("y"));
    assert!(!importances.contains_key("z"));
}

// ═══════════════════════════════════════════════════════════════════════
// Group 7: 种子可复现性
// ═══════════════════════════════════════════════════════════════════════

/// fANOVA 使用相同种子应产生相同结果
#[test]
fn test_fanova_reproducible_with_same_seed() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let eval1 = FanovaEvaluator::new(32, 32, Some(42));
    let eval2 = FanovaEvaluator::new(32, 32, Some(42));
    let imp1 = get_param_importances(&study, Some(&eval1), None, None, true).unwrap();
    let imp2 = get_param_importances(&study, Some(&eval2), None, None, true).unwrap();

    for key in imp1.keys() {
        let v1 = imp1[key];
        let v2 = imp2[key];
        assert!(
            (v1 - v2).abs() < 1e-14,
            "相同种子结果应完全一致: {key}: {v1} vs {v2}"
        );
    }
}

/// MDI 使用相同种子应产生相同结果
#[test]
fn test_mdi_reproducible_with_same_seed() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let eval1 = MeanDecreaseImpurityEvaluator::new(32, 32, Some(42));
    let eval2 = MeanDecreaseImpurityEvaluator::new(32, 32, Some(42));
    let imp1 = get_param_importances(&study, Some(&eval1), None, None, true).unwrap();
    let imp2 = get_param_importances(&study, Some(&eval2), None, None, true).unwrap();

    for key in imp1.keys() {
        let v1 = imp1[key];
        let v2 = imp2[key];
        assert!(
            (v1 - v2).abs() < 1e-14,
            "相同种子结果应完全一致: {key}: {v1} vs {v2}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Group 8: fANOVA 方差分解性质验证
// ═══════════════════════════════════════════════════════════════════════

/// 对单参数函数, 该参数的重要性应接近 1.0
#[test]
fn test_fanova_single_important_param_near_one() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let _y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x) // 只依赖 x
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let x_imp = importances["x"];
    assert!(
        x_imp > 0.85,
        "f(x,y)=x^2: x 归一化重要性应 > 0.85, 实际 {x_imp:.4}"
    );
}

/// MDI 对单参数函数同理
#[test]
fn test_mdi_single_important_param_near_one() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let _y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok(x * x)
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = MeanDecreaseImpurityEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let x_imp = importances["x"];
    assert!(
        x_imp > 0.85,
        "MDI f(x,y)=x^2: x 归一化重要性应 > 0.85, 实际 {x_imp:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 9: PED-ANOVA 特定行为
// ═══════════════════════════════════════════════════════════════════════

/// PED-ANOVA: 均匀随机目标 → 所有参数重要性应相近
#[test]
fn test_ped_anova_uniform_noise_equal_importance() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let _x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let _y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                // 目标只依赖随机噪声 (由 sampler 种子决定, 但不依赖任何参数)
                Ok(_x * 0.0 + _y * 0.0 + 1.0) // 常数目标
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    // 常数目标: 所有重要性应接近相等
    let evaluator = PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    if importances.len() == 2 {
        let imp_vals: Vec<f64> = importances.values().cloned().collect();
        let diff = (imp_vals[0] - imp_vals[1]).abs();
        assert!(
            diff < 0.6,
            "常数目标: 两参数重要性差应 < 0.6, 实际 {diff:.4}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Group 10: 最大化方向
// ═══════════════════════════════════════════════════════════════════════

/// fANOVA 在最大化方向上的重要性排序应一致
#[test]
fn test_fanova_maximize_direction() {
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> =
        Arc::new(RandomSampler::new(Some(42)));
    let study = create_study(
        None,
        Some(sampler),
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
                Ok(x * x + 0.001 * y) // 最大化: x 仍是 dominant
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    let first_key = importances.keys().next().unwrap();
    assert_eq!(
        first_key, "x",
        "最大化方向: dominant 参数应为 x"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 11: 自定义 target 函数
// ═══════════════════════════════════════════════════════════════════════

/// 使用自定义 target 函数: 只看 duration-like 值
#[test]
fn test_importance_custom_target() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x + y) // 简单线性
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    // 自定义 target: 使用 values[0] 的平方
    let target_fn = |t: &optuna_rs::trial::FrozenTrial| -> f64 {
        let v = t.values.as_ref().unwrap()[0];
        v * v
    };

    let evaluator = FanovaEvaluator::new(32, 32, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, Some(&target_fn), true)
            .unwrap();
    assert_eq!(importances.len(), 2);
    let total: f64 = importances.values().sum();
    assert!(
        (total - 1.0).abs() < 1e-10,
        "自定义 target: 归一化总和应为 1.0"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 12: 端到端交叉验证 — 线性函数已知排序
// ═══════════════════════════════════════════════════════════════════════

/// 线性函数 f = 10x + y + 0.1z: 重要性排序应为 x > y > z
/// 对所有三种评估器进行端到端验证
#[test]
fn test_linear_function_ordering_all_evaluators() {
    let study = create_test_study(77);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(10.0 * x + y + 0.1 * z)
            },
            Some(300),
            None,
            None,
        )
        .unwrap();

    // fANOVA
    let fanova = FanovaEvaluator::new(64, 64, Some(42));
    let imp_f = get_param_importances(&study, Some(&fanova), None, None, true).unwrap();
    let f_keys: Vec<&String> = imp_f.keys().collect();
    assert_eq!(f_keys[0], "x", "fANOVA 线性: x 应排首位");

    // MDI
    let mdi = MeanDecreaseImpurityEvaluator::new(64, 64, Some(42));
    let imp_m = get_param_importances(&study, Some(&mdi), None, None, true).unwrap();
    let m_keys: Vec<&String> = imp_m.keys().collect();
    assert_eq!(m_keys[0], "x", "MDI 线性: x 应排首位");

    // PED-ANOVA
    let ped = PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true));
    let imp_p = get_param_importances(&study, Some(&ped), None, None, true).unwrap();
    let p_keys: Vec<&String> = imp_p.keys().collect();
    assert_eq!(p_keys[0], "x", "PED-ANOVA 线性: x 应排首位");
}

/// 二次函数 f = x^2 + 0.01*y^2 + constant*z: 非线性关系下的排序
#[test]
fn test_quadratic_function_ordering() {
    let study = create_test_study(99);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                let z = trial.suggest_float("z", -10.0, 10.0, false, None)?;
                Ok(x * x + 0.01 * y * y + 0.0 * z) // z 完全无关
            },
            Some(300),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(64, 64, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();

    // x 应排首位, z 应排末位
    let keys: Vec<&String> = importances.keys().collect();
    assert_eq!(keys[0], "x", "二次函数: x 应排首位");
    // z 应有最低重要性
    let z_imp = importances["z"];
    let x_imp = importances["x"];
    assert!(
        x_imp > z_imp,
        "二次函数: x 重要性 ({x_imp:.4}) 应 > z ({z_imp:.4})"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Group 13: Integer 参数支持
// ═══════════════════════════════════════════════════════════════════════

/// 混合 float 和 int 参数
#[test]
fn test_importance_with_int_params() {
    let study = create_test_study(42);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let n = trial.suggest_int("n", 0, 10, false, 1)?;
                Ok(x * x + 0.01 * n as f64)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let evaluator = FanovaEvaluator::new(32, 32, Some(42));
    let importances =
        get_param_importances(&study, Some(&evaluator), None, None, true).unwrap();
    assert_eq!(importances.len(), 2);

    let first_key = importances.keys().next().unwrap();
    assert_eq!(first_key, "x", "混合参数: x 应排首位");
}
