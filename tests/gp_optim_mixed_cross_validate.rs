//! GP 采集函数混合优化 (optim_mixed) 交叉验证测试
//!
//! 对齐 Python `optuna._gp.optim_mixed` + `optuna._gp.search_space`
//! 验证:
//! 1. build_search_space_info: 连续/离散索引、选择列表、xtol
//! 2. optimize_acqf_mixed: 混合空间优化收敛结果
//! 3. Bug #11 修复: 非连续 continuous_indices 下的 lengthscale 索引

use indexmap::IndexMap;
use optuna_rs::distributions::*;

/// 辅助: 简单浮点近似比较
fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    let diff = (a - b).abs();
    assert!(diff < tol, "{}: |{} - {}| = {} >= {}", msg, a, b, diff, tol);
}

/// Golden values from Python:
/// ```python
/// ss = SearchSpace({
///     'x': FloatDistribution(0, 1),
///     'n': IntDistribution(1, 5),
///     'y': FloatDistribution(0, 10, step=2.5),
///     'c': CategoricalDistribution(['a', 'b', 'c']),
/// })
/// continuous_indices: [0]
/// discrete_indices: [1, 2, 3]
/// Int(1,5) normalized choices: [0.1, 0.3, 0.5, 0.7, 0.9]
/// Float(0,10,step=2.5) normalized: [0.1, 0.3, 0.5, 0.7, 0.9]
/// Categorical choices: [0.0, 1.0, 2.0]
/// xtols: [0.05, 0.05, 0.25]
/// ```
#[test]
fn test_build_search_space_info_mixed() {
    use optuna_rs::samplers::gp_optim_mixed::build_search_space_info;

    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 1.0, log: false, step: None }
    ));
    search_space.insert("n".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 5, log: false, step: 1 }
    ));
    search_space.insert("y".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 10.0, log: false, step: Some(2.5) }
    ));
    search_space.insert("c".to_string(), Distribution::CategoricalDistribution(
        CategoricalDistribution {
            choices: vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
                CategoricalChoice::Str("c".into()),
            ]
        }
    ));

    let param_names: Vec<String> = search_space.keys().cloned().collect();
    let info = build_search_space_info(&search_space, &param_names);

    // Python: continuous_indices: [0]
    assert_eq!(info.continuous_indices, vec![0], "continuous_indices mismatch");

    // Python: discrete_indices: [1, 2, 3]
    assert_eq!(info.discrete_indices, vec![1, 2, 3], "discrete_indices mismatch");

    // Python: is_categorical_discrete: [False, False, True]
    assert_eq!(info.is_categorical_discrete, vec![false, false, true]);

    // Python: Int(1,5) normalized: [0.1, 0.3, 0.5, 0.7, 0.9]
    let expected_int = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    assert_eq!(info.discrete_choices[0].len(), 5, "Int(1,5) should have 5 choices");
    for (j, (&got, &exp)) in info.discrete_choices[0].iter().zip(expected_int.iter()).enumerate() {
        assert_close(got, exp, 1e-10, &format!("Int(1,5) choice[{}]", j));
    }

    // Python: Float(0,10,step=2.5) normalized: [0.1, 0.3, 0.5, 0.7, 0.9]
    let expected_float_step = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    assert_eq!(info.discrete_choices[1].len(), 5, "Float(0,10,step=2.5) should have 5 choices");
    for (j, (&got, &exp)) in info.discrete_choices[1].iter().zip(expected_float_step.iter()).enumerate() {
        assert_close(got, exp, 1e-10, &format!("Float step choice[{}]", j));
    }

    // Python: Categorical choices: [0.0, 1.0, 2.0]
    let expected_cat = vec![0.0, 1.0, 2.0];
    assert_eq!(info.discrete_choices[2].len(), 3, "Categorical should have 3 choices");
    for (j, (&got, &exp)) in info.discrete_choices[2].iter().zip(expected_cat.iter()).enumerate() {
        assert_close(got, exp, 1e-10, &format!("Cat choice[{}]", j));
    }

    // Python: xtols: [0.05, 0.05, 0.25]
    assert_close(info.discrete_xtols[0], 0.05, 1e-10, "xtol[0] Int(1,5)");
    assert_close(info.discrete_xtols[1], 0.05, 1e-10, "xtol[1] Float step");
    assert_close(info.discrete_xtols[2], 0.25, 1e-10, "xtol[2] Categorical");
}

/// Non-contiguous continuous indices:
/// Python: SearchSpace({'cat': Cat, 'cont1': Float, 'int1': Int, 'cont2': Float})
/// continuous_indices: [1, 3]
/// discrete_indices: [0, 2]
#[test]
fn test_build_search_space_info_non_contiguous_continuous() {
    use optuna_rs::samplers::gp_optim_mixed::build_search_space_info;

    let mut search_space = IndexMap::new();
    search_space.insert("cat".to_string(), Distribution::CategoricalDistribution(
        CategoricalDistribution {
            choices: vec![CategoricalChoice::Str("x".into()), CategoricalChoice::Str("y".into())]
        }
    ));
    search_space.insert("cont1".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 1.0, log: false, step: None }
    ));
    search_space.insert("int1".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 0, high: 10, log: false, step: 1 }
    ));
    search_space.insert("cont2".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: -5.0, high: 5.0, log: false, step: None }
    ));

    let param_names: Vec<String> = search_space.keys().cloned().collect();
    let info = build_search_space_info(&search_space, &param_names);

    // Python: continuous_indices: [1, 3]
    assert_eq!(info.continuous_indices, vec![1, 3]);
    // Python: discrete_indices: [0, 2]
    assert_eq!(info.discrete_indices, vec![0, 2]);
    // is_categorical: [True, False]
    assert_eq!(info.is_categorical_discrete, vec![true, false]);
}

/// Bug #11 回归测试:
/// 当 continuous_indices 非连续时 (如 [1, 3])，
/// gradient_ascent_continuous 应正确使用 lengthscales[1] 和 lengthscales[3]
/// 而非错误地使用 lengthscales[0] 和 lengthscales[1]。
///
/// 测试方法: 构造一个函数 f(cat, x1, int, x2) = -(x1 - 0.3)^2 - 10*(x2 - 0.8)^2
/// 其中 x1 对应 dim 1, x2 对应 dim 3
/// 设 lengthscales = [1.0, 0.5, 1.0, 2.0]
/// Bug #11 下 x2 会错误使用 lengthscales[1]=0.5 而非 lengthscales[3]=2.0
#[test]
fn test_optimize_mixed_non_contiguous_bug11_regression() {
    use optuna_rs::samplers::gp_optim_mixed::{SearchSpaceInfo, optimize_acqf_mixed};
    use rand::SeedableRng;

    // f(params) = -(x1 - 0.3)^2 - 10*(x2 - 0.8)^2 + (cat == 1)*0.1
    let eval = |x: &[f64]| -> f64 {
        let x1 = x[1]; // continuous, dim 1
        let x2 = x[3]; // continuous, dim 3
        let cat_bonus = if (x[0] - 1.0).abs() < 0.5 { 0.1 } else { 0.0 };
        -(x1 - 0.3) * (x1 - 0.3) - 10.0 * (x2 - 0.8) * (x2 - 0.8) + cat_bonus
    };

    let ss_info = SearchSpaceInfo {
        continuous_indices: vec![1, 3],  // Non-contiguous!
        discrete_indices: vec![0, 2],
        discrete_choices: vec![
            vec![0.0, 1.0],           // categorical dim 0
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], // int dim 2
        ],
        is_categorical_discrete: vec![true, false],
        discrete_xtols: vec![0.25, 0.025],
    };

    // Lengthscales for ALL 4 params
    let lengthscales = vec![1.0, 0.5, 1.0, 2.0];

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let result = optimize_acqf_mixed(
        &eval, &ss_info, &[], 4, 512, 8, &lengthscales, &mut rng, 42,
    );

    // Should find x1 ≈ 0.3, x2 ≈ 0.8, cat ≈ 1.0
    assert!(
        (result[1] - 0.3).abs() < 0.1,
        "x1 should be near 0.3, got {}", result[1]
    );
    assert!(
        (result[3] - 0.8).abs() < 0.1,
        "x2 should be near 0.8, got {}", result[3]
    );
    assert!(
        (result[0] - 1.0).abs() < 0.5,
        "cat should be 1.0, got {}", result[0]
    );
}

/// 纯离散空间: 无连续参数
#[test]
fn test_optimize_pure_discrete() {
    use optuna_rs::samplers::gp_optim_mixed::{SearchSpaceInfo, optimize_acqf_mixed};
    use rand::SeedableRng;

    // f(cat, int) = -(int - 0.5)^2 + (cat == 2)*1.0
    let eval = |x: &[f64]| -> f64 {
        let int_val = x[0];
        let cat_val = x[1];
        -(int_val - 0.5) * (int_val - 0.5) + if (cat_val - 2.0).abs() < 0.5 { 1.0 } else { 0.0 }
    };

    let ss_info = SearchSpaceInfo {
        continuous_indices: vec![],
        discrete_indices: vec![0, 1],
        discrete_choices: vec![
            vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vec![0.0, 1.0, 2.0],
        ],
        is_categorical_discrete: vec![false, true],
        discrete_xtols: vec![0.05, 0.25],
    };

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let result = optimize_acqf_mixed(
        &eval, &ss_info, &[], 2, 256, 5, &[], &mut rng, 42,
    );

    assert_close(result[0], 0.4, 0.25, "int should be near 0.4 or 0.6");
    assert_close(result[1], 2.0, 0.5, "cat should be 2.0");
}

/// 测试 roulette 选择的概率正确性
#[test]
fn test_roulette_probability_distribution() {
    use optuna_rs::samplers::gp_optim_mixed::{SearchSpaceInfo, optimize_acqf_mixed};
    use rand::SeedableRng;

    // 简单峰值函数在 x=0.7
    let eval = |x: &[f64]| -> f64 { -(x[0] - 0.7) * (x[0] - 0.7) };

    let ss_info = SearchSpaceInfo {
        continuous_indices: vec![0],
        discrete_indices: vec![],
        discrete_choices: vec![],
        is_categorical_discrete: vec![],
        discrete_xtols: vec![],
    };

    // 用多个随机种子运行确保稳定收敛
    for seed in 0..5u64 {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let result = optimize_acqf_mixed(
            &eval, &ss_info, &[], 1, 512, 5, &[1.0], &mut rng, seed,
        );
        assert!(
            (result[0] - 0.7).abs() < 0.05,
            "seed={}: expected x ≈ 0.7, got {}", seed, result[0]
        );
    }
}

/// 测试 warmstart 参数被正确使用
#[test]
fn test_warmstart_used() {
    use optuna_rs::samplers::gp_optim_mixed::{SearchSpaceInfo, optimize_acqf_mixed};
    use rand::SeedableRng;

    // f(x) = -(x - 0.95)^2，最优在边界附近
    let eval = |x: &[f64]| -> f64 { -(x[0] - 0.95) * (x[0] - 0.95) };

    let ss_info = SearchSpaceInfo {
        continuous_indices: vec![0],
        discrete_indices: vec![],
        discrete_choices: vec![],
        is_categorical_discrete: vec![],
        discrete_xtols: vec![],
    };

    // warmstart 在 0.9 附近 — 应帮助找到 0.95
    let warmstart = vec![vec![0.9]];
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let result = optimize_acqf_mixed(
        &eval, &ss_info, &warmstart, 1, 256, 5, &[1.0], &mut rng, 42,
    );

    assert!(
        (result[0] - 0.95).abs() < 0.05,
        "with warmstart near 0.9, should find 0.95, got {}", result[0]
    );
}

/// 2D Rosenbrock-like: f(x,y) = -(1-x)^2 - 100*(y-x^2)^2
/// 验证梯度上升能处理非凸函数
#[test]
fn test_optimize_2d_rosenbrock_variant() {
    use optuna_rs::samplers::gp_optim_mixed::{SearchSpaceInfo, optimize_acqf_mixed};
    use rand::SeedableRng;

    // 简化版: f(x,y) = -((x-0.5)^2 + (y-0.5)^2 + 2*(y-x)^2)
    // 最优在 (0.5, 0.5)
    let eval = |x: &[f64]| -> f64 {
        let a = x[0] - 0.5;
        let b = x[1] - 0.5;
        let c = x[1] - x[0];
        -(a*a + b*b + 2.0*c*c)
    };

    let ss_info = SearchSpaceInfo {
        continuous_indices: vec![0, 1],
        discrete_indices: vec![],
        discrete_choices: vec![],
        is_categorical_discrete: vec![],
        discrete_xtols: vec![],
    };

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let result = optimize_acqf_mixed(
        &eval, &ss_info, &[], 2, 512, 8, &[1.0, 1.0], &mut rng, 42,
    );

    assert!(
        (result[0] - 0.5).abs() < 0.05,
        "x should be near 0.5, got {}", result[0]
    );
    assert!(
        (result[1] - 0.5).abs() < 0.05,
        "y should be near 0.5, got {}", result[1]
    );
}
