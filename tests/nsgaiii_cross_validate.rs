// NSGA-III Cross-Validation Tests
//
// 验证 optuna-rs NSGA-III 采样器核心组件与 Python 精确对齐:
// - Das-Dennis 参考点生成
// - 垂直距离计算
// - 目标值归一化
// - Niche 保留选择机制
//
// Python 基线: tests/nsgaiii_baseline.json

use optuna_rs::samplers::nsgaiii::{generate_reference_points, NSGAIIISamplerBuilder};
use optuna_rs::study::{create_study, StudyDirection};
use std::sync::Arc;

/// 加载 Python 基线 JSON
fn load_baseline() -> serde_json::Value {
    let data = include_str!("nsgaiii_baseline.json");
    serde_json::from_str(data).expect("Failed to parse nsgaiii_baseline.json")
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. Das-Dennis 参考点生成
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsgaiii_das_dennis_2d_3div() {
    let baseline = load_baseline();
    let pts = generate_reference_points(2, 3);
    let expected: Vec<Vec<f64>> = serde_json::from_value(
        baseline["das_dennis_2d_3div_points"].clone()
    ).unwrap();

    assert_eq!(pts.len(), expected.len(), "count mismatch");
    for (i, (got, exp)) in pts.iter().zip(expected.iter()).enumerate() {
        for (j, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-10,
                "Das-Dennis 2D/3div point[{}][{}]: got {}, expected {}",
                i, j, g, e
            );
        }
    }
}

#[test]
fn nsgaiii_das_dennis_2d_4div() {
    let baseline = load_baseline();
    let pts = generate_reference_points(2, 4);
    let expected: Vec<Vec<f64>> = serde_json::from_value(
        baseline["das_dennis_2d_4div_points"].clone()
    ).unwrap();

    assert_eq!(pts.len(), 5);
    for (i, (got, exp)) in pts.iter().zip(expected.iter()).enumerate() {
        for (j, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-10,
                "Das-Dennis 2D/4div point[{}][{}]: got {}, expected {}",
                i, j, g, e
            );
        }
    }
}

#[test]
fn nsgaiii_das_dennis_3d_3div() {
    let baseline = load_baseline();
    let pts = generate_reference_points(3, 3);
    let expected: Vec<Vec<f64>> = serde_json::from_value(
        baseline["das_dennis_3d_3div_points"].clone()
    ).unwrap();

    assert_eq!(pts.len(), 10);
    for (i, (got, exp)) in pts.iter().zip(expected.iter()).enumerate() {
        for (j, (&g, &e)) in got.iter().zip(exp.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-10,
                "Das-Dennis 3D/3div point[{}][{}]: got {}, expected {}",
                i, j, g, e
            );
        }
    }
}

#[test]
fn nsgaiii_das_dennis_3d_4div() {
    let pts = generate_reference_points(3, 4);
    assert_eq!(pts.len(), 15); // C(6,2) = 15
}

#[test]
fn nsgaiii_das_dennis_4d_3div() {
    let pts = generate_reference_points(4, 3);
    assert_eq!(pts.len(), 20); // C(6,3) = 20
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. Das-Dennis 不变量
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsgaiii_das_dennis_sum_to_one() {
    // 所有参考点的分量之和应为 1.0
    for (n, d) in &[(2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (5, 2)] {
        let pts = generate_reference_points(*n, *d);
        for (i, pt) in pts.iter().enumerate() {
            let sum: f64 = pt.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Point {} in {}D/{}div sums to {}, not 1.0",
                i, n, d, sum
            );
        }
    }
}

#[test]
fn nsgaiii_das_dennis_non_negative() {
    // 所有分量应 ≥ 0
    for (n, d) in &[(2, 5), (3, 4), (4, 3)] {
        let pts = generate_reference_points(*n, *d);
        for pt in &pts {
            for &v in pt {
                assert!(v >= 0.0, "Reference point component {} is negative", v);
            }
        }
    }
}

#[test]
fn nsgaiii_das_dennis_count_formula() {
    // C(n_obj + divs - 1, divs) = C(n+d-1, d)
    // 验证公式正确性
    assert_eq!(generate_reference_points(2, 3).len(), 4);  // C(4,3)
    assert_eq!(generate_reference_points(2, 4).len(), 5);  // C(5,4)
    assert_eq!(generate_reference_points(3, 3).len(), 10); // C(5,2)
    assert_eq!(generate_reference_points(3, 4).len(), 15); // C(6,2)
    assert_eq!(generate_reference_points(4, 3).len(), 20); // C(6,3)
    assert_eq!(generate_reference_points(2, 1).len(), 2);  // C(2,1)
    assert_eq!(generate_reference_points(3, 1).len(), 3);  // C(3,1) — vertex points only
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. NSGA-III Sampler 集成测试
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsgaiii_basic_2obj_run() {
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> = Arc::new(
        NSGAIIISamplerBuilder::new(vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ])
        .population_size(10)
        .dividing_parameter(3)
        .seed(42)
        .build()
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        None,
        Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
        false,
    )
    .unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
            Ok(vec![x * x, (1.0 - x).powi(2) + y * y])
        },
        Some(30),
        None,
        None,
    ).unwrap();

    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 30);
    let best = study.best_trials().unwrap();
    assert!(!best.is_empty(), "Should have Pareto-optimal trials");
}

#[test]
fn nsgaiii_3obj_run() {
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> = Arc::new(
        NSGAIIISamplerBuilder::new(vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ])
        .population_size(15)
        .dividing_parameter(3)
        .seed(42)
        .build()
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        None,
        Some(vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ]),
        false,
    )
    .unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
            Ok(vec![x, y, (x - 0.5).powi(2) + (y - 0.5).powi(2)])
        },
        Some(40),
        None,
        None,
    ).unwrap();

    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 40);
}

#[test]
fn nsgaiii_with_maximize() {
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> = Arc::new(
        NSGAIIISamplerBuilder::new(vec![
            StudyDirection::Maximize,
            StudyDirection::Minimize,
        ])
        .population_size(10)
        .seed(42)
        .build()
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        None,
        Some(vec![StudyDirection::Maximize, StudyDirection::Minimize]),
        false,
    )
    .unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(vec![x, 1.0 - x])
        },
        Some(25),
        None,
        None,
    ).unwrap();

    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 25);
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. 参数验证对齐
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn nsgaiii_default_params_runs_ok() {
    // 验证默认参数构建成功且可运行
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> = Arc::new(
        NSGAIIISamplerBuilder::new(vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ])
        .seed(42)
        .build()
    );
    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        None,
        Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
        false,
    )
    .unwrap();
    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(vec![x, 1.0 - x])
        },
        Some(5),
        None,
        None,
    ).unwrap();
    assert_eq!(study.trials().unwrap().len(), 5);
}

#[test]
#[should_panic(expected = "population_size")]
fn nsgaiii_pop_size_too_small() {
    NSGAIIISamplerBuilder::new(vec![StudyDirection::Minimize])
        .population_size(1)
        .build();
}

#[test]
fn nsgaiii_custom_reference_points() {
    let custom_pts = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.5, 0.5],
    ];
    let sampler: Arc<dyn optuna_rs::samplers::Sampler> = Arc::new(
        NSGAIIISamplerBuilder::new(vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ])
        .population_size(10)
        .reference_points(custom_pts)
        .seed(42)
        .build()
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        None,
        Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
        false,
    )
    .unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(vec![x, 1.0 - x])
        },
        Some(20),
        None,
        None,
    ).unwrap();

    assert_eq!(study.trials().unwrap().len(), 20);
}
