//! PartialFixedSampler 与 SearchSpaceTransform 交叉验证测试。
//!
//! 覆盖: PartialFixedSampler 固定参数行为、SearchSpaceTransform 编码/解码精度。
//! 所有参考值来自 Python optuna。

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use optuna_rs::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution,
    IntDistribution, ParamValue,
};
use optuna_rs::samplers::{PartialFixedSampler, RandomSampler, Sampler};
use optuna_rs::search_space::SearchSpaceTransform;
use optuna_rs::study::{create_study, StudyDirection};

// ============================================================================
// 1. PartialFixedSampler 基本行为
// ============================================================================

/// Python 参考: 固定 x=3.0, 其余参数由 base sampler 采样
/// 所有 trial 的 x 应等于 3.0
#[test]
fn test_partial_fixed_float_param() {
    let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let mut fixed = HashMap::new();
    fixed.insert("x".to_string(), 3.0);
    let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 10.0, false, None)?;
            Ok((x - 3.0).powi(2) + y.powi(2))
        },
        Some(20), None, None,
    ).unwrap();

    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 20);

    // 所有 trial 的 x 应固定为 3.0
    for t in &trials {
        if let Some(ParamValue::Float(x)) = t.params.get("x") {
            assert!(
                (*x - 3.0).abs() < 1e-10,
                "x should be fixed at 3.0, got {}",
                x
            );
        } else {
            panic!("x param not found or not float");
        }
    }

    // y 应有变化 (不同 trial 的 y 不应全相同)
    let y_vals: Vec<f64> = trials.iter()
        .filter_map(|t| match t.params.get("y") {
            Some(ParamValue::Float(y)) => Some(*y),
            _ => None,
        })
        .collect();
    let y_unique: std::collections::HashSet<u64> = y_vals.iter()
        .map(|&y| y.to_bits())
        .collect();
    assert!(y_unique.len() > 1, "y values should vary across trials");
}

/// 固定整数参数
#[test]
fn test_partial_fixed_int_param() {
    let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let mut fixed = HashMap::new();
    fixed.insert("n".to_string(), 5.0); // int 5 stored as f64
    let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let n = trial.suggest_int("n", 0, 10, false, 1)?;
            let x = trial.suggest_float("x", 0.0, 5.0, false, None)?;
            Ok((n as f64 - 5.0).powi(2) + x.powi(2))
        },
        Some(10), None, None,
    ).unwrap();

    let trials = study.trials().unwrap();
    for t in &trials {
        match t.params.get("n") {
            Some(ParamValue::Int(n)) => assert_eq!(*n, 5, "n should be fixed at 5"),
            Some(ParamValue::Float(n)) => assert!((*n - 5.0).abs() < 1e-10, "n should be 5.0"),
            _ => panic!("n not found"),
        }
    }
}

/// 固定未知参数 (不在搜索空间中) — 应被忽略
#[test]
fn test_partial_fixed_unknown_param() {
    let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let mut fixed = HashMap::new();
    fixed.insert("nonexistent".to_string(), 99.0);
    let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(x * x)
        },
        Some(5), None, None,
    ).unwrap();

    assert_eq!(study.trials().unwrap().len(), 5);
}

// ============================================================================
// 2. SearchSpaceTransform Float 编码/解码
// ============================================================================

/// Python 参考: float [2, 8], transform_log=true, transform_step=true, transform_0_1=false
/// transform(v) = v (non-log, non-step → identity)
#[test]
fn test_sst_float_identity() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 2.0, high: 8.0, log: false, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    let cases: Vec<(f64, f64)> = vec![
        (2.0, 2.0), (5.0, 5.0), (8.0, 8.0), (3.5, 3.5),
    ];

    for (val, expected_encoded) in &cases {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(*val));
        let encoded = t.transform(&params);
        assert!(
            (encoded[0] - expected_encoded).abs() < 1e-14,
            "transform({val}) = {}, expected {expected_encoded}",
            encoded[0]
        );

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("x") {
            Some(ParamValue::Float(v)) => assert!(
                (*v - val).abs() < 1e-12,
                "untransform mismatch: {v} vs {val}"
            ),
            _ => panic!("untransform failed"),
        }
    }
}

/// Float log scale: transform(v) = ln(v)
#[test]
fn test_sst_float_log() {
    let mut ss = IndexMap::new();
    ss.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 1e-5, high: 1.0, log: true, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    let cases: Vec<(f64, f64)> = vec![
        (1e-5, -1.1512925464970229e+01),
        (0.001, -6.9077552789821368e+00),
        (0.01, -4.6051701859880909e+00),
        (0.1, -2.3025850929940455e+00),
        (1.0, 0.0),
    ];

    for (val, expected_encoded) in &cases {
        let mut params = IndexMap::new();
        params.insert("lr".to_string(), ParamValue::Float(*val));
        let encoded = t.transform(&params);
        assert!(
            (encoded[0] - expected_encoded).abs() < 1e-10,
            "transform({val}) = {:.17e}, expected {:.17e}",
            encoded[0], expected_encoded
        );

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("lr") {
            Some(ParamValue::Float(v)) => {
                assert!(
                    ((*v - val) / val).abs() < 1e-10,
                    "untransform: {} vs {val}",
                    v
                );
            }
            _ => panic!("untransform failed"),
        }
    }
}

// ============================================================================
// 3. SearchSpaceTransform Int 编码/解码
// ============================================================================

/// Int [0, 10]: transform(v) = v as f64
#[test]
fn test_sst_int_identity() {
    let mut ss = IndexMap::new();
    ss.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution {
            low: 0, high: 10, log: false, step: 1,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    for val in [0, 3, 7, 10] {
        let mut params = IndexMap::new();
        params.insert("n".to_string(), ParamValue::Int(val));
        let encoded = t.transform(&params);
        assert!(
            (encoded[0] - val as f64).abs() < 1e-14,
            "transform({val}) = {}",
            encoded[0]
        );

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("n") {
            Some(ParamValue::Int(v)) => assert_eq!(*v, val, "untransform mismatch"),
            _ => panic!("untransform failed"),
        }
    }
}

// ============================================================================
// 4. SearchSpaceTransform Categorical 编码/解码 (one-hot)
// ============================================================================

/// Python 参考: categorical ['a','b','c'] → one-hot [1,0,0], [0,1,0], [0,0,1]
#[test]
fn test_sst_categorical_one_hot() {
    let mut ss = IndexMap::new();
    ss.insert(
        "c".to_string(),
        Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".to_string()),
                CategoricalChoice::Str("b".to_string()),
                CategoricalChoice::Str("c".to_string()),
            ]).unwrap()
        ),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // n_encoded should be 3 (one column per choice)
    assert_eq!(t.n_encoded(), 3);

    // 'a' → [1, 0, 0]
    let mut params_a = IndexMap::new();
    params_a.insert("c".to_string(), ParamValue::Categorical(CategoricalChoice::Str("a".to_string())));
    let enc_a = t.transform(&params_a);
    assert_eq!(enc_a, vec![1.0, 0.0, 0.0]);

    // 'b' → [0, 1, 0]
    let mut params_b = IndexMap::new();
    params_b.insert("c".to_string(), ParamValue::Categorical(CategoricalChoice::Str("b".to_string())));
    let enc_b = t.transform(&params_b);
    assert_eq!(enc_b, vec![0.0, 1.0, 0.0]);

    // 'c' → [0, 0, 1]
    let mut params_c = IndexMap::new();
    params_c.insert("c".to_string(), ParamValue::Categorical(CategoricalChoice::Str("c".to_string())));
    let enc_c = t.transform(&params_c);
    assert_eq!(enc_c, vec![0.0, 0.0, 1.0]);

    // Roundtrip
    let dec_a = t.untransform(&enc_a).unwrap();
    match dec_a.get("c") {
        Some(ParamValue::Categorical(CategoricalChoice::Str(s))) => assert_eq!(s, "a"),
        _ => panic!("untransform failed for 'a'"),
    }
    let dec_b = t.untransform(&enc_b).unwrap();
    match dec_b.get("c") {
        Some(ParamValue::Categorical(CategoricalChoice::Str(s))) => assert_eq!(s, "b"),
        _ => panic!("untransform failed for 'b'"),
    }
}

// ============================================================================
// 5. SearchSpaceTransform 0-1 缩放
// ============================================================================

/// Python 参考: float [2, 8], transform_0_1=true
/// transform(2.0) = 0.0, transform(5.0) = 0.5, transform(8.0) = 1.0
#[test]
fn test_sst_transform_0_1() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 2.0, high: 8.0, log: false, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, true);

    let cases: Vec<(f64, f64)> = vec![
        (2.0, 0.0), (5.0, 0.5), (8.0, 1.0),
    ];

    for (val, expected_01) in &cases {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(*val));
        let encoded = t.transform(&params);
        assert!(
            (encoded[0] - expected_01).abs() < 1e-14,
            "0-1 transform({val}) = {}, expected {expected_01}",
            encoded[0]
        );

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("x") {
            Some(ParamValue::Float(v)) => assert!(
                (*v - val).abs() < 1e-10,
                "0-1 untransform: {v} vs {val}"
            ),
            _ => panic!("untransform failed"),
        }
    }

    // Bounds should all be [0, 1]
    let bounds = t.bounds();
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0], [0.0, 1.0]);
}

// ============================================================================
// 6. SearchSpaceTransform 混合搜索空间
// ============================================================================

/// 混合 float + int + categorical 搜索空间
#[test]
fn test_sst_mixed_space() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 0.0, high: 1.0, log: false, step: None,
        }),
    );
    ss.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution {
            low: 1, high: 5, log: false, step: 1,
        }),
    );
    ss.insert(
        "cat".to_string(),
        Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".to_string()),
                CategoricalChoice::Str("b".to_string()),
            ]).unwrap()
        ),
    );

    let t = SearchSpaceTransform::new(ss, true, true, false);

    // n_encoded: float(1) + int(1) + cat(2) = 4
    assert_eq!(t.n_encoded(), 4);

    // column_to_encoded_columns
    let c2e = t.column_to_encoded_columns();
    assert_eq!(c2e[0], 0..1); // x
    assert_eq!(c2e[1], 1..2); // n
    assert_eq!(c2e[2], 2..4); // cat (2 choices)

    // encoded_column_to_column
    let e2c = t.encoded_column_to_column();
    assert_eq!(e2c, &[0, 1, 2, 2]); // 列0→x, 列1→n, 列2,3→cat

    // Transform roundtrip
    let mut params = IndexMap::new();
    params.insert("x".to_string(), ParamValue::Float(0.5));
    params.insert("n".to_string(), ParamValue::Int(3));
    params.insert("cat".to_string(), ParamValue::Categorical(CategoricalChoice::Str("b".to_string())));

    let encoded = t.transform(&params);
    assert_eq!(encoded.len(), 4);

    let decoded = t.untransform(&encoded).unwrap();
    match decoded.get("x") {
        Some(ParamValue::Float(v)) => assert!((v - 0.5).abs() < 1e-10),
        _ => panic!("x decode fail"),
    }
    match decoded.get("n") {
        Some(ParamValue::Int(v)) => assert_eq!(*v, 3),
        _ => panic!("n decode fail"),
    }
    match decoded.get("cat") {
        Some(ParamValue::Categorical(CategoricalChoice::Str(s))) => assert_eq!(s, "b"),
        _ => panic!("cat decode fail"),
    }
}

// ============================================================================
// 7. SearchSpaceTransform float with step
// ============================================================================

/// Float with step: bounds should include ±half_step padding
#[test]
fn test_sst_float_step() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 0.0, high: 1.0, log: false, step: Some(0.25),
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // Bounds should be padded: [0 - 0.125, 1 + 0.125] = [-0.125, 1.125]
    let bounds = t.bounds();
    let [lo, hi] = bounds[0];
    assert!(
        (lo - (-0.125)).abs() < 1e-14,
        "step lower bound: {lo}"
    );
    assert!(
        (hi - 1.125).abs() < 1e-14,
        "step upper bound: {hi}"
    );

    // Transform/untransform roundtrip
    let mut params = IndexMap::new();
    params.insert("x".to_string(), ParamValue::Float(0.5));
    let encoded = t.transform(&params);
    let decoded = t.untransform(&encoded).unwrap();
    match decoded.get("x") {
        Some(ParamValue::Float(v)) => assert!((*v - 0.5).abs() < 1e-10),
        _ => panic!("step untransform fail"),
    }
}

// ============================================================================
// 8. SearchSpaceTransform bounds 验证
// ============================================================================

/// 非 0-1 模式下 bounds 应反映原始分布范围
#[test]
fn test_sst_bounds_raw() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: -5.0, high: 5.0, log: false, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    let bounds = t.bounds();
    assert_eq!(bounds.len(), 1);
    assert!((bounds[0][0] - (-5.0)).abs() < 1e-14);
    assert!((bounds[0][1] - 5.0).abs() < 1e-14);
}

/// Log 分布 bounds 应是 ln 空间
#[test]
fn test_sst_bounds_log() {
    let mut ss = IndexMap::new();
    ss.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 1e-5, high: 1.0, log: true, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    let bounds = t.bounds();
    let [lo, hi] = bounds[0];
    assert!(
        (lo - 1e-5_f64.ln()).abs() < 1e-10,
        "log lower bound: {lo} vs {}",
        1e-5_f64.ln()
    );
    assert!(
        (hi - 1.0_f64.ln()).abs() < 1e-10,
        "log upper bound: {hi} vs {}",
        1.0_f64.ln()
    );
}

// ============================================================================
// 9. PartialFixedSampler 多参数固定
// ============================================================================

/// 同时固定多个参数
#[test]
fn test_partial_fixed_multiple_params() {
    let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let mut fixed = HashMap::new();
    fixed.insert("x".to_string(), 1.0);
    fixed.insert("y".to_string(), 2.0);
    let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 5.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 5.0, false, None)?;
            let z = trial.suggest_float("z", 0.0, 5.0, false, None)?;
            Ok(x + y + z)
        },
        Some(10), None, None,
    ).unwrap();

    let trials = study.trials().unwrap();
    for t in &trials {
        match t.params.get("x") {
            Some(ParamValue::Float(v)) => assert!((v - 1.0).abs() < 1e-10, "x={v}"),
            _ => panic!("x missing"),
        }
        match t.params.get("y") {
            Some(ParamValue::Float(v)) => assert!((v - 2.0).abs() < 1e-10, "y={v}"),
            _ => panic!("y missing"),
        }
    }

    // z should vary
    let z_vals: Vec<f64> = trials.iter()
        .filter_map(|t| match t.params.get("z") {
            Some(ParamValue::Float(z)) => Some(*z),
            _ => None,
        })
        .collect();
    let z_unique: std::collections::HashSet<u64> = z_vals.iter()
        .map(|&z| z.to_bits())
        .collect();
    assert!(z_unique.len() > 1, "z should vary");
}

// ============================================================================
// 10. SearchSpaceTransform roundtrip 精度
// ============================================================================

/// 多参数 roundtrip 精度验证
#[test]
fn test_sst_roundtrip_precision() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: -100.0, high: 100.0, log: false, step: None,
        }),
    );
    ss.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 1e-8, high: 10.0, log: true, step: None,
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // Test extreme values
    let test_cases: Vec<(f64, f64)> = vec![
        (-100.0, 1e-8),
        (0.0, 0.001),
        (42.5, 1.0),
        (100.0, 10.0),
    ];

    for (x_val, lr_val) in test_cases {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(x_val));
        params.insert("lr".to_string(), ParamValue::Float(lr_val));

        let encoded = t.transform(&params);
        let decoded = t.untransform(&encoded).unwrap();

        match decoded.get("x") {
            Some(ParamValue::Float(v)) => assert!(
                (v - x_val).abs() < 1e-10,
                "x roundtrip: {} vs {x_val}",
                v
            ),
            _ => panic!("x decode fail"),
        }
        match decoded.get("lr") {
            Some(ParamValue::Float(v)) => assert!(
                ((v - lr_val) / lr_val).abs() < 1e-10,
                "lr roundtrip: {} vs {lr_val}",
                v
            ),
            _ => panic!("lr decode fail"),
        }
    }
}

// ============================================================================
// 11. PartialFixedSampler::from_param_values
// ============================================================================

/// from_param_values 带 categorical 固定值
#[test]
fn test_partial_fixed_from_param_values_categorical() {
    let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));

    let mut fixed = HashMap::new();
    fixed.insert(
        "cat".to_string(),
        ParamValue::Categorical(CategoricalChoice::Str("b".to_string())),
    );

    let mut dists = HashMap::new();
    dists.insert(
        "cat".to_string(),
        Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".to_string()),
                CategoricalChoice::Str("b".to_string()),
                CategoricalChoice::Str("c".to_string()),
            ]).unwrap()
        ),
    );

    let sampler: Arc<dyn Sampler> = Arc::new(
        PartialFixedSampler::from_param_values(fixed, &dists, base).unwrap()
    );

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let cat = trial.suggest_categorical(
                "cat",
                vec![
                    CategoricalChoice::Str("a".to_string()),
                    CategoricalChoice::Str("b".to_string()),
                    CategoricalChoice::Str("c".to_string()),
                ],
            )?;
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            // cat internal repr for "b" is 1.0 (index)
            let cat_val = match &cat {
                CategoricalChoice::Str(s) => if s == "b" { 0.0 } else { 1.0 },
                _ => 1.0,
            };
            Ok(cat_val + x)
        },
        Some(10), None, None,
    ).unwrap();

    let trials = study.trials().unwrap();
    for t in &trials {
        match t.params.get("cat") {
            Some(ParamValue::Categorical(CategoricalChoice::Str(s))) => {
                assert_eq!(s, "b", "cat should be fixed to 'b', got '{s}'");
            }
            other => panic!("cat not found or unexpected: {:?}", other),
        }
    }
}

// ============================================================================
// 12. SearchSpaceTransform with_defaults
// ============================================================================

/// with_defaults should give (true, true, false) = (log, step, no 0-1)
#[test]
fn test_sst_with_defaults() {
    let mut ss = IndexMap::new();
    ss.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 1e-5, high: 1.0, log: true, step: None,
        }),
    );
    let t = SearchSpaceTransform::with_defaults(ss);

    // Log transform should be applied
    let mut params = IndexMap::new();
    params.insert("lr".to_string(), ParamValue::Float(0.01));
    let encoded = t.transform(&params);
    // Should be ln(0.01) ≈ -4.605
    assert!(
        (encoded[0] - 0.01_f64.ln()).abs() < 1e-10,
        "with_defaults log transform: {} vs {}",
        encoded[0], 0.01_f64.ln()
    );

    // 0-1 scaling should NOT be applied
    let bounds = t.bounds();
    // Should be raw log bounds, not [0,1]
    assert!((bounds[0][0] - 1e-5_f64.ln()).abs() < 1e-10);
    assert!((bounds[0][1] - 1.0_f64.ln()).abs() < 1e-10);
}
