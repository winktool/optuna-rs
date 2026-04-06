/// SearchSpaceTransform 精确交叉验证
/// Golden values 来源: Python optuna._transform._SearchSpaceTransform
///
/// 覆盖:
///   - Float with step: bounds, transform, untransform
///   - Int with step: bounds, transform, untransform
///   - Log float: bounds, transform, roundtrip
///   - Log int: bounds, transform, roundtrip
///   - Categorical: one-hot encode/decode
///   - transform_0_1 模式
///   - Mixed space roundtrip

use indexmap::IndexMap;
use optuna_rs::distributions::*;
use optuna_rs::search_space::transform::SearchSpaceTransform;

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    let diff = (a - b).abs();
    let rel = if b.abs() > 1e-15 { diff / b.abs() } else { diff };
    assert!(rel < tol || diff < 1e-15, "{}: a={}, b={}, rel={}", msg, a, b, rel);
}

// ═══════════════════════════════════════════════════════════════
// Float with step
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_float_step_transform_bounds() {
    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 1.0, log: false, step: Some(0.1) },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);
    let b = t.bounds();
    assert_close(b[0][0], -0.05, 1e-12, "float_step low");
    assert_close(b[0][1],  1.05, 1e-12, "float_step high");
}

#[test]
fn test_float_step_transform_roundtrip() {
    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 1.0, log: false, step: Some(0.1) },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // x=0.35 -> enc=0.35 -> untransform -> 0.3 (rounded to step)
    let mut params = IndexMap::new();
    params.insert("x".to_string(), ParamValue::Float(0.35));
    let enc = t.transform(&params);
    assert_close(enc[0], 0.35, 1e-12, "enc(0.35)");
    let dec = t.untransform(&enc).unwrap();
    match &dec["x"] {
        ParamValue::Float(v) => assert_close(*v, 0.3, 1e-12, "dec(0.35)"),
        _ => panic!("expected Float"),
    }

    // x=0.5 -> enc=0.5 -> untransform -> 0.5
    let mut params2 = IndexMap::new();
    params2.insert("x".to_string(), ParamValue::Float(0.5));
    let enc2 = t.transform(&params2);
    let dec2 = t.untransform(&enc2).unwrap();
    match &dec2["x"] {
        ParamValue::Float(v) => assert_close(*v, 0.5, 1e-12, "dec(0.5)"),
        _ => panic!("expected Float"),
    }

    // x=0.95 -> enc=0.95 -> untransform -> 0.9
    let mut params3 = IndexMap::new();
    params3.insert("x".to_string(), ParamValue::Float(0.95));
    let enc3 = t.transform(&params3);
    let dec3 = t.untransform(&enc3).unwrap();
    match &dec3["x"] {
        ParamValue::Float(v) => assert_close(*v, 0.9, 1e-12, "dec(0.95)"),
        _ => panic!("expected Float"),
    }
}

// ═══════════════════════════════════════════════════════════════
// Int with step
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_int_step_transform_bounds() {
    let mut ss = IndexMap::new();
    ss.insert("n".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 9, log: false, step: 2 },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);
    let b = t.bounds();
    assert_close(b[0][0], 0.0, 1e-12, "int_step low");
    assert_close(b[0][1], 10.0, 1e-12, "int_step high");
}

#[test]
fn test_int_step_untransform_golden() {
    let mut ss = IndexMap::new();
    ss.insert("n".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 9, log: false, step: 2 },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // Python golden: untransform(0.5)->1, (2.0)->1, (4.0)->5, (6.0)->5, (8.0)->9, (10.0)->9
    let cases: Vec<(f64, i64)> = vec![
        (0.5, 1), (2.0, 1), (4.0, 5), (6.0, 5), (8.0, 9), (10.0, 9),
    ];
    for (enc_val, expected) in cases {
        let dec = t.untransform(&[enc_val]).unwrap();
        match &dec["n"] {
            ParamValue::Int(v) => assert_eq!(*v, expected, "untransform({}) should be {}", enc_val, expected),
            _ => panic!("expected Int"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Log float
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_float_transform_bounds() {
    let mut ss = IndexMap::new();
    ss.insert("lr".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 1e-5, high: 1.0, log: true, step: None },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);
    let b = t.bounds();
    assert_close(b[0][0], -11.512925464970229, 1e-12, "log_float low");
    assert_close(b[0][1], 0.0, 1e-12, "log_float high");
}

#[test]
fn test_log_float_transform_roundtrip() {
    let mut ss = IndexMap::new();
    ss.insert("lr".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 1e-5, high: 1.0, log: true, step: None },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);

    for &v in &[1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
        let mut params = IndexMap::new();
        params.insert("lr".to_string(), ParamValue::Float(v));
        let enc = t.transform(&params);
        assert_close(enc[0], v.ln(), 1e-12, &format!("enc({})", v));
        let dec = t.untransform(&enc).unwrap();
        match &dec["lr"] {
            ParamValue::Float(dv) => assert_close(*dv, v, 1e-10, &format!("roundtrip({})", v)),
            _ => panic!("expected Float"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Log int
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_int_transform_bounds() {
    let mut ss = IndexMap::new();
    ss.insert("k".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 100, log: true, step: 1 },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);
    let b = t.bounds();
    // Python: bounds=[[-0.6931471805599453, 4.61015772749913]]
    assert_close(b[0][0], (0.5_f64).ln(), 1e-12, "log_int low");
    assert_close(b[0][1], (100.5_f64).ln(), 1e-12, "log_int high");
}

#[test]
fn test_log_int_transform_roundtrip() {
    let mut ss = IndexMap::new();
    ss.insert("k".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 100, log: true, step: 1 },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);

    for &v in &[1_i64, 2, 10, 50, 100] {
        let mut params = IndexMap::new();
        params.insert("k".to_string(), ParamValue::Int(v));
        let enc = t.transform(&params);
        assert_close(enc[0], (v as f64).ln(), 1e-12, &format!("enc({})", v));
        let dec = t.untransform(&enc).unwrap();
        match &dec["k"] {
            ParamValue::Int(dv) => assert_eq!(*dv, v, "roundtrip({})", v),
            _ => panic!("expected Int"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Categorical
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_categorical_transform_onehot() {
    let mut ss = IndexMap::new();
    ss.insert("algo".to_string(), Distribution::CategoricalDistribution(
        CategoricalDistribution {
            choices: vec![
                CategoricalChoice::Str("sgd".to_string()),
                CategoricalChoice::Str("adam".to_string()),
                CategoricalChoice::Str("rmsprop".to_string()),
            ],
        },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);
    assert_eq!(t.n_encoded(), 3);

    // sgd -> [1,0,0]
    let mut p1 = IndexMap::new();
    p1.insert("algo".to_string(), ParamValue::Categorical(CategoricalChoice::Str("sgd".to_string())));
    let enc1 = t.transform(&p1);
    assert_eq!(enc1, vec![1.0, 0.0, 0.0]);
    let dec1 = t.untransform(&enc1).unwrap();
    assert_eq!(dec1["algo"], ParamValue::Categorical(CategoricalChoice::Str("sgd".to_string())));

    // adam -> [0,1,0]
    let mut p2 = IndexMap::new();
    p2.insert("algo".to_string(), ParamValue::Categorical(CategoricalChoice::Str("adam".to_string())));
    let enc2 = t.transform(&p2);
    assert_eq!(enc2, vec![0.0, 1.0, 0.0]);

    // rmsprop -> [0,0,1]
    let mut p3 = IndexMap::new();
    p3.insert("algo".to_string(), ParamValue::Categorical(CategoricalChoice::Str("rmsprop".to_string())));
    let enc3 = t.transform(&p3);
    assert_eq!(enc3, vec![0.0, 0.0, 1.0]);
}

// ═══════════════════════════════════════════════════════════════
// transform_0_1 模式
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_transform_0_1_mode() {
    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 2.0, high: 8.0, log: false, step: None },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, true);
    let b = t.bounds();
    assert_close(b[0][0], 0.0, 1e-12, "0_1 bound low");
    assert_close(b[0][1], 1.0, 1e-12, "0_1 bound high");

    // x=2.0 -> 0.0
    let mut p1 = IndexMap::new();
    p1.insert("x".to_string(), ParamValue::Float(2.0));
    let enc1 = t.transform(&p1);
    assert_close(enc1[0], 0.0, 1e-12, "0_1 enc(2.0)");

    // x=5.0 -> 0.5
    let mut p2 = IndexMap::new();
    p2.insert("x".to_string(), ParamValue::Float(5.0));
    let enc2 = t.transform(&p2);
    assert_close(enc2[0], 0.5, 1e-12, "0_1 enc(5.0)");

    // x=8.0 -> 1.0
    let mut p3 = IndexMap::new();
    p3.insert("x".to_string(), ParamValue::Float(8.0));
    let enc3 = t.transform(&p3);
    assert_close(enc3[0], 1.0, 1e-12, "0_1 enc(8.0)");
}

// ═══════════════════════════════════════════════════════════════
// Mixed space roundtrip
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_mixed_space_roundtrip() {
    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), Distribution::FloatDistribution(
        FloatDistribution { low: 0.0, high: 1.0, log: false, step: None },
    ));
    ss.insert("n".to_string(), Distribution::IntDistribution(
        IntDistribution { low: 1, high: 5, log: false, step: 1 },
    ));
    ss.insert("c".to_string(), Distribution::CategoricalDistribution(
        CategoricalDistribution {
            choices: vec![
                CategoricalChoice::Str("a".to_string()),
                CategoricalChoice::Str("b".to_string()),
            ],
        },
    ));
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // Python: bounds=[[0.0, 1.0], [0.5, 5.5], [0.0, 1.0], [0.0, 1.0]]
    let b = t.bounds();
    assert_eq!(b.len(), 4);
    assert_close(b[0][0], 0.0, 1e-12, "mixed x low");
    assert_close(b[0][1], 1.0, 1e-12, "mixed x high");
    assert_close(b[1][0], 0.5, 1e-12, "mixed n low");
    assert_close(b[1][1], 5.5, 1e-12, "mixed n high");

    // Python: enc=[0.7, 3.0, 0.0, 1.0]
    let mut params = IndexMap::new();
    params.insert("x".to_string(), ParamValue::Float(0.7));
    params.insert("n".to_string(), ParamValue::Int(3));
    params.insert("c".to_string(), ParamValue::Categorical(CategoricalChoice::Str("b".to_string())));
    let enc = t.transform(&params);
    assert_close(enc[0], 0.7, 1e-12, "mixed enc x");
    assert_close(enc[1], 3.0, 1e-12, "mixed enc n");
    assert_close(enc[2], 0.0, 1e-12, "mixed enc c[0]");
    assert_close(enc[3], 1.0, 1e-12, "mixed enc c[1]");

    let dec = t.untransform(&enc).unwrap();
    match &dec["x"] {
        ParamValue::Float(v) => assert_close(*v, 0.7, 1e-12, "mixed dec x"),
        _ => panic!("expected Float"),
    }
    match &dec["n"] {
        ParamValue::Int(v) => assert_eq!(*v, 3, "mixed dec n"),
        _ => panic!("expected Int"),
    }
    match &dec["c"] {
        ParamValue::Categorical(CategoricalChoice::Str(s)) => assert_eq!(s, "b"),
        _ => panic!("expected Categorical Str"),
    }
}
