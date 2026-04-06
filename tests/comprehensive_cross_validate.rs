//! 综合交叉验证测试
//!
//! 使用 Python 生成的精确参考值验证 Rust 实现。
//! 参考值来源: tests/comprehensive_baseline.json
//!   生成脚本: tests/generate_comprehensive_baseline.py
//!
//! 覆盖模块:
//!   1. Distributions (contains / single)
//!   2. Hyperband CRC32 哈希对齐
//!   3. Percentile 计算
//!   4. Search Space Transform (编码/解码/往返)
//!   5. TPE 权重/gamma
//!   6. QMC 低差异序列 (Van der Corput / Halton)

use indexmap::IndexMap;
use serde_json::Value;

use optuna_rs::distributions::*;
use optuna_rs::samplers::qmc::{halton_point, van_der_corput};
use optuna_rs::samplers::tpe::parzen_estimator::{default_gamma, default_weights};
use optuna_rs::search_space::SearchSpaceTransform;

const TOL: f64 = 1e-10;

fn load_baseline() -> Value {
    let data = include_str!("comprehensive_baseline.json");
    serde_json::from_str(data).expect("Failed to parse comprehensive_baseline.json")
}

fn get_bool_vec(v: &Value, section: &str, key: &str) -> Vec<bool> {
    v[section][key]
        .as_array()
        .unwrap_or_else(|| panic!("{section}.{key} not an array"))
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect()
}

fn get_f64(v: &Value, section: &str, key: &str) -> f64 {
    v[section][key]
        .as_f64()
        .unwrap_or_else(|| panic!("{section}.{key} not a number"))
}

fn get_f64_vec(v: &Value, section: &str, key: &str) -> Vec<f64> {
    v[section][key]
        .as_array()
        .unwrap_or_else(|| panic!("{section}.{key} not an array"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn get_u32(v: &Value, section: &str, key: &str) -> u32 {
    v[section][key]
        .as_u64()
        .unwrap_or_else(|| panic!("{section}.{key} not a number")) as u32
}

fn get_i64_vec(v: &Value, section: &str, key: &str) -> Vec<i64> {
    v[section][key]
        .as_array()
        .unwrap_or_else(|| panic!("{section}.{key} not an array"))
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    if expected.is_infinite() {
        assert!(
            actual.is_infinite() && actual.signum() == expected.signum(),
            "{msg}: expected {expected}, got {actual}"
        );
        return;
    }
    if expected.is_nan() {
        assert!(actual.is_nan(), "{msg}: expected NaN, got {actual}");
        return;
    }
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{msg}: expected {expected}, got {actual}, diff={diff}"
    );
}

// ═══════════════════════════════════════════════════════════════
//  1. Distributions — contains
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_float_contains_basic() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "float_contains_0");
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    let values = [0.0, 0.5, 1.0, -0.1, 1.1, f64::NAN, f64::INFINITY];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "float_contains_0[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_float_contains_log() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "float_contains_1");
    let d = FloatDistribution::new(1e-5, 1.0, true, None).unwrap();
    let values = [1e-5, 0.01, 1.0, 0.0, -1.0, 2.0];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "float_contains_1[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_float_contains_step() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "float_contains_2");
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
    let values = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.3];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "float_contains_2[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_int_contains_basic() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "int_contains_0");
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    let values: Vec<f64> = vec![0.0, 5.0, 10.0, -1.0, 11.0, 3.0];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "int_contains_0[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_int_contains_log() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "int_contains_1");
    let d = IntDistribution::new(1, 100, true, 1).unwrap();
    let values: Vec<f64> = vec![1.0, 10.0, 100.0, 0.0, 101.0];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "int_contains_1[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_int_contains_step() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "int_contains_2");
    let d = IntDistribution::new(0, 10, false, 2).unwrap();
    let values: Vec<f64> = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "int_contains_2[{i}]: value={v}, expected={exp}"
        );
    }
}

#[test]
fn cv_cat_contains() {
    let b = load_baseline();
    let expected = get_bool_vec(&b, "distributions", "cat_contains");
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".to_string()),
        CategoricalChoice::Str("b".to_string()),
        CategoricalChoice::Str("c".to_string()),
    ])
    .unwrap();
    // contains 接受 index (0, 1, 2 有效，3, -1 无效)
    let values: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, -1.0];
    for (i, (&v, &exp)) in values.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            d.contains(v), exp,
            "cat_contains[{i}]: value={v}, expected={exp}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  1b. Distributions — single
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_float_single() {
    let b = load_baseline();

    // Float(0,1) → not single
    let d1 = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert_eq!(d1.single(), b["distributions"]["float_single_0_1"].as_bool().unwrap());

    // Float(5,5) → single
    let d2 = FloatDistribution::new(5.0, 5.0, false, None).unwrap();
    assert_eq!(d2.single(), b["distributions"]["float_single_5_5"].as_bool().unwrap());

    // Float(0,0,step=0.5) → single
    let d3 = FloatDistribution::new(0.0, 0.0, false, Some(0.5)).unwrap();
    assert_eq!(d3.single(), b["distributions"]["float_single_step"].as_bool().unwrap());
}

#[test]
fn cv_int_single() {
    // Int(5,5) → single
    let d1 = IntDistribution::new(5, 5, false, 1).unwrap();
    assert!(d1.single());

    // Int(0,10) → not single
    let d2 = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(!d2.single());

    // Int(0,10,step=11) → not single (0 and 11 out of range)
    // Int(0,1,step=2) → single (only 0)
    let d3 = IntDistribution::new(0, 1, false, 2).unwrap();
    assert!(d3.single());
}

#[test]
fn cv_cat_single() {
    // 单个选项 → single
    let d1 = CategoricalDistribution::new(vec![CategoricalChoice::Str("only".to_string())]).unwrap();
    assert!(d1.single());

    // 多个选项 → not single
    let d2 = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".to_string()),
        CategoricalChoice::Str("b".to_string()),
    ])
    .unwrap();
    assert!(!d2.single());
}

// ═══════════════════════════════════════════════════════════════
//  2. CRC32 哈希对齐
// ═══════════════════════════════════════════════════════════════

/// CRC32 标准测试 — 内联实现（与 hyperband.rs 内部相同的算法）
fn crc32_hash(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

#[test]
fn cv_crc32_basic() {
    let b = load_baseline();

    let test_strings = [
        "my_study_0", "my_study_1", "my_study_99",
        "test_5", "study_abc_42", "123456789",
        "", "a", "hello_world_12345",
        "study_with_underscores_100",
        "s_0", "s_1", "s_2", "s_3", "s_4",
    ];

    for s in &test_strings {
        let key = format!("crc32_{s}");
        let expected = get_u32(&b, "crc32", &key);
        let actual = crc32_hash(s.as_bytes());
        assert_eq!(actual, expected, "CRC32({s:?}): expected={expected}, got={actual}");
    }
}

#[test]
fn cv_crc32_hyperband_bracket_assignment() {
    let b = load_baseline();
    let expected = get_i64_vec(&b, "crc32", "bracket_assignments_100");

    let budgets: Vec<usize> = vec![81, 27, 9, 3];
    let total: usize = budgets.iter().sum();

    for i in 0..100 {
        let hash_input = format!("hyperband_study_{i}");
        let h = crc32_hash(hash_input.as_bytes());
        let mut n = (h as usize) % total;
        let mut bracket_id = 0;
        for (bid, &b_budget) in budgets.iter().enumerate() {
            if n < b_budget {
                bracket_id = bid;
                break;
            }
            n -= b_budget;
        }
        assert_eq!(
            bracket_id as i64,
            expected[i],
            "bracket_assignment[{i}]: expected={}, got={bracket_id}",
            expected[i]
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  3. Percentile 计算
// ═══════════════════════════════════════════════════════════════

/// 线性插值百分位数（与 numpy nanpercentile 一致）
fn nanpercentile(data: &[f64], percentile: f64) -> f64 {
    let mut valid: Vec<f64> = data.iter().copied().filter(|x| !x.is_nan()).collect();
    if valid.is_empty() {
        return f64::NAN;
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = valid.len();
    if n == 1 {
        return valid[0];
    }
    let idx = percentile / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi {
        valid[lo]
    } else {
        valid[lo] * (1.0 - frac) + valid[hi] * frac
    }
}

#[test]
fn cv_percentile_basic() {
    let b = load_baseline();
    let data_sets: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
        vec![100.0, 1.0, 50.0, 25.0, 75.0],
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
        vec![f64::NAN, f64::NAN, f64::NAN, 1.0],
    ];
    let percentiles = [25, 50, 75, 90];

    for (i, data) in data_sets.iter().enumerate() {
        for &p in &percentiles {
            let key = format!("percentile_{i}_p{p}");
            let expected = get_f64(&b, "percentile", &key);
            let actual = nanpercentile(data, p as f64);

            if expected.is_nan() {
                assert!(actual.is_nan(), "{key}: expected NaN, got {actual}");
            } else {
                assert_close(actual, expected, TOL, &key);
            }
        }
    }
}

#[test]
fn cv_percentile_interpolation() {
    let b = load_baseline();
    let data = [1.0, 2.0, 3.0, 4.0];

    let expected_33 = get_f64(&b, "percentile", "percentile_interp_33");
    let actual_33 = nanpercentile(&data, 33.33);
    assert_close(actual_33, expected_33, TOL, "percentile_interp_33");

    let expected_66 = get_f64(&b, "percentile", "percentile_interp_66");
    let actual_66 = nanpercentile(&data, 66.67);
    assert_close(actual_66, expected_66, TOL, "percentile_interp_66");
}

// ═══════════════════════════════════════════════════════════════
//  4. Search Space Transform
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_transform_float_linear() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, false, false, false);

    for &v in &[0.0, 0.5, 1.0] {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        assert_close(encoded[0], v, TOL, &format!("linear_transform({v})"));
    }
}

#[test]
fn cv_transform_float_log() {
    let b = load_baseline();
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(1e-5, 1.0, true, None).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, false, false);

    let inputs = [1e-5, 0.001, 0.01, 0.1, 1.0];
    let expected: Vec<f64> = b["search_space_transform"]["transform_float_log"]["transformed"]
        .as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();

    for (i, (&v, &exp)) in inputs.iter().zip(expected.iter()).enumerate() {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        assert_close(
            encoded[0], exp, TOL,
            &format!("log_transform[{i}]({v})")
        );
    }
}

#[test]
fn cv_transform_int_linear() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, false, false, false);

    for &v in &[0i64, 1, 2, 5, 10] {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Int(v));
        let encoded = t.transform(&params);
        assert_close(encoded[0], v as f64, TOL, &format!("int_linear({v})"));
    }
}

#[test]
fn cv_transform_int_log() {
    let b = load_baseline();
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, false, false);

    let inputs = [1i64, 2, 5, 10, 100];
    let expected: Vec<f64> = b["search_space_transform"]["transform_int_log"]["transformed"]
        .as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();

    for (i, (&v, &exp)) in inputs.iter().zip(expected.iter()).enumerate() {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Int(v));
        let encoded = t.transform(&params);
        assert_close(
            encoded[0], exp, TOL,
            &format!("int_log_transform[{i}]({v})")
        );
    }
}

#[test]
fn cv_transform_round_trip_linear() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, false, false, false);

    for &v in &[0.0, 0.123456789, 0.5, 1.0] {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        let decoded = t.untransform(&encoded).unwrap();
        let recovered = match &decoded["x"] {
            ParamValue::Float(f) => *f,
            _ => panic!("Expected Float"),
        };
        assert_close(recovered, v, TOL, &format!("round_trip_linear({v})"));
    }
}

#[test]
fn cv_transform_round_trip_log() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(1e-5, 1.0, true, None).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, false, false);

    for &v in &[1e-5, 0.001, 0.5, 1.0] {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        let decoded = t.untransform(&encoded).unwrap();
        let recovered = match &decoded["x"] {
            ParamValue::Float(f) => *f,
            _ => panic!("Expected Float"),
        };
        let error = (recovered - v).abs();
        assert!(
            error < 1e-10,
            "round_trip_log({v}): recovered={recovered}, error={error}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  5. TPE gamma / weights
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_tpe_gamma() {
    let b = load_baseline();
    let test_n = [1, 4, 9, 16, 25, 50, 100, 1000];
    for &n in &test_n {
        let key = format!("gamma_{n}");
        let expected = b["tpe_supplementary"][&key].as_i64().unwrap() as usize;
        let actual = default_gamma(n);
        assert_eq!(actual, expected, "gamma({n}): expected={expected}, got={actual}");
    }
}

#[test]
fn cv_tpe_weights() {
    let b = load_baseline();
    let test_n = [1, 3, 5, 10, 25];
    for &n in &test_n {
        let key = format!("weights_{n}");
        let expected = get_f64_vec(&b, "tpe_supplementary", &key);
        let actual = default_weights(n);
        assert_eq!(
            actual.len(),
            expected.len(),
            "weights({n}): length mismatch"
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, 1e-12, &format!("weights({n})[{i}]"));
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  6. QMC — Van der Corput
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_vdc_base2() {
    let b = load_baseline();
    let expected = get_f64_vec(&b, "qmc", "vdc_base2");
    for (i, &exp) in expected.iter().enumerate() {
        let n = (i + 1) as u64;
        let actual = van_der_corput(n, 2);
        assert_close(actual, exp, TOL, &format!("vdc(base=2, n={n})"));
    }
}

#[test]
fn cv_vdc_base3() {
    let b = load_baseline();
    let expected = get_f64_vec(&b, "qmc", "vdc_base3");
    for (i, &exp) in expected.iter().enumerate() {
        let n = (i + 1) as u64;
        let actual = van_der_corput(n, 3);
        assert_close(actual, exp, TOL, &format!("vdc(base=3, n={n})"));
    }
}

#[test]
fn cv_vdc_base5() {
    let b = load_baseline();
    let expected = get_f64_vec(&b, "qmc", "vdc_base5");
    for (i, &exp) in expected.iter().enumerate() {
        let n = (i + 1) as u64;
        let actual = van_der_corput(n, 5);
        assert_close(actual, exp, TOL, &format!("vdc(base=5, n={n})"));
    }
}

#[test]
fn cv_vdc_base7() {
    let b = load_baseline();
    let expected = get_f64_vec(&b, "qmc", "vdc_base7");
    for (i, &exp) in expected.iter().enumerate() {
        let n = (i + 1) as u64;
        let actual = van_der_corput(n, 7);
        assert_close(actual, exp, TOL, &format!("vdc(base=7, n={n})"));
    }
}

#[test]
fn cv_halton_2d() {
    let b = load_baseline();
    let points = b["qmc"]["halton_2d"]
        .as_array()
        .expect("halton_2d not an array");

    for (i, pt) in points.iter().enumerate() {
        let expected: Vec<f64> = pt.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let n = (i + 1) as u64;
        let actual = halton_point(n, 2, false, 0);
        assert_eq!(actual.len(), expected.len(), "halton_2d[{i}] dim mismatch");
        for (d, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, TOL, &format!("halton_2d[{i}][{d}]"));
        }
    }
}

#[test]
fn cv_halton_3d() {
    let b = load_baseline();
    let points = b["qmc"]["halton_3d"]
        .as_array()
        .expect("halton_3d not an array");

    for (i, pt) in points.iter().enumerate() {
        let expected: Vec<f64> = pt.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let n = (i + 1) as u64;
        let actual = halton_point(n, 3, false, 0);
        assert_eq!(actual.len(), expected.len(), "halton_3d[{i}] dim mismatch");
        for (d, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, TOL, &format!("halton_3d[{i}][{d}]"));
        }
    }
}

#[test]
fn cv_halton_5d() {
    let b = load_baseline();
    let points = b["qmc"]["halton_5d"]
        .as_array()
        .expect("halton_5d not an array");

    for (i, pt) in points.iter().enumerate() {
        let expected: Vec<f64> = pt.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let n = (i + 1) as u64;
        let actual = halton_point(n, 5, false, 0);
        assert_eq!(actual.len(), expected.len(), "halton_5d[{i}] dim mismatch");
        for (d, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, TOL, &format!("halton_5d[{i}][{d}]"));
        }
    }
}

#[test]
fn cv_halton_10d() {
    let b = load_baseline();
    let points = b["qmc"]["halton_10d"]
        .as_array()
        .expect("halton_10d not an array");

    for (i, pt) in points.iter().enumerate() {
        let expected: Vec<f64> = pt
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let n = (i + 1) as u64;
        let actual = halton_point(n, 10, false, 0);
        assert_eq!(actual.len(), expected.len(), "halton_10d[{i}] dim mismatch");
        for (d, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, TOL, &format!("halton_10d[{i}][{d}]"));
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  7. Distribution 序列化/反序列化
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_distribution_serde_round_trip() {
    // FloatDistribution
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let d2 = json_to_distribution(&json).unwrap();
    match (&d, &d2) {
        (Distribution::FloatDistribution(a), Distribution::FloatDistribution(b)) => {
            assert_eq!(a.low, b.low);
            assert_eq!(a.high, b.high);
            assert_eq!(a.log, b.log);
            assert_eq!(a.step, b.step);
        }
        _ => panic!("Type mismatch after serde"),
    }

    // FloatDistribution with log
    let d = Distribution::FloatDistribution(FloatDistribution::new(1e-5, 1.0, true, None).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let d2 = json_to_distribution(&json).unwrap();
    match (&d, &d2) {
        (Distribution::FloatDistribution(a), Distribution::FloatDistribution(b)) => {
            assert_eq!(a.low, b.low);
            assert_eq!(a.high, b.high);
            assert_eq!(a.log, b.log);
        }
        _ => panic!("Type mismatch after serde"),
    }

    // IntDistribution
    let d = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 5).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let d2 = json_to_distribution(&json).unwrap();
    match (&d, &d2) {
        (Distribution::IntDistribution(a), Distribution::IntDistribution(b)) => {
            assert_eq!(a.low, b.low);
            assert_eq!(a.high, b.high);
            assert_eq!(a.log, b.log);
            assert_eq!(a.step, b.step);
        }
        _ => panic!("Type mismatch after serde"),
    }

    // CategoricalDistribution
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".to_string()),
            CategoricalChoice::Int(42),
            CategoricalChoice::Float(3.14),
            CategoricalChoice::Bool(true),
            CategoricalChoice::None,
        ])
        .unwrap(),
    );
    let json = distribution_to_json(&d).unwrap();
    let d2 = json_to_distribution(&json).unwrap();
    match (&d, &d2) {
        (Distribution::CategoricalDistribution(a), Distribution::CategoricalDistribution(b)) => {
            assert_eq!(a.choices.len(), b.choices.len());
        }
        _ => panic!("Type mismatch after serde"),
    }
}

// ═══════════════════════════════════════════════════════════════
//  8. Distribution 兼容性检查
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_distribution_compatibility() {
    // 相同分布 → 兼容
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok());

    // 不同类型 → 不兼容
    let d = Distribution::IntDistribution(IntDistribution::new(0, 1, false, 1).unwrap());
    assert!(check_distribution_compatibility(&a, &d).is_err());

    // 不同 log 属性 → 不兼容
    let e = Distribution::FloatDistribution(FloatDistribution::new(0.1, 1.0, true, None).unwrap());
    let f = Distribution::FloatDistribution(FloatDistribution::new(0.1, 1.0, false, None).unwrap());
    assert!(check_distribution_compatibility(&e, &f).is_err());
}

// ═══════════════════════════════════════════════════════════════
//  9. Internal/External Repr
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_float_internal_external_repr() {
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    for &v in &[0.0, 0.5, 1.0] {
        let internal = d.to_internal_repr(v).unwrap();
        let external = d.to_external_repr(internal);
        assert_close(external, v, TOL, &format!("float_repr({v})"));
    }

    // log 分布
    let d_log = FloatDistribution::new(1e-5, 1.0, true, None).unwrap();
    for &v in &[1e-5, 0.001, 1.0] {
        let internal = d_log.to_internal_repr(v).unwrap();
        let external = d_log.to_external_repr(internal);
        assert_close(external, v, 1e-8, &format!("float_log_repr({v})"));
    }
}

#[test]
fn cv_int_internal_external_repr() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    for &v in &[0, 5, 10] {
        let internal = d.to_internal_repr(v).unwrap();
        let external = d.to_external_repr(internal).unwrap();
        assert_eq!(external, v, "int_repr({v}): got {external}");
    }

    // log 分布
    let d_log = IntDistribution::new(1, 100, true, 1).unwrap();
    for &v in &[1, 10, 100] {
        let internal = d_log.to_internal_repr(v).unwrap();
        let external = d_log.to_external_repr(internal).unwrap();
        assert_eq!(external, v, "int_log_repr({v}): got {external}");
    }
}

#[test]
fn cv_cat_internal_external_repr() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".to_string()),
        CategoricalChoice::Str("b".to_string()),
        CategoricalChoice::Str("c".to_string()),
    ])
    .unwrap();

    let v = CategoricalChoice::Str("b".to_string());
    let internal = d.to_internal_repr(&v).unwrap();
    assert_close(internal, 1.0, TOL, "cat_repr(b) → 1.0");

    let external = d.to_external_repr(1.0).unwrap();
    match external {
        CategoricalChoice::Str(s) => assert_eq!(s, "b"),
        _ => panic!("Expected Str('b')"),
    }
}

// ═══════════════════════════════════════════════════════════════
//  10. 数值精度压力测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_vdc_precision_stress() {
    // Van der Corput — 2^20 = 1048576 的二进制是 1 后跟 20 个 0
    // VdC(2^20, 2) 反转后 = 0.000...01 (21 位) → 2^(-21)
    // 因为 van_der_corput 对 n=1048576 (= 0b100000000000000000000)
    // 反转后得到 0.0...01 即 2^(-21)
    let actual = van_der_corput(1048576, 2);
    // 1048576 = 2^20, 二进制 = 1 后 20 个 0
    // VdC 反转: 0. + 0*2^-1 + ... + 0*2^-20 + 1*2^-21 = 2^-21
    let expected = 1.0 / (1u64 << 21) as f64;
    assert_close(actual, expected, 1e-15, "vdc(2^20, 2)");

    // VdC(1000000, 2) 累积误差检查
    let actual = van_der_corput(1000000, 2);
    assert!(actual > 0.0 && actual < 1.0, "vdc(1M, 2) out of [0,1): {actual}");
}

#[test]
fn cv_float_extreme_ranges() {
    // 非常小的范围
    let d = FloatDistribution::new(1e-300, 1e-299, false, None).unwrap();
    assert!(d.contains(5e-300));
    assert!(!d.contains(1e-298));

    // 非常大的范围
    let d = FloatDistribution::new(-1e300, 1e300, false, None).unwrap();
    assert!(d.contains(0.0));
    assert!(d.contains(1e299));
}

#[test]
fn cv_log_distribution_edge_cases() {
    // log 分布的极端边界
    let d = FloatDistribution::new(1e-10, 1e10, true, None).unwrap();
    assert!(d.contains(1e-10));
    assert!(d.contains(1e10));
    assert!(d.contains(1.0));
    assert!(!d.contains(0.0));
    assert!(!d.contains(-1.0));
}
