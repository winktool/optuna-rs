//! Distribution 模块深度交叉验证测试。
//!
//! 所有参考值由 Python optuna (tests/golden_distributions_deep.py) 生成，
//! 共 140 个黄金值，覆盖 13 个测试组。
//!
//! 测试覆盖：
//! - FloatDistribution: adjust_high (14 cases)、single (12 cases)、contains (24 cases)、repr (12 cases)
//! - IntDistribution: adjust_high (11 cases)、single (8 cases)、contains (13 cases)、repr (3 cases)
//! - CategoricalDistribution: repr、contains、single、NaN 处理
//! - check_distribution_compatibility (12 cases)
//! - get_single_value (5 cases)
//! - is_log (5 cases)
//! - JSON 序列化/反序列化 往返 (9 cases)

use optuna_rs::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    check_distribution_compatibility, distribution_to_json, json_to_distribution,
};

// ============================================================================
// 1. FloatDistribution: _adjust_discrete_uniform_high 精确对比
//    Python 使用 Decimal(str(x)) 精确运算，Rust 使用缩放整数近似。
// ============================================================================

/// Python: FloatDistribution(0.0, 1.0, step=0.3) → high=0.9
#[test]
fn test_float_adjust_high_step03() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
    assert!((d.high - 0.9).abs() < 1e-12, "Python: high=0.9, got {}", d.high);
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=0.25) → high=1.0 (整除)
#[test]
fn test_float_adjust_high_step025() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
    assert_eq!(d.high, 1.0);
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=0.7) → high=0.7
#[test]
fn test_float_adjust_high_step07() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.7)).unwrap();
    assert!((d.high - 0.7).abs() < 1e-12, "Python: high=0.7, got {}", d.high);
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=0.1) → high=1.0 (整除)
#[test]
fn test_float_adjust_high_step01() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap();
    assert_eq!(d.high, 1.0);
}

/// Python: FloatDistribution(0.0, 1.0, step=0.15) → high=0.9
#[test]
fn test_float_adjust_high_step015() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.15)).unwrap();
    assert!((d.high - 0.9).abs() < 1e-12, "Python: high=0.9, got {}", d.high);
}

/// Python: FloatDistribution(0.0, 1.0, step=0.4) → high=0.8
#[test]
fn test_float_adjust_high_step04() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.4)).unwrap();
    assert!((d.high - 0.8).abs() < 1e-12, "Python: high=0.8, got {}", d.high);
}

/// Python: FloatDistribution(0.1, 0.95, step=0.3) → high=0.7
#[test]
fn test_float_adjust_high_offset_step03() {
    let d = FloatDistribution::new(0.1, 0.95, false, Some(0.3)).unwrap();
    assert!((d.high - 0.7).abs() < 1e-12, "Python: high=0.7, got {}", d.high);
}

/// Python: FloatDistribution(0.05, 1.0, step=0.3) → high=0.95
#[test]
fn test_float_adjust_high_offset005_step03() {
    let d = FloatDistribution::new(0.05, 1.0, false, Some(0.3)).unwrap();
    assert!((d.high - 0.95).abs() < 1e-12, "Python: high=0.95, got {}", d.high);
}

/// Python: FloatDistribution(0.0, 10.0, step=3.0) → high=9.0
#[test]
fn test_float_adjust_high_large_step3() {
    let d = FloatDistribution::new(0.0, 10.0, false, Some(3.0)).unwrap();
    assert!((d.high - 9.0).abs() < 1e-12, "Python: high=9.0, got {}", d.high);
}

/// Python: FloatDistribution(0.0, 10.0, step=0.3) → high=9.9
#[test]
fn test_float_adjust_high_large_step03() {
    let d = FloatDistribution::new(0.0, 10.0, false, Some(0.3)).unwrap();
    assert!((d.high - 9.9).abs() < 1e-12, "Python: high=9.9, got {}", d.high);
}

/// Python: FloatDistribution(1.5, 5.5, step=0.7) → high=5.0
#[test]
fn test_float_adjust_high_mid_range_step07() {
    let d = FloatDistribution::new(1.5, 5.5, false, Some(0.7)).unwrap();
    assert!((d.high - 5.0).abs() < 1e-12, "Python: high=5.0, got {}", d.high);
}

/// Python: FloatDistribution(0.0, 0.1, step=0.3) → high=0.0 (range < step)
#[test]
fn test_float_adjust_high_tiny_range() {
    let d = FloatDistribution::new(0.0, 0.1, false, Some(0.3)).unwrap();
    assert!((d.high - 0.0).abs() < 1e-12, "Python: high=0.0, got {}", d.high);
    assert!(d.single());
}

/// Python: FloatDistribution(0.0, 0.3, step=0.3) → high=0.3 (恰好整除)
#[test]
fn test_float_adjust_high_exact_step() {
    let d = FloatDistribution::new(0.0, 0.3, false, Some(0.3)).unwrap();
    assert!((d.high - 0.3).abs() < 1e-12, "Python: high=0.3, got {}", d.high);
    assert!(!d.single(), "step == range → 2 values (0.0, 0.3) → NOT single");
}

/// Python: FloatDistribution(0.0, 0.0, step=0.3) → high=0.0
#[test]
fn test_float_adjust_high_zero_range() {
    let d = FloatDistribution::new(0.0, 0.0, false, Some(0.3)).unwrap();
    assert_eq!(d.high, 0.0);
    assert!(d.single());
}

// ============================================================================
// 2. FloatDistribution: single() 精度边界
// ============================================================================

/// Python: FloatDistribution(0.0, 1.0, step=0.3) → adjusted 0.9, single=false
#[test]
fn test_float_single_case0() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 0.1, step=0.3) → adjusted 0.0, single=true
#[test]
fn test_float_single_case1() {
    let d = FloatDistribution::new(0.0, 0.1, false, Some(0.3)).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(0.0, 0.3, step=0.3) → high=0.3, single=false
#[test]
fn test_float_single_case2() {
    let d = FloatDistribution::new(0.0, 0.3, false, Some(0.3)).unwrap();
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 0.29, step=0.3) → adjusted 0.0, single=true
#[test]
fn test_float_single_case3() {
    let d = FloatDistribution::new(0.0, 0.29, false, Some(0.3)).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(5.0, 5.0) → single=true
#[test]
fn test_float_single_equal_no_step() {
    let d = FloatDistribution::new(5.0, 5.0, false, None).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(5.0, 5.0, step=0.1) → single=true
#[test]
fn test_float_single_equal_with_step() {
    let d = FloatDistribution::new(5.0, 5.0, false, Some(0.1)).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(1e-10, 1e-10) → single=true
#[test]
fn test_float_single_very_small_equal() {
    let d = FloatDistribution::new(1e-10, 1e-10, false, None).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(0.0, 1e-15) → single=false (非常小但不相等)
#[test]
fn test_float_single_very_close_not_equal() {
    let d = FloatDistribution::new(0.0, 1e-15, false, None).unwrap();
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=2.0) → adjusted 0.0, single=true
#[test]
fn test_float_single_step_larger_than_range() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(2.0)).unwrap();
    assert!(d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=1.0) → high=1.0, single=false
#[test]
fn test_float_single_step_equals_range() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(1.0)).unwrap();
    assert!(!d.single());
}

/// Python: FloatDistribution(0.0, 1.0, step=0.999) → high=0.999, single=false
#[test]
fn test_float_single_step_0999() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.999)).unwrap();
    assert!((d.high - 0.999).abs() < 1e-12);
    assert!(!d.single(), "step == adjusted range → NOT single");
}

/// Python: FloatDistribution(0.0, 1.0, step=1.001) → adjusted 0.0, single=true
#[test]
fn test_float_single_step_1001() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(1.001)).unwrap();
    assert!(d.single());
}

// ============================================================================
// 3. FloatDistribution: contains() 精度边界
// ============================================================================

/// 连续 FloatDistribution(0.0, 1.0) 的 contains 边界
#[test]
fn test_float_contains_continuous() {
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert!(d.contains(0.0));
    assert!(d.contains(0.5));
    assert!(d.contains(1.0));
    assert!(!d.contains(-1e-15), "Python: just below low → false");
    assert!(!d.contains(1.0 + 1e-15), "Python: just above high → false");
    assert!(!d.contains(f64::NAN), "Python: NaN → false");
    assert!(!d.contains(f64::INFINITY), "Python: Inf → false");
    assert!(!d.contains(f64::NEG_INFINITY), "Python: -Inf → false");
}

/// step 分布 contains 1e-8 容差
#[test]
fn test_float_contains_step_tolerance() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
    // 精确网格点
    assert!(d.contains(0.0));
    assert!(d.contains(0.25));
    assert!(d.contains(0.5));
    assert!(d.contains(0.75));
    assert!(d.contains(1.0));
    // 非网格点
    assert!(!d.contains(0.1));
    assert!(!d.contains(0.3));
    assert!(!d.contains(0.125));
    // 1e-9 偏差 → 在 1e-8 容差内
    assert!(d.contains(0.25 + 1e-9), "Python: 1e-9 offset passes 1e-8 tolerance");
    assert!(d.contains(0.25 - 1e-9), "Python: -1e-9 offset passes");
    // 1e-6 偏差 → 超出 1e-8 容差
    assert!(!d.contains(0.25 + 1e-6), "Python: 1e-6 offset fails");
    assert!(!d.contains(0.25 - 1e-6), "Python: -1e-6 offset fails");
}

/// log 分布 contains
#[test]
fn test_float_contains_log() {
    let d = FloatDistribution::new(0.001, 100.0, true, None).unwrap();
    assert!(d.contains(0.001));
    assert!(d.contains(1.0));
    assert!(d.contains(100.0));
    assert!(!d.contains(0.0), "log: 0 below low");
    assert!(!d.contains(-1.0), "log: negative");
    assert!(!d.contains(100.001), "log: above high");
}

// ============================================================================
// 4. FloatDistribution: to_internal_repr / to_external_repr 往返
// ============================================================================

/// Python: FloatDistribution(0.0, 10.0) repr 恒等映射
#[test]
fn test_float_repr_identity() {
    let d = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    for v in [0.0, 1.5, 5.0, 10.0, 3.14159265, 1e-10, 9.999999] {
        let internal = d.to_internal_repr(v).unwrap();
        let external = d.to_external_repr(internal);
        assert!((external - v).abs() < 1e-15, "roundtrip failed for {v}");
    }
}

/// Python: FloatDistribution(0.001, 100.0, log=True) repr 恒等映射
#[test]
fn test_float_repr_log() {
    let d = FloatDistribution::new(0.001, 100.0, true, None).unwrap();
    for v in [0.001, 0.1, 1.0, 50.0, 100.0] {
        let internal = d.to_internal_repr(v).unwrap();
        let external = d.to_external_repr(internal);
        assert!((external - v).abs() < 1e-15);
    }
}

/// log 分布拒绝 0 和负数
#[test]
fn test_float_repr_log_rejects() {
    let d = FloatDistribution::new(0.001, 100.0, true, None).unwrap();
    assert!(d.to_internal_repr(0.0).is_err());
    assert!(d.to_internal_repr(-1.0).is_err());
    assert!(d.to_internal_repr(f64::NAN).is_err());
}

// ============================================================================
// 5. IntDistribution: _adjust_int_uniform_high 精确对比
// ============================================================================

#[test]
fn test_int_adjust_high_step3() {
    let d = IntDistribution::new(0, 10, false, 3).unwrap();
    assert_eq!(d.high, 9, "Python: high=9");
    assert!(!d.single());
}

#[test]
fn test_int_adjust_high_step2() {
    let d = IntDistribution::new(0, 10, false, 2).unwrap();
    assert_eq!(d.high, 10, "Python: high=10 (整除)");
}

#[test]
fn test_int_adjust_high_step1() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    assert_eq!(d.high, 10);
}

#[test]
fn test_int_adjust_high_step7() {
    let d = IntDistribution::new(0, 20, false, 7).unwrap();
    assert_eq!(d.high, 14, "Python: high=14");
}

#[test]
fn test_int_adjust_high_step30() {
    let d = IntDistribution::new(0, 100, false, 30).unwrap();
    assert_eq!(d.high, 90, "Python: high=90");
}

#[test]
fn test_int_adjust_high_offset5_step4() {
    let d = IntDistribution::new(5, 25, false, 4).unwrap();
    assert_eq!(d.high, 25, "Python: high=25 (20%4=0, 整除)");
}

#[test]
fn test_int_adjust_high_offset5_step4_adj() {
    let d = IntDistribution::new(5, 26, false, 4).unwrap();
    assert_eq!(d.high, 25, "Python: high=25 (21%4=1, 调整)");
}

#[test]
fn test_int_adjust_high_tiny_range() {
    let d = IntDistribution::new(0, 2, false, 5).unwrap();
    assert_eq!(d.high, 0, "Python: high=0 (2%5=2, 0*5+0=0)");
    assert!(d.single());
}

#[test]
fn test_int_adjust_high_zero_range() {
    let d = IntDistribution::new(0, 0, false, 1).unwrap();
    assert_eq!(d.high, 0);
    assert!(d.single());
}

#[test]
fn test_int_adjust_high_large_range() {
    let d = IntDistribution::new(1, 1000, false, 1).unwrap();
    assert_eq!(d.high, 1000);
}

#[test]
fn test_int_adjust_high_large_step7() {
    let d = IntDistribution::new(1, 1000, false, 7).unwrap();
    assert_eq!(d.high, 995, "Python: high=995 (999//7=142, 142*7+1=995)");
}

// ============================================================================
// 6. IntDistribution: single()
// ============================================================================

#[test]
fn test_int_single_basic_false() {
    assert!(!IntDistribution::new(0, 10, false, 1).unwrap().single());
}

#[test]
fn test_int_single_equal_true() {
    assert!(IntDistribution::new(5, 5, false, 1).unwrap().single());
}

#[test]
fn test_int_single_log_equal_true() {
    assert!(IntDistribution::new(5, 5, true, 1).unwrap().single());
}

#[test]
fn test_int_single_log_range_false() {
    assert!(!IntDistribution::new(1, 100, true, 1).unwrap().single());
}

#[test]
fn test_int_single_step_gt_range() {
    let d = IntDistribution::new(0, 2, false, 5).unwrap();
    assert!(d.single(), "Python: adjusted high=0 → single");
}

#[test]
fn test_int_single_step_eq_range() {
    // (10-0) / 5 = 2, NOT single (3 values: 0, 5, 10)
    assert!(!IntDistribution::new(0, 10, false, 5).unwrap().single());
}

#[test]
fn test_int_single_exact_two_values() {
    // (5-0) / 5 = 1, NOT single (2 values: 0, 5)
    assert!(!IntDistribution::new(0, 5, false, 5).unwrap().single());
}

#[test]
fn test_int_single_adjusted_to_zero() {
    // 4 % 5 != 0 → high adjusted to 0, single
    assert!(IntDistribution::new(0, 4, false, 5).unwrap().single());
}

// ============================================================================
// 7. IntDistribution: contains()
// ============================================================================

#[test]
fn test_int_contains_step3() {
    let d = IntDistribution::new(0, 10, false, 3).unwrap();
    // adjusted high = 9
    assert!(d.contains(0.0));
    assert!(d.contains(3.0));
    assert!(d.contains(6.0));
    assert!(d.contains(9.0));
    assert!(!d.contains(10.0), "Python: 10 > adjusted high 9");
    assert!(!d.contains(1.0), "Python: not on step grid");
    assert!(!d.contains(2.0));
    assert!(!d.contains(-1.0));
    assert!(!d.contains(12.0));
    assert!(!d.contains(0.5), "Python: non-integer");
}

#[test]
fn test_int_contains_nan_inf() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(!d.contains(f64::NAN), "Python: NaN → ValueError → false");
    assert!(!d.contains(f64::INFINITY), "Python: Inf → false");
    assert!(!d.contains(f64::NEG_INFINITY));
}

// ============================================================================
// 8. IntDistribution: repr 往返
// ============================================================================

#[test]
fn test_int_repr_roundtrip() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    for v in [0i64, 5, 10] {
        let internal = d.to_internal_repr(v).unwrap();
        let external = d.to_external_repr(internal).unwrap();
        assert_eq!(v, external, "Python: roundtrip({v})");
    }
}

#[test]
fn test_int_repr_nan_inf_error() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(d.to_external_repr(f64::NAN).is_err());
    assert!(d.to_external_repr(f64::INFINITY).is_err());
    assert!(d.to_external_repr(f64::NEG_INFINITY).is_err());
}

// ============================================================================
// 9. CategoricalDistribution: repr、contains、single
// ============================================================================

#[test]
fn test_categorical_string_repr_deep() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".into()),
        CategoricalChoice::Str("b".into()),
        CategoricalChoice::Str("c".into()),
    ]).unwrap();
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Str("a".into())).unwrap(), 0.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Str("b".into())).unwrap(), 1.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Str("c".into())).unwrap(), 2.0);
    assert_eq!(d.to_external_repr(0.0).unwrap(), CategoricalChoice::Str("a".into()));
    assert_eq!(d.to_external_repr(2.0).unwrap(), CategoricalChoice::Str("c".into()));
    assert!(!d.single());
}

#[test]
fn test_categorical_mixed_types() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::None,
        CategoricalChoice::Bool(true),
        CategoricalChoice::Int(42),
        CategoricalChoice::Float(3.14),
        CategoricalChoice::Str("hello".into()),
    ]).unwrap();
    assert_eq!(d.to_internal_repr(&CategoricalChoice::None).unwrap(), 0.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Bool(true)).unwrap(), 1.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Int(42)).unwrap(), 2.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Float(3.14)).unwrap(), 3.0);
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Str("hello".into())).unwrap(), 4.0);
    assert!(!d.single());
}

#[test]
fn test_categorical_contains_deep() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("x".into()),
        CategoricalChoice::Str("y".into()),
        CategoricalChoice::Str("z".into()),
    ]).unwrap();
    assert!(d.contains(0.0));
    assert!(d.contains(1.0));
    assert!(d.contains(2.0));
    assert!(!d.contains(3.0));
    assert!(!d.contains(-1.0));
    // Python: int(0.5)=0 → valid index
    assert!(d.contains(0.5), "Python: int(0.5)=0 → valid");
    // Python: int(2.9)=2 → valid index
    assert!(d.contains(2.9), "Python: int(2.9)=2 → valid");
    // NaN, Inf
    assert!(!d.contains(f64::NAN));
    assert!(!d.contains(f64::INFINITY));
}

#[test]
fn test_categorical_single_deep() {
    let d = CategoricalDistribution::new(vec![CategoricalChoice::Int(42)]).unwrap();
    assert!(d.single());
    assert_eq!(d.to_internal_repr(&CategoricalChoice::Int(42)).unwrap(), 0.0);
    assert_eq!(d.to_external_repr(0.0).unwrap(), CategoricalChoice::Int(42));
}

#[test]
fn test_categorical_nan_repr_deep() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Float(f64::NAN),
        CategoricalChoice::Float(1.0),
    ]).unwrap();
    // NaN 在 choices 中应被正确索引
    let idx = d.to_internal_repr(&CategoricalChoice::Float(f64::NAN)).unwrap();
    assert_eq!(idx, 0.0, "Python: NaN → index 0");
    let ext = d.to_external_repr(0.0).unwrap();
    match ext {
        CategoricalChoice::Float(v) => assert!(v.is_nan()),
        _ => panic!("Expected Float(NaN)"),
    }
    // to_external_repr 拒绝 NaN 和负数
    assert!(d.to_external_repr(f64::NAN).is_err());
    assert!(d.to_external_repr(-1.0).is_err());
}

// ============================================================================
// 10. check_distribution_compatibility
// ============================================================================

#[test]
fn test_compat_float_same_log_false() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok());
}

#[test]
fn test_compat_float_diff_log() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, true, None).unwrap());
    let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

#[test]
fn test_compat_float_diff_step_ok() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
    let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.2)).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok(), "Python 不检查 step");
}

#[test]
fn test_compat_float_step_vs_none_ok() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
    let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok(), "Python 不检查 step");
}

#[test]
fn test_compat_int_same_log_false() {
    let a = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    let b = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 1).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok());
}

#[test]
fn test_compat_int_diff_log() {
    let a = Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap());
    let b = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 1).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

#[test]
fn test_compat_int_diff_step_ok() {
    let a = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    let b = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 2).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_ok(), "Python 不检查 step");
}

#[test]
fn test_compat_cat_same() {
    let a = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ]).unwrap(),
    );
    let b = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ]).unwrap(),
    );
    assert!(check_distribution_compatibility(&a, &b).is_ok());
}

#[test]
fn test_compat_cat_diff() {
    let a = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ]).unwrap(),
    );
    let b = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("c".into()),
        ]).unwrap(),
    );
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

#[test]
fn test_compat_float_vs_int() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let b = Distribution::IntDistribution(IntDistribution::new(0, 1, false, 1).unwrap());
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

#[test]
fn test_compat_float_vs_cat() {
    let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let b = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
    );
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

#[test]
fn test_compat_int_vs_cat() {
    let a = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    let b = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
    );
    assert!(check_distribution_compatibility(&a, &b).is_err());
}

// ============================================================================
// 11. get_single_value
// ============================================================================

#[test]
fn test_get_single_value_float() {
    use optuna_rs::distributions::ParamValue;
    let d = Distribution::FloatDistribution(FloatDistribution::new(5.0, 5.0, false, None).unwrap());
    assert!(d.single());
    assert_eq!(d.get_single_value().unwrap(), ParamValue::Float(5.0));
}

#[test]
fn test_get_single_value_float_step() {
    use optuna_rs::distributions::ParamValue;
    let d = Distribution::FloatDistribution(FloatDistribution::new(3.0, 3.1, false, Some(0.5)).unwrap());
    // adjusted high = 3.0 (since (3.1-3.0)=0.1 < 0.5)
    assert!(d.single());
    assert_eq!(d.get_single_value().unwrap(), ParamValue::Float(3.0));
}

#[test]
fn test_get_single_value_int() {
    use optuna_rs::distributions::ParamValue;
    let d = Distribution::IntDistribution(IntDistribution::new(7, 7, false, 1).unwrap());
    assert!(d.single());
    assert_eq!(d.get_single_value().unwrap(), ParamValue::Int(7));
}

#[test]
fn test_get_single_value_int_step() {
    use optuna_rs::distributions::ParamValue;
    let d = Distribution::IntDistribution(IntDistribution::new(0, 2, false, 5).unwrap());
    // adjusted high = 0
    assert!(d.single());
    assert_eq!(d.get_single_value().unwrap(), ParamValue::Int(0));
}

#[test]
fn test_get_single_value_categorical() {
    use optuna_rs::distributions::ParamValue;
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![CategoricalChoice::Str("only".into())]).unwrap(),
    );
    assert!(d.single());
    assert_eq!(
        d.get_single_value().unwrap(),
        ParamValue::Categorical(CategoricalChoice::Str("only".into()))
    );
}

// ============================================================================
// 12. is_log
// ============================================================================

#[test]
fn test_is_log_float_false() {
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    assert!(!d.is_log());
}

#[test]
fn test_is_log_float_true() {
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, true, None).unwrap());
    assert!(d.is_log());
}

#[test]
fn test_is_log_int_false() {
    let d = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    assert!(!d.is_log());
}

#[test]
fn test_is_log_int_true() {
    let d = Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap());
    assert!(d.is_log());
}

#[test]
fn test_is_log_categorical() {
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
    );
    assert!(!d.is_log());
}

// ============================================================================
// 13. JSON 序列化/反序列化 往返
// ============================================================================

#[test]
fn test_json_roundtrip_float_basic() {
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_float_log() {
    let d = Distribution::FloatDistribution(FloatDistribution::new(1e-5, 1.0, true, None).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_float_step() {
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_float_step03() {
    // step=0.3 → high adjusted to 0.9
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
    if let Distribution::FloatDistribution(fd) = &back {
        assert!((fd.high - 0.9).abs() < 1e-12, "JSON roundtrip preserves adjusted high");
    } else {
        panic!("Expected FloatDistribution");
    }
}

#[test]
fn test_json_roundtrip_int_basic() {
    let d = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 1).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_int_log() {
    let d = Distribution::IntDistribution(IntDistribution::new(1, 1000, true, 1).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_int_step3() {
    // step=3 → high adjusted to 9
    let d = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 3).unwrap());
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
    if let Distribution::IntDistribution(id) = &back {
        assert_eq!(id.high, 9, "JSON roundtrip preserves adjusted high");
    }
}

#[test]
fn test_json_roundtrip_cat_str() {
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("x".into()),
            CategoricalChoice::Str("y".into()),
            CategoricalChoice::Str("z".into()),
        ]).unwrap(),
    );
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

#[test]
fn test_json_roundtrip_cat_mixed() {
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::None,
            CategoricalChoice::Int(1),
            CategoricalChoice::Float(2.5),
            CategoricalChoice::Str("hi".into()),
        ]).unwrap(),
    );
    let json = distribution_to_json(&d).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(d, back);
}

// ============================================================================
// 14. 验证约束 — 对齐 Python 构造器错误检查
// ============================================================================

#[test]
fn test_float_validation_errors() {
    // log + step → 报错
    assert!(FloatDistribution::new(0.1, 1.0, true, Some(0.1)).is_err());
    // low > high → 报错
    assert!(FloatDistribution::new(2.0, 1.0, false, None).is_err());
    // log + low <= 0 → 报错
    assert!(FloatDistribution::new(0.0, 1.0, true, None).is_err());
    assert!(FloatDistribution::new(-1.0, 1.0, true, None).is_err());
    // step <= 0 → 报错
    assert!(FloatDistribution::new(0.0, 1.0, false, Some(0.0)).is_err());
    assert!(FloatDistribution::new(0.0, 1.0, false, Some(-0.1)).is_err());
}

#[test]
fn test_int_validation_errors() {
    // log + step != 1 → 报错
    assert!(IntDistribution::new(1, 10, true, 2).is_err());
    // low > high → 报错
    assert!(IntDistribution::new(10, 1, false, 1).is_err());
    // log + low < 1 → 报错
    assert!(IntDistribution::new(0, 10, true, 1).is_err());
    // step <= 0 → 报错
    assert!(IntDistribution::new(0, 10, false, 0).is_err());
    assert!(IntDistribution::new(0, 10, false, -1).is_err());
}

#[test]
fn test_categorical_validation_errors() {
    // 空选项 → 报错
    assert!(CategoricalDistribution::new(vec![]).is_err());
}

// ============================================================================
// 15. Distribution enum: to_internal_repr 跨类型转换
// ============================================================================

#[test]
fn test_distribution_enum_cross_type_int_to_float() {
    use optuna_rs::distributions::ParamValue;
    // ParamValue::Int(5) → FloatDistribution → should coerce to 5.0
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
    assert_eq!(d.to_internal_repr(&ParamValue::Int(5)).unwrap(), 5.0);
}

#[test]
fn test_distribution_enum_cross_type_float_to_int() {
    use optuna_rs::distributions::ParamValue;
    // ParamValue::Float(5.0) → IntDistribution → should coerce to int 5
    let d = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    assert_eq!(d.to_internal_repr(&ParamValue::Float(5.0)).unwrap(), 5.0);
}

#[test]
fn test_distribution_enum_int_to_categorical() {
    use optuna_rs::distributions::ParamValue;
    // ParamValue::Int(42) → CategoricalDistribution with Int choices
    let d = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Int(10),
            CategoricalChoice::Int(42),
        ]).unwrap(),
    );
    assert_eq!(d.to_internal_repr(&ParamValue::Int(42)).unwrap(), 1.0);
}

#[test]
fn test_distribution_enum_type_mismatch_error() {
    use optuna_rs::distributions::ParamValue;
    // ParamValue::Categorical → FloatDistribution → type mismatch
    let d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    assert!(d.to_internal_repr(&ParamValue::Categorical(CategoricalChoice::Str("x".into()))).is_err());
}

// ============================================================================
// 16. Legacy JSON format 兼容
// ============================================================================

#[test]
fn test_legacy_uniform() {
    let json = r#"{"name":"UniformDistribution","attributes":{"low":0.0,"high":1.0}}"#;
    let dist = json_to_distribution(json).unwrap();
    if let Distribution::FloatDistribution(fd) = dist {
        assert_eq!(fd.low, 0.0);
        assert_eq!(fd.high, 1.0);
        assert!(!fd.log);
        assert!(fd.step.is_none());
    } else {
        panic!("Expected FloatDistribution");
    }
}

#[test]
fn test_legacy_log_uniform() {
    let json = r#"{"name":"LogUniformDistribution","attributes":{"low":0.0000001,"high":1.0}}"#;
    let dist = json_to_distribution(json).unwrap();
    if let Distribution::FloatDistribution(fd) = dist {
        assert!(fd.log);
    } else {
        panic!("Expected FloatDistribution");
    }
}

#[test]
fn test_legacy_discrete_uniform() {
    let json = r#"{"name":"DiscreteUniformDistribution","attributes":{"low":0.0,"high":1.0,"q":0.1}}"#;
    let dist = json_to_distribution(json).unwrap();
    if let Distribution::FloatDistribution(fd) = dist {
        assert!(!fd.log);
        assert_eq!(fd.step, Some(0.1));
    } else {
        panic!("Expected FloatDistribution");
    }
}

#[test]
fn test_legacy_int_uniform() {
    let json = r#"{"name":"IntUniformDistribution","attributes":{"low":0,"high":10,"step":1}}"#;
    let dist = json_to_distribution(json).unwrap();
    if let Distribution::IntDistribution(id) = dist {
        assert_eq!(id.low, 0);
        assert_eq!(id.high, 10);
        assert!(!id.log);
    } else {
        panic!("Expected IntDistribution");
    }
}

#[test]
fn test_legacy_int_log_uniform() {
    let json = r#"{"name":"IntLogUniformDistribution","attributes":{"low":1,"high":100,"step":1}}"#;
    let dist = json_to_distribution(json).unwrap();
    if let Distribution::IntDistribution(id) = dist {
        assert!(id.log);
        assert_eq!(id.low, 1);
        assert_eq!(id.high, 100);
    } else {
        panic!("Expected IntDistribution");
    }
}
