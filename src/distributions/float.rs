use serde::{Deserialize, Serialize};

use crate::error::{OptunaError, Result};
use crate::optuna_warn;

/// A distribution over floating-point values.
///
/// Corresponds to Python `optuna.distributions.FloatDistribution`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatDistribution {
    pub low: f64,
    pub high: f64,
    #[serde(default)]
    pub log: bool,
    #[serde(default)]
    pub step: Option<f64>,
}

impl FloatDistribution {
    /// Create a new `FloatDistribution` with validation.
    pub fn new(low: f64, high: f64, log: bool, step: Option<f64>) -> Result<Self> {
        if log && step.is_some() {
            return Err(OptunaError::InvalidDistribution(
                "cannot combine log-scale with discretization step".into(),
            ));
        }
        if low > high {
            return Err(OptunaError::InvalidDistribution(format!(
                "low must be <= high, got low={low}, high={high}"
            )));
        }
        if log && low <= 0.0 {
            return Err(OptunaError::InvalidDistribution(format!(
                "low must be > 0 for log-scale, got low={low}"
            )));
        }
        if let Some(s) = step
            && s <= 0.0
        {
            return Err(OptunaError::InvalidDistribution(format!(
                "step must be > 0, got step={s}"
            )));
        }

        // 对应 Python `high = _adjust_discrete_uniform_high(low, high, step)`
        // 当 (high - low) 不能被 step 整除时，向下调整 high 到最近的 step 网格点
        let adjusted_high = if let Some(s) = step {
            adjust_discrete_uniform_high(low, high, s)
        } else {
            high
        };

        Ok(Self {
            low,
            high: adjusted_high,
            log,
            step,
        })
    }

    /// Check if `value` (in internal representation) is contained in this distribution.
    /// 对齐 Python: 严格 low <= value <= high（无容差）
    pub fn contains(&self, value: f64) -> bool {
        if value < self.low || value > self.high {
            return false;
        }
        if let Some(step) = self.step {
            // 对齐 Python `_contains`: (value - low) % step == 0
            // 使用浮点容差: k = (value - low) / step; abs(k - round(k)) < 1e-8
            let k = (value - self.low) / step;
            (k - k.round()).abs() < 1.0e-8
        } else {
            true
        }
    }

    /// Convert an external value to internal representation (f64).
    pub fn to_internal_repr(&self, value: f64) -> Result<f64> {
        if value.is_nan() {
            return Err(OptunaError::ValueError("NaN is not allowed".into()));
        }
        if self.log && value <= 0.0 {
            return Err(OptunaError::ValueError(format!(
                "value must be > 0 for log-scale, got {value}"
            )));
        }
        Ok(value)
    }

    /// Convert an internal representation back to an external value.
    pub fn to_external_repr(&self, value: f64) -> f64 {
        value
    }

    /// True if this distribution contains exactly one value.
    ///
    /// 对应 Python `FloatDistribution.single()`:
    /// - 无 step: `self.low == self.high`
    /// - 有 step: `self.low == self.high` 或 `(high - low) < step`
    pub fn single(&self) -> bool {
        if let Some(step) = self.step {
            if self.low == self.high {
                return true;
            }
            // 对应 Python: `return (high - low) < step`（使用 Decimal 精度）
            (self.high - self.low) < step
        } else {
            self.low == self.high
        }
    }
}

/// 对应 Python `_adjust_discrete_uniform_high(low, high, step)`。
/// 当 (high - low) 不能被 step 整除时，向下调整 high 到最近的 step 网格点。
/// 对齐 Python: 使用与 Decimal(str(...)) 等价的精确计算方式。
fn adjust_discrete_uniform_high(low: f64, high: f64, step: f64) -> f64 {
    // 模拟 Python Decimal(str(x)) 算术:
    // n = floor((high - low) / step), adjusted = low + n * step
    // 由于 f64 直接运算会有精度问题 (如 3.0*0.3 = 0.8999...),
    // 先计算 n，然后用 round-trip 通过字符串精确表示
    let r = high - low;
    let n = (r / step).floor();
    let n_steps = n as i64;
    let remainder_check = r - n * step;

    // 检查 remainder 是否为零（含浮点容差）
    if remainder_check.abs() < 1e-12 || (step - remainder_check.abs()).abs() < 1e-12 {
        return high;
    }

    // 使用与 Python Decimal(str(x)) 等价的方法：
    // 通过格式化为字符串并重新解析来获得精确结果
    // Python: float(Decimal(str(n)) * Decimal(str(step)) + Decimal(str(low)))
    // 在 Rust 中用有限精度的 f64 近似:
    // 由于 n 是整数, step 和 low 是用户提供的 "简单" 浮点数,
    // 使用 format!("{}", x) (Display trait) 模拟 str(x) 的行为
    let step_str = format!("{}", step);
    let low_str = format!("{}", low);

    // 解析小数位数以确定精度
    let step_decimals = step_str.find('.').map_or(0, |p| step_str.len() - p - 1);
    let low_decimals = low_str.find('.').map_or(0, |p| low_str.len() - p - 1);
    let max_decimals = step_decimals.max(low_decimals);

    // 缩放到整数域进行精确计算
    let scale = 10_f64.powi(max_decimals as i32);
    let step_scaled = (step * scale).round() as i64;
    let low_scaled = (low * scale).round() as i64;
    let adjusted_scaled = n_steps * step_scaled + low_scaled;
    let adjusted = adjusted_scaled as f64 / scale;

    optuna_warn!(
        "The distribution is specified by [{}, {}] and step={}, but the range is \
         not divisible by `step`. It will be replaced with [{}, {}].",
        low,
        high,
        step,
        low,
        adjusted
    );
    adjusted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_distribution() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert_eq!(d.low, 0.0);
        assert_eq!(d.high, 1.0);
        assert!(!d.log);
        assert!(d.step.is_none());
    }

    #[test]
    fn test_log_with_step_rejected() {
        assert!(FloatDistribution::new(0.1, 1.0, true, Some(0.1)).is_err());
    }

    #[test]
    fn test_low_greater_than_high() {
        assert!(FloatDistribution::new(2.0, 1.0, false, None).is_err());
    }

    #[test]
    fn test_log_non_positive_low() {
        assert!(FloatDistribution::new(0.0, 1.0, true, None).is_err());
        assert!(FloatDistribution::new(-1.0, 1.0, true, None).is_err());
    }

    #[test]
    fn test_negative_step() {
        assert!(FloatDistribution::new(0.0, 1.0, false, Some(-0.1)).is_err());
        assert!(FloatDistribution::new(0.0, 1.0, false, Some(0.0)).is_err());
    }

    #[test]
    fn test_contains() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
        assert!(!d.contains(-0.1));
        assert!(!d.contains(1.1));
    }

    #[test]
    fn test_contains_with_step() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(0.25));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
        assert!(!d.contains(0.1));
    }

    #[test]
    fn test_single() {
        assert!(FloatDistribution::new(1.0, 1.0, false, None).unwrap().single());
        assert!(!FloatDistribution::new(0.0, 1.0, false, None).unwrap().single());
    }

    /// 对应 Python: `FloatDistribution(0.0, 0.1, step=0.2).single() == True`
    /// 因为 (0.1 - 0.0) < 0.2
    #[test]
    fn test_single_with_step() {
        // (high - low) < step → single
        let d = FloatDistribution::new(0.0, 0.1, false, Some(0.2)).unwrap();
        assert!(d.single(), "(0.1 - 0.0) < 0.2 → should be single");

        // (high - low) == step → NOT single (range contains exactly 2 values: low and low+step)
        let d2 = FloatDistribution::new(0.0, 0.5, false, Some(0.5)).unwrap();
        assert!(!d2.single(), "(0.5 - 0.0) == 0.5 → NOT single");

        // (high - low) > step → NOT single
        let d3 = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        assert!(!d3.single());
    }

    /// 对应 Python `_adjust_discrete_uniform_high`。
    /// FloatDistribution(0.0, 1.0, step=0.3) → high 应被调整为 0.9
    #[test]
    fn test_high_adjustment_float() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
        assert!(
            (d.high - 0.9).abs() < 1e-10,
            "high should be adjusted to 0.9, got {}",
            d.high
        );
    }

    /// 范围刚好整除 step 时不应调整
    #[test]
    fn test_no_adjustment_when_divisible() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        assert_eq!(d.high, 1.0, "high should NOT be adjusted when range is divisible by step");
    }

    /// 对应 Python: 验证 high 调整后 contains 行为正确
    #[test]
    fn test_contains_after_adjustment() {
        // step=0.3: high adjusted from 1.0 to ~0.9
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(0.3));
        assert!(d.contains(0.6));
        // 使用调整后的 high 值检查（避免浮点精度问题）
        assert!(d.contains(d.high), "adjusted high should be valid");
        assert!(!d.contains(1.0), "1.0 should not be in range after adjustment");
    }

    #[test]
    fn test_to_internal_repr_nan() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert!(d.to_internal_repr(f64::NAN).is_err());
    }

    /// to_internal_repr / to_external_repr 往返一致
    #[test]
    fn test_repr_roundtrip() {
        let d = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
        for v in [0.0, 1.5, 5.0, 10.0] {
            let internal = d.to_internal_repr(v).unwrap();
            let external = d.to_external_repr(internal);
            assert!((v - external).abs() < 1e-15);
        }
    }

    /// log 分布 to_internal_repr 拒绝非正值
    #[test]
    fn test_log_to_internal_repr_rejects_non_positive() {
        let d = FloatDistribution::new(0.1, 10.0, true, None).unwrap();
        assert!(d.to_internal_repr(0.0).is_err());
        assert!(d.to_internal_repr(-1.0).is_err());
        assert!(d.to_internal_repr(1.0).is_ok());
    }

    // ========================================================================
    // 以下测试使用 Python optuna 生成的精确参考值进行交叉验证。
    // 每个 assert 旁的注释标注了 Python 的对应输出。
    // ========================================================================

    /// Python 交叉验证: FloatDistribution(0.0, 1.0, step=0.3)
    /// Python: high=0.9, single=false
    #[test]
    fn test_python_cross_float_step03() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
        assert!((d.high - 0.9).abs() < 1e-10, "Python: high=0.9, got {}", d.high);
        assert!(!d.single(), "Python: single=false");
        assert!(d.contains(0.9),  "Python: contains(0.9)=true");
        assert!(!d.contains(1.0), "Python: contains(1.0)=false");
        assert!(d.contains(0.6),  "Python: contains(0.6)=true");
        assert!(d.contains(0.3),  "Python: contains(0.3)=true");
        assert!(d.contains(0.0),  "Python: contains(0.0)=true");
        assert!(!d.contains(0.4), "Python: contains(0.4)=false");
    }

    /// Python 交叉验证: FloatDistribution(0.0, 1.0, step=0.25)
    /// Python: high=1.0, single=false
    #[test]
    fn test_python_cross_float_step025() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        assert_eq!(d.high, 1.0, "Python: high=1.0");
        assert!(!d.single(), "Python: single=false");
    }

    /// 验证 contains() 使用 1e-8 容差（对齐 Python _contains）。
    /// 容差公式: abs((value - low) / step - round((value - low) / step)) < 1e-8
    /// step=0.25 时, offset=1e-9 → k_offset=4e-9 < 1e-8 → 通过
    ///              offset=5e-9 → k_offset=2e-8 > 1e-8 → 拒绝
    #[test]
    fn test_contains_tolerance_1e8() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        // 偏差 1e-9 → k_offset = 1e-9 / 0.25 = 4e-9 < 1e-8 → 应通过
        assert!(d.contains(0.25 + 1e-9), "1e-9 offset should pass 1e-8 tolerance");
        assert!(d.contains(0.25 - 1e-9), "1e-9 negative offset should pass");
        // 偏差 5e-9 → k_offset = 5e-9 / 0.25 = 2e-8 > 1e-8 → 应拒绝
        assert!(!d.contains(0.25 + 5e-9), "5e-9 offset should fail (k_offset=2e-8 > 1e-8)");
        assert!(!d.contains(0.25 - 5e-9), "5e-9 negative offset should fail");
    }

    /// Python 交叉验证: FloatDistribution(0.0, 0.1, step=0.2)
    /// Python: high=0.0 (调整), single=true
    #[test]
    fn test_python_cross_float_small_range_step() {
        let d = FloatDistribution::new(0.0, 0.1, false, Some(0.2)).unwrap();
        assert!((d.high - 0.0).abs() < 1e-10, "Python: high=0.0, got {}", d.high);
        assert!(d.single(), "Python: single=true");
    }

    /// Python 交叉验证: FloatDistribution(0.001, 10.0, log=True)
    /// Python: single=false
    #[test]
    fn test_python_cross_float_log() {
        let d = FloatDistribution::new(0.001, 10.0, true, None).unwrap();
        assert!(!d.single(), "Python: single=false");
    }

    /// Python 交叉验证: FloatDistribution(5.0, 5.0)
    /// Python: single=true
    #[test]
    fn test_python_cross_float_equal() {
        let d = FloatDistribution::new(5.0, 5.0, false, None).unwrap();
        assert!(d.single(), "Python: single=true");
    }

    /// Python 交叉验证: FloatDistribution(0.0, 1.0, step=0.7)
    /// Python: high=0.7, contains(0.7)=true, contains(0.0)=true, contains(1.0)=false
    #[test]
    fn test_python_cross_float_step07() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.7)).unwrap();
        assert!((d.high - 0.7).abs() < 1e-12, "Python: high=0.7");
        assert!(d.contains(0.7), "Python: contains(0.7)=true");
        assert!(d.contains(0.0), "Python: contains(0.0)=true");
        assert!(!d.contains(1.0), "Python: contains(1.0)=false");
    }

    /// Python 交叉验证: repr 往返一致性
    /// Python: to_external_repr(to_internal_repr(v)) == v
    #[test]
    fn test_python_cross_float_repr() {
        let d = FloatDistribution::new(1.0, 10.0, false, None).unwrap();
        for v in [1.0, 5.5, 9.999] {
            let internal = d.to_internal_repr(v).unwrap();
            let external = d.to_external_repr(internal);
            assert!((external - v).abs() < 1e-12, "roundtrip failed for {v}");
        }
    }

    /// Python 交叉验证: log repr 往返
    /// Python: to_external_repr(to_internal_repr(v)) == v for log dist
    #[test]
    fn test_python_cross_float_log_repr() {
        let d = FloatDistribution::new(0.01, 100.0, true, None).unwrap();
        for v in [0.01, 1.0, 50.0, 100.0] {
            let internal = d.to_internal_repr(v).unwrap();
            let external = d.to_external_repr(internal);
            assert!((external - v).abs() < 1e-12);
        }
    }
}
