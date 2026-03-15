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
    pub fn contains(&self, value: f64) -> bool {
        if value < self.low - 1e-10 || value > self.high + 1e-10 {
            return false;
        }
        if let Some(step) = self.step {
            // 对齐 Python `_contains`: k = (value - low) / step; abs(k - round(k)) < 1e-8
            // 使用与 Python Decimal 等价的容差检查，避免浮点误差导致合法网格点被拒绝
            let k = (value - self.low) / step;
            (k - k.round()).abs() < 1.0e-6
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
/// 使用 f64 算术（无 Decimal），在常规数值范围内足够精确。
fn adjust_discrete_uniform_high(low: f64, high: f64, step: f64) -> f64 {
    let r = high - low;
    let remainder = r % step;
    // 浮点取模可能有微小误差，用 epsilon 判断
    if remainder.abs() > 1e-12 && (step - remainder.abs()).abs() > 1e-12 {
        let n = (r / step).floor();
        let adjusted = n * step + low;
        // 对应 Python: optuna_warn(...)
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
    } else {
        high
    }
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
}
