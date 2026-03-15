use serde::{Deserialize, Serialize};

use crate::error::{OptunaError, Result};
use crate::optuna_warn;

/// A distribution over integer values.
///
/// Corresponds to Python `optuna.distributions.IntDistribution`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntDistribution {
    pub low: i64,
    pub high: i64,
    #[serde(default)]
    pub log: bool,
    #[serde(default = "default_step")]
    pub step: i64,
}

fn default_step() -> i64 {
    1
}

impl IntDistribution {
    /// Create a new `IntDistribution` with validation.
    pub fn new(low: i64, high: i64, log: bool, step: i64) -> Result<Self> {
        if log && step != 1 {
            return Err(OptunaError::InvalidDistribution(
                "cannot combine log-scale with step != 1".into(),
            ));
        }
        if low > high {
            return Err(OptunaError::InvalidDistribution(format!(
                "low must be <= high, got low={low}, high={high}"
            )));
        }
        if log && low < 1 {
            return Err(OptunaError::InvalidDistribution(format!(
                "low must be >= 1 for log-scale, got low={low}"
            )));
        }
        if step <= 0 {
            return Err(OptunaError::InvalidDistribution(format!(
                "step must be > 0, got step={step}"
            )));
        }

        // 对应 Python `self.high = _adjust_int_uniform_high(self.low, high, self.step)`
        // 当 (high - low) 不能被 step 整除时，向下调整 high
        let adjusted_high = adjust_int_uniform_high(low, high, step);

        Ok(Self {
            low,
            high: adjusted_high,
            log,
            step,
        })
    }

    /// Convenience constructor with default step=1 and log=false.
    pub fn with_range(low: i64, high: i64) -> Result<Self> {
        Self::new(low, high, false, 1)
    }

    /// Check if `value` (in internal representation as f64) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        let v = value as i64;
        if (v as f64 - value).abs() > 1e-8 {
            return false;
        }
        v >= self.low && v <= self.high && (v - self.low) % self.step == 0
    }

    /// Convert an external value to internal representation (f64).
    pub fn to_internal_repr(&self, value: i64) -> Result<f64> {
        let f = value as f64;
        if f.is_nan() {
            return Err(OptunaError::ValueError("NaN is not allowed".into()));
        }
        if self.log && value <= 0 {
            return Err(OptunaError::ValueError(format!(
                "value must be > 0 for log-scale, got {value}"
            )));
        }
        Ok(f)
    }

    /// Convert an internal representation back to an external value.
    pub fn to_external_repr(&self, value: f64) -> i64 {
        value as i64
    }

    /// True if this distribution contains exactly one value.
    ///
    /// 对应 Python `IntDistribution.single()`:
    /// - log=true: `self.low == self.high`
    /// - log=false: `self.low == self.high` 或 `(high - low) < step`
    pub fn single(&self) -> bool {
        if self.log {
            return self.low == self.high;
        }
        if self.low == self.high {
            return true;
        }
        // 对应 Python: `return (self.high - self.low) < self.step`
        (self.high - self.low) < self.step
    }
}

/// 对应 Python `_adjust_int_uniform_high(low, high, step)`。
/// 当 (high - low) 不能被 step 整除时，向下调整 high 到最近的 step 网格点。
fn adjust_int_uniform_high(low: i64, high: i64, step: i64) -> i64 {
    let r = high - low;
    if r % step != 0 {
        let adjusted = r / step * step + low;
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
        let d = IntDistribution::new(1, 10, false, 1).unwrap();
        assert_eq!(d.low, 1);
        assert_eq!(d.high, 10);
    }

    #[test]
    fn test_log_with_step_rejected() {
        assert!(IntDistribution::new(1, 10, true, 2).is_err());
    }

    #[test]
    fn test_low_greater_than_high() {
        assert!(IntDistribution::new(10, 1, false, 1).is_err());
    }

    #[test]
    fn test_log_low_less_than_one() {
        assert!(IntDistribution::new(0, 10, true, 1).is_err());
        assert!(IntDistribution::new(-1, 10, true, 1).is_err());
    }

    #[test]
    fn test_non_positive_step() {
        assert!(IntDistribution::new(0, 10, false, 0).is_err());
        assert!(IntDistribution::new(0, 10, false, -1).is_err());
    }

    #[test]
    fn test_contains() {
        let d = IntDistribution::new(0, 10, false, 2).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(2.0));
        assert!(d.contains(10.0));
        assert!(!d.contains(1.0));
        assert!(!d.contains(0.5));
        assert!(!d.contains(-1.0));
        assert!(!d.contains(11.0));
    }

    #[test]
    fn test_single() {
        assert!(IntDistribution::new(5, 5, false, 1).unwrap().single());
        assert!(!IntDistribution::new(1, 10, false, 1).unwrap().single());
    }

    /// 对应 Python: `IntDistribution(0, 2, step=5).single() == True`
    /// 因为 (2 - 0) < 5
    #[test]
    fn test_single_with_step() {
        let d = IntDistribution::new(0, 2, false, 5).unwrap();
        assert!(d.single(), "(2 - 0) < 5 → should be single");

        let d2 = IntDistribution::new(0, 10, false, 5).unwrap();
        assert!(!d2.single(), "(10 - 0) >= 5 → NOT single");
    }

    /// 对应 Python: `IntDistribution(0, 2, step=5)` → high 仍为 0 (向下调整)
    #[test]
    fn test_single_by_high_adjustment() {
        // high=2, step=5 → range=2, 2%5 != 0 → adjusted high = 0/5*5+0 = 0
        let d = IntDistribution::new(0, 2, false, 5).unwrap();
        assert_eq!(d.high, 0, "high should be adjusted to 0 (2//5*5+0=0)");
        assert!(d.single());
    }

    /// 对应 Python `_adjust_int_uniform_high`
    /// IntDistribution(0, 10, step=3) → high 应被调整为 9
    #[test]
    fn test_high_adjustment_int() {
        let d = IntDistribution::new(0, 10, false, 3).unwrap();
        assert_eq!(d.high, 9, "high should be adjusted from 10 to 9");
    }

    /// 范围刚好整除 step 时不应调整
    #[test]
    fn test_no_adjustment_when_divisible_int() {
        let d = IntDistribution::new(0, 10, false, 2).unwrap();
        assert_eq!(d.high, 10, "high should NOT be adjusted when range is divisible by step");
    }

    /// contains 应在 high 调整后正确工作
    #[test]
    fn test_contains_after_adjustment_int() {
        let d = IntDistribution::new(0, 10, false, 3).unwrap();
        // high adjusted to 9
        assert!(d.contains(0.0));
        assert!(d.contains(3.0));
        assert!(d.contains(6.0));
        assert!(d.contains(9.0));
        assert!(!d.contains(10.0), "10 should not be in range after adjustment to 9");
    }

    /// log 分布的 single() 只看 low == high
    #[test]
    fn test_single_log() {
        assert!(IntDistribution::new(5, 5, true, 1).unwrap().single());
        assert!(!IntDistribution::new(1, 10, true, 1).unwrap().single());
    }

    #[test]
    fn test_to_external_repr() {
        let d = IntDistribution::new(0, 10, false, 1).unwrap();
        assert_eq!(d.to_external_repr(5.0), 5);
    }

    /// to_internal_repr / to_external_repr 往返一致
    #[test]
    fn test_repr_roundtrip() {
        let d = IntDistribution::new(0, 100, false, 1).unwrap();
        for v in [0, 25, 50, 100] {
            let internal = d.to_internal_repr(v).unwrap();
            let external = d.to_external_repr(internal);
            assert_eq!(v, external);
        }
    }

    // ========================================================================
    // Python 交叉验证测试
    // ========================================================================

    /// Python 交叉验证: IntDistribution(0, 10, step=3)
    /// Python: high=9, single=false
    #[test]
    fn test_python_cross_int_step3() {
        let d = IntDistribution::new(0, 10, false, 3).unwrap();
        assert_eq!(d.high, 9,  "Python: high=9");
        assert!(!d.single(),   "Python: single=false");
        assert!(d.contains(0.0),  "Python: contains(0)=true");
        assert!(d.contains(3.0),  "Python: contains(3)=true");
        assert!(d.contains(6.0),  "Python: contains(6)=true");
        assert!(d.contains(9.0),  "Python: contains(9)=true");
        assert!(!d.contains(10.0), "Python: contains(10)=false");
        assert!(!d.contains(1.0),  "Python: contains(1)=false");
    }

    /// Python 交叉验证: IntDistribution(0, 10, step=2)
    /// Python: high=10
    #[test]
    fn test_python_cross_int_step2() {
        let d = IntDistribution::new(0, 10, false, 2).unwrap();
        assert_eq!(d.high, 10, "Python: high=10 (无调整)");
    }

    /// Python 交叉验证: IntDistribution(0, 2, step=5)
    /// Python: high=0, single=true
    #[test]
    fn test_python_cross_int_small_range_step() {
        let d = IntDistribution::new(0, 2, false, 5).unwrap();
        assert_eq!(d.high, 0, "Python: high=0");
        assert!(d.single(),  "Python: single=true");
    }

    /// Python 交叉验证: IntDistribution(1, 100, log=True)
    /// Python: single=false
    #[test]
    fn test_python_cross_int_log() {
        let d = IntDistribution::new(1, 100, true, 1).unwrap();
        assert!(!d.single(), "Python: single=false");
    }

    /// log 分布的 to_internal_repr 拒绝非正值
    #[test]
    fn test_log_to_internal_repr_rejects_non_positive() {
        let d = IntDistribution::new(1, 100, true, 1).unwrap();
        assert!(d.to_internal_repr(0).is_err());
        assert!(d.to_internal_repr(-1).is_err());
        assert!(d.to_internal_repr(1).is_ok());
    }
}
