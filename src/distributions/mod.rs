mod categorical;
mod float;
mod int;

pub use categorical::{CategoricalChoice, CategoricalDistribution};
pub use float::FloatDistribution;
pub use int::IntDistribution;

use serde::{Deserialize, Serialize};

use crate::error::{OptunaError, Result};

/// A parameter value as stored in `FrozenTrial::params`.
///
/// This represents the external (user-facing) value of a parameter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Categorical(CategoricalChoice),
}

/// A unified distribution type covering all optuna distributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", content = "attributes")]
pub enum Distribution {
    FloatDistribution(FloatDistribution),
    IntDistribution(IntDistribution),
    CategoricalDistribution(CategoricalDistribution),
}

impl Distribution {
    /// Check whether `value` (in internal representation) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        match self {
            Self::FloatDistribution(d) => d.contains(value),
            Self::IntDistribution(d) => d.contains(value),
            Self::CategoricalDistribution(d) => d.contains(value),
        }
    }

    /// True if this distribution contains exactly one value.
    pub fn single(&self) -> bool {
        match self {
            Self::FloatDistribution(d) => d.single(),
            Self::IntDistribution(d) => d.single(),
            Self::CategoricalDistribution(d) => d.single(),
        }
    }

    /// 对齐 Python `_get_single_value()`: 返回唯一值（调用前需确保 `single()` 为 true）。
    pub fn get_single_value(&self) -> Result<ParamValue> {
        match self {
            Self::FloatDistribution(d) => Ok(ParamValue::Float(d.low)),
            Self::IntDistribution(d) => Ok(ParamValue::Int(d.low)),
            Self::CategoricalDistribution(d) => Ok(ParamValue::Categorical(d.choices[0].clone())),
        }
    }

    /// 对齐 Python `_is_distribution_log()`: 返回分布是否为对数尺度。
    pub fn is_log(&self) -> bool {
        match self {
            Self::FloatDistribution(d) => d.log,
            Self::IntDistribution(d) => d.log,
            Self::CategoricalDistribution(_) => false,
        }
    }

    /// Convert an external `ParamValue` to an internal `f64` representation.
    ///
    /// 对齐 Python: CategoricalDistribution 接受任意类型的值并在 choices 中查找索引。
    /// 当 ParamValue 通过 JSON 反序列化时，整数/浮点分类选项可能被解析为
    /// `ParamValue::Int(1)` 而非 `ParamValue::Categorical(CategoricalChoice::Int(1))`。
    pub fn to_internal_repr(&self, value: &ParamValue) -> Result<f64> {
        match (self, value) {
            (Self::FloatDistribution(d), ParamValue::Float(v)) => d.to_internal_repr(*v),
            (Self::FloatDistribution(d), ParamValue::Int(v)) => d.to_internal_repr(*v as f64),
            (Self::IntDistribution(d), ParamValue::Int(v)) => d.to_internal_repr(*v),
            (Self::IntDistribution(d), ParamValue::Float(v)) => d.to_internal_repr(*v as i64),
            (Self::CategoricalDistribution(d), ParamValue::Categorical(v)) => {
                d.to_internal_repr(v)
            }
            // 对齐 Python: 当分类选项为 Int/Float/Bool 时，ParamValue 可能不是 Categorical 变体
            (Self::CategoricalDistribution(d), ParamValue::Int(v)) => {
                d.to_internal_repr(&CategoricalChoice::Int(*v))
            }
            (Self::CategoricalDistribution(d), ParamValue::Float(v)) => {
                d.to_internal_repr(&CategoricalChoice::Float(*v))
            }
            _ => Err(OptunaError::ValueError(
                "parameter value type mismatch for distribution".to_string(),
            )),
        }
    }

    /// Convert an internal `f64` representation back to an external `ParamValue`.
    pub fn to_external_repr(&self, value: f64) -> Result<ParamValue> {
        match self {
            Self::FloatDistribution(d) => Ok(ParamValue::Float(d.to_external_repr(value))),
            Self::IntDistribution(d) => Ok(ParamValue::Int(d.to_external_repr(value))),
            Self::CategoricalDistribution(d) => {
                Ok(ParamValue::Categorical(d.to_external_repr(value)?))
            }
        }
    }
}

/// Serialize a distribution to JSON in the Python-compatible format.
///
/// Produces: `{"name": "<DistributionType>", "attributes": { ... }}`
pub fn distribution_to_json(dist: &Distribution) -> Result<String> {
    serde_json::to_string(dist)
        .map_err(|e| OptunaError::StorageInternalError(format!("JSON serialization failed: {e}")))
}

/// Deserialize a distribution from JSON.
///
/// Accepts the current format: `{"name": "<Type>", "attributes": { ... }}`
pub fn json_to_distribution(json: &str) -> Result<Distribution> {
    serde_json::from_str(json)
        .map_err(|e| OptunaError::StorageInternalError(format!("JSON deserialization failed: {e}")))
}

/// 检查两个分布是否兼容。
///
/// 对应 Python `optuna.distributions.check_distribution_compatibility()`。
/// 兼容规则（严格对齐 Python）：
/// - 同一类型的分布（Float/Int/Categorical）兼容（允许 range/step 不同）
/// - Float/Int: 仅检查 log 属性必须相同
/// - Categorical: choices 必须完全相同
/// - 不同类型的分布不兼容
/// 
/// 注意：Python 不检查 step 是否一致，仅检查 log。
pub fn check_distribution_compatibility(
    dist_a: &Distribution,
    dist_b: &Distribution,
) -> Result<()> {
    match (dist_a, dist_b) {
        // 对齐 Python: Float 仅检查 log，不检查 step
        (Distribution::FloatDistribution(a), Distribution::FloatDistribution(b)) => {
            if a.log != b.log {
                return Err(OptunaError::ValueError(
                    "Cannot set different log configuration to the same parameter name.".to_string(),
                ));
            }
            Ok(())
        }
        // 对齐 Python: Int 仅检查 log，不检查 step
        (Distribution::IntDistribution(a), Distribution::IntDistribution(b)) => {
            if a.log != b.log {
                return Err(OptunaError::ValueError(
                    "Cannot set different log configuration to the same parameter name.".to_string(),
                ));
            }
            Ok(())
        }
        (Distribution::CategoricalDistribution(a), Distribution::CategoricalDistribution(b)) => {
            if a.choices != b.choices {
                return Err(OptunaError::ValueError(
                    "CategoricalDistribution does not support dynamic value space.".to_string(),
                ));
            }
            Ok(())
        }
        _ => {
            Err(OptunaError::ValueError(
                "Cannot set different distribution kind to the same parameter name.".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_json_roundtrip() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let json = distribution_to_json(&dist).unwrap();
        let parsed = json_to_distribution(&json).unwrap();
        assert_eq!(dist, parsed);
    }

    #[test]
    fn test_int_json_roundtrip() {
        let dist =
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap());
        let json = distribution_to_json(&dist).unwrap();
        let parsed = json_to_distribution(&json).unwrap();
        assert_eq!(dist, parsed);
    }

    #[test]
    fn test_categorical_json_roundtrip() {
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
            ])
            .unwrap(),
        );
        let json = distribution_to_json(&dist).unwrap();
        let parsed = json_to_distribution(&json).unwrap();
        assert_eq!(dist, parsed);
    }

    #[test]
    fn test_to_internal_external_roundtrip_float() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let val = ParamValue::Float(0.5);
        let internal = dist.to_internal_repr(&val).unwrap();
        assert_eq!(internal, 0.5);
        let external = dist.to_external_repr(internal).unwrap();
        assert_eq!(external, ParamValue::Float(0.5));
    }

    #[test]
    fn test_to_internal_external_roundtrip_int() {
        let dist =
            Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
        let val = ParamValue::Int(5);
        let internal = dist.to_internal_repr(&val).unwrap();
        assert_eq!(internal, 5.0);
        let external = dist.to_external_repr(internal).unwrap();
        assert_eq!(external, ParamValue::Int(5));
    }

    #[test]
    fn test_to_internal_external_roundtrip_categorical() {
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("x".into()),
                CategoricalChoice::Str("y".into()),
            ])
            .unwrap(),
        );
        let val = ParamValue::Categorical(CategoricalChoice::Str("y".into()));
        let internal = dist.to_internal_repr(&val).unwrap();
        assert_eq!(internal, 1.0);
        let external = dist.to_external_repr(internal).unwrap();
        assert_eq!(external, ParamValue::Categorical(CategoricalChoice::Str("y".into())));
    }

    #[test]
    fn test_type_mismatch_error() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let val = ParamValue::Categorical(CategoricalChoice::Str("oops".into()));
        assert!(dist.to_internal_repr(&val).is_err());
    }

    // === check_distribution_compatibility 对齐 Python 测试 ===

    #[test]
    fn test_compat_float_same_log() {
        // 对齐 Python: 同类型同 log 属性即兼容，range 不同也 OK
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_float_different_log() {
        // log 不同则不兼容
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, true, None).unwrap());
        let b = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, false, None).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_err());
    }

    #[test]
    fn test_compat_float_different_step_ok() {
        // 对齐 Python: Float 的 step 不同也兼容（Python 不检查 step）
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
        let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.2)).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_float_step_vs_none_ok() {
        // 对齐 Python: step=Some vs step=None 也兼容（Python 不检查 step）
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
        let b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_int_same_log() {
        let a = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
        let b = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 1).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_int_different_step_ok() {
        // 对齐 Python: Int 的 step 不同也兼容（Python 不检查 step）
        let a = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
        let b = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 2).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_int_different_log() {
        let a = Distribution::IntDistribution(IntDistribution::new(1, 10, true, 1).unwrap());
        let b = Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_err());
    }

    #[test]
    fn test_compat_categorical_same() {
        let a = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
        );
        let b = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
        );
        assert!(check_distribution_compatibility(&a, &b).is_ok());
    }

    #[test]
    fn test_compat_categorical_different() {
        let a = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
        );
        let b = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("b".into())]).unwrap(),
        );
        assert!(check_distribution_compatibility(&a, &b).is_err());
    }

    #[test]
    fn test_compat_different_types() {
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let b = Distribution::IntDistribution(IntDistribution::new(0, 1, false, 1).unwrap());
        assert!(check_distribution_compatibility(&a, &b).is_err());
    }

    // ── 新增测试：对齐 Python 审计修复 ──

    /// 对齐 Python: get_single_value() 返回唯一值
    #[test]
    fn test_get_single_value_float() {
        let d = Distribution::FloatDistribution(FloatDistribution::new(1.5, 1.5, false, None).unwrap());
        assert!(d.single());
        assert_eq!(d.get_single_value().unwrap(), ParamValue::Float(1.5));
    }

    #[test]
    fn test_get_single_value_int() {
        let d = Distribution::IntDistribution(IntDistribution::new(3, 3, false, 1).unwrap());
        assert!(d.single());
        assert_eq!(d.get_single_value().unwrap(), ParamValue::Int(3));
    }

    #[test]
    fn test_get_single_value_categorical() {
        let d = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("only".into())]).unwrap(),
        );
        assert!(d.single());
        assert_eq!(
            d.get_single_value().unwrap(),
            ParamValue::Categorical(CategoricalChoice::Str("only".into()))
        );
    }

    /// 对齐 Python: is_log() 方法
    #[test]
    fn test_is_log() {
        let d1 = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, true, None).unwrap());
        assert!(d1.is_log());
        let d2 = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        assert!(!d2.is_log());
        let d3 = Distribution::IntDistribution(IntDistribution::new(1, 10, true, 1).unwrap());
        assert!(d3.is_log());
        let d4 = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Int(1)]).unwrap(),
        );
        assert!(!d4.is_log());
    }

    /// 对齐 Python: check_distribution_compatibility 错误消息完全匹配
    #[test]
    fn test_compat_error_messages_match_python() {
        // 不同类型
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let b = Distribution::IntDistribution(IntDistribution::new(0, 1, false, 1).unwrap());
        let err = check_distribution_compatibility(&a, &b).unwrap_err();
        assert!(err.to_string().contains("Cannot set different distribution kind to the same parameter name."));

        // 不同 log
        let a = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, true, None).unwrap());
        let b = Distribution::FloatDistribution(FloatDistribution::new(0.01, 1.0, false, None).unwrap());
        let err = check_distribution_compatibility(&a, &b).unwrap_err();
        assert!(err.to_string().contains("Cannot set different log configuration to the same parameter name."));

        // 不同 choices
        let a = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("a".into())]).unwrap(),
        );
        let b = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![CategoricalChoice::Str("b".into())]).unwrap(),
        );
        let err = check_distribution_compatibility(&a, &b).unwrap_err();
        assert!(err.to_string().contains("CategoricalDistribution does not support dynamic value space."));
    }
}
