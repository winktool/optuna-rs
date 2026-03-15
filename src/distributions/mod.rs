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

    /// Convert an external `ParamValue` to an internal `f64` representation.
    pub fn to_internal_repr(&self, value: &ParamValue) -> Result<f64> {
        match (self, value) {
            (Self::FloatDistribution(d), ParamValue::Float(v)) => d.to_internal_repr(*v),
            (Self::FloatDistribution(d), ParamValue::Int(v)) => d.to_internal_repr(*v as f64),
            (Self::IntDistribution(d), ParamValue::Int(v)) => d.to_internal_repr(*v),
            (Self::IntDistribution(d), ParamValue::Float(v)) => d.to_internal_repr(*v as i64),
            (Self::CategoricalDistribution(d), ParamValue::Categorical(v)) => {
                d.to_internal_repr(v)
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
/// 兼容规则：
/// - 同一类型的分布（Float/Int/Categorical）兼容（允许 range 不同）
/// - Float: log 属性必须相同，step 的 Some/None 状态必须相同
/// - Int: log 属性必须相同，step 值必须相同
/// - Categorical: choices 必须完全相同
/// - 不同类型的分布不兼容
pub fn check_distribution_compatibility(
    dist_a: &Distribution,
    dist_b: &Distribution,
) -> Result<()> {
    match (dist_a, dist_b) {
        (Distribution::FloatDistribution(a), Distribution::FloatDistribution(b)) => {
            if a.log != b.log {
                return Err(OptunaError::ValueError(format!(
                    "Cannot set different `log` values for the same parameter: {} vs {}",
                    a.log, b.log
                )));
            }
            if a.step.is_some() != b.step.is_some() {
                return Err(OptunaError::ValueError(
                    "Cannot use both discrete and continuous distributions for the same parameter".into()
                ));
            }
            Ok(())
        }
        (Distribution::IntDistribution(a), Distribution::IntDistribution(b)) => {
            if a.log != b.log {
                return Err(OptunaError::ValueError(format!(
                    "Cannot set different `log` values for the same parameter: {} vs {}",
                    a.log, b.log
                )));
            }
            if a.step != b.step {
                return Err(OptunaError::ValueError(format!(
                    "Cannot set different `step` values for the same parameter: {} vs {}",
                    a.step, b.step
                )));
            }
            Ok(())
        }
        (Distribution::CategoricalDistribution(a), Distribution::CategoricalDistribution(b)) => {
            if a.choices != b.choices {
                return Err(OptunaError::ValueError(format!(
                    "Cannot set different `choices` for the same parameter: {:?} vs {:?}",
                    a.choices, b.choices
                )));
            }
            Ok(())
        }
        _ => {
            Err(OptunaError::ValueError(format!(
                "Cannot use different distribution types for the same parameter: {:?} vs {:?}",
                std::mem::discriminant(dist_a),
                std::mem::discriminant(dist_b),
            )))
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
}
