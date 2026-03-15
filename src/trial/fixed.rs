use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    ParamValue,
};
use crate::error::{OptunaError, Result};

/// A trial with pre-fixed parameter values, used for deployment and testing.
///
/// Corresponds to Python `optuna.trial.FixedTrial`.
///
/// `suggest_*` methods look up values from the fixed params dict rather than
/// sampling from distributions.
pub struct FixedTrial {
    params: HashMap<String, ParamValue>,
    suggested_params: HashMap<String, ParamValue>,
    distributions: HashMap<String, Distribution>,
    user_attrs: HashMap<String, serde_json::Value>,
    datetime_start: DateTime<Utc>,
    number: i64,
}

impl FixedTrial {
    /// Create a new `FixedTrial` with pre-set parameter values.
    pub fn new(params: HashMap<String, ParamValue>, number: i64) -> Self {
        Self {
            params,
            suggested_params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            datetime_start: Utc::now(),
            number,
        }
    }

    /// Suggest a floating-point parameter.
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Result<f64> {
        let dist = Distribution::FloatDistribution(FloatDistribution::new(low, high, log, step)?);
        let value = self.suggest(name, &dist)?;
        match value {
            ParamValue::Float(v) => Ok(v),
            ParamValue::Int(v) => Ok(v as f64),
            _ => Err(OptunaError::ValueError(format!(
                "param '{name}' is not a float"
            ))),
        }
    }

    /// Suggest an integer parameter.
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        log: bool,
        step: i64,
    ) -> Result<i64> {
        let dist = Distribution::IntDistribution(IntDistribution::new(low, high, log, step)?);
        let value = self.suggest(name, &dist)?;
        match value {
            ParamValue::Int(v) => Ok(v),
            ParamValue::Float(v) => Ok(v as i64),
            _ => Err(OptunaError::ValueError(format!(
                "param '{name}' is not an int"
            ))),
        }
    }

    /// Suggest a categorical parameter.
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<CategoricalChoice>,
    ) -> Result<CategoricalChoice> {
        let dist = Distribution::CategoricalDistribution(CategoricalDistribution::new(choices)?);
        let value = self.suggest(name, &dist)?;
        match value {
            ParamValue::Categorical(c) => Ok(c),
            _ => Err(OptunaError::ValueError(format!(
                "param '{name}' is not categorical"
            ))),
        }
    }

    /// Core suggest logic: look up the fixed value and validate it.
    ///
    /// 对齐 Python FixedTrial._suggest():
    /// - 值越界时发出 warning 但仍然返回该值（不报错）
    /// - 重复 suggest 同名参数时检查分布兼容性
    fn suggest(&mut self, name: &str, dist: &Distribution) -> Result<ParamValue> {
        let value = self
            .params
            .get(name)
            .ok_or_else(|| {
                OptunaError::ValueError(format!(
                    "fixed param '{name}' not found in the provided params"
                ))
            })?
            .clone();

        // 对齐 Python: 值越界时仅 warn 不报错
        let internal = dist.to_internal_repr(&value)?;
        if !dist.contains(internal) {
            crate::optuna_warn!(
                "Fixed param '{}' value is out of the distribution range.",
                name
            );
        }

        // 对齐 Python: 重复 suggest 时检查分布兼容性
        if let Some(existing_dist) = self.distributions.get(name) {
            crate::distributions::check_distribution_compatibility(existing_dist, dist)?;
        }

        self.suggested_params.insert(name.to_string(), value.clone());
        self.distributions.insert(name.to_string(), dist.clone());
        Ok(value)
    }

    /// Report an intermediate value (no-op for FixedTrial).
    pub fn report(&self, _value: f64, _step: i64) {}

    /// Check if the trial should be pruned (always false for FixedTrial).
    pub fn should_prune(&self) -> bool {
        false
    }

    /// Set a user attribute.
    pub fn set_user_attr(&mut self, key: String, value: serde_json::Value) {
        self.user_attrs.insert(key, value);
    }

    /// Get the trial number.
    pub fn number(&self) -> i64 {
        self.number
    }

    /// Get the suggested params.
    pub fn suggested_params(&self) -> &HashMap<String, ParamValue> {
        &self.suggested_params
    }

    /// Get the distributions.
    pub fn distributions(&self) -> &HashMap<String, Distribution> {
        &self.distributions
    }

    /// Get the user attributes.
    pub fn user_attrs(&self) -> &HashMap<String, serde_json::Value> {
        &self.user_attrs
    }

    /// Get the start time.
    pub fn datetime_start(&self) -> DateTime<Utc> {
        self.datetime_start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_float() {
        let mut params = HashMap::new();
        params.insert("lr".into(), ParamValue::Float(0.01));
        let mut trial = FixedTrial::new(params, 0);

        let lr = trial.suggest_float("lr", 0.001, 0.1, false, None).unwrap();
        assert_eq!(lr, 0.01);
        assert!(trial.suggested_params().contains_key("lr"));
    }

    #[test]
    fn test_suggest_int() {
        let mut params = HashMap::new();
        params.insert("n".into(), ParamValue::Int(5));
        let mut trial = FixedTrial::new(params, 0);

        let n = trial.suggest_int("n", 1, 10, false, 1).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_suggest_categorical() {
        let mut params = HashMap::new();
        params.insert(
            "opt".into(),
            ParamValue::Categorical(CategoricalChoice::Str("sgd".into())),
        );
        let mut trial = FixedTrial::new(params, 0);

        let opt = trial
            .suggest_categorical(
                "opt",
                vec![
                    CategoricalChoice::Str("sgd".into()),
                    CategoricalChoice::Str("adam".into()),
                ],
            )
            .unwrap();
        assert_eq!(opt, CategoricalChoice::Str("sgd".into()));
    }

    #[test]
    fn test_missing_param() {
        let params = HashMap::new();
        let mut trial = FixedTrial::new(params, 0);
        assert!(trial.suggest_float("x", 0.0, 1.0, false, None).is_err());
    }

    #[test]
    fn test_out_of_range() {
        // 对齐 Python: 值越界时不报错，只 warn，仍然返回该值
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(2.0));
        let mut trial = FixedTrial::new(params, 0);
        let result = trial.suggest_float("x", 0.0, 1.0, false, None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2.0); // 返回越界的值
    }

    #[test]
    fn test_should_prune_always_false() {
        let trial = FixedTrial::new(HashMap::new(), 0);
        assert!(!trial.should_prune());
    }

    #[test]
    fn test_distribution_compatibility_check() {
        // 重复 suggest 同名参数但分布兼容（同类型不同范围 ok）
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        let mut trial = FixedTrial::new(params, 0);
        assert!(trial.suggest_float("x", 0.0, 1.0, false, None).is_ok());
        // 再次 suggest 同名参数，不同 range 但同 type/log/step → 兼容
        assert!(trial.suggest_float("x", 0.0, 2.0, false, None).is_ok());
        // 不兼容：不同 log 设置
        let result = trial.suggest_float("x", 0.0, 1.0, true, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_suggest_int_out_of_range_warns() {
        let mut params = HashMap::new();
        params.insert("n".into(), ParamValue::Int(20));
        let mut trial = FixedTrial::new(params, 0);
        // 越界只 warn 不报错
        let result = trial.suggest_int("n", 1, 10, false, 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 20);
    }

    #[test]
    fn test_report_noop() {
        let trial = FixedTrial::new(HashMap::new(), 0);
        trial.report(1.0, 0); // 不应 panic
    }

    #[test]
    fn test_set_user_attr() {
        let mut trial = FixedTrial::new(HashMap::new(), 0);
        trial.set_user_attr("key".into(), serde_json::json!(42));
        assert_eq!(trial.user_attrs()["key"], serde_json::json!(42));
    }

    #[test]
    fn test_number() {
        let trial = FixedTrial::new(HashMap::new(), 7);
        assert_eq!(trial.number(), 7);
    }
}
