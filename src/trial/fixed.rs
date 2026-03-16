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
    /// 固定参数值（构造时传入）。
    fixed_params: HashMap<String, ParamValue>,
    /// 已 suggest 的参数值（对齐 Python `_suggested_params`）。
    suggested_params: HashMap<String, ParamValue>,
    /// 已 suggest 的参数分布（对齐 Python `_distributions`）。
    distributions: HashMap<String, Distribution>,
    /// 用户属性。
    user_attrs: HashMap<String, serde_json::Value>,
    /// 系统属性（对齐 Python `_system_attrs`）。
    system_attrs: HashMap<String, serde_json::Value>,
    /// 试验开始时间。
    datetime_start: DateTime<Utc>,
    /// 试验编号。
    number: i64,
}

impl FixedTrial {
    /// Create a new `FixedTrial` with pre-set parameter values.
    ///
    /// 对齐 Python `FixedTrial.__init__(params, number=0)`。
    pub fn new(params: HashMap<String, ParamValue>, number: i64) -> Self {
        Self {
            fixed_params: params,
            suggested_params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
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

    /// 对齐 Python `FixedTrial.suggest_float(name, low, high, step=None, log=False)`。
    pub fn suggest_float_py(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> Result<f64> {
        self.suggest_float(name, low, high, log, step)
    }

    /// 对齐 Python 默认参数形式：`suggest_float(name, low, high)`。
    pub fn suggest_float_default(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_py(name, low, high, None, false)
    }

    /// 对齐 Python `suggest_float(..., step=...)` 的便捷入口。
    pub fn suggest_float_step(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: f64,
    ) -> Result<f64> {
        self.suggest_float_py(name, low, high, Some(step), false)
    }

    /// 对齐 Python `suggest_float(..., log=True)` 的便捷入口。
    pub fn suggest_float_log(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_py(name, low, high, None, true)
    }

    /// 对齐 Python 已弃用别名 `suggest_uniform()`。
    pub fn suggest_uniform(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_default(name, low, high)
    }

    /// 对齐 Python 已弃用别名 `suggest_loguniform()`。
    pub fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_log(name, low, high)
    }

    /// 对齐 Python 已弃用别名 `suggest_discrete_uniform()`。
    pub fn suggest_discrete_uniform(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        q: f64,
    ) -> Result<f64> {
        self.suggest_float_step(name, low, high, q)
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

    /// 对齐 Python `FixedTrial.suggest_int(name, low, high, step=1, log=False)`。
    pub fn suggest_int_py(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    ) -> Result<i64> {
        self.suggest_int(name, low, high, log, step)
    }

    /// 对齐 Python 默认参数形式：`suggest_int(name, low, high)`。
    pub fn suggest_int_default(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int_py(name, low, high, 1, false)
    }

    /// 对齐 Python `suggest_int(..., step=...)` 的便捷入口。
    pub fn suggest_int_step(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
    ) -> Result<i64> {
        self.suggest_int_py(name, low, high, step, false)
    }

    /// 对齐 Python `suggest_int(..., log=True)` 的便捷入口。
    pub fn suggest_int_log(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int_py(name, low, high, 1, true)
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
            .fixed_params
            .get(name)
            .ok_or_else(|| {
                OptunaError::ValueError(format!(
                    "The value of the parameter '{name}' is not found. Please set it at the construction of the FixedTrial object."
                ))
            })?
            .clone();

        // 对齐 Python: 值越界时仅 warn 不报错
        let internal = dist.to_internal_repr(&value)?;
        if !dist.contains(internal) {
            crate::optuna_warn!(
                "The value {:?} of the parameter '{}' is out of the range of the distribution {:?}.",
                value,
                name,
                dist
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
    /// 对齐 Python `FixedTrial.set_user_attr(key, value)`。
    pub fn set_user_attr(&mut self, key: String, value: serde_json::Value) {
        self.user_attrs.insert(key, value);
    }

    /// Set a system attribute.
    /// 对齐 Python `FixedTrial.set_system_attr(key, value)`（deprecated in Python 3.1.0）。
    #[deprecated(note = "Use user_attrs instead. Will be removed in a future version.")]
    pub fn set_system_attr(&mut self, key: String, value: serde_json::Value) {
        self.system_attrs.insert(key, value);
    }

    /// Get the trial number.
    pub fn number(&self) -> i64 {
        self.number
    }

    /// Get the suggested params.
    /// 对齐 Python `FixedTrial.params` 属性（Python 返回 `_suggested_params`）。
    pub fn params(&self) -> &HashMap<String, ParamValue> {
        &self.suggested_params
    }

    /// 别名：与 `params()` 相同（向后兼容）。
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

    /// Get the system attributes.
    /// 对齐 Python `FixedTrial.system_attrs` 属性。
    pub fn system_attrs(&self) -> &HashMap<String, serde_json::Value> {
        &self.system_attrs
    }

    /// Get the start time.
    /// 对齐 Python `FixedTrial.datetime_start` 属性。
    pub fn datetime_start(&self) -> Option<DateTime<Utc>> {
        Some(self.datetime_start)
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

    #[test]
    fn test_python_compat_float_wrappers() {
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.25));
        params.insert("y".into(), ParamValue::Float(2.0));
        params.insert("z".into(), ParamValue::Float(0.75));
        params.insert("w".into(), ParamValue::Float(1.5));
        let mut trial = FixedTrial::new(params, 0);

        assert_eq!(trial.suggest_float_default("x", 0.0, 1.0).unwrap(), 0.25);
        assert_eq!(trial.suggest_loguniform("y", 1.0, 10.0).unwrap(), 2.0);
        assert_eq!(trial.suggest_discrete_uniform("z", 0.0, 1.0, 0.25).unwrap(), 0.75);
        assert_eq!(trial.suggest_float_py("w", 1.0, 2.0, Some(0.5), false).unwrap(), 1.5);

        match trial.distributions().get("x").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("x should use float distribution"),
        }
        match trial.distributions().get("y").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("y should use float distribution"),
        }
        match trial.distributions().get("z").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.25));
            }
            _ => panic!("z should use float distribution"),
        }
        match trial.distributions().get("w").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.5));
            }
            _ => panic!("w should use float distribution"),
        }
    }

    #[test]
    fn test_python_compat_int_wrappers() {
        let mut params = HashMap::new();
        params.insert("a".into(), ParamValue::Int(3));
        params.insert("b".into(), ParamValue::Int(4));
        params.insert("c".into(), ParamValue::Int(6));
        params.insert("d".into(), ParamValue::Int(9));
        let mut trial = FixedTrial::new(params, 0);

        assert_eq!(trial.suggest_int_default("a", 1, 5).unwrap(), 3);
        assert_eq!(trial.suggest_int_log("b", 1, 8).unwrap(), 4);
        assert_eq!(trial.suggest_int_step("c", 0, 10, 2).unwrap(), 6);
        assert_eq!(trial.suggest_int_py("d", 3, 9, 3, false).unwrap(), 9);

        match trial.distributions().get("a").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("a should use int distribution"),
        }
        match trial.distributions().get("b").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("b should use int distribution"),
        }
        match trial.distributions().get("c").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 2);
            }
            _ => panic!("c should use int distribution"),
        }
        match trial.distributions().get("d").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 3);
            }
            _ => panic!("d should use int distribution"),
        }
    }

    // ========== 对齐 Python: system_attrs 测试 ==========

    /// 测试 FixedTrial 的 system_attrs 初始化为空。
    /// 对应 Python: `assert trial.system_attrs == {}`
    #[test]
    fn test_fixed_trial_system_attrs_initially_empty() {
        let params = HashMap::new();
        let trial = FixedTrial::new(params, 0);
        assert!(trial.system_attrs().is_empty());
    }

    /// 测试 set_system_attr (deprecated) 和 system_attrs getter。
    /// 对应 Python:
    /// ```python
    /// trial.set_system_attr("key", "value")
    /// assert trial.system_attrs["key"] == "value"
    /// ```
    #[test]
    #[allow(deprecated)]
    fn test_fixed_trial_set_and_get_system_attr() {
        let params = HashMap::new();
        let mut trial = FixedTrial::new(params, 0);
        trial.set_system_attr("runner".to_string(), serde_json::json!("optuna-rs"));
        assert_eq!(
            trial.system_attrs().get("runner").unwrap(),
            &serde_json::json!("optuna-rs")
        );
    }

    /// 测试 params() 别名返回与 suggested_params() 相同的引用。
    /// 对应 Python: `trial.params` 属性。
    #[test]
    fn test_fixed_trial_params_alias() {
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(1.5));
        let mut trial = FixedTrial::new(params, 0);

        // 先 suggest 一个参数，使其记录到 suggested_params
        trial.suggest_float("x", 0.0, 10.0, false, None).unwrap();

        // params() 和 suggested_params() 应该返回相同的内容
        assert_eq!(trial.params(), trial.suggested_params());
        match trial.params().get("x").unwrap() {
            ParamValue::Float(v) => assert_eq!(*v, 1.5),
            _ => panic!("expected Float param"),
        }
    }

    /// 测试 datetime_start 返回 Some。
    /// FixedTrial 在构造时设置 datetime_start 为 Utc::now()。
    #[test]
    fn test_fixed_trial_datetime_start_is_some() {
        let params = HashMap::new();
        let trial = FixedTrial::new(params, 0);
        assert!(trial.datetime_start().is_some());
    }

    /// 测试 suggest 不存在的参数时报错。
    /// 对应 Python: `FixedTrial({"x": 1.0}).suggest_float("y", 0, 1)` → ValueError
    #[test]
    fn test_fixed_trial_missing_param_error() {
        let params = HashMap::new();
        let mut trial = FixedTrial::new(params, 0);
        let result = trial.suggest_float("missing", 0.0, 1.0, false, None);
        assert!(result.is_err());
    }
}
