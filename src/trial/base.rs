use std::collections::HashMap;

use crate::distributions::{CategoricalChoice, Distribution, ParamValue};
use crate::error::Result;

/// 统一的 Trial 接口。
///
/// 对应 Python `optuna.trial.BaseTrial` 抽象类。
/// Trial、FrozenTrial、FixedTrial 均实现此 trait，
/// 使得接受任意试验类型的泛型代码成为可能。
pub trait BaseTrial {
    /// 建议一个浮点参数。
    fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> Result<f64>;

    /// 建议一个整数参数。
    fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    ) -> Result<i64>;

    /// 建议一个分类参数。
    fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<CategoricalChoice>,
    ) -> Result<CategoricalChoice>;

    /// 报告中间值（FrozenTrial/FixedTrial 为 no-op）。
    fn report(&mut self, value: f64, step: i64) -> Result<()>;

    /// 判断是否应该剪枝（FrozenTrial/FixedTrial 始终返回 false）。
    fn should_prune(&self) -> Result<bool>;

    /// 设置用户属性。
    fn set_user_attr(&mut self, key: &str, value: serde_json::Value) -> Result<()>;

    /// 试验编号。
    fn number(&self) -> i64;

    /// 当前参数。
    fn params(&self) -> HashMap<String, ParamValue>;

    /// 当前分布。
    fn distributions(&self) -> HashMap<String, Distribution>;

    /// 用户属性。
    fn user_attrs(&self) -> Result<HashMap<String, serde_json::Value>>;

    /// 系统属性。
    /// 对应 Python `BaseTrial.system_attrs` (deprecated in Python 3.1.0)。
    fn system_attrs(&self) -> Result<HashMap<String, serde_json::Value>>;

    /// 设置系统属性。
    /// 对应 Python `BaseTrial.set_system_attr()` (deprecated in Python 3.1.0)。
    fn set_system_attr(&mut self, key: &str, value: serde_json::Value) -> Result<()>;

    /// 试验开始时间。
    fn datetime_start(&self) -> Option<chrono::DateTime<chrono::Utc>>;
}
