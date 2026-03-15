mod fixed;
mod frozen;
mod state;
mod handle;

pub use fixed::FixedTrial;
pub use frozen::FrozenTrial;
pub use handle::Trial;
pub use state::TrialState;

use std::collections::HashMap;

/// 创建一个 FrozenTrial 实例。
///
/// 对应 Python `optuna.trial.create_trial()`。
/// 用于手动构建试验对象，方便批量导入历史数据。
///
/// # 参数
/// * `state` - 试验状态（默认 Complete）
/// * `value` - 单目标值（与 values 互斥）
/// * `values` - 多目标值（与 value 互斥）
/// * `params` - 参数字典
/// * `distributions` - 参数分布字典
/// * `user_attrs` - 用户属性
/// * `system_attrs` - 系统属性
/// * `intermediate_values` - 中间值
///
/// # 行为对齐 Python
/// - 同时传入 value 和 values 时报错（对齐 Python ValueError）
/// - 内部调用 `FrozenTrial::validate()` 进行状态一致性校验
/// - `number` 和 `trial_id` 设为 -1（表示"尚未分配"），add_trial() 时由 storage 分配。
/// - 非 WAITING 状态自动设 `datetime_start = now()`。
/// - 已完成状态（Complete/Pruned/Fail）自动设 `datetime_complete = now()`。
pub fn create_trial(
    state: Option<TrialState>,
    value: Option<f64>,
    values: Option<Vec<f64>>,
    params: Option<HashMap<String, crate::distributions::ParamValue>>,
    distributions: Option<HashMap<String, crate::distributions::Distribution>>,
    user_attrs: Option<HashMap<String, serde_json::Value>>,
    system_attrs: Option<HashMap<String, serde_json::Value>>,
    intermediate_values: Option<HashMap<i64, f64>>,
) -> crate::error::Result<FrozenTrial> {
    let s = state.unwrap_or(TrialState::Complete);
    let now = chrono::Utc::now();

    // 对齐 Python: 非 WAITING 状态自动设 datetime_start
    let datetime_start = if s != TrialState::Waiting {
        Some(now)
    } else {
        None
    };

    // 对齐 Python: 已完成的状态自动设 datetime_complete
    let datetime_complete = if s.is_finished() {
        Some(now)
    } else {
        None
    };

    // 通过 FrozenTrial::new 构建，内部会：
    // 1. 检查 value / values 互斥（同时传入则报 ValueError）
    // 2. 调用 validate() 校验状态一致性
    FrozenTrial::new(
        -1,          // number: 尚未分配
        s,
        value,
        values,
        datetime_start,
        datetime_complete,
        params.unwrap_or_default(),
        distributions.unwrap_or_default(),
        user_attrs.unwrap_or_default(),
        system_attrs.unwrap_or_default(),
        intermediate_values.unwrap_or_default(),
        -1,          // trial_id: 尚未分配
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_trial_defaults() {
        // 默认状态 Complete，number/trial_id = -1
        let t = create_trial(None, Some(1.0), None, None, None, None, None, None).unwrap();
        assert_eq!(t.state, TrialState::Complete);
        assert_eq!(t.number, -1);
        assert_eq!(t.trial_id, -1);
        assert_eq!(t.values, Some(vec![1.0]));
    }

    #[test]
    fn test_create_trial_datetime_auto_set() {
        // 非 WAITING 状态自动设 datetime_start
        let t = create_trial(
            Some(TrialState::Complete), Some(1.0), None,
            None, None, None, None, None,
        ).unwrap();
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_some());
    }

    #[test]
    fn test_create_trial_waiting_no_datetime() {
        // WAITING 状态不设 datetime
        let t = create_trial(
            Some(TrialState::Waiting), None, None,
            None, None, None, None, None,
        ).unwrap();
        assert!(t.datetime_start.is_none());
        assert!(t.datetime_complete.is_none());
    }

    #[test]
    fn test_create_trial_running_has_start_no_complete() {
        let t = create_trial(
            Some(TrialState::Running), None, None,
            None, None, None, None, None,
        ).unwrap();
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_none());
    }

    #[test]
    fn test_create_trial_with_values() {
        let t = create_trial(
            None, None, Some(vec![1.0, 2.0]),
            None, None, None, None, None,
        ).unwrap();
        assert_eq!(t.values, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_create_trial_value_wraps_to_vec() {
        let t = create_trial(None, Some(42.0), None, None, None, None, None, None).unwrap();
        assert_eq!(t.values, Some(vec![42.0]));
    }

    // === 对齐 Python: value/values 互斥 + validate ===

    #[test]
    fn test_create_trial_value_values_mutual_exclusion() {
        // 对齐 Python: 同时传入 value 和 values 报 ValueError
        let result = create_trial(
            None, Some(1.0), Some(vec![2.0]),
            None, None, None, None, None,
        );
        assert!(result.is_err(), "同时传入 value 和 values 应报错");
    }

    #[test]
    fn test_create_trial_complete_without_values_error() {
        // 对齐 Python validate(): Complete 状态必须有 values
        let result = create_trial(
            Some(TrialState::Complete), None, None,
            None, None, None, None, None,
        );
        assert!(result.is_err(), "Complete 无 values 应报错");
    }

    #[test]
    fn test_create_trial_fail_with_values_error() {
        // 对齐 Python validate(): Fail 状态不能有 values
        let result = create_trial(
            Some(TrialState::Fail), Some(1.0), None,
            None, None, None, None, None,
        );
        assert!(result.is_err(), "Fail 有 values 应报错");
    }

    #[test]
    fn test_create_trial_complete_with_nan_error() {
        // 对齐 Python validate(): Complete 值中不能有 NaN
        let result = create_trial(
            None, Some(f64::NAN), None,
            None, None, None, None, None,
        );
        assert!(result.is_err(), "Complete 含 NaN 应报错");
    }

    #[test]
    fn test_create_trial_pruned_no_values_ok() {
        // Pruned 状态可以没有 values
        let result = create_trial(
            Some(TrialState::Pruned), None, None,
            None, None, None, None, None,
        );
        assert!(result.is_ok());
    }
}
