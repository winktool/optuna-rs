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
/// * `value` - 单目标值
/// * `values` - 多目标值
/// * `params` - 参数字典
/// * `distributions` - 参数分布字典
/// * `user_attrs` - 用户属性
/// * `system_attrs` - 系统属性
/// * `intermediate_values` - 中间值
///
/// # 行为对齐 Python
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
) -> FrozenTrial {
    // value 和 values 互斥（对齐 Python: 若两者都给，values 优先）
    let final_values = if let Some(v) = value {
        Some(vec![v])
    } else {
        values
    };

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

    FrozenTrial {
        number: -1,
        trial_id: -1,
        state: s,
        values: final_values,
        datetime_start,
        datetime_complete,
        params: params.unwrap_or_default(),
        distributions: distributions.unwrap_or_default(),
        user_attrs: user_attrs.unwrap_or_default(),
        system_attrs: system_attrs.unwrap_or_default(),
        intermediate_values: intermediate_values.unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_trial_defaults() {
        // 默认状态 Complete，number/trial_id = -1
        let t = create_trial(None, Some(1.0), None, None, None, None, None, None);
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
        );
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_some());
    }

    #[test]
    fn test_create_trial_waiting_no_datetime() {
        // WAITING 状态不设 datetime
        let t = create_trial(
            Some(TrialState::Waiting), None, None,
            None, None, None, None, None,
        );
        assert!(t.datetime_start.is_none());
        assert!(t.datetime_complete.is_none());
    }

    #[test]
    fn test_create_trial_running_has_start_no_complete() {
        let t = create_trial(
            Some(TrialState::Running), None, None,
            None, None, None, None, None,
        );
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_none());
    }

    #[test]
    fn test_create_trial_with_values() {
        let t = create_trial(
            None, None, Some(vec![1.0, 2.0]),
            None, None, None, None, None,
        );
        assert_eq!(t.values, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_create_trial_value_wraps_to_vec() {
        let t = create_trial(None, Some(42.0), None, None, None, None, None, None);
        assert_eq!(t.values, Some(vec![42.0]));
    }
}
