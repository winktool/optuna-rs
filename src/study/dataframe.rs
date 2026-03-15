//! DataFrame 输出模块。
//!
//! 对应 Python `optuna.study._dataframe.trials_dataframe()`。
//! 使用 [`polars`](https://docs.rs/polars) crate 实现 DataFrame 支持。
//!
//! # 使用方式
//! 需要启用 `dataframe` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["dataframe"] }
//! ```

#[cfg(feature = "dataframe")]
use polars::prelude::*;

#[cfg(feature = "dataframe")]
use crate::distributions::ParamValue;
#[cfg(feature = "dataframe")]
use crate::error::{OptunaError, Result};
#[cfg(feature = "dataframe")]
use crate::trial::FrozenTrial;

/// Python 中 `trials_dataframe` 支持的默认 attrs。
/// 对应 Python `(number, value, datetime_start, datetime_complete, duration, params,
///               user_attrs, system_attrs, state)`。
#[cfg(feature = "dataframe")]
pub const DEFAULT_ATTRS: &[&str] = &[
    "number",
    "value",
    "datetime_start",
    "datetime_complete",
    "duration",
    "params",
    "user_attrs",
    "system_attrs",
    "state",
];

/// 将试验列表转换为 polars DataFrame。
///
/// 对应 Python `study.trials_dataframe(attrs, multi_index)`。
///
/// # 参数
/// * `trials` - 试验列表
/// * `multi_objective` - 是否多目标
/// * `metric_names` - 目标名（多目标场景）
/// * `attrs` - 要包含的列名列表。`None` 使用 `DEFAULT_ATTRS`。
///   支持: `number`, `value`/`values`, `datetime_start`, `datetime_complete`,
///   `duration`, `params`, `user_attrs`, `system_attrs`, `state`,
///   `intermediate_values`, `trial_id`。
/// * `multi_index` - 是否使用 MultiIndex 风格列名 (e.g. `params|x` 而非 `params_x`)
#[cfg(feature = "dataframe")]
pub fn trials_to_dataframe(
    trials: &[FrozenTrial],
    multi_objective: bool,
    metric_names: Option<&[String]>,
    attrs: Option<&[&str]>,
    multi_index: bool,
) -> Result<DataFrame> {
    if trials.is_empty() {
        return Ok(DataFrame::empty());
    }

    let effective_attrs: Vec<&str> = if let Some(a) = attrs {
        // 对应 Python: if "value" in attrs and study._is_multi_objective(): replace with "values"
        a.iter()
            .map(|&attr| {
                if attr == "value" && multi_objective {
                    "values"
                } else {
                    attr
                }
            })
            .collect()
    } else {
        let mut da: Vec<&str> = DEFAULT_ATTRS.to_vec();
        if multi_objective {
            if let Some(pos) = da.iter().position(|&a| a == "value") {
                da[pos] = "values";
            }
        }
        da
    };

    // 分隔符: multi_index → "|", 否则 → "_"
    let sep = if multi_index { "|" } else { "_" };

    let mut columns: Vec<Column> = Vec::new();

    for &attr in &effective_attrs {
        match attr {
            "number" => {
                let numbers: Vec<i64> = trials.iter().map(|t| t.number).collect();
                columns.push(Column::new("number".into(), &numbers));
            }
            "trial_id" => {
                let ids: Vec<i64> = trials.iter().map(|t| t.trial_id).collect();
                columns.push(Column::new("trial_id".into(), &ids));
            }
            "value" => {
                // 单目标
                let vals: Vec<Option<f64>> = trials
                    .iter()
                    .map(|t| t.values.as_ref().and_then(|v| v.first().copied()))
                    .collect();
                let col_name = if multi_index {
                    if let Some(names) = metric_names {
                        format!("value|{}", names.first().map(|s| s.as_str()).unwrap_or(""))
                    } else {
                        "value".to_string()
                    }
                } else {
                    "value".to_string()
                };
                columns.push(Column::new(col_name.into(), &vals));
            }
            "values" => {
                let n_obj = trials
                    .iter()
                    .filter_map(|t| t.values.as_ref())
                    .map(|v| v.len())
                    .max()
                    .unwrap_or(0);
                for i in 0..n_obj {
                    let col_name = if let Some(names) = metric_names {
                        if i < names.len() {
                            format!("values{sep}{}", names[i])
                        } else {
                            format!("values{sep}{i}")
                        }
                    } else {
                        format!("values{sep}{i}")
                    };
                    let vals: Vec<Option<f64>> = trials
                        .iter()
                        .map(|t| t.values.as_ref().and_then(|v| v.get(i).copied()))
                        .collect();
                    columns.push(Column::new(col_name.into(), &vals));
                }
            }
            "datetime_start" => {
                let starts: Vec<Option<i64>> = trials
                    .iter()
                    .map(|t| t.datetime_start.map(|dt| dt.timestamp_millis()))
                    .collect();
                columns.push(Column::new("datetime_start".into(), &starts));
            }
            "datetime_complete" => {
                let completes: Vec<Option<i64>> = trials
                    .iter()
                    .map(|t| t.datetime_complete.map(|dt| dt.timestamp_millis()))
                    .collect();
                columns.push(Column::new("datetime_complete".into(), &completes));
            }
            "duration" => {
                let durations: Vec<Option<f64>> = trials
                    .iter()
                    .map(|t| t.duration().map(|d| d.num_milliseconds() as f64 / 1000.0))
                    .collect();
                columns.push(Column::new("duration".into(), &durations));
            }
            "params" => {
                let all_param_names: Vec<String> = {
                    let mut names = std::collections::BTreeSet::new();
                    for t in trials {
                        for k in t.params.keys() {
                            names.insert(k.clone());
                        }
                    }
                    names.into_iter().collect()
                };
                for name in &all_param_names {
                    let col_name = format!("params{sep}{name}");
                    let vals: Vec<Option<String>> = trials
                        .iter()
                        .map(|t| t.params.get(name).map(param_value_to_string))
                        .collect();
                    columns.push(Column::new(col_name.into(), &vals));
                }
            }
            "user_attrs" => {
                let all_keys: Vec<String> = {
                    let mut keys = std::collections::BTreeSet::new();
                    for t in trials {
                        for k in t.user_attrs.keys() {
                            keys.insert(k.clone());
                        }
                    }
                    keys.into_iter().collect()
                };
                for key in &all_keys {
                    let col_name = format!("user_attrs{sep}{key}");
                    let vals: Vec<Option<String>> = trials
                        .iter()
                        .map(|t| t.user_attrs.get(key).map(|v| v.to_string()))
                        .collect();
                    columns.push(Column::new(col_name.into(), &vals));
                }
            }
            "system_attrs" => {
                let all_keys: Vec<String> = {
                    let mut keys = std::collections::BTreeSet::new();
                    for t in trials {
                        for k in t.system_attrs.keys() {
                            keys.insert(k.clone());
                        }
                    }
                    keys.into_iter().collect()
                };
                for key in &all_keys {
                    let col_name = format!("system_attrs{sep}{key}");
                    let vals: Vec<Option<String>> = trials
                        .iter()
                        .map(|t| t.system_attrs.get(key).map(|v| v.to_string()))
                        .collect();
                    columns.push(Column::new(col_name.into(), &vals));
                }
            }
            "state" => {
                let states: Vec<String> = trials.iter().map(|t| t.state.to_string()).collect();
                columns.push(Column::new("state".into(), &states));
            }
            "intermediate_values" => {
                // 收集所有 step key
                let all_steps: Vec<i64> = {
                    let mut steps = std::collections::BTreeSet::new();
                    for t in trials {
                        for &s in t.intermediate_values.keys() {
                            steps.insert(s);
                        }
                    }
                    steps.into_iter().collect()
                };
                for step in &all_steps {
                    let col_name = format!("intermediate_values{sep}{step}");
                    let vals: Vec<Option<f64>> = trials
                        .iter()
                        .map(|t| t.intermediate_values.get(step).copied())
                        .collect();
                    columns.push(Column::new(col_name.into(), &vals));
                }
            }
            _ => {
                // 不识别的 attr 忽略
            }
        }
    }

    if columns.is_empty() {
        return Ok(DataFrame::empty());
    }

    DataFrame::new(columns)
        .map_err(|e| OptunaError::ValueError(format!("创建 DataFrame 失败: {e}")))
}

#[cfg(feature = "dataframe")]
fn param_value_to_string(v: &ParamValue) -> String {
    match v {
        ParamValue::Float(f) => f.to_string(),
        ParamValue::Int(i) => i.to_string(),
        ParamValue::Categorical(c) => format!("{c:?}"),
    }
}

#[cfg(test)]
#[cfg(feature = "dataframe")]
mod tests {
    use super::*;
    use crate::testing::create_frozen_trial;
    use std::collections::HashMap;

    #[test]
    fn test_empty_trials() {
        let df = trials_to_dataframe(&[], false, None, None, false).unwrap();
        assert_eq!(df.height(), 0);
    }

    #[test]
    fn test_single_objective() {
        let trials = vec![
            create_frozen_trial(0, Some(vec![1.0]), None, None, None, None),
            create_frozen_trial(1, Some(vec![2.0]), None, None, None, None),
        ];
        let df = trials_to_dataframe(&trials, false, None, None, false).unwrap();
        assert_eq!(df.height(), 2);
        assert!(df.get_column_names().contains(&&PlSmallStr::from("value")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("number")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("state")));
    }

    #[test]
    fn test_multi_objective() {
        let trials = vec![
            create_frozen_trial(0, Some(vec![1.0, 2.0]), None, None, None, None),
            create_frozen_trial(1, Some(vec![3.0, 4.0]), None, None, None, None),
        ];
        let df = trials_to_dataframe(&trials, true, None, None, false).unwrap();
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values_0")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values_1")));
    }

    #[test]
    fn test_with_metric_names() {
        let trials = vec![
            create_frozen_trial(0, Some(vec![1.0, 2.0]), None, None, None, None),
        ];
        let names = vec!["accuracy".to_string(), "loss".to_string()];
        let df = trials_to_dataframe(&trials, true, Some(&names), None, false).unwrap();
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values_accuracy")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values_loss")));
    }

    #[test]
    fn test_attrs_filtering() {
        let trials = vec![
            create_frozen_trial(0, Some(vec![1.0]), None, None, None, None),
        ];
        // 只请求 number 和 state
        let df = trials_to_dataframe(&trials, false, None, Some(&["number", "state"]), false).unwrap();
        assert_eq!(df.width(), 2);
        assert!(df.get_column_names().contains(&&PlSmallStr::from("number")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("state")));
    }

    #[test]
    fn test_multi_index() {
        let trials = vec![
            create_frozen_trial(0, Some(vec![1.0, 2.0]), None, None, None, None),
        ];
        let names = vec!["accuracy".to_string(), "loss".to_string()];
        let df = trials_to_dataframe(&trials, true, Some(&names), None, true).unwrap();
        // multi_index 模式用 "|" 分隔
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values|accuracy")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("values|loss")));
    }

    #[test]
    fn test_intermediate_values() {
        use std::collections::HashMap;
        let now = chrono::Utc::now();
        let mut iv = HashMap::new();
        iv.insert(1_i64, 0.5);
        iv.insert(2_i64, 0.3);
        let trial = FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: 0,
        };
        let df = trials_to_dataframe(
            &[trial],
            false,
            None,
            Some(&["number", "intermediate_values"]),
            false,
        ).unwrap();
        assert!(df.get_column_names().contains(&&PlSmallStr::from("intermediate_values_1")));
        assert!(df.get_column_names().contains(&&PlSmallStr::from("intermediate_values_2")));
    }

    /// 对齐 Python: Pruned/Failed 试验的 values = None
    #[test]
    fn test_pruned_trial_null_values() {
        let now = chrono::Utc::now();
        let trial = FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Pruned,
            values: None,
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let df = trials_to_dataframe(&[trial], false, None, None, false).unwrap();
        assert_eq!(df.height(), 1);
        // value 列应为 null
        let col = df.column("value").unwrap();
        assert!(col.f64().unwrap().get(0).is_none());
    }

    /// 对齐 Python: user_attrs 跨试验不一致
    #[test]
    fn test_user_attrs_inconsistent_across_trials() {
        let now = chrono::Utc::now();
        let mut ua1 = HashMap::new();
        ua1.insert("x".to_string(), serde_json::json!(1));
        let t1 = FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: ua1,
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let t2 = FrozenTrial {
            number: 1,
            state: crate::trial::TrialState::Complete,
            values: Some(vec![2.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(), // 没有 "x" attr
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 1,
        };
        let df = trials_to_dataframe(&[t1, t2], false, None, None, false).unwrap();
        assert_eq!(df.height(), 2);
        // user_attrs_x 列应该存在
        assert!(df.get_column_names().contains(&&PlSmallStr::from("user_attrs_x")));
    }

    /// 对齐 Python: duration 列
    #[test]
    fn test_duration_column() {
        let start = chrono::Utc::now();
        let end = start + chrono::Duration::seconds(10);
        let trial = FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(start),
            datetime_complete: Some(end),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let df = trials_to_dataframe(
            &[trial], false, None,
            Some(&["number", "duration"]), false,
        ).unwrap();
        assert!(df.get_column_names().contains(&&PlSmallStr::from("duration")));
        let dur = df.column("duration").unwrap().f64().unwrap().get(0).unwrap();
        // duration 应约为 10 秒
        assert!((dur - 10.0).abs() < 0.1, "duration should be ~10s, got {dur}");
    }
}
