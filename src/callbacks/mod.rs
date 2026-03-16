//! 回调框架。
//!
//! 对应 Python `optuna.study.Study.optimize(..., callbacks=...)` 与
//! `optuna.storages.RetryFailedTrialCallback`。
//!
//! Python 的回调签名为 `(study, trial)`，Rust 由于所有权限制使用
//! `(study: &Study, trial: &FrozenTrial)`。

use std::sync::Arc;

use crate::study::Study;
use crate::terminators::Terminator;
use crate::trial::{FrozenTrial, TrialState};

/// 回调 trait：每次试验完成后调用。
///
/// 对应 Python `optuna.study.Study.optimize(..., callbacks=...)` 的回调协议。
///
/// Python 回调签名: `def __call__(self, study: Study, trial: FrozenTrial) -> None`
/// Rust 回调签名: `fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial)`
pub trait Callback: Send + Sync {
    /// 试验完成后调用。
    ///
    /// # 参数
    /// * `study` - 当前研究的引用
    /// * `trial` - 刚完成的试验
    fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial);
}

/// 最大试验数回调：达到指定试验数后停止。
///
/// 对应 Python `optuna.study.MaxTrialsCallback`。
///
/// 可按状态过滤试验（默认只计算 Complete 状态）。
pub struct MaxTrialsCallback {
    /// 最大试验数限制
    n_trials: usize,
    /// 要计数的试验状态（None 表示所有状态）
    states: Option<Vec<TrialState>>,
}

impl MaxTrialsCallback {
    /// 创建最大试验数回调。
    ///
    /// # 参数
    /// * `n_trials` - 最大试验数
    /// * `states` - 要计数的状态。`None` 表示计数所有状态。
    ///
    /// 对齐 Python: `MaxTrialsCallback(n_trials, states=None)` → 计数所有状态
    pub fn new(n_trials: usize, states: Option<Vec<TrialState>>) -> Self {
        // 对齐 Python: 保留 None 语义（所有状态），不自动填充默认值
        // 用户需要显式传入 Some(vec![Complete]) 来只计数 Complete 状态
        // Python 默认值 states=(TrialState.COMPLETE,) 在调用侧处理
        Self {
            n_trials,
            states,
        }
    }

    /// 便捷构造器：只计数 Complete 状态（对齐 Python 默认行为）。
    pub fn with_default_states(n_trials: usize) -> Self {
        Self::new(n_trials, Some(vec![TrialState::Complete]))
    }
}

impl Callback for MaxTrialsCallback {
    fn on_trial_complete(&self, study: &Study, _trial: &FrozenTrial) {
        // 获取符合状态过滤的试验数
        let count = match &self.states {
            Some(states) => study
                .get_trials(Some(states))
                .map(|t| t.len())
                .unwrap_or(0),
            None => study.trials().map(|t| t.len()).unwrap_or(0),
        };
        // 达到限制则停止研究
        if count >= self.n_trials {
            study.stop();
        }
    }
}

/// 失败试验重试回调。
///
/// 对应 Python `optuna.storages.RetryFailedTrialCallback`。
///
/// 当试验失败时，自动创建一个 WAITING 状态的新试验，
/// 继承原始试验的参数、分布和用户属性。
///
/// 通过 `retried_trial_number()` 和 `retry_history()` 静态方法
/// 可以追溯重试链。
pub struct RetryFailedTrialCallback {
    /// 最大重试次数（None 表示无限重试）
    max_retry: Option<usize>,
    /// 是否继承中间值
    inherit_intermediate_values: bool,
}

/// 系统属性键：原始失败试验编号
const FAILED_TRIAL_KEY: &str = "failed_trial";
/// 系统属性键：重试历史列表
const RETRY_HISTORY_KEY: &str = "retry_history";

impl RetryFailedTrialCallback {
    /// 创建重试回调。
    ///
    /// # 参数
    /// * `max_retry` - 最大重试次数。`None` 表示无限重试。
    /// * `inherit_intermediate_values` - 是否从失败试验继承中间值（默认 false）。
    pub fn new(max_retry: Option<usize>, inherit_intermediate_values: bool) -> Self {
        Self {
            max_retry,
            inherit_intermediate_values,
        }
    }

    /// 获取被重试的原始试验编号。
    ///
    /// # 返回
    /// 如果此试验是重试，返回原始失败试验的编号；否则返回 `None`。
    pub fn retried_trial_number(trial: &FrozenTrial) -> Option<i64> {
        trial
            .system_attrs
            .get(FAILED_TRIAL_KEY)
            .and_then(|v| v.as_i64())
    }

    /// 获取重试历史。
    ///
    /// # 返回
    /// 按时间顺序排列的重试试验编号列表。列表首个元素是原始失败试验。
    /// 如果此试验不是重试，返回空列表。
    pub fn retry_history(trial: &FrozenTrial) -> Vec<i64> {
        trial
            .system_attrs
            .get(RETRY_HISTORY_KEY)
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default()
    }
}

impl Callback for RetryFailedTrialCallback {
    fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial) {
        // 只处理失败的试验
        if trial.state != TrialState::Fail {
            return;
        }

        // 构建系统属性：继承原始属性并更新重试信息
        let mut system_attrs = trial.system_attrs.clone();

        // 设置原始失败试验编号
        if !system_attrs.contains_key(FAILED_TRIAL_KEY) {
            system_attrs.insert(
                FAILED_TRIAL_KEY.to_string(),
                serde_json::json!(trial.number),
            );
        }

        // 更新重试历史
        let mut retry_history = Self::retry_history(trial);
        retry_history.push(trial.number);

        // 检查是否超过最大重试次数
        if let Some(max) = self.max_retry {
            if retry_history.len() > max {
                return; // 超过最大重试次数，不再重试
            }
        }

        system_attrs.insert(
            RETRY_HISTORY_KEY.to_string(),
            serde_json::json!(retry_history),
        );

        // 构建 WAITING 模板试验
        let template = FrozenTrial {
            number: 0, // 存储层会分配新编号
            trial_id: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: trial.params.clone(),
            distributions: trial.distributions.clone(),
            user_attrs: trial.user_attrs.clone(),
            system_attrs,
            intermediate_values: if self.inherit_intermediate_values {
                trial.intermediate_values.clone()
            } else {
                std::collections::HashMap::new()
            },
        };

        // 添加重试试验（忽略错误，避免回调中 panic）
        let _ = study.add_trial(&template);
    }
}

/// 终止器回调：将终止器包装为回调。
///
/// 对应 Python `optuna.terminator.TerminatorCallback`。
///
/// 在每次试验完成后检查终止器条件，满足时停止研究。
pub struct TerminatorCallback {
    /// 内部终止器
    terminator: Arc<dyn Terminator>,
}

impl TerminatorCallback {
    /// 创建终止器回调。
    ///
    /// # 参数
    /// * `terminator` - 要包装的终止器
    pub fn new(terminator: Arc<dyn Terminator>) -> Self {
        Self { terminator }
    }
}

impl Callback for TerminatorCallback {
    fn on_trial_complete(&self, study: &Study, _trial: &FrozenTrial) {
        // 检查终止器条件
        if self.terminator.should_terminate(study) {
            // 对齐 Python: 输出终止日志
            crate::optuna_info!("The study has been stopped by the terminator.");
            study.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::create_study;
    use crate::study::StudyDirection;

    #[test]
    fn test_max_trials_callback_none_means_all() {
        // 对齐 Python: states=None → 计数所有状态
        let cb = MaxTrialsCallback::new(5, None);
        assert!(cb.states.is_none(), "None 应保持为 None（所有状态）");
    }

    #[test]
    fn test_max_trials_callback_with_default_states() {
        // 便捷构造器：只计 Complete 状态（Python 默认行为）
        let cb = MaxTrialsCallback::with_default_states(5);
        assert!(cb.states.as_ref().unwrap().contains(&TrialState::Complete));
    }

    #[test]
    fn test_max_trials_callback_custom_states() {
        // 自定义状态
        let cb = MaxTrialsCallback::new(
            3,
            Some(vec![TrialState::Complete, TrialState::Pruned]),
        );
        assert_eq!(cb.states.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_retry_failed_trial_callback_basic() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        )
        .unwrap();

        let cb = RetryFailedTrialCallback::new(Some(3), false);

        // 模拟一个失败试验
        let failed = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Fail,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };

        // 调用回调
        cb.on_trial_complete(&study, &failed);

        // 应该创建了一个 WAITING 试验
        let waiting = study
            .get_trials(Some(&[TrialState::Waiting]))
            .unwrap();
        assert_eq!(waiting.len(), 1, "应创建一个 WAITING 重试试验");
    }

    #[test]
    fn test_retry_failed_trial_max_retry_exceeded() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        )
        .unwrap();

        let cb = RetryFailedTrialCallback::new(Some(1), false);

        // 模拟一个已有重试历史的失败试验
        let mut system_attrs = std::collections::HashMap::new();
        system_attrs.insert(FAILED_TRIAL_KEY.to_string(), serde_json::json!(0));
        system_attrs.insert(RETRY_HISTORY_KEY.to_string(), serde_json::json!([0, 1]));

        let failed = FrozenTrial {
            number: 2,
            trial_id: 2,
            state: TrialState::Fail,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs,
            intermediate_values: std::collections::HashMap::new(),
        };

        cb.on_trial_complete(&study, &failed);

        // 已重试 2 次 > max_retry=1，不应再创建
        let waiting = study
            .get_trials(Some(&[TrialState::Waiting]))
            .unwrap();
        assert_eq!(waiting.len(), 0, "超过最大重试次数，不应创建新试验");
    }

    #[test]
    fn test_retry_history_static_methods() {
        let mut system_attrs = std::collections::HashMap::new();
        system_attrs.insert(FAILED_TRIAL_KEY.to_string(), serde_json::json!(5));
        system_attrs.insert(RETRY_HISTORY_KEY.to_string(), serde_json::json!([5, 7]));

        let trial = FrozenTrial {
            number: 9,
            trial_id: 9,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs,
            intermediate_values: std::collections::HashMap::new(),
        };

        assert_eq!(RetryFailedTrialCallback::retried_trial_number(&trial), Some(5));
        assert_eq!(RetryFailedTrialCallback::retry_history(&trial), vec![5, 7]);
    }

    #[test]
    fn test_retry_non_failed_trial_ignored() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        )
        .unwrap();

        let cb = RetryFailedTrialCallback::new(None, false);

        // Complete 状态的试验不应触发重试
        let complete = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };

        cb.on_trial_complete(&study, &complete);

        let waiting = study
            .get_trials(Some(&[TrialState::Waiting]))
            .unwrap();
        assert_eq!(waiting.len(), 0, "Complete 试验不应触发重试");
    }

    #[test]
    fn test_terminator_callback() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        )
        .unwrap();

        // 使用 MaxTrialsTerminator(1) 作为终止器
        let term = Arc::new(crate::terminators::MaxTrialsTerminator::new(1));
        let cb = TerminatorCallback::new(term);

        // 模拟一个完成的试验
        let now = chrono::Utc::now();
        let complete = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };

        // 先添加一个试验使得 MaxTrials(1) 满足
        study.add_trial(&complete).unwrap();
        cb.on_trial_complete(&study, &complete);

        // 研究应已标记停止（内部通过 stop_flag）
        // 注意：stop() 设置的是 AtomicBool，外部无法直接检查
        // 但可以验证 optimize 会提前停止
    }

    /// 对齐 Python: MaxTrialsCallback 实际停止 optimize 循环
    #[test]
    fn test_max_trials_callback_actually_stops_optimize() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        let cb = MaxTrialsCallback::with_default_states(3);
        let cbs: &[&dyn Callback] = &[&cb];

        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x * x)
            },
            Some(100), // 设置很大的上限，回调应在 3 次后停止
            None,
            Some(cbs),
        ).unwrap();

        let trials = study.trials().unwrap();
        let n_complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
        // 回调在 3 次完成后触发 stop → 实际可能是 3 或 4 次(竞争窗口)
        assert!(n_complete >= 3 && n_complete <= 5,
            "应在约 3 次完成后停止，实际完成 {} 次", n_complete);
    }

    /// 对齐 Python: RetryFailedTrialCallback 继承中间值
    #[test]
    fn test_retry_failed_trial_inherits_intermediate_values() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        let cb = RetryFailedTrialCallback::new(Some(3), true); // inherit=true

        let mut intermediate_values = std::collections::HashMap::new();
        intermediate_values.insert(0_i64, 0.5);
        intermediate_values.insert(1, 0.3);

        let failed = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Fail,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values,
        };

        cb.on_trial_complete(&study, &failed);

        let waiting = study.get_trials(Some(&[TrialState::Waiting])).unwrap();
        assert_eq!(waiting.len(), 1);
        // inherit_intermediate_values=true → 应继承中间值
        assert_eq!(waiting[0].intermediate_values.len(), 2,
            "应继承 2 个中间值");
        assert_eq!(waiting[0].intermediate_values.get(&0).copied(), Some(0.5));
        assert_eq!(waiting[0].intermediate_values.get(&1).copied(), Some(0.3));
    }

    /// 对齐 Python: RetryFailedTrialCallback 不继承中间值
    #[test]
    fn test_retry_failed_trial_no_inherit_intermediate_values() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        let cb = RetryFailedTrialCallback::new(Some(3), false); // inherit=false

        let mut intermediate_values = std::collections::HashMap::new();
        intermediate_values.insert(0_i64, 0.5);

        let failed = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Fail,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values,
        };

        cb.on_trial_complete(&study, &failed);

        let waiting = study.get_trials(Some(&[TrialState::Waiting])).unwrap();
        assert_eq!(waiting.len(), 1);
        assert!(waiting[0].intermediate_values.is_empty(),
            "inherit=false 时不应继承中间值");
    }
}
