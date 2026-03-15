// 异步逐次减半剪枝器 (ASHA - Asynchronous Successive Halving Algorithm)
// 对应 Python `optuna.pruners.SuccessiveHalvingPruner`
//
// 算法原理：
// - 试验按 "rung"（层级）晋升，每个 rung 有一个晋升步骤
// - 在每个 rung，只有排名前 1/reduction_factor 的试验能晋升到下一层
// - 晋升步骤计算：min_resource * reduction_factor^(min_early_stopping_rate + rung)
//
// 与 Python 的关键区别：
// Python 使用 study.system_attrs 存储 rung 信息（在 trial 的 system_attrs 中）
// Rust 版本在 FrozenTrial 的 system_attrs 中存储同样的信息

use std::sync::Mutex;

use crate::error::Result;
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// 逐次减半剪枝器的 rung 键前缀
const COMPLETED_RUNG_KEY_PREFIX: &str = "completed_rung_";

/// 异步逐次减半剪枝器 (ASHA)。
///
/// 对应 Python `optuna.pruners.SuccessiveHalvingPruner`。
///
/// 通过逐层竞争来决定是否继续训练试验：
/// 每一层只允许排名前 1/reduction_factor 的试验晋升。
#[derive(Debug)]
pub struct SuccessiveHalvingPruner {
    /// 最小资源量（步数）。如果为 None，则自动估计。
    min_resource: Option<i64>,
    /// 自动估计后缓存的值（对齐 Python：一旦估计就不再变化）
    cached_min_resource: Mutex<Option<i64>>,
    /// 缩减因子（每层保留 1/reduction_factor 的试验通过）
    reduction_factor: i64,
    /// 最小早停率 (s in the paper)，控制初始 rung 的晋升步骤
    min_early_stopping_rate: i64,
    /// 引导计数：每个 rung 至少需要此数量的试验才进行排名
    bootstrap_count: usize,
    /// 优化方向
    direction: StudyDirection,
}

impl SuccessiveHalvingPruner {
    /// 创建新的逐次减半剪枝器。
    ///
    /// # 参数
    /// * `min_resource` - 最小资源量。`None` 表示自动估计（基于已完成试验）。
    /// * `reduction_factor` - 缩减因子 (>= 2)。
    /// * `min_early_stopping_rate` - 最小早停率 (>= 0)。
    /// * `bootstrap_count` - 引导样本数 (>= 0)。
    /// * `direction` - 优化方向。
    pub fn new(
        min_resource: Option<i64>,
        reduction_factor: i64,
        min_early_stopping_rate: i64,
        bootstrap_count: usize,
        direction: StudyDirection,
    ) -> Self {
        assert!(reduction_factor >= 2, "reduction_factor 必须 >= 2");
        assert!(
            min_early_stopping_rate >= 0,
            "min_early_stopping_rate 必须 >= 0"
        );
        if let Some(mr) = min_resource {
            assert!(mr >= 1, "min_resource 必须 >= 1");
        }
        // 对齐 Python: bootstrap_count > 0 与 min_resource = auto(None) 互斥
        assert!(
            !(bootstrap_count > 0 && min_resource.is_none()),
            "bootstrap_count > 0 和 min_resource = auto(None) 互不兼容"
        );

        Self {
            min_resource,
            cached_min_resource: Mutex::new(None),
            reduction_factor,
            min_early_stopping_rate,
            bootstrap_count,
            direction,
        }
    }

    /// 估计最小资源量：基于已完成试验的最大步骤数 / 100。
    /// 对应 Python `_estimate_min_resource()`。
    fn estimate_min_resource(completed_trials: &[&FrozenTrial]) -> Option<i64> {
        let max_step = completed_trials
            .iter()
            .filter_map(|t| t.last_step())
            .max();

        max_step.map(|s| (s / 100).max(1))
    }

    /// 获取当前试验已完成的 rung 数量。
    /// 对应 Python `_get_current_rung(trial)`。
    fn get_current_rung(trial: &FrozenTrial) -> i64 {
        let mut rung = 0i64;
        loop {
            let key = format!("{}{}", COMPLETED_RUNG_KEY_PREFIX, rung);
            if trial.system_attrs.contains_key(&key) {
                rung += 1;
            } else {
                break;
            }
        }
        rung
    }

    /// 生成 rung 键名。
    fn completed_rung_key(rung: i64) -> String {
        format!("{}{}", COMPLETED_RUNG_KEY_PREFIX, rung)
    }

    /// 收集与该试验在同一 rung 竞争的所有值。
    /// 对应 Python `_get_competing_values()`。
    fn get_competing_values(
        study_trials: &[FrozenTrial],
        current_value: f64,
        rung_key: &str,
    ) -> Vec<f64> {
        let mut values: Vec<f64> = study_trials
            .iter()
            .filter_map(|t| {
                t.system_attrs.get(rung_key).and_then(|v| v.as_f64())
            })
            .collect();
        // 加入当前试验的值
        values.push(current_value);
        values
    }

    /// 判断试验是否可以晋升到下一个 rung。
    /// 对应 Python `_is_trial_promotable_to_next_rung()`。
    fn is_promotable(
        competing_values: &mut Vec<f64>,
        value: f64,
        reduction_factor: i64,
        direction: StudyDirection,
    ) -> bool {
        let n = competing_values.len();
        // 可晋升的试验数量
        let mut promotable_idx = (n / reduction_factor as usize) as i64 - 1;
        if promotable_idx < 0 {
            // 竞争者不足 reduction_factor → 只晋升最优的一个
            promotable_idx = 0;
        }
        let promotable_idx = promotable_idx as usize;

        // 排序竞争值
        competing_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        match direction {
            StudyDirection::Maximize => {
                // 最大化：取倒数第 (promotable_idx + 1) 个值
                let threshold = competing_values[n - 1 - promotable_idx];
                value >= threshold
            }
            _ => {
                // 最小化：取第 promotable_idx 个值
                let threshold = competing_values[promotable_idx];
                value <= threshold
            }
        }
    }
}

impl Pruner for SuccessiveHalvingPruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial, storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        // 获取当前步骤
        let step = match trial.last_step() {
            Some(s) => s,
            None => return Ok(false), // 没有中间值 → 不剪枝
        };

        // 获取当前 rung 级别
        let mut rung = Self::get_current_rung(trial);

        // 获取当前步骤的值
        let value = trial.intermediate_values[&step];

        // 获取已完成的试验列表（用于估计 min_resource）
        let completed: Vec<&FrozenTrial> = study_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        // 循环：可能连续晋升多个 rung
        loop {
            // 确定 min_resource（对齐 Python：自动估计后缓存）
            let min_resource = if let Some(mr) = self.min_resource {
                mr
            } else {
                let mut cached = self.cached_min_resource.lock().unwrap();
                if let Some(mr) = *cached {
                    mr
                } else {
                    match Self::estimate_min_resource(&completed) {
                        Some(mr) => {
                            *cached = Some(mr);
                            mr
                        }
                        None => return Ok(false),
                    }
                }
            };

            // 计算当前 rung 的晋升步骤
            let exponent = self.min_early_stopping_rate + rung;
            let rung_promotion_step = min_resource * self.reduction_factor.pow(exponent as u32);

            // 尚未到达晋升步骤 → 不剪枝
            if step < rung_promotion_step {
                return Ok(false);
            }

            // NaN 值 → 剪枝
            if value.is_nan() {
                return Ok(true);
            }

            // 对齐 Python: 通过 storage 写入 system_attrs 记录 rung 竞争值
            let rung_key = Self::completed_rung_key(rung);
            if let Some(st) = storage {
                st.set_trial_system_attr(
                    trial.trial_id,
                    &rung_key,
                    serde_json::Value::from(value),
                )?;
            }

            // 获取竞争值（包括刚写入的当前试验的值）
            let mut competing_values =
                Self::get_competing_values(study_trials, value, &rung_key);

            // 引导阶段：竞争者不足 → 剪枝（等待更多试验）
            if competing_values.len() <= self.bootstrap_count {
                return Ok(true);
            }

            // 检查是否可晋升
            let promotable = Self::is_promotable(
                &mut competing_values,
                value,
                self.reduction_factor,
                self.direction,
            );

            if !promotable {
                return Ok(true);
            }

            // 晋升成功 → 检查下一个 rung
            rung += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 辅助函数：创建带有中间值和 system_attrs 的 FrozenTrial
    fn make_trial_with_attrs(
        number: i64,
        intermediate: Vec<(i64, f64)>,
        system_attrs: HashMap<String, serde_json::Value>,
        state: TrialState,
    ) -> FrozenTrial {
        let mut iv = HashMap::new();
        for (step, val) in intermediate {
            iv.insert(step, val);
        }
        FrozenTrial {
            number,
            state,
            values: if state == TrialState::Complete {
                Some(vec![0.0])
            } else {
                None
            },
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: if state == TrialState::Complete {
                Some(chrono::Utc::now())
            } else {
                None
            },
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs,
            intermediate_values: iv,
            trial_id: number,
        }
    }

    fn make_running_trial(number: i64, intermediate: Vec<(i64, f64)>) -> FrozenTrial {
        make_trial_with_attrs(number, intermediate, HashMap::new(), TrialState::Running)
    }

    fn make_completed_trial(number: i64, intermediate: Vec<(i64, f64)>) -> FrozenTrial {
        make_trial_with_attrs(number, intermediate, HashMap::new(), TrialState::Complete)
    }

    #[test]
    fn test_no_intermediate_values() {
        // 无中间值 → 不剪枝
        let pruner =
            SuccessiveHalvingPruner::new(Some(1), 4, 0, 0, StudyDirection::Minimize);
        let trial = make_running_trial(0, vec![]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_not_reached_rung() {
        // 尚未到达第一个 rung 的晋升步骤 → 不剪枝
        // min_resource=10, reduction_factor=4, rate=0 → 第一个 rung 在步骤 10
        let pruner =
            SuccessiveHalvingPruner::new(Some(10), 4, 0, 0, StudyDirection::Minimize);
        let trial = make_running_trial(0, vec![(5, 1.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_nan_value_pruned() {
        // NaN 值在到达 rung 后 → 剪枝
        let pruner =
            SuccessiveHalvingPruner::new(Some(1), 4, 0, 0, StudyDirection::Minimize);
        let completed = make_completed_trial(0, vec![(0, 1.0), (100, 1.0)]);
        let trial = make_running_trial(1, vec![(1, f64::NAN)]);
        assert!(pruner.prune(&[completed], &trial, None).unwrap());
    }

    #[test]
    fn test_auto_min_resource() {
        // min_resource=None → 自动估计
        let pruner =
            SuccessiveHalvingPruner::new(None, 4, 0, 0, StudyDirection::Minimize);
        // 没有已完成试验 → 不剪枝
        let trial = make_running_trial(0, vec![(5, 1.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_get_current_rung() {
        // 测试 rung 计数
        let trial = make_running_trial(0, vec![]);
        assert_eq!(SuccessiveHalvingPruner::get_current_rung(&trial), 0);

        let mut attrs = HashMap::new();
        attrs.insert(
            "completed_rung_0".to_string(),
            serde_json::Value::from(1.0),
        );
        attrs.insert(
            "completed_rung_1".to_string(),
            serde_json::Value::from(0.5),
        );
        let trial = make_trial_with_attrs(0, vec![], attrs, TrialState::Running);
        assert_eq!(SuccessiveHalvingPruner::get_current_rung(&trial), 2);
    }

    /// 对齐 Python: bootstrap_count > 0 与 min_resource=None 互斥
    #[test]
    #[should_panic(expected = "互不兼容")]
    fn test_bootstrap_auto_incompatible() {
        // Python: ValueError("bootstrap_count > 0 and min_resource == 'auto' are mutually incompatible")
        SuccessiveHalvingPruner::new(None, 4, 0, 1, StudyDirection::Minimize);
    }

    /// 验证 reduction_factor < 2 被拒绝
    #[test]
    #[should_panic(expected = "reduction_factor")]
    fn test_reduction_factor_too_small() {
        SuccessiveHalvingPruner::new(Some(1), 1, 0, 0, StudyDirection::Minimize);
    }

    /// 验证 min_resource=0 被拒绝
    #[test]
    #[should_panic(expected = "min_resource")]
    fn test_min_resource_zero() {
        SuccessiveHalvingPruner::new(Some(0), 4, 0, 0, StudyDirection::Minimize);
    }

    /// 验证 min_early_stopping_rate 负值被拒绝
    #[test]
    #[should_panic(expected = "min_early_stopping_rate")]
    fn test_negative_early_stopping_rate() {
        SuccessiveHalvingPruner::new(Some(1), 4, -1, 0, StudyDirection::Minimize);
    }

    /// estimate_min_resource：正常估计
    #[test]
    fn test_estimate_min_resource() {
        // 最大步骤 = 100 → min_resource = 100/100 = 1
        let t = make_completed_trial(0, vec![(100, 1.0)]);
        assert_eq!(SuccessiveHalvingPruner::estimate_min_resource(&[&t]), Some(1));
        // 最大步骤 = 500 → min_resource = 500/100 = 5
        let t2 = make_completed_trial(1, vec![(500, 1.0)]);
        assert_eq!(SuccessiveHalvingPruner::estimate_min_resource(&[&t2]), Some(5));
    }

    /// estimate_min_resource：无试验时返回 None
    #[test]
    fn test_estimate_min_resource_empty() {
        let trials: Vec<&FrozenTrial> = vec![];
        assert_eq!(SuccessiveHalvingPruner::estimate_min_resource(&trials), None);
    }
}
