// 阈值剪枝器 - 当中间值超出给定阈值范围时剪枝
// 对应 Python `optuna.pruners.ThresholdPruner`
//
// 算法逻辑：
// 1. 检查试验是否已报告中间值
// 2. 检查是否在预热阶段
// 3. 检查当前步骤是否在检查间隔内
// 4. 获取最新报告的中间值
// 5. 如果值为 NaN → 剪枝
// 6. 如果值 < lower → 剪枝
// 7. 如果值 > upper → 剪枝
// 8. 否则不剪枝

use crate::error::Result;
use crate::pruners::Pruner;
use crate::trial::FrozenTrial;

/// 阈值剪枝器：当中间值超出 [lower, upper] 范围或为 NaN 时进行剪枝。
///
/// 对应 Python `optuna.pruners.ThresholdPruner`。
///
/// # 参数
/// * `lower` - 最小允许值，低于此值则剪枝。`None` 表示负无穷。
/// * `upper` - 最大允许值，高于此值则剪枝。`None` 表示正无穷。
/// * `n_warmup_steps` - 预热步数，在此之前不进行剪枝。
/// * `interval_steps` - 检查间隔步数。
#[derive(Debug, Clone)]
pub struct ThresholdPruner {
    /// 最小允许值（含）
    lower: f64,
    /// 最大允许值（含）
    upper: f64,
    /// 预热步数
    n_warmup_steps: i64,
    /// 检查间隔
    interval_steps: i64,
}

impl ThresholdPruner {
    /// 创建新的阈值剪枝器。
    ///
    /// # 参数
    /// * `lower` - 最小允许值。`None` 表示不设下界。
    /// * `upper` - 最大允许值。`None` 表示不设上界。
    /// * `n_warmup_steps` - 预热步数 (>= 0)。
    /// * `interval_steps` - 检查间隔 (>= 1)。
    ///
    /// # Panics
    /// 如果 `lower` 和 `upper` 都为 `None`，或者 `lower > upper`。
    pub fn new(
        lower: Option<f64>,
        upper: Option<f64>,
        n_warmup_steps: i64,
        interval_steps: i64,
    ) -> Self {
        // 与 Python 一致：lower 和 upper 不能同时为 None
        assert!(
            lower.is_some() || upper.is_some(),
            "至少需要指定 lower 或 upper 之一"
        );
        assert!(interval_steps >= 1, "interval_steps 必须 >= 1");
        assert!(n_warmup_steps >= 0, "n_warmup_steps 必须 >= 0");

        let lower_val = lower.unwrap_or(f64::NEG_INFINITY);
        let upper_val = upper.unwrap_or(f64::INFINITY);

        assert!(
            lower_val <= upper_val,
            "lower ({}) 必须 <= upper ({})",
            lower_val,
            upper_val
        );

        Self {
            lower: lower_val,
            upper: upper_val,
            n_warmup_steps,
            interval_steps,
        }
    }
}

impl Pruner for ThresholdPruner {
    fn prune(&self, _study_trials: &[FrozenTrial], trial: &FrozenTrial, _storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        // 获取最新报告的步骤
        let step = match trial.last_step() {
            Some(s) => s,
            None => return Ok(false), // 没有中间值 → 不剪枝
        };

        // 检查预热阶段
        if step < self.n_warmup_steps {
            return Ok(false);
        }

        // 检查是否在检查间隔内
        if !super::is_first_in_interval_step(
            step,
            &trial.intermediate_values,
            self.n_warmup_steps,
            self.interval_steps,
        ) {
            return Ok(false);
        }

        // 获取最新报告的中间值
        let latest_value = trial.intermediate_values[&step];

        // NaN 值 → 剪枝（与 Python 一致）
        if latest_value.is_nan() {
            return Ok(true);
        }

        // 低于下界 → 剪枝
        if latest_value < self.lower {
            return Ok(true);
        }

        // 高于上界 → 剪枝
        if latest_value > self.upper {
            return Ok(true);
        }

        // 在范围内 → 不剪枝
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialState;
    use std::collections::HashMap;

    /// 辅助函数：创建带有中间值的 FrozenTrial
    fn make_trial(intermediate: Vec<(i64, f64)>) -> FrozenTrial {
        let mut iv = HashMap::new();
        for (step, val) in intermediate {
            iv.insert(step, val);
        }
        FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: 0,
        }
    }

    #[test]
    fn test_lower_threshold() {
        // 低于下界的值应该被剪枝
        let pruner = ThresholdPruner::new(Some(1.0), None, 0, 1);
        let trial = make_trial(vec![(0, 0.5)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_upper_threshold() {
        // 高于上界的值应该被剪枝
        let pruner = ThresholdPruner::new(None, Some(2.0), 0, 1);
        let trial = make_trial(vec![(0, 3.0)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_within_range() {
        // 在范围内的值不应该被剪枝
        let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
        let trial = make_trial(vec![(0, 3.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_nan_value() {
        // NaN 值应该被剪枝
        let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
        let trial = make_trial(vec![(0, f64::NAN)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_warmup_steps() {
        // 预热阶段不应该剪枝
        let pruner = ThresholdPruner::new(Some(1.0), None, 5, 1);
        let trial = make_trial(vec![(3, 0.1)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_no_intermediate_values() {
        // 没有中间值不应该剪枝
        let pruner = ThresholdPruner::new(Some(1.0), None, 0, 1);
        let trial = make_trial(vec![]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_boundary_values() {
        // 边界值不应该被剪枝（包含边界）
        let pruner = ThresholdPruner::new(Some(1.0), Some(5.0), 0, 1);
        let trial_lower = make_trial(vec![(0, 1.0)]);
        assert!(!pruner.prune(&[], &trial_lower, None).unwrap());
        let trial_upper = make_trial(vec![(0, 5.0)]);
        assert!(!pruner.prune(&[], &trial_upper, None).unwrap());
    }

    #[test]
    #[should_panic]
    fn test_both_none_panics() {
        // lower 和 upper 都为 None 应该 panic
        ThresholdPruner::new(None, None, 0, 1);
    }
}
