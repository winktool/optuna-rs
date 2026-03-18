// Hyperband 剪枝器 - 多括号逐次减半策略
// 对应 Python `optuna.pruners.HyperbandPruner`
//
// 算法原理：
// - Hyperband 是多个 SuccessiveHalving 的组合（不同括号）
// - 每个括号使用不同的 min_early_stopping_rate (即 bracket_id)
// - 括号分配通过确定性哈希实现（CRC32(study_name + trial_number) % total_budget）
// - 括号数量 = floor(log_eta(max_resource / min_resource)) + 1

use crate::error::Result;
use crate::pruners::successive_halving::SuccessiveHalvingPruner;
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};
use std::sync::Mutex;

/// Hyperband 剪枝器：通过多括号逐次减半来平衡探索与利用。
///
/// 对应 Python `optuna.pruners.HyperbandPruner`。
///
/// 每个括号是一个独立的 SuccessiveHalvingPruner，具有不同的早停率。
/// 试验通过确定性哈希被分配到不同的括号中。
pub struct HyperbandPruner {
    /// 最小资源量 (r in the paper)
    min_resource: i64,
    /// 最大资源量 (R in the paper)。None 表示自动估计。
    max_resource: Option<i64>,
    /// 缩减因子 (eta in the paper)
    reduction_factor: i64,
    /// 引导计数
    bootstrap_count: usize,
    /// 优化方向
    direction: StudyDirection,
    /// Study 名称，用于括号分配哈希（对应 Python study.study_name）
    study_name: String,
    /// 内部状态（延迟初始化）
    inner: Mutex<HyperbandInner>,
}

/// Hyperband 内部状态，延迟初始化
struct HyperbandInner {
    /// 各括号的 SuccessiveHalvingPruner
    pruners: Vec<SuccessiveHalvingPruner>,
    /// 各括号的试验分配预算
    trial_allocation_budgets: Vec<usize>,
    /// 括号数量
    n_brackets: Option<usize>,
}

impl HyperbandPruner {
    /// 创建新的 Hyperband 剪枝器。
    ///
    /// # 参数
    /// * `min_resource` - 最小资源量 (>= 1)。
    /// * `max_resource` - 最大资源量。`None` 表示从第一个完成的试验自动估计。
    /// * `reduction_factor` - 缩减因子 (>= 2)。
    /// * `bootstrap_count` - 引导样本数 (>= 0)。
    /// * `direction` - 优化方向。
    pub fn new(
        min_resource: i64,
        max_resource: Option<i64>,
        reduction_factor: i64,
        bootstrap_count: usize,
        direction: StudyDirection,
        study_name: &str,
    ) -> Self {
        assert!(min_resource >= 1, "`min_resource` must be >= 1.");
        assert!(reduction_factor >= 2, "`reduction_factor` must be >= 2.");
        if let Some(mr) = max_resource {
            assert!(
                mr >= min_resource,
                "`max_resource` must be >= `min_resource`."
            );
        }
        // 对齐 Python: bootstrap_count > 0 与 max_resource="auto"(None) 互斥
        assert!(
            !(bootstrap_count > 0 && max_resource.is_none()),
            "`bootstrap_count` > 0 and `max_resource` = auto(None) are mutually exclusive."
        );

        Self {
            min_resource,
            max_resource,
            reduction_factor,
            bootstrap_count,
            direction,
            study_name: study_name.to_string(),
            inner: Mutex::new(HyperbandInner {
                pruners: Vec::new(),
                trial_allocation_budgets: Vec::new(),
                n_brackets: None,
            }),
        }
    }

    /// 尝试初始化内部状态。
    /// 对应 Python `_try_initialization(study)`。
    fn try_initialization(
        &self,
        study_trials: &[FrozenTrial],
        inner: &mut HyperbandInner,
    ) {
        // 确定 max_resource
        let max_resource = match self.max_resource {
            Some(mr) => mr,
            None => {
                // 自动估计：从已完成试验中获取最大步骤 + 1
                let completed: Vec<&FrozenTrial> = study_trials
                    .iter()
                    .filter(|t| t.state == TrialState::Complete)
                    .collect();
                match completed.iter().filter_map(|t| t.last_step()).max() {
                    Some(max_step) => max_step + 1,
                    None => return, // 没有已完成试验 → 无法初始化
                }
            }
        };

        // 计算括号数量：floor(log_eta(max_resource / min_resource)) + 1
        let n_brackets = if inner.n_brackets.is_some() {
            inner.n_brackets.unwrap()
        } else {
            let ratio = max_resource as f64 / self.min_resource as f64;
            let n = (ratio.ln() / (self.reduction_factor as f64).ln()).floor() as usize + 1;
            inner.n_brackets = Some(n);
            n
        };

        // 防止重复初始化：如果已经有 pruners 则跳过
        if !inner.pruners.is_empty() {
            return;
        }

        // 为每个括号创建 SuccessiveHalvingPruner
        for bracket_id in 0..n_brackets {
            let budget = self.calculate_trial_allocation_budget(bracket_id, n_brackets);
            inner.trial_allocation_budgets.push(budget);

            let pruner = SuccessiveHalvingPruner::new(
                Some(self.min_resource),
                self.reduction_factor,
                bracket_id as i64, // min_early_stopping_rate = bracket_id
                self.bootstrap_count,
                self.direction,
            );
            inner.pruners.push(pruner);
        }
    }

    /// 计算括号的试验分配预算。
    /// 对应 Python `_calculate_trial_allocation_budget()`。
    fn calculate_trial_allocation_budget(&self, bracket_id: usize, n_brackets: usize) -> usize {
        let s = n_brackets - 1 - bracket_id;
        let eta = self.reduction_factor as f64;
        let n = n_brackets as f64;
        // budget = ceil(n_brackets * eta^s / (s + 1))
        (n * eta.powi(s as i32) / (s as f64 + 1.0)).ceil() as usize
    }

    /// 确定试验所属的括号 ID。
    /// 使用 CRC32 哈希实现确定性分配。
    /// 对应 Python `_get_bracket_id(study, trial)`。
    fn get_bracket_id(
        &self,
        trial_number: i64,
        study_name: &str,
        budgets: &[usize],
    ) -> usize {
        // 计算哈希值
        let hash_input = format!("{}_{}", study_name, trial_number);
        let hash = crc32_hash(hash_input.as_bytes());

        // 确定性括号分配
        let total_budget: usize = budgets.iter().sum();
        let mut n = (hash as usize) % total_budget;

        for (bracket_id, &budget) in budgets.iter().enumerate() {
            if n < budget {
                return bracket_id;
            }
            n -= budget;
        }

        // 不应该到达这里
        budgets.len() - 1
    }
}

/// 简单的 CRC32 实现（与 Python binascii.crc32 对齐）
fn crc32_hash(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

impl Pruner for HyperbandPruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial, storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        let mut inner = self.inner.lock().unwrap();

        // 延迟初始化
        if inner.pruners.is_empty() {
            self.try_initialization(study_trials, &mut inner);
            if inner.pruners.is_empty() {
                return Ok(false); // 无法初始化 → 不剪枝
            }
        }

        // 确定括号 ID（使用 study_name 参与哈希，与 Python 一致）
        let bracket_id = self.get_bracket_id(
            trial.number,
            &self.study_name,
            &inner.trial_allocation_budgets,
        );

        // 筛选同一括号的试验
        let bracket_trials: Vec<FrozenTrial> = study_trials
            .iter()
            .filter(|t| {
                let t_bracket = self.get_bracket_id(
                    t.number,
                    &self.study_name,
                    &inner.trial_allocation_budgets,
                );
                t_bracket == bracket_id
            })
            .cloned()
            .collect();

        // 委托给对应括号的 SuccessiveHalvingPruner
        inner.pruners[bracket_id].prune(&bracket_trials, trial, storage)
    }

    /// 对齐 Python `_filter_study` / `_create_bracket_study`:
    /// 仅返回与给定试验同一括号的试验。
    fn filter_trials(&self, trials: &[FrozenTrial], trial: &FrozenTrial) -> Vec<FrozenTrial> {
        let inner = self.inner.lock().unwrap();
        if inner.trial_allocation_budgets.is_empty() {
            // 尚未初始化 → 返回全部（无法过滤）
            return trials.to_vec();
        }

        let bracket_id = self.get_bracket_id(
            trial.number,
            &self.study_name,
            &inner.trial_allocation_budgets,
        );

        trials
            .iter()
            .filter(|t| {
                let t_bracket = self.get_bracket_id(
                    t.number,
                    &self.study_name,
                    &inner.trial_allocation_budgets,
                );
                t_bracket == bracket_id
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_completed_trial(number: i64, max_step: i64) -> FrozenTrial {
        let mut iv = HashMap::new();
        for s in 0..=max_step {
            iv.insert(s, 1.0);
        }
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: Some(chrono::Utc::now()),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: number,
        }
    }

    fn make_running_trial(number: i64, step: i64, value: f64) -> FrozenTrial {
        let mut iv = HashMap::new();
        iv.insert(step, value);
        FrozenTrial {
            number,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: number,
        }
    }

    #[test]
    fn test_crc32_hash() {
        // 验证 CRC32 实现正确性
        assert_eq!(crc32_hash(b""), 0x00000000);
        assert_eq!(crc32_hash(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_no_completed_trials_auto() {
        // max_resource=None + 无已完成试验 → 不剪枝
        let pruner =
            HyperbandPruner::new(1, None, 3, 0, StudyDirection::Minimize, "test_study");
        let trial = make_running_trial(0, 5, 1.0);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_with_completed_trials() {
        // 有已完成试验时可以初始化
        let pruner =
            HyperbandPruner::new(1, Some(27), 3, 0, StudyDirection::Minimize, "test_study");
        let completed = make_completed_trial(0, 26);
        let trial = make_running_trial(1, 0, 1.0);
        // 不管结果如何，不应 panic
        let _ = pruner.prune(&[completed], &trial, None);
    }

    #[test]
    fn test_bracket_allocation() {
        let pruner =
            HyperbandPruner::new(1, Some(27), 3, 0, StudyDirection::Minimize, "test_study");
        // n_brackets = floor(log_3(27/1)) + 1 = 3 + 1 = 4
        let budget_0 = pruner.calculate_trial_allocation_budget(0, 4);
        let budget_1 = pruner.calculate_trial_allocation_budget(1, 4);
        let budget_2 = pruner.calculate_trial_allocation_budget(2, 4);
        let budget_3 = pruner.calculate_trial_allocation_budget(3, 4);
        assert!(budget_0 > 0);
        assert!(budget_1 > 0);
        assert!(budget_2 > 0);
        assert!(budget_3 > 0);
    }

    /// 对齐 Python: bootstrap_count > 0 + max_resource=None 互斥
    #[test]
    #[should_panic(expected = "mutually exclusive")]
    fn test_bootstrap_auto_incompatible() {
        HyperbandPruner::new(1, None, 3, 1, StudyDirection::Minimize, "test");
    }

    /// 对齐 Python: 重复调用 prune 不会重复初始化括号
    #[test]
    fn test_no_double_initialization() {
        let pruner =
            HyperbandPruner::new(1, Some(27), 3, 0, StudyDirection::Minimize, "test_study");
        let completed = make_completed_trial(0, 26);
        let trial = make_running_trial(1, 0, 1.0);
        // 第一次调用初始化
        let _ = pruner.prune(&[completed.clone()], &trial, None);
        // 第二次调用不应重复
        let _ = pruner.prune(&[completed.clone()], &trial, None);
        // 验证  pruners 数量正确：n_brackets = 4
        let inner = pruner.inner.lock().unwrap();
        assert_eq!(inner.pruners.len(), 4);
        assert_eq!(inner.trial_allocation_budgets.len(), 4);
    }

    /// 对齐 Python: max_resource=None 自动估计
    #[test]
    fn test_auto_max_resource() {
        let pruner =
            HyperbandPruner::new(1, None, 3, 0, StudyDirection::Minimize, "test_study");
        // 第一个已完成试验 max_step=26 → max_resource=27
        let completed = make_completed_trial(0, 26);
        let trial = make_running_trial(1, 0, 1.0);
        let _ = pruner.prune(&[completed], &trial, None);
        // 验证初始化成功：n_brackets = floor(log_3(27)) + 1 = 4
        let inner = pruner.inner.lock().unwrap();
        assert_eq!(inner.n_brackets, Some(4));
    }

    /// 对齐 Python: 多个试验被分配到不同括号
    #[test]
    fn test_different_brackets_for_different_trials() {
        let pruner =
            HyperbandPruner::new(1, Some(27), 3, 0, StudyDirection::Minimize, "test_study");
        let completed = make_completed_trial(0, 26);
        // 初始化
        let trial0 = make_running_trial(1, 0, 1.0);
        let _ = pruner.prune(&[completed.clone()], &trial0, None);
        // 获取预算分配
        let inner = pruner.inner.lock().unwrap();
        let budgets = inner.trial_allocation_budgets.clone();
        drop(inner);
        // 检查多个试验的括号分配
        let mut bracket_ids = std::collections::HashSet::new();
        for trial_num in 0..100 {
            let bid = pruner.get_bracket_id(trial_num, "test_study", &budgets);
            bracket_ids.insert(bid);
        }
        // 100 个试验应该分布在多个括号中
        assert!(bracket_ids.len() > 1, "试验应分布在多个括号中");
    }

    /// 对齐 Python: CRC32 与 Python binascii.crc32 一致
    #[test]
    fn test_crc32_cross_python() {
        // Python 交叉验证:
        // binascii.crc32(b"my_study_0") = 2668288816
        // binascii.crc32(b"my_study_1") = 3893226406
        // binascii.crc32(b"test_5") = 826366991
        // binascii.crc32(b"study_name_42") = 3374298732
        assert_eq!(crc32_hash(b"my_study_0"), 2668288816);
        assert_eq!(crc32_hash(b"my_study_1"), 3893226406);
        assert_eq!(crc32_hash(b"test_5"), 826366991);
        assert_eq!(crc32_hash(b"study_name_42"), 3374298732);
        // 标准 CRC32 测试向量
        assert_eq!(crc32_hash(b"123456789"), 0xCBF43926);
        assert_eq!(crc32_hash(b"test"), 0xD87F7E0C);
    }
}
