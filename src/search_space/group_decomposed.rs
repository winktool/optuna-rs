//! 分组分解搜索空间。
//!
//! 对应 Python `optuna.search_space._GroupDecomposedSearchSpace`。
//!
//! 处理条件/动态参数空间的分组。当不同试验使用不同的参数子集时，
//! 此模块将参数分解为独立的搜索空间组，每组内的参数总是共同出现。

use std::collections::{HashMap, HashSet};

use crate::distributions::Distribution;
use crate::trial::{FrozenTrial, TrialState};

/// 搜索空间组：管理多个独立的参数子空间。
///
/// 对应 Python `_SearchSpaceGroup`。
///
/// 每个组是一组总是一起出现的参数。不同试验可能激活不同的参数组。
#[derive(Debug, Clone)]
pub struct SearchSpaceGroup {
    /// 参数子空间列表，每个子空间是参数名到分布的映射
    search_spaces: Vec<HashMap<String, Distribution>>,
}

impl SearchSpaceGroup {
    /// 创建空的搜索空间组。
    pub fn new() -> Self {
        Self {
            search_spaces: Vec::new(),
        }
    }

    /// 获取所有搜索子空间。
    pub fn search_spaces(&self) -> &[HashMap<String, Distribution>] {
        &self.search_spaces
    }

    /// 添加一组分布，动态分裂或合并现有子空间。
    ///
    /// 对应 Python `_SearchSpaceGroup.add_distributions()`。
    ///
    /// 算法：
    /// 1. 对每个已有子空间，将其参数分为：与新分布交集部分、差集部分
    /// 2. 交集和差集各自成为新的子空间
    /// 3. 新分布中不属于任何已有子空间的参数组成额外子空间
    pub fn add_distributions(&mut self, distributions: &HashMap<String, Distribution>) {
        // 收集新分布的参数名集合
        let mut dist_keys: HashSet<String> = distributions.keys().cloned().collect();
        let mut next_search_spaces = Vec::new();

        // 遍历每个已有子空间
        for search_space in &self.search_spaces {
            let keys: HashSet<String> = search_space.keys().cloned().collect();

            // 与新分布的交集 → 保留共同出现的参数
            let intersection: HashMap<String, Distribution> = keys
                .intersection(&dist_keys)
                .filter_map(|name| search_space.get(name).map(|d| (name.clone(), d.clone())))
                .collect();

            // 已有子空间独有的参数 → 拆分为独立组
            let difference: HashMap<String, Distribution> = keys
                .difference(&dist_keys)
                .filter_map(|name| search_space.get(name).map(|d| (name.clone(), d.clone())))
                .collect();

            // 非空则加入
            if !intersection.is_empty() {
                next_search_spaces.push(intersection);
            }
            if !difference.is_empty() {
                next_search_spaces.push(difference);
            }

            // 从待处理集中移除已覆盖的参数
            dist_keys = dist_keys.difference(&keys).cloned().collect();
        }

        // 剩余的新参数组成额外子空间
        if !dist_keys.is_empty() {
            let remainder: HashMap<String, Distribution> = dist_keys
                .into_iter()
                .filter_map(|name| distributions.get(&name).map(|d| (name, d.clone())))
                .collect();
            next_search_spaces.push(remainder);
        }

        // 过滤掉空子空间
        self.search_spaces = next_search_spaces
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect();
    }
}

impl Default for SearchSpaceGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// 分组分解搜索空间。
///
/// 对应 Python `optuna.search_space._GroupDecomposedSearchSpace`。
///
/// 从研究的试验中自动将参数分组。当参数在某些试验中共同出现
/// 而在其他试验中不出现时，它们会被分到不同的组。
#[derive(Debug)]
pub struct GroupDecomposedSearchSpace {
    /// 内部搜索空间组
    search_space: SearchSpaceGroup,
    /// 绑定的研究 ID（用于检查一致性）
    study_id: Option<i64>,
    /// 是否包含被剪枝的试验
    include_pruned: bool,
}

impl GroupDecomposedSearchSpace {
    /// 创建新的分组分解搜索空间。
    ///
    /// # 参数
    /// * `include_pruned` - 是否包含 Pruned 状态的试验（默认 false）
    pub fn new(include_pruned: bool) -> Self {
        Self {
            search_space: SearchSpaceGroup::new(),
            study_id: None,
            include_pruned,
        }
    }

    /// 从研究的试验计算分组搜索空间。
    ///
    /// 返回深拷贝的 `SearchSpaceGroup`。
    ///
    /// # 参数
    /// * `study_id` - 研究 ID（用于一致性检查）
    /// * `trials` - 所有试验列表
    pub fn calculate(
        &mut self,
        study_id: i64,
        trials: &[FrozenTrial],
    ) -> SearchSpaceGroup {
        // 检查研究 ID 一致性
        if let Some(cached_id) = self.study_id {
            // 对齐 Python: raise ValueError 而非 panic
            if cached_id != study_id {
                panic!("GroupDecomposedSearchSpace 不能处理多个研究");
            }
        } else {
            self.study_id = Some(study_id);
        }

        // 确定感兴趣的状态
        for trial in trials {
            // 检查状态是否符合
            let is_of_interest = match trial.state {
                TrialState::Complete => true,
                TrialState::Pruned => self.include_pruned,
                _ => false,
            };
            if !is_of_interest {
                continue;
            }
            // 将此试验的分布添加到搜索空间组
            self.search_space.add_distributions(&trial.distributions);
        }

        // 返回深拷贝
        self.search_space.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{FloatDistribution, IntDistribution, CategoricalDistribution, CategoricalChoice};

    /// 辅助函数：创建浮点分布
    fn float_dist(low: f64, high: f64) -> Distribution {
        Distribution::FloatDistribution(FloatDistribution { low, high, log: false, step: None })
    }

    /// 辅助函数：创建整数分布
    fn int_dist(low: i64, high: i64) -> Distribution {
        Distribution::IntDistribution(IntDistribution { low, high, log: false, step: 1 })
    }

    /// 辅助函数：创建分类分布
    fn cat_dist(choices: Vec<&str>) -> Distribution {
        Distribution::CategoricalDistribution(CategoricalDistribution {
            choices: choices.into_iter().map(|s| CategoricalChoice::Str(s.to_string())).collect(),
        })
    }

    #[test]
    fn test_search_space_group_empty() {
        let group = SearchSpaceGroup::new();
        assert!(group.search_spaces().is_empty());
    }

    #[test]
    fn test_search_space_group_single_add() {
        let mut group = SearchSpaceGroup::new();
        let mut dists = HashMap::new();
        dists.insert("x".into(), float_dist(0.0, 1.0));
        dists.insert("y".into(), float_dist(-1.0, 1.0));
        group.add_distributions(&dists);
        // 第一次添加应产生一个组，包含 x 和 y
        assert_eq!(group.search_spaces().len(), 1);
        assert!(group.search_spaces()[0].contains_key("x"));
        assert!(group.search_spaces()[0].contains_key("y"));
    }

    #[test]
    fn test_search_space_group_split_on_subset() {
        let mut group = SearchSpaceGroup::new();
        // 第一次：{x, y, z}
        let mut dists1 = HashMap::new();
        dists1.insert("x".into(), float_dist(0.0, 1.0));
        dists1.insert("y".into(), float_dist(0.0, 1.0));
        dists1.insert("z".into(), float_dist(0.0, 1.0));
        group.add_distributions(&dists1);

        // 第二次：{x, y}（缺少 z）→ 应分裂为 {x,y} 和 {z}
        let mut dists2 = HashMap::new();
        dists2.insert("x".into(), float_dist(0.0, 1.0));
        dists2.insert("y".into(), float_dist(0.0, 1.0));
        group.add_distributions(&dists2);

        assert_eq!(group.search_spaces().len(), 2);
        // 找到包含 z 的组和包含 x,y 的组
        let has_z = group.search_spaces().iter().any(|s| s.contains_key("z") && s.len() == 1);
        let has_xy = group.search_spaces().iter().any(|s| s.contains_key("x") && s.contains_key("y"));
        assert!(has_z, "z 应在独立组中");
        assert!(has_xy, "x 和 y 应在同一组中");
    }

    #[test]
    fn test_search_space_group_new_params() {
        let mut group = SearchSpaceGroup::new();
        // 第一次：{x}
        let mut dists1 = HashMap::new();
        dists1.insert("x".into(), float_dist(0.0, 1.0));
        group.add_distributions(&dists1);

        // 第二次：{y}（全新参数）
        let mut dists2 = HashMap::new();
        dists2.insert("y".into(), float_dist(0.0, 1.0));
        group.add_distributions(&dists2);

        // 应有 2 个独立组
        assert_eq!(group.search_spaces().len(), 2);
    }

    #[test]
    fn test_search_space_group_complex_split() {
        let mut group = SearchSpaceGroup::new();
        // 试验 1: {a, b, c}
        let mut d1 = HashMap::new();
        d1.insert("a".into(), float_dist(0.0, 1.0));
        d1.insert("b".into(), float_dist(0.0, 1.0));
        d1.insert("c".into(), float_dist(0.0, 1.0));
        group.add_distributions(&d1);

        // 试验 2: {b, c, d} → 分裂为 {a}, {b,c}, {d}
        let mut d2 = HashMap::new();
        d2.insert("b".into(), float_dist(0.0, 1.0));
        d2.insert("c".into(), float_dist(0.0, 1.0));
        d2.insert("d".into(), float_dist(0.0, 1.0));
        group.add_distributions(&d2);

        // 试验 3: {a, d} → 已有 {a} 和 {d} 不变, {b,c} 不变
        let mut d3 = HashMap::new();
        d3.insert("a".into(), float_dist(0.0, 1.0));
        d3.insert("d".into(), float_dist(0.0, 1.0));
        group.add_distributions(&d3);

        // b 和 c 应在同一组中
        let bc_group = group.search_spaces().iter().find(|s| s.contains_key("b"));
        assert!(bc_group.is_some());
        assert!(bc_group.unwrap().contains_key("c"));
    }

    #[test]
    fn test_group_decomposed_search_space_basic() {
        let mut gdss = GroupDecomposedSearchSpace::new(false);

        // 模拟试验
        let trials = vec![
            FrozenTrial {
                number: 0,
                trial_id: 0,
                state: TrialState::Complete,
                values: Some(vec![1.0]),
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d.insert("y".into(), int_dist(0, 10));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
            FrozenTrial {
                number: 1,
                trial_id: 1,
                state: TrialState::Complete,
                values: Some(vec![2.0]),
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d.insert("z".into(), cat_dist(vec!["a", "b"]));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
        ];

        let result = gdss.calculate(1, &trials);
        // x 在两个试验中都出现，y 和 z 各在一个中
        // 应分为 {x}, {y}, {z} 三组
        assert_eq!(result.search_spaces().len(), 3);
    }

    #[test]
    fn test_group_decomposed_ignores_pruned_by_default() {
        let mut gdss = GroupDecomposedSearchSpace::new(false);

        let trials = vec![
            FrozenTrial {
                number: 0,
                trial_id: 0,
                state: TrialState::Complete,
                values: Some(vec![1.0]),
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
            FrozenTrial {
                number: 1,
                trial_id: 1,
                state: TrialState::Pruned,
                values: None,
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d.insert("y".into(), float_dist(0.0, 1.0));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
        ];

        let result = gdss.calculate(1, &trials);
        // Pruned 试验被忽略，只处理 Complete 的 {x}
        assert_eq!(result.search_spaces().len(), 1);
        assert!(result.search_spaces()[0].contains_key("x"));
        assert!(!result.search_spaces()[0].contains_key("y"));
    }

    #[test]
    fn test_group_decomposed_includes_pruned() {
        let mut gdss = GroupDecomposedSearchSpace::new(true);

        let trials = vec![
            FrozenTrial {
                number: 0,
                trial_id: 0,
                state: TrialState::Complete,
                values: Some(vec![1.0]),
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
            FrozenTrial {
                number: 1,
                trial_id: 1,
                state: TrialState::Pruned,
                values: None,
                datetime_start: None,
                datetime_complete: None,
                params: HashMap::new(),
                distributions: {
                    let mut d = HashMap::new();
                    d.insert("x".into(), float_dist(0.0, 1.0));
                    d.insert("y".into(), float_dist(0.0, 1.0));
                    d
                },
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
            },
        ];

        let result = gdss.calculate(1, &trials);
        // 包含 Pruned → 处理 {x} 和 {x,y} → 分裂为 {x} 和 {y}
        assert_eq!(result.search_spaces().len(), 2);
    }
}
