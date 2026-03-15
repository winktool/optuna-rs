//! Parameter importance analysis.
//!
//! Provides tools for evaluating which hyperparameters have the most
//! impact on objective values. This is useful for understanding search
//! spaces and pruning unimportant parameters.

use indexmap::IndexMap;

use crate::distributions::{CategoricalChoice, ParamValue};
use crate::error::{OptunaError, Result};
use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

/// Evaluator for computing parameter importance scores.
pub trait ImportanceEvaluator: Send + Sync {
    /// Evaluate the importance of each parameter.
    ///
    /// Returns a map from parameter name to importance score (0.0 to 1.0),
    /// ordered by decreasing importance. Scores are normalized to sum to 1.0.
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>>;
}

/// Functional ANOVA (fANOVA) importance evaluator.
///
/// Estimates parameter importance by computing between-group variance
/// of objective values when trials are grouped by discretized parameter
/// values. Parameters that produce large variance between groups are
/// considered more important.
pub struct FanovaEvaluator {
    /// Number of bins for discretizing continuous parameters.
    n_bins: usize,
}

impl Default for FanovaEvaluator {
    fn default() -> Self {
        Self { n_bins: 16 }
    }
}

impl FanovaEvaluator {
    /// Create a new evaluator with the given number of bins.
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }
}

impl ImportanceEvaluator for FanovaEvaluator {
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>> {
        if trials.is_empty() || params.is_empty() {
            return Ok(IndexMap::new());
        }

        let global_mean: f64 = target_values.iter().sum::<f64>() / target_values.len() as f64;

        let mut raw_importances: Vec<(String, f64)> = Vec::new();

        for param_name in params {
            // Collect (param_value, objective_value) pairs
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (i, trial) in trials.iter().enumerate() {
                if let Some(pv) = trial.params.get(param_name) {
                    let internal = param_value_to_f64(pv);
                    pairs.push((internal, target_values[i]));
                }
            }

            if pairs.is_empty() {
                raw_importances.push((param_name.clone(), 0.0));
                continue;
            }

            // Discretize into bins and compute between-group variance
            let importance = between_group_variance(&pairs, self.n_bins, global_mean);
            raw_importances.push((param_name.clone(), importance));
        }

        // Normalize importances to sum to 1.0
        let total: f64 = raw_importances.iter().map(|(_, v)| *v).sum();
        let mut result = IndexMap::new();

        if total > 0.0 {
            // Sort by importance descending
            raw_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (name, imp) in raw_importances {
                result.insert(name, imp / total);
            }
        } else {
            // All importances are zero — assign equal weight
            let uniform = 1.0 / params.len() as f64;
            for name in params {
                result.insert(name.clone(), uniform);
            }
        }

        Ok(result)
    }
}

/// Convert a ParamValue to f64 for importance computation.
/// 分类变量使用 choice index（而非 hash），对齐 Python 的 one-hot 策略中的顺序索引。
fn param_value_to_f64(pv: &ParamValue) -> f64 {
    match pv {
        ParamValue::Float(v) => *v,
        ParamValue::Int(v) => *v as f64,
        ParamValue::Categorical(c) => {
            // 使用 choice 的内部表示值（索引）而非 hash
            // CategoricalChoice 在分布的 choices 中有确定位置
            match c {
                CategoricalChoice::Int(v) => *v as f64,
                CategoricalChoice::Float(v) => *v,
                CategoricalChoice::Str(s) => {
                    // 字符串类型用确定性 hash （保持一致性）
                    use std::hash::{Hash, Hasher};
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    s.hash(&mut hasher);
                    hasher.finish() as f64
                }
                CategoricalChoice::Bool(b) => if *b { 1.0 } else { 0.0 },
                CategoricalChoice::None => f64::NAN,
            }
        }
    }
}

/// Compute between-group variance for a set of (param_value, objective_value) pairs.
///
/// Groups values into `n_bins` equally-spaced bins based on param_value,
/// then computes weighted variance of group means around the global mean.
fn between_group_variance(pairs: &[(f64, f64)], n_bins: usize, global_mean: f64) -> f64 {
    if pairs.len() <= 1 {
        return 0.0;
    }

    let min_val = pairs.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
    let max_val = pairs
        .iter()
        .map(|(v, _)| *v)
        .fold(f64::NEG_INFINITY, f64::max);

    // If all param values are the same, this parameter has no importance
    let range = max_val - min_val;
    if range < 1e-14 {
        return 0.0;
    }

    // Group into bins
    let mut bin_sums = vec![0.0_f64; n_bins];
    let mut bin_counts = vec![0_usize; n_bins];

    for &(param_val, obj_val) in pairs {
        let bin = ((param_val - min_val) / range * (n_bins as f64 - 1.0)).round() as usize;
        let bin = bin.min(n_bins - 1);
        bin_sums[bin] += obj_val;
        bin_counts[bin] += 1;
    }

    // Compute between-group variance: sum of n_k * (mean_k - global_mean)^2
    let n_total = pairs.len() as f64;
    let mut variance = 0.0;
    for k in 0..n_bins {
        if bin_counts[k] > 0 {
            let group_mean = bin_sums[k] / bin_counts[k] as f64;
            let diff = group_mean - global_mean;
            variance += (bin_counts[k] as f64 / n_total) * diff * diff;
        }
    }

    variance
}

// ============================================================================
// MeanDecreaseImpurity (MDI) 评估器 — 基于随机森林的特征重要性
// ============================================================================

/// Mean Decrease Impurity (MDI) 参数重要性评估器。
///
/// 对应 Python `optuna.importance.MeanDecreaseImpurityImportanceEvaluator`。
///
/// 使用随机森林回归模型拟合目标值，然后通过特征重要性（MDI）来计算参数重要性。
/// Python 版本使用 sklearn 的 RandomForestRegressor，Rust 版本实现了等效的纯 Rust 决策树和随机森林。
pub struct MeanDecreaseImpurityEvaluator {
    /// 随机森林中的决策树数量
    n_trees: usize,
    /// 每棵树的最大深度
    max_depth: usize,
    /// 节点分裂所需的最小样本数
    min_samples_split: usize,
    /// 叶子节点最少样本数
    min_samples_leaf: usize,
    /// 随机种子
    seed: Option<u64>,
}

impl MeanDecreaseImpurityEvaluator {
    /// 创建 MDI 评估器。
    ///
    /// # 参数
    /// * `n_trees` - 随机森林树数量（默认 64）
    /// * `max_depth` - 最大深度（默认 64）
    /// * `seed` - 随机种子
    pub fn new(n_trees: usize, max_depth: usize, seed: Option<u64>) -> Self {
        Self {
            n_trees,
            max_depth,
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed,
        }
    }
}

impl Default for MeanDecreaseImpurityEvaluator {
    fn default() -> Self {
        Self::new(64, 64, None)
    }
}

impl ImportanceEvaluator for MeanDecreaseImpurityEvaluator {
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>> {
        if trials.is_empty() || params.is_empty() {
            return Ok(IndexMap::new());
        }

        let n_features = params.len();

        // 构建特征矩阵 X[n_samples, n_features]\n        // 对齐 Python: 只保留包含所有指定参数的试验（过滤而非填0）
        let mut x_matrix: Vec<Vec<f64>> = Vec::new();
        let mut filtered_targets: Vec<f64> = Vec::new();
        for (i, trial) in trials.iter().enumerate() {
            let has_all = params.iter().all(|name| trial.params.contains_key(name));
            if !has_all {
                continue;
            }
            let mut row = Vec::with_capacity(n_features);
            for param_name in params {
                let val = trial.params.get(param_name).map(param_value_to_f64).unwrap();
                row.push(val);
            }
            x_matrix.push(row);
            filtered_targets.push(target_values[i]);
        }

        if x_matrix.is_empty() {
            return Ok(IndexMap::new());
        }

        // 使用随机森林计算特征重要性
        let importances = random_forest_feature_importances(
            &x_matrix,
            &filtered_targets,
            self.n_trees,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self.seed,
        );

        // 归一化并排序
        let total: f64 = importances.iter().sum();
        let mut result = IndexMap::new();
        if total > 0.0 {
            let mut indexed: Vec<(String, f64)> = params
                .iter()
                .zip(importances.iter())
                .map(|(name, &imp)| (name.clone(), imp / total))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (name, imp) in indexed {
                result.insert(name, imp);
            }
        } else {
            let uniform = 1.0 / params.len() as f64;
            for name in params {
                result.insert(name.clone(), uniform);
            }
        }

        Ok(result)
    }
}

// ── 随机森林与决策树实现 ──

/// 决策树节点。
enum TreeNode {
    /// 叶子节点：存储预测值和样本数
    Leaf {
        _value: f64,
        _n_samples: usize,
    },
    /// 内部节点：存储分裂特征、分裂阈值、左右子树、不纯度减少量
    Internal {
        feature: usize,
        _threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        impurity_decrease: f64,
        _n_samples: usize,
    },
}

/// 计算样本方差（MSE 不纯度）。
fn variance_impurity(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64
}

/// 在给定索引集上构建决策树。
fn build_tree(
    x: &[Vec<f64>],
    y: &[f64],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    feature_subset: &[usize],
    rng: &mut SimpleRng,
) -> TreeNode {
    let n = indices.len();
    // 收集当前节点的目标值
    let values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    let node_mean = values.iter().sum::<f64>() / n as f64;

    // 终止条件：达到最大深度、样本不足、方差为零
    if depth >= max_depth || n < min_samples_split || n <= 1 {
        return TreeNode::Leaf {
            _value: node_mean,
            _n_samples: n,
        };
    }

    let parent_impurity = variance_impurity(&values);
    if parent_impurity < 1e-14 {
        return TreeNode::Leaf {
            _value: node_mean,
            _n_samples: n,
        };
    }

    // 搜索最佳分裂
    let mut best_feature = 0;
    let mut best_threshold = 0.0;
    let mut best_impurity_decrease = f64::NEG_INFINITY;
    let mut best_left_indices = Vec::new();
    let mut best_right_indices = Vec::new();

    // 随机选择特征子集（sqrt(n_features)）
    let n_features_to_try = feature_subset.len();
    let mut features_to_try: Vec<usize> = feature_subset.to_vec();
    // Fisher-Yates shuffle 取前 max_features 个
    let max_features = (n_features_to_try as f64).sqrt().ceil() as usize;
    let max_features = max_features.max(1).min(n_features_to_try);
    for i in 0..max_features.min(features_to_try.len()) {
        let j = i + (rng.next_u64() as usize % (features_to_try.len() - i));
        features_to_try.swap(i, j);
    }

    for &feat_idx in features_to_try.iter().take(max_features) {
        // 收集此特征的所有值
        let mut feat_values: Vec<f64> = indices.iter().map(|&i| x[i][feat_idx]).collect();
        feat_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        feat_values.dedup();

        if feat_values.len() <= 1 {
            continue; // 特征值全部相同
        }

        // 尝试中点分裂
        for w in feat_values.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;

            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            for &i in indices {
                if x[i][feat_idx] <= threshold {
                    left_idx.push(i);
                } else {
                    right_idx.push(i);
                }
            }

            // 检查最小叶子样本约束
            if left_idx.len() < min_samples_leaf || right_idx.len() < min_samples_leaf {
                continue;
            }

            // 计算不纯度减少
            let left_values: Vec<f64> = left_idx.iter().map(|&i| y[i]).collect();
            let right_values: Vec<f64> = right_idx.iter().map(|&i| y[i]).collect();
            let left_impurity = variance_impurity(&left_values);
            let right_impurity = variance_impurity(&right_values);
            let weighted_impurity = (left_idx.len() as f64 * left_impurity
                + right_idx.len() as f64 * right_impurity)
                / n as f64;
            let impurity_decrease = parent_impurity - weighted_impurity;

            if impurity_decrease > best_impurity_decrease {
                best_impurity_decrease = impurity_decrease;
                best_feature = feat_idx;
                best_threshold = threshold;
                best_left_indices = left_idx;
                best_right_indices = right_idx;
            }
        }
    }

    // 无有效分裂则返回叶子
    if best_impurity_decrease <= 0.0 || best_left_indices.is_empty() || best_right_indices.is_empty()
    {
        return TreeNode::Leaf {
            _value: node_mean,
            _n_samples: n,
        };
    }

    // 递归构建子树
    let left = build_tree(
        x, y, &best_left_indices, depth + 1, max_depth,
        min_samples_split, min_samples_leaf, feature_subset, rng,
    );
    let right = build_tree(
        x, y, &best_right_indices, depth + 1, max_depth,
        min_samples_split, min_samples_leaf, feature_subset, rng,
    );

    TreeNode::Internal {
        feature: best_feature,
        _threshold: best_threshold,
        left: Box::new(left),
        right: Box::new(right),
        impurity_decrease: best_impurity_decrease * n as f64,
        _n_samples: n,
    }
}

/// 从决策树中提取每个特征的加权不纯度减少。
fn extract_feature_importances(node: &TreeNode, importances: &mut [f64]) {
    if let TreeNode::Internal {
        feature,
        left,
        right,
        impurity_decrease,
        ..
    } = node
    {
        importances[*feature] += *impurity_decrease;
        extract_feature_importances(left, importances);
        extract_feature_importances(right, importances);
    }
}

/// 简单的线性同余随机数生成器（避免引入额外依赖）。
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// 从 [0, n) 中均匀采样一个索引
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// 随机森林特征重要性计算。
///
/// 对应 sklearn `RandomForestRegressor.feature_importances_`。
fn random_forest_feature_importances(
    x: &[Vec<f64>],
    y: &[f64],
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    let n_samples = x.len();
    let n_features = if n_samples > 0 { x[0].len() } else { 0 };
    if n_samples == 0 || n_features == 0 {
        return vec![0.0; n_features];
    }

    let base_seed = seed.unwrap_or(42);
    let mut total_importances = vec![0.0_f64; n_features];
    let feature_indices: Vec<usize> = (0..n_features).collect();

    for tree_idx in 0..n_trees {
        let mut rng = SimpleRng::new(base_seed.wrapping_add(tree_idx as u64));

        // Bootstrap 采样（有放回）
        let bootstrap_indices: Vec<usize> = (0..n_samples)
            .map(|_| rng.next_usize(n_samples))
            .collect();

        // 构建决策树
        let tree = build_tree(
            x, y, &bootstrap_indices, 0, max_depth,
            min_samples_split, min_samples_leaf, &feature_indices, &mut rng,
        );

        // 提取特征重要性
        let mut tree_importances = vec![0.0_f64; n_features];
        extract_feature_importances(&tree, &mut tree_importances);

        // 归一化单棵树的重要性
        let tree_total: f64 = tree_importances.iter().sum();
        if tree_total > 0.0 {
            for imp in &mut tree_importances {
                *imp /= tree_total;
            }
        }

        // 累加到总重要性
        for (i, imp) in tree_importances.iter().enumerate() {
            total_importances[i] += imp;
        }
    }

    // 取所有树的平均值
    for imp in &mut total_importances {
        *imp /= n_trees as f64;
    }

    total_importances
}

/// Compute parameter importances for a study.
///
/// Returns a map from parameter name to importance score, ordered by
/// decreasing importance. Scores are normalized to sum to 1.0.
///
/// # Arguments
///
/// * `study` - The study to analyze.
/// * `evaluator` - The importance evaluator to use. Defaults to [`FanovaEvaluator`].
/// * `params` - Optional subset of parameter names to evaluate. If `None`,
///   all parameters from completed trials are used.
/// * `target` - Optional function to extract a scalar target value from each trial.
///   If `None`, uses the first objective value.
/// * `normalize` - Whether to normalize importances to sum to 1.0. Default: `true`.
pub fn get_param_importances(
    study: &Study,
    evaluator: Option<&dyn ImportanceEvaluator>,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    normalize: bool,
) -> Result<IndexMap<String, f64>> {
    let default_evaluator = FanovaEvaluator::default();
    let evaluator = evaluator.unwrap_or(&default_evaluator);

    let trials: Vec<FrozenTrial> = study
        .get_trials(Some(&[TrialState::Complete]))?
        .into_iter()
        .filter(|t| t.values.is_some())
        // 对齐 Python: 过滤 NaN/Inf 目标值
        .filter(|t| {
            let val = target.map_or_else(
                || t.values.as_ref().unwrap()[0],
                |f| f(t),
            );
            val.is_finite()
        })
        .collect();

    if trials.is_empty() {
        return Err(OptunaError::ValueError(
            "study has no completed trials with finite target values".into(),
        ));
    }

    // 对齐 Python: 至少需要 2 个试验
    if trials.len() == 1 {
        return Err(OptunaError::ValueError(
            "Cannot evaluate parameter importances with only a single trial.".into(),
        ));
    }

    // Collect parameter names: 对齐 Python 使用 intersection_search_space（取交集）
    let param_names: Vec<String> = if let Some(names) = params {
        names.iter().map(|s| s.to_string()).collect()
    } else {
        // 取所有已完成试验的参数交集（而非并集）
        if trials.is_empty() {
            Vec::new()
        } else {
            let first_params: std::collections::HashSet<&String> =
                trials[0].params.keys().collect();
            let mut intersection: std::collections::HashSet<String> =
                first_params.iter().map(|s| s.to_string()).collect();
            for trial in &trials[1..] {
                let trial_params: std::collections::HashSet<String> =
                    trial.params.keys().cloned().collect();
                intersection = intersection.intersection(&trial_params).cloned().collect();
            }
            let mut sorted: Vec<String> = intersection.into_iter().collect();
            sorted.sort();
            sorted
        }
    };

    if param_names.is_empty() {
        return Ok(IndexMap::new());
    }

    // Extract target values using the provided function or default (first objective)
    let target_values: Vec<f64> = trials
        .iter()
        .map(|t| {
            target.map_or_else(
                || t.values.as_ref().unwrap()[0],
                |f| f(t),
            )
        })
        .collect();

    let mut importances = evaluator.evaluate(&trials, &param_names, &target_values)?;

    // Normalize if requested
    if normalize {
        let total: f64 = importances.values().sum();
        if total > 0.0 {
            for val in importances.values_mut() {
                *val /= total;
            }
        }
    }

    Ok(importances)
}

// ============================================================================
// PED-ANOVA 重要性评估器 — 基于 Pearson 散度 + Parzen 估计
// ============================================================================

/// PED-ANOVA 参数重要性评估器。
///
/// 对应 Python `optuna.importance.PedAnovaImportanceEvaluator`。
///
/// 使用分位数过滤 + Scott-Parzen 估计器 + Pearson χ² 散度来评估参数重要性。
/// 核心思想：如果某个参数对目标值有影响，则"好"试验和"全部"试验在该参数上的分布应有显著差异。
pub struct PedAnovaEvaluator {
    /// 目标分位数 γ'（取前 target_quantile 比例的好试验）
    target_quantile: f64,
    /// 区域分位数 γ（用于局部评估的试验范围，1.0 表示全部）
    region_quantile: f64,
    /// 是否在局部分布上评估（true 使用 region_trials 构建基准分布）
    evaluate_on_local: bool,
    /// 离散化步数
    n_steps: usize,
    /// 先验权重
    prior_weight: f64,
    /// 最少 top 试验数
    min_n_top_trials: usize,
    /// 是否小值更好（对齐 Python `is_lower_better`）
    /// 默认 true（最小化）；最大化研究应设为 false
    pub is_lower_better: bool,
}

impl PedAnovaEvaluator {
    /// 创建 PED-ANOVA 评估器。
    ///
    /// # 参数
    /// * `target_quantile` - 目标分位数（默认 0.1）
    /// * `region_quantile` - 区域分位数（默认 1.0）
    /// * `evaluate_on_local` - 是否局部评估（默认 true）
    pub fn new(
        target_quantile: Option<f64>,
        region_quantile: Option<f64>,
        evaluate_on_local: Option<bool>,
    ) -> Self {
        let tq = target_quantile.unwrap_or(0.1);
        let rq = region_quantile.unwrap_or(1.0);
        assert!(tq > 0.0 && tq < rq, "需要 0 < target_quantile < region_quantile");
        assert!(rq <= 1.0, "region_quantile <= 1.0");
        Self {
            target_quantile: tq,
            region_quantile: rq,
            evaluate_on_local: evaluate_on_local.unwrap_or(true),
            n_steps: 50,
            prior_weight: 1.0,
            min_n_top_trials: 2,
            is_lower_better: true,
        }
    }
}

impl Default for PedAnovaEvaluator {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

/// 分位数过滤：取目标值前 quantile 比例的试验。
/// 最少保留 min_n_top 个。
fn quantile_filter(
    values: &[f64],
    quantile: f64,
    is_lower_better: bool,
    min_n_top: usize,
) -> Vec<usize> {
    let n = values.len();
    if n == 0 { return vec![]; }

    // 转换为 loss（越小越好）
    let losses: Vec<f64> = if is_lower_better {
        values.to_vec()
    } else {
        values.iter().map(|v| -v).collect()
    };

    // 排序获取阈值
    let mut sorted = losses.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // quantile 位置处的值
    let q_idx = ((n as f64 * quantile).ceil() as usize).max(min_n_top).min(n);
    let cutoff = sorted[q_idx - 1];

    // 选择 loss <= cutoff 的试验索引
    let mut indices: Vec<usize> = (0..n)
        .filter(|&i| losses[i] <= cutoff)
        .collect();

    // 如果太少，按 loss 排序取前 min_n_top 个
    if indices.len() < min_n_top {
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| losses[a].partial_cmp(&losses[b]).unwrap_or(std::cmp::Ordering::Equal));
        indices = sorted_idx[..min_n_top.min(n)].to_vec();
    }

    indices
}

/// Scott's Rule 带宽的离散截断正态 Parzen 估计器。
///
/// 对应 Python `_ScottParzenEstimator`。
struct ScottParzenEstimator {
    /// 各核心的均值（grid 索引）
    mus: Vec<f64>,
    /// 各核心的标准差
    sigmas: Vec<f64>,
    /// 各核心的权重
    weights: Vec<f64>,
    /// 离散域下界
    low: f64,
    /// 离散域上界
    high: f64,
}

impl ScottParzenEstimator {
    /// 从计数向量构建 Scott-Parzen 估计器。
    ///
    /// # 参数
    /// * `counts` - 每个 grid 点的样本计数
    /// * `n_steps` - grid 点总数
    /// * `prior_weight` - 先验均匀分布的权重
    fn from_counts(counts: &[usize], n_steps: usize, prior_weight: f64) -> Self {
        let low = 0.0_f64;
        let high = (n_steps - 1) as f64;
        let n_total: usize = counts.iter().sum();

        // 非零计数的 grid 点
        let mut mus = Vec::new();
        let mut counts_nz = Vec::new();
        for (i, &c) in counts.iter().enumerate() {
            if c > 0 {
                mus.push(i as f64);
                counts_nz.push(c as f64);
            }
        }

        let sigma = if mus.is_empty() || n_total <= 1 {
            // 只有先验
            1.0 * (high - low + 1.0)
        } else {
            // Scott's rule 带宽
            let n = n_total as f64;
            let weights: Vec<f64> = counts_nz.iter().map(|c| c / n).collect();

            // 加权均值
            let mean_est: f64 = mus.iter().zip(weights.iter()).map(|(m, w)| m * w).sum();

            // 加权标准差（样本方差）
            let var_est: f64 = mus.iter().zip(counts_nz.iter())
                .map(|(m, c)| (m - mean_est).powi(2) * c)
                .sum::<f64>() / (n - 1.0).max(1.0);
            let sigma_est = var_est.sqrt();

            // IQR 计算
            let cum: Vec<f64> = {
                let mut c = Vec::with_capacity(counts_nz.len());
                let mut acc = 0.0;
                for &cnt in &counts_nz {
                    acc += cnt;
                    c.push(acc);
                }
                c
            };
            let q25_target = n / 4.0;
            let q75_target = n * 3.0 / 4.0;
            let idx_q25 = cum.iter().position(|&c| c >= q25_target).unwrap_or(0);
            let idx_q75 = cum.iter().position(|&c| c >= q75_target).unwrap_or(mus.len() - 1);
            let iqr = mus[idx_q75.min(mus.len() - 1)] - mus[idx_q25];

            // Scott: h = 1.059 * min(IQR/1.34, σ) * n^{-0.2}
            let sigma_choice = if iqr > 0.0 {
                (iqr / 1.34).min(sigma_est)
            } else {
                sigma_est
            };
            let h = 1.059 * sigma_choice * n.powf(-0.2);

            // 最小带宽: 确保 90% 的核质量落在 [0, n_steps-1] 的一个 grid 内
            let sigma_min = 0.5 / 1.64;
            h.max(sigma_min)
        };

        // 为每个非零 grid 点分配带宽
        let mut sigmas: Vec<f64> = vec![sigma; mus.len()];

        // 加上均匀先验核
        let prior_mu = (low + high) / 2.0;
        let prior_sigma = 1.0 * (high - low + 1.0);
        mus.push(prior_mu);
        sigmas.push(prior_sigma);

        // 权重：非零计数 + 先验权重
        let mut all_weights: Vec<f64> = counts_nz.clone();
        all_weights.push(prior_weight);
        let w_sum: f64 = all_weights.iter().sum();
        let weights: Vec<f64> = all_weights.iter().map(|w| w / w_sum).collect();

        Self { mus, sigmas, weights, low, high }
    }

    /// 在 grid 点上计算 PDF。
    ///
    /// 返回长度为 n_steps 的 PDF 向量。
    fn pdf(&self, n_steps: usize) -> Vec<f64> {
        let mut result = vec![0.0_f64; n_steps];

        for (k, ((mu, sigma), w)) in self.mus.iter()
            .zip(self.sigmas.iter())
            .zip(self.weights.iter())
            .enumerate()
        {
            let _ = k; // 避免未使用警告
            // 离散截断正态的 PDF
            // P(x) = φ((x - μ) / σ) / (Φ((high - μ) / σ) - Φ((low - μ) / σ)) for x in Z ∩ [low, high]
            let norm_cdf_high = crate::samplers::gp::normal_cdf((self.high - mu) / sigma);
            let norm_cdf_low = crate::samplers::gp::normal_cdf((self.low - mu) / sigma);
            let denom = (norm_cdf_high - norm_cdf_low).max(1e-14);

            for x in 0..n_steps {
                let z = (x as f64 - mu) / sigma;
                let p = crate::samplers::gp::normal_pdf(z) / sigma / denom;
                result[x] += w * p;
            }
        }

        // 归一化使 PDF 和为 1
        let total: f64 = result.iter().sum();
        if total > 1e-14 {
            for v in &mut result {
                *v /= total;
            }
        }

        result
    }
}

/// 将参数值离散化到 grid 点。
///
/// 返回 (grid_indices, n_steps_actual) — 每个试验的 grid 索引和实际使用的步数。
fn discretize_param(
    values: &[f64],
    low: f64,
    high: f64,
    n_steps: usize,
    is_log: bool,
) -> (Vec<usize>, usize) {
    if (high - low).abs() < 1e-14 {
        return (vec![0; values.len()], 1);
    }

    // 对 log 域进行变换
    let (s_low, s_high) = if is_log {
        (low.max(1e-300).ln(), high.max(1e-300).ln())
    } else {
        (low, high)
    };

    let grids: Vec<f64> = (0..n_steps)
        .map(|i| s_low + (s_high - s_low) * i as f64 / (n_steps - 1).max(1) as f64)
        .collect();

    let indices: Vec<usize> = values.iter().map(|&v| {
        let sv = if is_log { v.max(1e-300).ln() } else { v };
        // 找最近的 grid 点
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, &g) in grids.iter().enumerate() {
            let d = (sv - g).abs();
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        best_idx
    }).collect();

    (indices, n_steps)
}

/// 统计每个 grid 点的计数
fn count_in_grid(indices: &[usize], n_steps: usize) -> Vec<usize> {
    let mut counts = vec![0usize; n_steps];
    for &idx in indices {
        if idx < n_steps {
            counts[idx] += 1;
        }
    }
    counts
}

/// 计算 Pearson χ² 散度: D_χ²(p || q) = Σ q(x) * ((p(x)/q(x)) - 1)²
fn pearson_divergence(pdf_p: &[f64], pdf_q: &[f64]) -> f64 {
    let eps = 1e-12;
    pdf_p.iter().zip(pdf_q.iter()).map(|(&p, &q)| {
        let q_safe = q + eps;
        let p_safe = p + eps;
        q_safe * ((p_safe / q_safe) - 1.0).powi(2)
    }).sum()
}

impl ImportanceEvaluator for PedAnovaEvaluator {
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>> {
        if trials.is_empty() || params.is_empty() {
            return Ok(IndexMap::new());
        }

        let is_lower_better = self.is_lower_better;

        // 区域过滤：取前 region_quantile 的试验
        let region_indices = quantile_filter(
            target_values,
            self.region_quantile,
            is_lower_better,
            self.min_n_top_trials,
        );
        if region_indices.len() <= self.min_n_top_trials {
            // 不够试验，返回全 0
            let mut result = IndexMap::new();
            for name in params {
                result.insert(name.clone(), 0.0);
            }
            return Ok(result);
        }

        let region_values: Vec<f64> = region_indices.iter().map(|&i| target_values[i]).collect();

        // 目标过滤：从 region 中取前 target_quantile 的试验
        let target_indices_in_region = quantile_filter(
            &region_values,
            self.target_quantile / self.region_quantile,
            is_lower_better,
            self.min_n_top_trials,
        );

        // 映射回全局索引
        let target_indices: Vec<usize> = target_indices_in_region.iter()
            .map(|&i| region_indices[i])
            .collect();

        let quantile_ratio = target_indices.len() as f64 / region_indices.len() as f64;

        let mut raw_importances: Vec<(String, f64)> = Vec::new();

        for param_name in params {
            // 收集参数值
            let target_vals: Vec<f64> = target_indices.iter()
                .filter_map(|&i| {
                    trials[i].params.get(param_name).map(|pv| param_value_to_f64(pv))
                })
                .collect();
            let region_vals: Vec<f64> = region_indices.iter()
                .filter_map(|&i| {
                    trials[i].params.get(param_name).map(|pv| param_value_to_f64(pv))
                })
                .collect();

            if target_vals.is_empty() || region_vals.is_empty() {
                raw_importances.push((param_name.clone(), 0.0));
                continue;
            }

            // 确定参数范围
            let all_vals: Vec<f64> = trials.iter()
                .filter_map(|t| t.params.get(param_name).map(|pv| param_value_to_f64(pv)))
                .collect();
            let low = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let high = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // 判断是否需要 log 变换
            let is_log = trials.iter()
                .find_map(|t| t.distributions.get(param_name))
                .map(|d| matches!(d, crate::distributions::Distribution::FloatDistribution(fd) if fd.log))
                .unwrap_or(false);

            // 离散化
            let (target_grid_idx, n_steps_actual) = discretize_param(&target_vals, low, high, self.n_steps, is_log);
            let target_counts = count_in_grid(&target_grid_idx, n_steps_actual);

            // 构建 target Parzen 估计器
            let pe_top = ScottParzenEstimator::from_counts(&target_counts, n_steps_actual, self.prior_weight);
            let pdf_top = pe_top.pdf(n_steps_actual);

            // 构建基准分布
            let pdf_base = if self.evaluate_on_local {
                let (region_grid_idx, _) = discretize_param(&region_vals, low, high, self.n_steps, is_log);
                let region_counts = count_in_grid(&region_grid_idx, n_steps_actual);
                let pe_local = ScottParzenEstimator::from_counts(&region_counts, n_steps_actual, self.prior_weight);
                pe_local.pdf(n_steps_actual)
            } else {
                // 均匀分布
                vec![1.0 / n_steps_actual as f64; n_steps_actual]
            };

            // 计算 Pearson 散度
            let divergence = pearson_divergence(&pdf_top, &pdf_base);
            let importance = quantile_ratio * quantile_ratio * divergence;

            raw_importances.push((param_name.clone(), importance));
        }

        // 归一化
        let total: f64 = raw_importances.iter().map(|(_, v)| *v).sum();
        let mut result = IndexMap::new();

        if total > 0.0 {
            raw_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (name, imp) in raw_importances {
                result.insert(name, imp / total);
            }
        } else {
            let uniform = 1.0 / params.len() as f64;
            for name in params {
                result.insert(name.clone(), uniform);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::{create_study, StudyDirection};
    use std::sync::Arc;

    #[test]
    fn test_fanova_evaluator_basic() {
        let evaluator = FanovaEvaluator::default();
        assert_eq!(evaluator.n_bins, 16);
    }

    #[test]
    fn test_fanova_evaluator_custom_bins() {
        let evaluator = FanovaEvaluator::new(8);
        assert_eq!(evaluator.n_bins, 8);
    }

    #[test]
    fn test_fanova_empty_trials() {
        let evaluator = FanovaEvaluator::default();
        let result = evaluator.evaluate(&[], &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_param_importances_quadratic() {
        let sampler: Arc<dyn crate::samplers::Sampler> =
            Arc::new(RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        // f(x, y) = x^2 + 0.01*y: x is much more important than y
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x + 0.01 * y)
                },
                Some(100),
                None,
                None,
            )
            .unwrap();

        let importances = get_param_importances(&study, None, None, None, true).unwrap();
        assert_eq!(importances.len(), 2);

        // Importances should sum to ~1.0
        let total: f64 = importances.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "importances should sum to 1.0, got {total}"
        );

        // x should be more important than y
        let x_imp = importances["x"];
        let y_imp = importances["y"];
        assert!(
            x_imp > y_imp,
            "x importance ({x_imp}) should be > y importance ({y_imp})"
        );
    }

    #[test]
    fn test_get_param_importances_with_subset() {
        let sampler: Arc<dyn crate::samplers::Sampler> =
            Arc::new(RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let _y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        // Only evaluate importance for "x"
        let importances =
            get_param_importances(&study, None, Some(&["x"]), None, true).unwrap();
        assert_eq!(importances.len(), 1);
        assert!(importances.contains_key("x"));
        assert!((importances["x"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_param_importances_no_completed_trials() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        let result = get_param_importances(&study, None, None, None, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_between_group_variance_identical() {
        // All same param value => zero importance
        let pairs = vec![(1.0, 2.0), (1.0, 3.0), (1.0, 4.0)];
        let v = between_group_variance(&pairs, 8, 3.0);
        assert!((v - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_between_group_variance_distinct() {
        // Two clearly separated groups
        let mut pairs = Vec::new();
        for _ in 0..10 {
            pairs.push((0.0, 1.0)); // group 1: low param, low obj
        }
        for _ in 0..10 {
            pairs.push((10.0, 100.0)); // group 2: high param, high obj
        }
        let global_mean = 50.5;
        let v = between_group_variance(&pairs, 8, global_mean);
        assert!(v > 0.0, "variance should be positive for distinct groups");
    }

    #[test]
    fn test_importance_three_params() {
        let sampler: Arc<dyn crate::samplers::Sampler> =
            Arc::new(RandomSampler::new(Some(123)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        // f(x, y, z) = 10*x^2 + y^2 + 0.001*z
        // Importance: x >> y >> z
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                    let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                    Ok(10.0 * x * x + y * y + 0.001 * z)
                },
                Some(200),
                None,
                None,
            )
            .unwrap();

        let importances = get_param_importances(&study, None, None, None, true).unwrap();
        assert_eq!(importances.len(), 3);

        // First key should be x (most important)
        let first_key = importances.keys().next().unwrap();
        assert_eq!(first_key, "x", "x should be most important");
    }

    // ── PedAnova 测试 ──

    #[test]
    fn test_ped_anova_creation() {
        let _eval = PedAnovaEvaluator::default();
        let _eval2 = PedAnovaEvaluator::new(Some(0.2), Some(0.8), Some(false));
    }

    #[test]
    fn test_ped_anova_empty() {
        let eval = PedAnovaEvaluator::default();
        let result = eval.evaluate(&[], &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_ped_anova_basic() {
        let eval = PedAnovaEvaluator::new(Some(0.3), Some(1.0), Some(true));
        let sampler: Arc<dyn crate::samplers::Sampler> =
            Arc::new(RandomSampler::new(Some(42)));
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        // f(x, y) = 10*x^2 + 0.001*y → x 远重要于 y
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(10.0 * x * x + 0.001 * y)
            },
            Some(100),
            None,
            None,
        ).unwrap();

        let importances = get_param_importances(
            &study,
            Some(&eval as &dyn ImportanceEvaluator),
            None,
            None,
            true,
        ).unwrap();
        assert_eq!(importances.len(), 2);

        // x 应比 y 更重要
        let imp_x = importances.get("x").unwrap();
        let imp_y = importances.get("y").unwrap();
        assert!(imp_x > imp_y, "x 重要性 ({imp_x}) 应大于 y ({imp_y})");
    }

    #[test]
    fn test_pearson_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let d = pearson_divergence(&p, &q);
        assert!(d < 1e-6, "相同分布的散度应接近 0: {d}");
    }

    #[test]
    fn test_pearson_divergence_different() {
        let p = vec![0.9, 0.03, 0.03, 0.04];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        let d = pearson_divergence(&p, &q);
        assert!(d > 0.0, "不同分布的散度应为正: {d}");
    }

    #[test]
    fn test_quantile_filter() {
        let values = vec![10.0, 3.0, 7.0, 1.0, 5.0];
        // 取前 40%（2 个），最小化时选最小的
        let indices = quantile_filter(&values, 0.4, true, 1);
        assert!(!indices.is_empty());
        // 应包含值为 1.0 和 3.0 的索引
        assert!(indices.contains(&3)); // value=1.0
        assert!(indices.contains(&1)); // value=3.0
    }

    #[test]
    fn test_scott_parzen_estimator() {
        let counts = vec![5, 0, 3, 0, 2];
        let pe = ScottParzenEstimator::from_counts(&counts, 5, 1.0);
        let pdf = pe.pdf(5);
        assert_eq!(pdf.len(), 5);
        let total: f64 = pdf.iter().sum();
        assert!((total - 1.0).abs() < 0.01, "PDF 总和应接近 1: {total}");
    }
}
