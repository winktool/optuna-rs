#!/usr/bin/env python3
"""
Importance 模块深度交叉验证 — 生成 golden values。

覆盖:
  1. weighted_variance — 加权方差
  2. FanovaTree — 方差分解 (total variance, marginal variance)
  3. ScottParzenEstimator — Scott's rule 带宽 + 截断正态 PDF
  4. QuantileFilter — 分位数过滤
  5. Pearson divergence — χ² 散度
  6. PedAnovaImportanceEvaluator._compute_pearson_divergence — 端到端
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../optuna"))

from optuna.importance._fanova._tree import _FanovaTree, _get_cardinality
from optuna.importance._ped_anova.evaluator import _QuantileFilter
from optuna.importance._ped_anova.scott_parzen_estimator import (
    _ScottParzenEstimator,
    _count_numerical_param_in_grid,
    _get_grids_and_grid_indices_of_trials,
)
from optuna.distributions import IntDistribution, FloatDistribution


def sanitize(v):
    """将 NaN/Inf 替换为 JSON 安全值。"""
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return v
    if isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    if isinstance(v, np.ndarray):
        return sanitize(v.tolist())
    if isinstance(v, dict):
        return {k: sanitize(vv) for k, vv in v.items()}
    if isinstance(v, (np.float64, np.float32)):
        return sanitize(float(v))
    if isinstance(v, (np.int64, np.int32)):
        return int(v)
    return v


results = {}


# ═══════════════════════════════════════════════════════════════════════
# Group 1: weighted_variance
# ═══════════════════════════════════════════════════════════════════════

def weighted_variance(values, weights):
    """Python 版 weighted variance (对齐 _FanovaTree.variance 计算方式)。"""
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return float(var)


cases = [
    {"values": [1.0, 3.0], "weights": [0.5, 0.5]},
    {"values": [1.0, 2.0, 4.0], "weights": [0.5, 0.15, 0.35]},
    {"values": [5.0, 5.0, 5.0], "weights": [1.0, 2.0, 3.0]},
    {"values": [0.0, 10.0], "weights": [0.9, 0.1]},
    {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "weights": [0.2, 0.2, 0.2, 0.2, 0.2]},
]

wv_results = []
for c in cases:
    var = weighted_variance(c["values"], c["weights"])
    wv_results.append({
        "values": c["values"],
        "weights": c["weights"],
        "variance": var,
    })

results["weighted_variance"] = wv_results
print("Group 1: weighted_variance — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 2: FanovaTree — variance decomposition
# ═══════════════════════════════════════════════════════════════════════

class MockTree:
    """模拟 sklearn 决策树结构，用于测试 _FanovaTree。"""
    def __init__(self, feature, threshold, children_left, children_right, value):
        self.feature = np.array(feature, dtype=np.intp)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.children_left = np.array(children_left, dtype=np.intp)
        self.children_right = np.array(children_right, dtype=np.intp)
        # sklearn tree.value shape: (node_count, n_outputs, max_n_classes)
        self.value = np.array(value, dtype=np.float64).reshape(-1, 1, 1)
        self.node_count = len(feature)
        self.n_features = max(f for f in feature if f >= 0) + 1 if any(f >= 0 for f in feature) else 1


# Tree A: 单特征, 2 叶子
# Root(feat=0, thr=0.5) → Left(leaf, val=1.0), Right(leaf, val=3.0)
# search_spaces=[[0,1]]
tree_a = MockTree(
    feature=[0, -1, -1],
    threshold=[0.5, -1, -1],
    children_left=[1, -1, -1],
    children_right=[2, -1, -1],
    value=[0.0, 1.0, 3.0],
)
ss_a = np.array([[0.0, 1.0]])
ftree_a = _FanovaTree(tree_a, ss_a)
var_a = ftree_a.variance
mv_a_0 = ftree_a.get_marginal_variance(np.array([0]))

# Tree B: 两个特征, 3 叶子
# Root(feat=0, thr=0.5)
#   Left: leaf, val=1.0
#   Right(feat=1, thr=0.3)
#     Left: leaf, val=2.0
#     Right: leaf, val=4.0
# search_spaces=[[0,1],[0,1]]
tree_b = MockTree(
    feature=[0, -1, 1, -1, -1],
    threshold=[0.5, -1, 0.3, -1, -1],
    children_left=[1, -1, 3, -1, -1],
    children_right=[2, -1, 4, -1, -1],
    value=[0.0, 1.0, 0.0, 2.0, 4.0],
)
ss_b = np.array([[0.0, 1.0], [0.0, 1.0]])
ftree_b = _FanovaTree(tree_b, ss_b)
var_b = ftree_b.variance
mv_b_0 = ftree_b.get_marginal_variance(np.array([0]))
mv_b_1 = ftree_b.get_marginal_variance(np.array([1]))
mv_b_01 = ftree_b.get_marginal_variance(np.array([0, 1]))

# Tree C: 三个特征, 深度 3, 7 节点
# Root(feat=0, thr=0.5)
#   Left(feat=1, thr=0.4)
#     LL: leaf val=-1.0
#     LR: leaf val=2.0
#   Right(feat=2, thr=0.6)
#     RL: leaf val=3.0
#     RR: leaf val=5.0
# search_spaces=[[0,1],[0,1],[0,1]]
tree_c = MockTree(
    feature=[0, 1, -1, -1, 2, -1, -1],
    threshold=[0.5, 0.4, -1, -1, 0.6, -1, -1],
    children_left=[1, 2, -1, -1, 5, -1, -1],
    children_right=[4, 3, -1, -1, 6, -1, -1],
    value=[0.0, 0.0, -1.0, 2.0, 0.0, 3.0, 5.0],
)
ss_c = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
ftree_c = _FanovaTree(tree_c, ss_c)
var_c = ftree_c.variance
mv_c_0 = ftree_c.get_marginal_variance(np.array([0]))
mv_c_1 = ftree_c.get_marginal_variance(np.array([1]))
mv_c_2 = ftree_c.get_marginal_variance(np.array([2]))

results["fanova_tree"] = {
    "tree_a": {
        "description": "单特征2叶子: root splits feat=0 at 0.5, leaves=[1.0, 3.0], ss=[[0,1]]",
        "total_variance": var_a,
        "marginal_variance_feat0": mv_a_0,
    },
    "tree_b": {
        "description": "两特征3叶子: root splits feat=0@0.5, right splits feat=1@0.3, leaves=[1,2,4], ss=[[0,1],[0,1]]",
        "total_variance": var_b,
        "marginal_variance_feat0": mv_b_0,
        "marginal_variance_feat1": mv_b_1,
        "marginal_variance_feat01": mv_b_01,
    },
    "tree_c": {
        "description": "三特征4叶子深度3: root@feat0=0.5, left@feat1=0.4, right@feat2=0.6, leaves=[-1,2,3,5]",
        "total_variance": var_c,
        "marginal_variance_feat0": mv_c_0,
        "marginal_variance_feat1": mv_c_1,
        "marginal_variance_feat2": mv_c_2,
    },
}
print("Group 2: FanovaTree — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 3: ScottParzenEstimator — PDF 计算
# ═══════════════════════════════════════════════════════════════════════

# 为了直接测试 Scott-Parzen 估计器, 构造 counts 并计算 PDF

# Case 1: uniform counts [2, 2, 2, 2, 2]
counts_1 = np.array([2, 2, 2, 2, 2], dtype=np.float64)
dist_1 = IntDistribution(low=0, high=4)
pe_1 = _ScottParzenEstimator("x", dist_1, counts_1, prior_weight=1.0)
pdf_1 = pe_1.pdf(np.arange(5))

# Case 2: peaked counts [10, 1, 0, 0, 0]
counts_2 = np.array([10, 1, 0, 0, 0], dtype=np.float64)
dist_2 = IntDistribution(low=0, high=4)
pe_2 = _ScottParzenEstimator("x", dist_2, counts_2, prior_weight=1.0)
pdf_2 = pe_2.pdf(np.arange(5))

# Case 3: bimodal [5, 0, 0, 0, 5]
counts_3 = np.array([5, 0, 0, 0, 5], dtype=np.float64)
dist_3 = IntDistribution(low=0, high=4)
pe_3 = _ScottParzenEstimator("x", dist_3, counts_3, prior_weight=1.0)
pdf_3 = pe_3.pdf(np.arange(5))

# Case 4: single peak [0, 0, 8, 0, 0]
counts_4 = np.array([0, 0, 8, 0, 0], dtype=np.float64)
dist_4 = IntDistribution(low=0, high=4)
pe_4 = _ScottParzenEstimator("x", dist_4, counts_4, prior_weight=1.0)
pdf_4 = pe_4.pdf(np.arange(5))

results["scott_parzen_pdf"] = {
    "case_uniform": {
        "counts": [2, 2, 2, 2, 2],
        "pdf": pdf_1.tolist(),
    },
    "case_peaked": {
        "counts": [10, 1, 0, 0, 0],
        "pdf": pdf_2.tolist(),
    },
    "case_bimodal": {
        "counts": [5, 0, 0, 0, 5],
        "pdf": pdf_3.tolist(),
    },
    "case_single_peak": {
        "counts": [0, 0, 8, 0, 0],
        "pdf": pdf_4.tolist(),
    },
}
print("Group 3: ScottParzenEstimator — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 4: QuantileFilter
# ═══════════════════════════════════════════════════════════════════════

from optuna.trial import create_trial, FrozenTrial
from optuna.trial import TrialState


def make_trial(value):
    """创建一个简单的 FrozenTrial。"""
    return FrozenTrial(
        number=0,
        value=value,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=0,
        state=TrialState.COMPLETE,
    )


# Case 1: 10 values, q=0.3, minimize
vals_q1 = [10.0, 3.0, 7.0, 1.0, 5.0, 2.0, 8.0, 4.0, 6.0, 9.0]
trials_q1 = [make_trial(v) for v in vals_q1]
qf_1 = _QuantileFilter(quantile=0.3, is_lower_better=True, min_n_top_trials=2, target=None)
top_q1 = qf_1.filter(trials_q1)
top_q1_vals = sorted([t.value for t in top_q1])

# Case 2: same values, q=0.3, maximize
qf_2 = _QuantileFilter(quantile=0.3, is_lower_better=False, min_n_top_trials=2, target=None)
top_q2 = qf_2.filter(trials_q1)
top_q2_vals = sorted([t.value for t in top_q2])

# Case 3: q=0.1, min_n_top=3 (forces at least 3)
qf_3 = _QuantileFilter(quantile=0.1, is_lower_better=True, min_n_top_trials=3, target=None)
top_q3 = qf_3.filter(trials_q1)
top_q3_vals = sorted([t.value for t in top_q3])

results["quantile_filter"] = {
    "values": vals_q1,
    "case_minimize_q03": {
        "quantile": 0.3,
        "is_lower_better": True,
        "min_n_top": 2,
        "top_values": top_q1_vals,
        "n_top": len(top_q1),
    },
    "case_maximize_q03": {
        "quantile": 0.3,
        "is_lower_better": False,
        "min_n_top": 2,
        "top_values": top_q2_vals,
        "n_top": len(top_q2),
    },
    "case_minimize_q01_min3": {
        "quantile": 0.1,
        "is_lower_better": True,
        "min_n_top": 3,
        "top_values": top_q3_vals,
        "n_top": len(top_q3),
    },
}
print("Group 4: QuantileFilter — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 5: Pearson divergence
# ═══════════════════════════════════════════════════════════════════════

def pearson_divergence_py(pdf_top, pdf_local):
    """对齐 Python PedAnovaImportanceEvaluator._compute_pearson_divergence。"""
    pdf_top = np.asarray(pdf_top) + 1e-12
    pdf_local = np.asarray(pdf_local) + 1e-12
    return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

# Case 1: identical
pd_identical = pearson_divergence_py([0.2, 0.2, 0.2, 0.2, 0.2],
                                      [0.2, 0.2, 0.2, 0.2, 0.2])

# Case 2: peaked vs uniform
pd_peaked_vs_uniform = pearson_divergence_py([0.8, 0.05, 0.05, 0.05, 0.05],
                                              [0.2, 0.2, 0.2, 0.2, 0.2])

# Case 3: bimodal vs uniform
pd_bimodal_vs_uniform = pearson_divergence_py([0.4, 0.05, 0.1, 0.05, 0.4],
                                               [0.2, 0.2, 0.2, 0.2, 0.2])

# Case 4: two peaked distributions
pd_two_peaked = pearson_divergence_py([0.7, 0.1, 0.1, 0.05, 0.05],
                                       [0.1, 0.1, 0.1, 0.1, 0.6])

results["pearson_divergence"] = {
    "case_identical": {
        "pdf_p": [0.2, 0.2, 0.2, 0.2, 0.2],
        "pdf_q": [0.2, 0.2, 0.2, 0.2, 0.2],
        "divergence": pd_identical,
    },
    "case_peaked_vs_uniform": {
        "pdf_p": [0.8, 0.05, 0.05, 0.05, 0.05],
        "pdf_q": [0.2, 0.2, 0.2, 0.2, 0.2],
        "divergence": pd_peaked_vs_uniform,
    },
    "case_bimodal_vs_uniform": {
        "pdf_p": [0.4, 0.05, 0.1, 0.05, 0.4],
        "pdf_q": [0.2, 0.2, 0.2, 0.2, 0.2],
        "divergence": pd_bimodal_vs_uniform,
    },
    "case_two_peaked": {
        "pdf_p": [0.7, 0.1, 0.1, 0.05, 0.05],
        "pdf_q": [0.1, 0.1, 0.1, 0.1, 0.6],
        "divergence": pd_two_peaked,
    },
}
print("Group 5: Pearson divergence — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 6: _get_cardinality (搜索空间体积)
# ═══════════════════════════════════════════════════════════════════════

card_1d = _get_cardinality(np.array([[0.0, 1.0]]))
card_2d = _get_cardinality(np.array([[0.0, 1.0], [0.0, 2.0]]))
card_3d = _get_cardinality(np.array([[-1.0, 1.0], [0.0, 0.5], [2.0, 5.0]]))
card_zero = _get_cardinality(np.array([[0.5, 0.5]]))

results["cardinality"] = {
    "case_1d": {"search_space": [[0.0, 1.0]], "cardinality": float(card_1d)},
    "case_2d": {"search_space": [[0.0, 1.0], [0.0, 2.0]], "cardinality": float(card_2d)},
    "case_3d": {"search_space": [[-1.0, 1.0], [0.0, 0.5], [2.0, 5.0]], "cardinality": float(card_3d)},
    "case_zero": {"search_space": [[0.5, 0.5]], "cardinality": float(card_zero)},
}
print("Group 6: Cardinality — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 7: Discretize param (grid mapping)
# ═══════════════════════════════════════════════════════════════════════

from optuna.importance._ped_anova.scott_parzen_estimator import _count_numerical_param_in_grid

# 构造 trials 用于 FrozenTrial
def make_param_trial(param_name, value, dist):
    return FrozenTrial(
        number=0,
        value=0.0,
        datetime_start=None,
        datetime_complete=None,
        params={param_name: value},
        distributions={param_name: dist},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=0,
        state=TrialState.COMPLETE,
    )

# Float 离散化: 10 个值映射到 5 步 grid
float_dist = FloatDistribution(low=0.0, high=1.0)
float_values = [0.0, 0.1, 0.2, 0.5, 0.5, 0.7, 0.9, 1.0, 0.3, 0.8]
float_trials = [make_param_trial("x", v, float_dist) for v in float_values]
float_counts = _count_numerical_param_in_grid("x", float_dist, float_trials, n_steps=5)

# Int 离散化
int_dist_small = IntDistribution(low=0, high=4)
int_values = [0, 1, 1, 2, 3, 3, 3, 4, 4, 4]
int_trials = [make_param_trial("n", v, int_dist_small) for v in int_values]
int_counts = _count_numerical_param_in_grid("n", int_dist_small, int_trials, n_steps=5)

results["discretize"] = {
    "float_case": {
        "values": float_values,
        "low": 0.0, "high": 1.0, "n_steps": 5,
        "counts": float_counts.tolist(),
    },
    "int_case": {
        "values": int_values,
        "low": 0, "high": 4, "n_steps": 5,
        "counts": int_counts.tolist(),
    },
}
print("Group 7: Discretize — OK")


# ═══════════════════════════════════════════════════════════════════════
# Group 8: FanovaTree split_midpoints 和 split_sizes
# ═══════════════════════════════════════════════════════════════════════

# 使用 Tree B
results["split_midpoints"] = {
    "tree_b": {
        "description": "Tree B 的分裂中点和大小",
        "feat0_midpoints": ftree_b._split_midpoints[0].tolist(),
        "feat0_sizes": ftree_b._split_sizes[0].tolist(),
        "feat1_midpoints": ftree_b._split_midpoints[1].tolist(),
        "feat1_sizes": ftree_b._split_sizes[1].tolist(),
    },
}
print("Group 8: Split midpoints — OK")


# ═══════════════════════════════════════════════════════════════════════
# Final: write JSON
# ═══════════════════════════════════════════════════════════════════════

output_path = os.path.join(os.path.dirname(__file__), "importance_deep_golden_values.json")
with open(output_path, "w") as f:
    json.dump(sanitize(results), f, indent=2, ensure_ascii=False)

print(f"\nGenerated {len(results)} groups → {output_path}")
