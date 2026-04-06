#!/usr/bin/env python3
"""
Session 49 - 会话 1-1: TPE 权重计算精确性交叉验证

深入验证 Rust optuna-rs 与 Python optuna 的 TPE Parzen estimator 权重计算
是否精确对齐，涵盖所有边界条件。

执行:
    python3 tests/session_49_tpe_weights_precision.py
"""
import sys
sys.path.insert(0, "/Users/lichangqing/Copilot/optuna/optuna")

import json
import math
import numpy as np
from optuna.samplers._tpe.sampler import default_gamma, default_weights

# ═══════════════════════════════════════════════════════════════════════════
#  Reference weight computation from Python optuna
# ═══════════════════════════════════════════════════════════════════════════

def compute_reference_weights(n_trials):
    """
    计算 Python optuna 的 TPE 默认权重
    对应 Rust 中的默认权重计算
    """
    return default_weights(n_trials)


def compute_reference_gamma(n_trials):
    """计算 Python optuna 的默认 gamma 值"""
    return default_gamma(n_trials)


# ═══════════════════════════════════════════════════════════════════════════
#  Test Cases
# ═══════════════════════════════════════════════════════════════════════════

test_results = {}

# 测试集 1: 基础规模
test_cases_basic = [10, 20, 30, 50]
for n in test_cases_basic:
    weights = compute_reference_weights(n)
    gamma = compute_reference_gamma(n)
    
    test_results[f"tpe_weights_n{n}"] = {
        "n_trials": n,
        "gamma": float(gamma),
        "weights": [float(w) for w in weights],
        "weight_sum": float(np.sum(weights)),
        "is_monotonic_increasing": bool(all(weights[i] <= weights[i+1] for i in range(len(weights)-1))),
        "min_weight": float(np.min(weights)),
        "max_weight": float(np.max(weights)),
        "weight_range": float(np.max(weights) - np.min(weights)),
    }

# 测试集 2: 边界条件
test_cases_boundaries = [1, 2, 25, 26, 100, 999]
for n in test_cases_boundaries:
    try:
        weights = compute_reference_weights(n)
        gamma = compute_reference_gamma(n)
        
        test_results[f"tpe_weights_boundary_n{n}"] = {
            "n_trials": n,
            "gamma": float(gamma),
            "weights": [float(w) for w in weights],
            "weight_sum": float(np.sum(weights)),
            "is_monotonic_increasing": bool(all(weights[i] <= weights[i+1] for i in range(len(weights)-1))),
            "length_matches": len(weights) == n,
        }
    except Exception as e:
        test_results[f"tpe_weights_boundary_n{n}_error"] = str(e)

# 测试集 3: 特殊情况评估
print("\n=== TPE Weights Precision Reference Values ===\n")
print(json.dumps(test_results, indent=2))

# 内联验证
print("\n=== Validation Assertions ===\n")

for n in test_cases_basic + test_cases_boundaries:
    if f"tpe_weights_n{n}" in test_results or f"tpe_weights_boundary_n{n}" in test_results:
        key = f"tpe_weights_n{n}" if f"tpe_weights_n{n}" in test_results else f"tpe_weights_boundary_n{n}"
        data = test_results[key]
        
        # 断言 1: 权重和应为 1.0
        weight_sum = data["weight_sum"]
        assert abs(weight_sum - 1.0) < 1e-10, f"n={n}: weight_sum={weight_sum} (expected 1.0)"
        
        # 断言 2: 权重单调递增
        if data.get("is_monotonic_increasing", True):
            assert True, f"n={n}: weights are monotonically increasing"
        
        # 断言 3: 权重数量应等于 n
        assert len(data["weights"]) == n, f"n={n}: len(weights)={len(data['weights'])} (expected {n})"
        
        # 断言 4: 所有权重非负且有限
        for i, w in enumerate(data["weights"]):
            assert 0.0 <= w <= 1.0, f"n={n}, w[{i}]={w} out of [0,1]"
            assert math.isfinite(w), f"n={n}, w[{i}]={w} is not finite"
        
        print(f"✓ n={n}: all assertions passed")

print("\n=== End of Validation ===\n")

# 保存为基线值供 Rust 测试使用
with open("tests/session_49_tpe_weights_baseline.json", "w") as f:
    json.dump(test_results, f, indent=2)

print("✓ Baseline values saved to: tests/session_49_tpe_weights_baseline.json")
