#!/usr/bin/env python3
"""
生成 _fast_non_domination_rank / _calculate_nondomination_rank 的金标准值。
"""
import numpy as np
from optuna.study._multi_objective import _fast_non_domination_rank

print("=== _fast_non_domination_rank (无约束) ===")

# Case 1: 简单 2D tradeoff
loss = np.array([
    [1.0, 4.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 1.0],
])
ranks = _fast_non_domination_rank(loss)
print(f"2D tradeoff: {ranks.tolist()}")

# Case 2: 有支配关系的 2D
loss2 = np.array([
    [1.0, 1.0],  # front 0
    [2.0, 2.0],  # front 1
    [3.0, 3.0],  # front 2
    [1.5, 0.5],  # front 0
])
ranks2 = _fast_non_domination_rank(loss2)
print(f"2D dominated:  {ranks2.tolist()}")

# Case 3: 单目标
loss3 = np.array([
    [3.0],
    [1.0],
    [2.0],
    [1.0],
    [4.0],
])
ranks3 = _fast_non_domination_rank(loss3)
print(f"1D: {ranks3.tolist()}")

# Case 4: 3D
loss4 = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 3.0],
    [3.0, 3.0, 1.0],
    [4.0, 4.0, 4.0],
])
ranks4 = _fast_non_domination_rank(loss4)
print(f"3D: {ranks4.tolist()}")

# Case 5: with n_below
loss5 = np.array([
    [1.0, 4.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 1.0],
    [5.0, 5.0],
    [0.5, 0.5],
])
ranks5_1 = _fast_non_domination_rank(loss5, n_below=1)
ranks5_2 = _fast_non_domination_rank(loss5, n_below=2)
ranks5_full = _fast_non_domination_rank(loss5)
print(f"n_below=1: {ranks5_1.tolist()}")
print(f"n_below=2: {ranks5_2.tolist()}")
print(f"n_below=None: {ranks5_full.tolist()}")

# Case 6: 重复值
loss6 = np.array([
    [1.0, 2.0],
    [1.0, 2.0],
    [3.0, 4.0],
])
ranks6 = _fast_non_domination_rank(loss6)
print(f"duplicates: {ranks6.tolist()}")

# Case 7: 带 penalty
penalty7 = np.array([0.0, 0.5, np.nan, 0.0, 1.0])
loss7 = np.array([
    [1.0, 1.0],  # feasible, penalty=0
    [2.0, 2.0],  # infeasible, penalty=0.5
    [0.5, 0.5],  # nan penalty (no constraints)
    [3.0, 3.0],  # feasible, penalty=0
    [1.5, 1.5],  # infeasible, penalty=1.0
])
ranks7 = _fast_non_domination_rank(loss7, penalty=penalty7)
print(f"with penalty: {ranks7.tolist()}")
