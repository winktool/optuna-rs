#!/usr/bin/env python3
"""
Generate Python reference values for NSGA-II cross-validation with Rust tests.
Covers: crowding distance, non-dominated sort, constrained dominance.

Run: python3 tests/nsga_cross_validate_baseline.py 2>/dev/null > tests/nsga_baseline_values.json
"""
import json, sys, math
sys.path.insert(0, "/Users/lichangqing/Copilot/optuna/optuna")

import numpy as np
from collections import defaultdict
from optuna.samplers.nsgaii._elite_population_selection_strategy import _calc_crowding_distance
from optuna.study._multi_objective import _fast_non_domination_rank, _calculate_nondomination_rank
from optuna.trial import FrozenTrial, create_trial
from optuna.study import StudyDirection

r = {}

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: Crowding distance
# ═══════════════════════════════════════════════════════════════════════════

def make_trial(number, values):
    t = create_trial(values=values)
    t.number = number
    return t

# 1a. Basic 2D case
trials_2d = [
    make_trial(0, [1.0, 5.0]),
    make_trial(1, [2.0, 3.0]),
    make_trial(2, [3.0, 2.0]),
    make_trial(3, [4.0, 1.0]),
    make_trial(4, [5.0, 0.0]),
]
cd_2d = _calc_crowding_distance(trials_2d)
# Sort by trial number for stability
r["cd_2d"] = [cd_2d[i] for i in range(5)]

# 1b. All same values in one dimension
trials_same = [
    make_trial(0, [1.0, 3.0]),
    make_trial(1, [1.0, 2.0]),
    make_trial(2, [1.0, 1.0]),
]
cd_same = _calc_crowding_distance(trials_same)
r["cd_same_dim"] = [cd_same[i] for i in range(3)]

# 1c. Two trials only
trials_two = [
    make_trial(0, [1.0, 3.0]),
    make_trial(1, [3.0, 1.0]),
]
cd_two = _calc_crowding_distance(trials_two)
r["cd_two"] = [cd_two[i] for i in range(2)]

# 1d. Single trial
trials_one = [make_trial(0, [2.0, 2.0])]
cd_one = _calc_crowding_distance(trials_one)
r["cd_one"] = [cd_one[0]]

# 1e. With inf values
trials_inf = [
    make_trial(0, [1.0, float('inf')]),
    make_trial(1, [2.0, 3.0]),
    make_trial(2, [3.0, 1.0]),
    make_trial(3, [float('inf'), 0.5]),
]
cd_inf = _calc_crowding_distance(trials_inf)
r["cd_inf"] = [cd_inf[i] for i in range(4)]

# 1f. 3D case
trials_3d = [
    make_trial(0, [1.0, 5.0, 3.0]),
    make_trial(1, [2.0, 3.0, 4.0]),
    make_trial(2, [3.0, 2.0, 2.0]),
    make_trial(3, [4.0, 1.0, 1.0]),
]
cd_3d = _calc_crowding_distance(trials_3d)
r["cd_3d"] = [cd_3d[i] for i in range(4)]

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: Non-dominated sort
# ═══════════════════════════════════════════════════════════════════════════

# 2a. Simple 2D case with clear fronts
loss_2d = np.array([
    [1.0, 4.0],   # Front 0
    [2.0, 2.0],   # Front 0
    [4.0, 1.0],   # Front 0
    [3.0, 3.0],   # Front 1
    [5.0, 5.0],   # Front 2
])
ranks_2d = _calculate_nondomination_rank(loss_2d)
r["nds_ranks_2d"] = ranks_2d.tolist()

# 2b. 3D case
loss_3d = np.array([
    [1.0, 3.0, 2.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
    [2.0, 2.0, 2.0],
    [4.0, 4.0, 4.0],
])
ranks_3d = _calculate_nondomination_rank(loss_3d)
r["nds_ranks_3d"] = ranks_3d.tolist()

# 2c. All same values
loss_same = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
])
ranks_same = _calculate_nondomination_rank(loss_same)
r["nds_ranks_same"] = ranks_same.tolist()

# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "NaN"
        return v
    if isinstance(obj, np.ndarray):
        return [convert(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [convert(x) for x in obj]
    return obj

# Pre-convert all values
converted = {}
for k, v in r.items():
    converted[k] = convert(v)

print(json.dumps(converted, indent=2))
