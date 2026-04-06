#!/usr/bin/env python3
"""
生成多目标优化模块的金标准参考值。
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/lichangqing/Copilot/optuna/optuna')

from optuna._hypervolume.wfg import compute_hypervolume, _compute_2d, _compute_3d
from optuna._hypervolume.hssp import _solve_hssp
from optuna._hypervolume.box_decomposition import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front

print("=" * 70)
print("1. 3D Hypervolume 精确参考值")
print("=" * 70)

cases_3d = [
    # (points, ref)
    (np.array([[1,1,1]], dtype=float), np.array([5,5,5], dtype=float)),
    (np.array([[1,3,2],[2,1,3],[3,2,1]], dtype=float), np.array([5,5,5], dtype=float)),
    (np.array([[0.5,0.5,4],[1,4,0.5],[4,1,1]], dtype=float), np.array([5,5,5], dtype=float)),
    (np.array([[1,1,1],[2,2,2]], dtype=float), np.array([5,5,5], dtype=float)),
    (np.array([[1,1,1],[1,1,1]], dtype=float), np.array([5,5,5], dtype=float)),
]

for i, (pts, ref) in enumerate(cases_3d):
    # Get Pareto front
    unique = np.unique(pts, axis=0)
    on_front = _is_pareto_front(unique, assume_unique_lexsorted=True)
    pareto = unique[on_front]
    sorted_pareto = pareto[pareto[:, 0].argsort()]
    hv = _compute_3d(sorted_pareto, ref)
    print(f"  Case {i}: pts={pts.tolist()}, ref={ref.tolist()}")
    print(f"           HV = {hv:.15e}")

print()
print("=" * 70)
print("2. 4D Hypervolume 精确参考值")
print("=" * 70)

cases_4d = [
    (np.array([[1,1,1,1]], dtype=float), np.array([5,5,5,5], dtype=float)),
    (np.array([[1,3,2,1],[2,1,3,2],[3,2,1,3]], dtype=float), np.array([5,5,5,5], dtype=float)),
    (np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]], dtype=float), np.array([5,5,5,5], dtype=float)),
]

for i, (pts, ref) in enumerate(cases_4d):
    hv = compute_hypervolume(pts, ref)
    print(f"  Case {i}: pts={pts.tolist()}, ref={ref.tolist()}")
    print(f"           HV = {hv:.15e}")

print()
print("=" * 70)
print("3. HSSP 2D 精确参考值")
print("=" * 70)

hssp_2d_cases = [
    # (loss_vals, subset_size, ref_point)
    (np.array([[1,5],[2,4],[3,3],[4,1]], dtype=float), 2, np.array([5,6], dtype=float)),
    (np.array([[1,5],[2,4],[3,3],[4,1]], dtype=float), 3, np.array([5,6], dtype=float)),
    (np.array([[1,8],[2,5],[4,4],[6,2],[7,1]], dtype=float), 2, np.array([10,10], dtype=float)),
    (np.array([[1,8],[2,5],[4,4],[6,2],[7,1]], dtype=float), 3, np.array([10,10], dtype=float)),
]

for i, (vals, k, ref) in enumerate(hssp_2d_cases):
    # Sort by first objective
    sorted_idx = np.argsort(vals[:, 0])
    sorted_vals = vals[sorted_idx]
    selected = _solve_hssp(sorted_vals, np.arange(len(vals))[sorted_idx], k, ref)
    # Map back to original indices
    selected_original = sorted_idx[np.array(selected)]
    selected_vals = vals[selected_original]
    print(f"  Case {i}: vals={vals.tolist()}, k={k}, ref={ref.tolist()}")
    print(f"           selected={sorted(selected_original.tolist())}, vals={selected_vals.tolist()}")

print()
print("=" * 70)
print("4. crowding_distance 精确参考值")
print("=" * 70)

# Manual crowding distance calculation matching Python implementation
def manual_crowding_distance(values, directions):
    """
    values: list of [v0, v1, ...] for each trial
    directions: list of 'minimize' or 'maximize'
    """
    n = len(values)
    n_obj = len(directions)
    distances = [0.0] * n
    
    for m in range(n_obj):
        indices = sorted(range(n), key=lambda i: values[i][m])
        vals = [values[indices[i]][m] for i in range(n)]
        vs = [float('-inf')] + vals + [float('inf')]
        
        if vs[1] == vs[n]:
            continue
        
        v_min = min(v for v in vs if v != float('-inf'))
        v_max = max(v for v in vs if v != float('inf'))
        width = v_max - v_min if v_max > v_min else 1.0
        
        for j in range(n):
            gap = 0.0 if vs[j] == vs[j+2] else vs[j+2] - vs[j]
            distances[indices[j]] += gap / width
    
    return distances

cd_cases = [
    ([[1, 5], [2, 4], [3, 3], [4, 1]], ['minimize', 'minimize']),
    ([[1, 3], [2, 2], [3, 1]], ['minimize', 'minimize']),
    ([[1, 5], [2, 3], [3, 2], [4, 1]], ['minimize', 'minimize']),
]

for i, (vals, dirs) in enumerate(cd_cases):
    cd = manual_crowding_distance(vals, dirs)
    print(f"  Case {i}: values={vals}")
    print(f"           distances={[f'{d:.15e}' for d in cd]}")

print()
print("=" * 70)
print("5. dominates 边界情况")
print("=" * 70)

# These are just logical tests
dom_cases = [
    ([1, 1], [2, 2], True, "A dominates B"),
    ([1, 1], [1, 2], True, "A dominates B (equal in one)"),
    ([1, 1], [1, 1], False, "Equal → not dominated"),
    ([1, 2], [2, 1], False, "Trade-off → not dominated"),
    ([2, 2], [1, 1], False, "B dominates A"),
]

for a, b, expected, label in dom_cases:
    print(f"  {label}: a={a}, b={b} → dominates={expected}")

print()
print("=" * 70)
print("6. non_dominated_sort 精确参考值 (fronts)")
print("=" * 70)

from optuna.study._multi_objective import _fast_non_dominated_sort

# Create mock trials for fast_non_dominated_sort
import optuna
nds_cases = [
    [[1,4],[2,3],[3,2],[4,4],[5,5]],  # 0,1,2 are front 0; 3 is front 1; 4 is front 2
    [[1,1],[2,2],[3,3],[4,4]],        # 0 is front 0; 1 front 1; etc.
    [[1,3],[3,1],[2,2],[1,1]],        # 3 alone is front 0; 0,1,2 are front 1? No. 3=(1,1) dominates (2,2)=2 and is part of front 0 with (1,3)=0 and (3,1)=1
]

for i, vals in enumerate(nds_cases):
    trials = []
    for j, v in enumerate(vals):
        trial = optuna.trial.create_trial(values=v)
        trial.number = j
        trials.append(trial)
    from optuna.study import StudyDirection
    dirs = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    fronts = _fast_non_dominated_sort([t for t in trials], dirs)
    front_indices = [[t.number for t in front] for front in fronts]
    print(f"  Case {i}: values={vals}")
    print(f"           fronts={front_indices}")

print()
print("=" * 70)
print("=== 完成 ===")
print("=" * 70)
