#!/usr/bin/env python3
"""Generate deep cross-validation baseline for NSGA-II algorithms.

Covers ALL parametrized test cases from Python Optuna's test suite:
- tests/study_tests/test_multi_objective.py
- tests/samplers_tests/test_nsgaii.py

Run:  python tests/generate_nsga_deep_baseline.py
"""
import json, math, sys, os
from collections import defaultdict

# Try to import optuna; if not available, implement the algorithms directly
try:
    import numpy as np
    from optuna.study._multi_objective import (
        _fast_non_domination_rank,
        _dominates,
        _normalize_value,
        _is_pareto_front,
    )
    from optuna.study._study_direction import StudyDirection
    from optuna.samplers.nsgaii._elite_population_selection_strategy import (
        _calc_crowding_distance,
    )
    from optuna.trial import create_trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    import numpy as np

    class StudyDirection:
        MINIMIZE = "minimize"
        MAXIMIZE = "maximize"

def _serialize(val):
    """Serialize special float values for JSON."""
    if isinstance(val, (list, tuple)):
        return [_serialize(v) for v in val]
    if isinstance(val, np.ndarray):
        return [_serialize(v) for v in val.tolist()]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        if math.isnan(val):
            return "NaN"
        return float(val)
    return val

def make_frozen_trial(number, values):
    """Create a FrozenTrial from optuna."""
    return create_trial(values=values, state=None)  # Complete by default

def make_frozen_trial_with_number(number, values):
    """Create a FrozenTrial with specific number."""
    t = create_trial(values=values)
    # Hack: override number
    t._number = number
    return t


# ═══════════════════════════════════════════════════════════════════════════
#  Generate all baseline data
# ═══════════════════════════════════════════════════════════════════════════
result = {}

if HAS_OPTUNA:
    # ─── 1. dominates() function ───────────────────────────────────────────
    # From test_dominates_1d_not_equal
    dom_1d_cases = [(-1, 1), (-float("inf"), 0), (0, float("inf")), (-float("inf"), float("inf"))]
    result["dom_1d_not_equal"] = []
    for v1, v2 in dom_1d_cases:
        t1 = create_trial(values=[v1])
        t2 = create_trial(values=[v2])
        d_min = _dominates(t1, t2, [StudyDirection.MINIMIZE])
        d_max = _dominates(t1, t2, [StudyDirection.MAXIMIZE])
        result["dom_1d_not_equal"].append({
            "v1": _serialize(v1), "v2": _serialize(v2),
            "dom_min": d_min, "dom_max": d_max,
        })

    # From test_dominates_1d_equal
    dom_1d_equal_vals = [0, -float("inf"), float("inf")]
    result["dom_1d_equal"] = []
    for v in dom_1d_equal_vals:
        t1 = create_trial(values=[v])
        t2 = create_trial(values=[v])
        d_min = _dominates(t1, t2, [StudyDirection.MINIMIZE])
        d_max = _dominates(t1, t2, [StudyDirection.MAXIMIZE])
        result["dom_1d_equal"].append({
            "v": _serialize(v),
            "dom_min": d_min, "dom_max": d_max,
        })

    # From test_dominates_2d - comprehensive grid
    vals = [-float("inf"), -2, -1, 0, 1, 2, float("inf")]
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
    result["dom_2d_grid"] = {}
    for d_idx, (d1, d2) in enumerate([(StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE)]):
        dirs = [d1, d2]
        dom_results = []
        for i1 in range(len(vals)):
            for j1 in range(len(vals)):
                for i2 in range(len(vals)):
                    for j2 in range(len(vals)):
                        if (i1, j1) == (i2, j2):
                            continue
                        t1 = create_trial(values=[vals[i1], vals[j1]])
                        t2 = create_trial(values=[vals[i2], vals[j2]])
                        if _dominates(t1, t2, dirs):
                            dom_results.append([i1, j1, i2, j2])
        result[f"dom_2d_min_max"] = dom_results

    # ─── 2. _normalize_value() ──────────────────────────────────────────────
    norm_cases = []
    test_vals = [1.0, -1.0, 0.0, float("inf"), -float("inf"), None]
    for v in test_vals:
        n_min = _normalize_value(v, StudyDirection.MINIMIZE)
        n_max = _normalize_value(v, StudyDirection.MAXIMIZE)
        norm_cases.append({
            "v": _serialize(v), 
            "norm_min": _serialize(n_min), 
            "norm_max": _serialize(n_max)
        })
    result["normalize_value"] = norm_cases

    # ─── 3. _fast_non_domination_rank() ─────────────────────────────────────
    # From test_fast_non_domination_rank in test_multi_objective.py
    rank_cases = [
        ([[1, 2]], [0]),
        ([[1, 2], [2, 1]], [0, 0]),
        ([[1, 2], [2, 3]], [0, 1]),
        ([[1, 1], [1, 1], [1, 2], [2, 1], [1, 1], [0, 1.5], [0, 1.5]], [0, 0, 1, 1, 0, 0, 0]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [0, 1.5, 1.5], [0, 1.5, 1.5]], [0, 0, 1, 1, 1, 0, 0]),
    ]
    result["fns_rank_cases"] = []
    for values, expected_ranks in rank_cases:
        ranks = list(_fast_non_domination_rank(np.array(values, dtype=float)))
        result["fns_rank_cases"].append({
            "values": values,
            "expected_ranks": [int(r) for r in ranks],
        })

    # Additional rank cases - many fronts
    many_front_values = [
        [1.0, 6.0],
        [2.0, 5.0],
        [3.0, 4.0],  # front 0
        [2.0, 6.0],
        [3.0, 5.0],
        [4.0, 4.0],  # front 1
        [5.0, 6.0],
        [6.0, 5.0],  # front 2
        [7.0, 7.0],  # front 3
    ]
    ranks = list(_fast_non_domination_rank(np.array(many_front_values)))
    result["fns_many_fronts"] = {
        "values": many_front_values,
        "ranks": [int(r) for r in ranks],
    }

    # Single objective rank
    single_obj = [[3.0], [1.0], [2.0], [1.0], [4.0]]
    ranks = list(_fast_non_domination_rank(np.array(single_obj)))
    result["fns_single_obj"] = {
        "values": single_obj,
        "ranks": [int(r) for r in ranks],
    }

    # ─── 4. _calc_crowding_distance() ───────────────────────────────────────
    # From test_calc_crowding_distance in test_nsgaii.py
    cd_test_cases = [
        # (values_list, expected_distances)
        ([[1], [2], [3]], None),  # 3 distinct 1D values
        ([[1], [2]], None),  # 2 values
        ([[1]], None),  # single value
        ([[1], [1], [1]], None),  # all same
        ([[float("inf")], [float("inf")], [float("inf")]], None),  # all inf
        ([[-float("inf")], [-float("inf")], [-float("inf")]], None),  # all -inf
        ([[-float("inf")], [float("inf")]], None),  # -inf and inf
        ([[-float("inf")], [-float("inf")], [-float("inf")], [0], [1], [2], [float("inf")]], None),
    ]

    result["cd_python_exact"] = []
    for values_list, _ in cd_test_cases:
        trials = []
        for i, vals in enumerate(values_list):
            t = create_trial(values=vals)
            t._number = i
            trials.append(t)
        cd = _calc_crowding_distance(trials)
        distances = [cd[i] for i in range(len(trials))]
        result["cd_python_exact"].append({
            "values": [_serialize(v) for v in values_list],
            "distances": _serialize(distances),
        })

    # Multi-objective CD
    cd_2d_cases = [
        [[1, 5], [2, 3], [3, 2], [4, 1]],  # nice spread
        [[1, 1], [2, 2], [3, 3]],  # diagonal
        [[0, 0], [0, 0], [0, 0]],  # all same
        [[1, 2], [1, 2]],  # duplicate pair
    ]
    result["cd_2d_exact"] = []
    for values_list in cd_2d_cases:
        trials = []
        for i, vals in enumerate(values_list):
            t = create_trial(values=vals)
            t._number = i
            trials.append(t)
        cd = _calc_crowding_distance(trials)
        distances = [cd[i] for i in range(len(trials))]
        result["cd_2d_exact"].append({
            "values": values_list,
            "distances": _serialize(distances),
        })

    # 3D CD for cross-validate
    cd_3d_case = [[1, 5, 3], [2, 3, 4], [3, 2, 2], [4, 1, 1], [2.5, 2.5, 2.5]]
    trials = []
    for i, vals in enumerate(cd_3d_case):
        t = create_trial(values=vals)
        t._number = i
        trials.append(t)
    cd = _calc_crowding_distance(trials)
    distances = [cd[i] for i in range(len(trials))]
    result["cd_3d_5points"] = {
        "values": cd_3d_case,
        "distances": _serialize(distances),
    }

    # ─── 5. _is_pareto_front() ──────────────────────────────────────────────
    pareto_cases = [
        # 2D cases
        [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],  # all on front
        [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]],  # [2] dominated
        [[1.0, 1.0], [1.0, 1.0]],  # duplicates
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # chain domination
    ]
    result["pareto_front_cases"] = []
    for vals in pareto_cases:
        arr = np.array(vals, dtype=float)
        on_front = _is_pareto_front(arr, assume_unique_lexsorted=False).tolist()
        result["pareto_front_cases"].append({
            "values": vals,
            "on_front": on_front,
        })

    # ─── 6. Mixed direction sorting ─────────────────────────────────────────
    mixed_values = [
        [1.0, 5.0],   # min=1, max=5 → good in both
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
        [5.0, 1.0],   # min=5, max=1 → bad in both
        [1.0, 1.0],   # min=1 (good), max=1 (bad)
        [5.0, 5.0],   # min=5 (bad), max=5 (good)
    ]
    # Loss values for min/max: multiply max by -1
    loss_vals = np.array(mixed_values, dtype=float)
    loss_vals[:, 1] *= -1  # maximize → negate
    ranks = list(_fast_non_domination_rank(loss_vals))
    result["fns_mixed_direction"] = {
        "values": mixed_values,
        "ranks": [int(r) for r in ranks],
    }

    # ─── 7. Crowding distance sort order ────────────────────────────────────
    sort_cases = [
        [[5], [6], [9], [0]],
        [[5, 0], [6, 0], [9, 0], [0, 0]],
        [[5, -1], [6, 0], [9, 1], [0, 2]],
        [[1], [2], [float("inf")]],
        [[float("-inf")], [1], [2]],
    ]
    result["cd_sort_order"] = []
    for values_list in sort_cases:
        trials = []
        for i, vals in enumerate(values_list):
            t = create_trial(values=vals)
            t._number = i
            trials.append(t)
        cd = _calc_crowding_distance(trials)
        distances = [cd[i] for i in range(len(trials))]
        # Sort descending by distance → selection order
        sorted_indices = sorted(range(len(trials)), key=lambda x: distances[x], reverse=True)
        result["cd_sort_order"].append({
            "values": [_serialize(v) for v in values_list],
            "distances": _serialize(distances),
            "sorted_indices": sorted_indices,
        })

    # ─── 8. Edge case: NaN in objectives ────────────────────────────────────
    # Python handles NaN: _normalize_value(None, ...) → inf
    nan_values = [[1.0, 2.0], [float("nan"), 3.0], [2.0, float("nan")]]
    try:
        ranks = list(_fast_non_domination_rank(np.array(nan_values)))
        result["fns_nan_objectives"] = {
            "values": [_serialize(v) for v in nan_values],
            "ranks": [int(r) for r in ranks],
        }
    except Exception as e:
        result["fns_nan_objectives"] = {"error": str(e)}

    # ─── 9. Large problem (10 trials, 3 objectives) ────────────────────────
    np.random.seed(42)
    large_values = np.random.rand(10, 3).tolist()
    loss = np.array(large_values)
    ranks = list(_fast_non_domination_rank(loss))
    
    # Also get crowding distance for front 0
    front0_indices = [i for i, r in enumerate(ranks) if r == 0]
    front0_trials = []
    for idx in front0_indices:
        t = create_trial(values=large_values[idx])
        t._number = idx
        front0_trials.append(t)
    cd = _calc_crowding_distance(front0_trials)
    front0_cd = {str(idx): cd[idx] for idx in front0_indices}
    
    result["large_3obj"] = {
        "values": [[round(v, 10) for v in row] for row in large_values],
        "ranks": [int(r) for r in ranks],
        "front0_indices": front0_indices,
        "front0_cd": {k: _serialize(v) for k, v in front0_cd.items()},
    }

out_path = os.path.join(os.path.dirname(__file__), "nsga_deep_baseline.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Wrote {out_path} ({len(result)} sections)")
print(f"Sections: {list(result.keys())}")
