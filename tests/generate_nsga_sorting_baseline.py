"""
Generate baseline values for NSGA-II/III sorting cross-validation tests.
Covers: hypervolume (4D, 5D), HSSP, box decomposition, perpendicular distance,
        and end-to-end elite selection flow.
"""
import json
import math
import sys
import os

# Add optuna source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'optuna'))

import numpy as np
from optuna._hypervolume.wfg import compute_hypervolume
from optuna._hypervolume.hssp import _solve_hssp
from optuna._hypervolume.box_decomposition import get_non_dominated_box_bounds
from optuna.study._multi_objective import (
    _fast_non_domination_rank,
    _is_pareto_front,
    _dominates,
    _normalize_value,
)


def sanitize_for_json(obj):
    """Recursively replace inf/nan with string representations for valid JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "NaN"
        return v
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def custom_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "NaN"
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float,)):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    raise TypeError(f"Not serializable: {type(obj)}")


results = {}

# ══════════════════════════════════════════════════════════════
# 1. Hypervolume 4D
# ══════════════════════════════════════════════════════════════
pts_4d = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 4.0, 3.0],
    [3.0, 3.0, 1.0, 2.0],
    [4.0, 4.0, 2.0, 1.0],
])
ref_4d = np.array([6.0, 6.0, 6.0, 6.0])
hv_4d = compute_hypervolume(pts_4d, ref_4d)
results["hv_4d"] = hv_4d
print(f"HV 4D: {hv_4d}")

# ══════════════════════════════════════════════════════════════
# 2. Hypervolume 5D (2 points)
# ══════════════════════════════════════════════════════════════
pts_5d = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0, 2.0],
])
ref_5d = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
hv_5d = compute_hypervolume(pts_5d, ref_5d)
results["hv_5d"] = hv_5d
print(f"HV 5D: {hv_5d}")

# ══════════════════════════════════════════════════════════════
# 3. Hypervolume 4D (single point)
# ══════════════════════════════════════════════════════════════
pts_4d_single = np.array([[1.0, 2.0, 3.0, 4.0]])
ref_4d_single = np.array([10.0, 10.0, 10.0, 10.0])
hv_4d_single = compute_hypervolume(pts_4d_single, ref_4d_single)
results["hv_4d_single"] = hv_4d_single
print(f"HV 4D single: {hv_4d_single}")

# ══════════════════════════════════════════════════════════════
# 4. Hypervolume 4D (3 points, some dominated)
# ══════════════════════════════════════════════════════════════
pts_4d_mixed = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0, 3.0],   # dominated by first two
])
ref_4d_mixed = np.array([5.0, 5.0, 5.0, 5.0])
hv_4d_mixed = compute_hypervolume(pts_4d_mixed, ref_4d_mixed)
results["hv_4d_mixed"] = hv_4d_mixed
print(f"HV 4D mixed: {hv_4d_mixed}")

# ══════════════════════════════════════════════════════════════
# 5. Hypervolume 3D large
# ══════════════════════════════════════════════════════════════
pts_3d_large = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 2.0, 2.0],
    [4.0, 1.0, 5.0],
    [5.0, 4.0, 1.0],
    [2.5, 2.5, 2.5],
])
ref_3d_large = np.array([8.0, 8.0, 8.0])
hv_3d_large = compute_hypervolume(pts_3d_large, ref_3d_large)
results["hv_3d_large"] = hv_3d_large
print(f"HV 3D large: {hv_3d_large}")

# ══════════════════════════════════════════════════════════════
# 6. HSSP 2D
# ══════════════════════════════════════════════════════════════
hssp_2d_points = np.array([
    [1.0, 5.0],
    [2.0, 3.0],
    [3.0, 2.0],
    [4.0, 4.0],  # dominated
    [5.0, 1.0],
])
hssp_2d_ref = np.array([7.0, 7.0])
# _solve_hssp returns indices
hssp_2d_result = _solve_hssp(
    rank_i_loss_vals=hssp_2d_points,
    rank_i_indices=np.arange(len(hssp_2d_points)),
    subset_size=3,
    reference_point=hssp_2d_ref,
)
results["hssp_2d_indices"] = hssp_2d_result.tolist()
print(f"HSSP 2D indices: {hssp_2d_result}")

# ══════════════════════════════════════════════════════════════
# 7. HSSP 3D
# ══════════════════════════════════════════════════════════════
hssp_3d_points = np.array([
    [1.0, 3.0, 2.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
    [1.5, 1.5, 1.5],
    [2.5, 2.5, 2.5],
])
hssp_3d_ref = np.array([5.0, 5.0, 5.0])
hssp_3d_result = _solve_hssp(
    rank_i_loss_vals=hssp_3d_points,
    rank_i_indices=np.arange(len(hssp_3d_points)),
    subset_size=3,
    reference_point=hssp_3d_ref,
)
results["hssp_3d_indices"] = hssp_3d_result.tolist()
print(f"HSSP 3D indices: {hssp_3d_result}")

# ══════════════════════════════════════════════════════════════
# 8. Box decomposition 2D
# ══════════════════════════════════════════════════════════════
box_2d_points = np.array([
    [1.0, 4.0],
    [2.0, 2.0],
    [4.0, 1.0],
])
box_2d_ref = np.array([6.0, 6.0])
box_2d_lower, box_2d_upper = get_non_dominated_box_bounds(box_2d_points, box_2d_ref)
results["box_2d_lower"] = box_2d_lower.tolist()
results["box_2d_upper"] = box_2d_upper.tolist()
print(f"Box 2D lower: {box_2d_lower.shape}, upper: {box_2d_upper.shape}")

# ══════════════════════════════════════════════════════════════
# 9. Box decomposition 3D
# ══════════════════════════════════════════════════════════════
box_3d_points = np.array([
    [1.0, 3.0, 2.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
])
box_3d_ref = np.array([5.0, 5.0, 5.0])
box_3d_lower, box_3d_upper = get_non_dominated_box_bounds(box_3d_points, box_3d_ref)
results["box_3d_lower"] = box_3d_lower.tolist()
results["box_3d_upper"] = box_3d_upper.tolist()
print(f"Box 3D lower: {box_3d_lower.shape}, upper: {box_3d_upper.shape}")

# ══════════════════════════════════════════════════════════════
# 10. Non-dominated sorting: large population, mixed directions
# ══════════════════════════════════════════════════════════════
nds_mixed_values = np.array([
    [1.0, 5.0],
    [2.0, 4.0],
    [3.0, 3.0],
    [4.0, 2.0],
    [5.0, 1.0],
    [2.5, 2.5],
    [1.5, 4.5],
    [4.5, 1.5],
    [3.0, 3.0],  # duplicate
    [6.0, 6.0],  # dominated by all
])
# Minimize obj0, Maximize obj1
# For ranking: convert maximize → negate
loss_vals_mixed = nds_mixed_values.copy()
loss_vals_mixed[:, 1] = -loss_vals_mixed[:, 1]  # negate obj1
ranks_mixed = _fast_non_domination_rank(loss_vals_mixed)
results["nds_ranks_mixed_dir"] = ranks_mixed.tolist()
print(f"NDS mixed dir ranks: {ranks_mixed}")

# ══════════════════════════════════════════════════════════════
# 11. Non-dominated sorting: 10 trials, 3 objectives
# ══════════════════════════════════════════════════════════════
nds_3obj_values = np.array([
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 5.0],
    [5.0, 2.0, 1.0],
    [3.0, 3.0, 3.0],
    [4.0, 4.0, 4.0],
    [1.5, 4.0, 2.0],
    [2.5, 2.5, 2.5],
    [6.0, 6.0, 6.0],
    [1.0, 1.0, 6.0],
    [3.5, 1.5, 1.5],
])
ranks_3obj = _fast_non_domination_rank(nds_3obj_values)
results["nds_ranks_3obj_10"] = ranks_3obj.tolist()
print(f"NDS 3-obj 10 ranks: {ranks_3obj}")

# ══════════════════════════════════════════════════════════════
# 12. Perpendicular distance (NSGA-III)
# ══════════════════════════════════════════════════════════════
def perpendicular_distance(point, direction):
    """Python reference: perpendicular distance from point to reference line."""
    point = np.array(point)
    direction = np.array(direction)
    norm_sq = np.dot(direction, direction)
    if norm_sq < 1e-30:
        return np.linalg.norm(point)
    proj = np.dot(point, direction) / norm_sq * direction
    return float(np.linalg.norm(point - proj))

perp_cases = [
    # (point, direction, expected_dist)
    ([1.0, 0.0], [1.0, 0.0]),
    ([1.0, 1.0], [1.0, 0.0]),
    ([0.5, 0.5], [1.0, 1.0]),
    ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ([3.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([1.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
]

perp_results = []
for pt, direction in perp_cases:
    d = perpendicular_distance(pt, direction)
    perp_results.append(d)
    print(f"  perp_dist({pt}, {direction}) = {d}")
results["perpendicular_distances"] = perp_results

# ══════════════════════════════════════════════════════════════
# 13. Das-Dennis reference points (cross-check counts)
# ══════════════════════════════════════════════════════════════
from itertools import combinations_with_replacement

def das_dennis_count(n_objectives, dividing_parameter):
    """Number of reference points = C(n_obj + div - 1, div)"""
    from math import comb
    return comb(n_objectives + dividing_parameter - 1, dividing_parameter)

def das_dennis_points(n_objectives, dividing_parameter):
    points = []
    for combo in combinations_with_replacement(range(dividing_parameter + 1), n_objectives - 1):
        padded = (0,) + combo + (dividing_parameter,)
        point = [(padded[i+1] - padded[i]) / dividing_parameter for i in range(n_objectives)]
        points.append(point)
    return points

# Test cases: (n_obj, div) -> count + first few points
dd_cases = [(2, 4), (3, 3), (4, 3), (5, 2)]
dd_counts = []
dd_points_samples = {}
for n_obj, div in dd_cases:
    count = das_dennis_count(n_obj, div)
    pts = das_dennis_points(n_obj, div)
    dd_counts.append(count)
    key = f"dd_{n_obj}d_{div}div"
    # Sort for stable comparison
    pts_sorted = sorted(pts)
    dd_points_samples[key] = pts_sorted
    print(f"  Das-Dennis({n_obj}, {div}): {count} points")
results["dd_counts"] = dd_counts
results["dd_points"] = dd_points_samples

# ══════════════════════════════════════════════════════════════
# 14. Pareto front identification (is_pareto_front)
# ══════════════════════════════════════════════════════════════
pf_vals = np.array([
    [1.0, 4.0],
    [2.0, 2.0],
    [4.0, 1.0],
    [3.0, 3.0],  # dominated
    [5.0, 5.0],  # dominated
    [1.5, 3.0],  # dominated by [1, 4]? No: 1.5>1 and 3<4 → non-dominated
])
pf_mask = _is_pareto_front(pf_vals, assume_unique_lexsorted=False)
results["pareto_front_mask"] = pf_mask.tolist()
print(f"Pareto front mask: {pf_mask}")

# Compute pareto front mask for 3D
pf_3d_vals = np.array([
    [1.0, 3.0, 2.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
    [2.0, 2.0, 2.0],  # dominated by none of above individually
    [4.0, 4.0, 4.0],  # dominated
])
pf_3d_mask = _is_pareto_front(pf_3d_vals, assume_unique_lexsorted=False)
results["pareto_front_3d_mask"] = pf_3d_mask.tolist()
print(f"Pareto front 3D mask: {pf_3d_mask}")

# ══════════════════════════════════════════════════════════════
# 15. Hypervolume contribution (for HSSP validation)
# ══════════════════════════════════════════════════════════════
hv_contrib_pts = np.array([
    [1.0, 4.0],
    [2.0, 2.0],
    [4.0, 1.0],
])
hv_contrib_ref = np.array([6.0, 6.0])
hv_total = compute_hypervolume(hv_contrib_pts, hv_contrib_ref)
hv_contribs = []
for i in range(len(hv_contrib_pts)):
    remaining = np.delete(hv_contrib_pts, i, axis=0)
    hv_without = compute_hypervolume(remaining, hv_contrib_ref)
    hv_contribs.append(hv_total - hv_without)
results["hv_contributions_2d"] = hv_contribs
results["hv_total_2d"] = hv_total
print(f"HV total: {hv_total}, contributions: {hv_contribs}")

# ══════════════════════════════════════════════════════════════
# 16. Large NDS with many fronts
# ══════════════════════════════════════════════════════════════
np.random.seed(42)
large_vals = np.random.rand(20, 2) * 10
large_ranks = _fast_non_domination_rank(large_vals)
results["large_nds_values"] = large_vals.tolist()
results["large_nds_ranks"] = large_ranks.tolist()
n_fronts = max(large_ranks) + 1
print(f"Large NDS: 20 trials, {n_fronts} fronts")

# ══════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════
output_path = os.path.join(os.path.dirname(__file__), "nsga_sorting_baseline.json")
sanitized = sanitize_for_json(results)
with open(output_path, "w") as f:
    json.dump(sanitized, f, indent=2)
print(f"\nSaved to {output_path}")
