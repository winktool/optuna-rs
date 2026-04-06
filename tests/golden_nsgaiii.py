#!/usr/bin/env python3
"""
生成 NSGA-III 参考点生成及关联的金标准值。
"""
import numpy as np
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    _generate_default_reference_point,
    _normalize_objective_values,
    _associate_individuals_with_reference_points,
    _filter_inf,
)

print("=== Reference Point Generation ===")

# 2 objectives, dividing_parameter=3
pts_2_3 = _generate_default_reference_point(2, 3)
print(f"2D, p=3: shape={pts_2_3.shape}")
for i, p in enumerate(pts_2_3):
    print(f"  [{i}] {p.tolist()}")

# 3 objectives, dividing_parameter=3
pts_3_3 = _generate_default_reference_point(3, 3)
print(f"3D, p=3: shape={pts_3_3.shape}")
for i, p in enumerate(pts_3_3):
    print(f"  [{i}] {p.tolist()}")

# 2 objectives, dividing_parameter=4
pts_2_4 = _generate_default_reference_point(2, 4)
print(f"2D, p=4: shape={pts_2_4.shape}")
for i, p in enumerate(pts_2_4):
    print(f"  [{i}] {p.tolist()}")

print("\n=== Normalization ===")

# Simple normalization test
obj_matrix = np.array([
    [1.0, 5.0],
    [2.0, 3.0],
    [4.0, 1.0],
    [3.0, 4.0],
])
norm_result = _normalize_objective_values(obj_matrix.copy())
print(f"Normalized:")
for i, row in enumerate(norm_result):
    print(f"  [{i}] {row.tolist()}")

print("\n=== Associate with Reference Points ===")
# Use normalized values from above and reference points
ref_pts = _generate_default_reference_point(2, 4)
# Use simple normalized values
norm_vals = np.array([
    [0.0, 1.0],  # ideal in obj 0
    [0.5, 0.5],  # balanced
    [1.0, 0.0],  # ideal in obj 1
    [0.25, 0.75],
])
closest, distances = _associate_individuals_with_reference_points(norm_vals, ref_pts)
print(f"Closest ref points: {closest.tolist()}")
print(f"Distances: {[f'{d:.6f}' for d in distances]}")
