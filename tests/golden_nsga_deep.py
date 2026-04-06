"""
NSGA-II/III 深度交叉验证黄金值生成器
生成用于 Rust 交叉验证测试的 Python 黄金值。
覆盖: Das-Dennis 参考点、参考点关联、ASF 归一化、拥挤距离值、约束违反度
"""
import json
import numpy as np
from itertools import combinations_with_replacement


def sanitize(val):
    """将 Python 浮点数转为 JSON 安全值"""
    if isinstance(val, float):
        if val == float('inf') or val > 1e308:
            return 1e308
        if val == float('-inf') or val < -1e308:
            return -1e308
        if val != val:  # NaN
            return None
        return val
    if isinstance(val, (list, tuple)):
        return [sanitize(v) for v in val]
    if isinstance(val, np.ndarray):
        return sanitize(val.tolist())
    if isinstance(val, (np.float64, np.float32)):
        return sanitize(float(val))
    if isinstance(val, (np.int64, np.int32)):
        return int(val)
    return val


# ──────────────────────────────────────────────
# 1. Das-Dennis 参考点
# ──────────────────────────────────────────────
def generate_reference_points(n_obj, div):
    """对齐 Rust: 坐标 = count / dividing_parameter, 所以和 = 1.0"""
    indices = np.array(list(combinations_with_replacement(range(n_obj), div)))
    row_indices = np.repeat(np.arange(len(indices)), div)
    col_indices = indices.flatten()
    ref_pts = np.zeros((len(indices), n_obj), dtype=float)
    np.add.at(ref_pts, (row_indices, col_indices), 1.0)
    ref_pts /= div  # 归一化使坐标和为 1.0
    return ref_pts


group_das_dennis = []
for n_obj, div in [(2, 3), (2, 5), (3, 3), (3, 4), (4, 3)]:
    pts = generate_reference_points(n_obj, div)
    group_das_dennis.append({
        "n_objectives": n_obj,
        "dividing_parameter": div,
        "count": len(pts),
        "points": sanitize(pts.tolist()),
    })


# ──────────────────────────────────────────────
# 2. 垂直距离 (perpendicular_distance)
# ──────────────────────────────────────────────
def perpendicular_distance(point, direction):
    norm_sq = np.dot(direction, direction)
    if norm_sq < 1e-14:
        return float('inf')
    dot = np.dot(point, direction)
    proj_scale = dot / norm_sq
    diff = point - proj_scale * np.array(direction)
    return float(np.sqrt(np.dot(diff, diff)))


group_perp_dist = []
test_cases = [
    ([1.0, 0.0], [1.0, 0.0]),   # 在方向上 → 距离 0
    ([0.0, 1.0], [1.0, 0.0]),   # 正交 → 距离 1
    ([1.0, 1.0], [1.0, 0.0]),   # 45° → 距离 1
    ([0.5, 0.5], [1.0, 1.0]),   # 在方向上 → 距离 0
    ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),  # 3D
    ([0.3, 0.7], [0.5, 0.5]),   # 非对齐
    ([0.0, 0.0], [1.0, 0.0]),   # 原点
]
for point, direction in test_cases:
    dist = perpendicular_distance(np.array(point), np.array(direction))
    group_perp_dist.append({
        "point": point,
        "direction": direction,
        "distance": sanitize(dist),
    })


# ──────────────────────────────────────────────
# 3. 拥挤距离黄金值
# ──────────────────────────────────────────────
def crowding_distance_values(population_values):
    """
    计算拥挤距离，返回与 population 对应的距离值列表。
    对齐 Python optuna._fast_non_domination_rank 的 crowding distance 实现。
    """
    n = len(population_values)
    if n == 0:
        return []
    n_obj = len(population_values[0])
    distances = [0.0] * n
    
    for m in range(n_obj):
        # 按目标 m 排序
        sorted_indices = sorted(range(n), key=lambda i: population_values[i][m])
        vals = [population_values[sorted_indices[i]][m] for i in range(n)]
        
        # 如果所有值相同，跳过
        if vals[0] == vals[-1]:
            continue
        
        # 插入哨兵
        vs = [float('-inf')] + vals + [float('inf')]
        v_min = next(x for x in vs if x != float('-inf'))
        v_max = next(x for x in reversed(vs) if x != float('inf'))
        width = v_max - v_min
        if width <= 0:
            width = 1.0
        
        for j in range(n):
            gap = 0.0 if vs[j] == vs[j + 2] else vs[j + 2] - vs[j]
            distances[sorted_indices[j]] += gap / width
    
    return distances


group_crowding = []

# Case 1: 2 目标 5 个体
pop_2d = [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]
cd_2d = crowding_distance_values(pop_2d)
group_crowding.append({
    "label": "2obj_5points_uniform",
    "values": pop_2d,
    "crowding_distances": sanitize(cd_2d),
})

# Case 2: 3 目标 4 个体
pop_3d = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.33, 0.33, 0.34]]
cd_3d = crowding_distance_values(pop_3d)
group_crowding.append({
    "label": "3obj_4points",
    "values": pop_3d,
    "crowding_distances": sanitize(cd_3d),
})

# Case 3: 重复值
pop_dup = [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
cd_dup = crowding_distance_values(pop_dup)
group_crowding.append({
    "label": "2obj_duplicates",
    "values": pop_dup,
    "crowding_distances": sanitize(cd_dup),
})


# ──────────────────────────────────────────────
# 4. 约束违反度 (penalty)
# ──────────────────────────────────────────────
group_penalty = []
test_constraints = [
    ([0.0, 0.0], 0.0),       # 全部满足
    ([-1.0, -0.5], 0.0),     # 全部满足 (负值)
    ([1.0, 2.0], 3.0),       # 全不满足 → sum = 3.0
    ([-0.5, 1.5, 0.0], 1.5), # 部分不满足 → sum = 1.5
    ([0.0, 0.0, 0.0], 0.0),  # 边界 (=0 视为满足)
]
for constraints, expected_penalty in test_constraints:
    actual = sum(v for v in constraints if v > 0)
    group_penalty.append({
        "constraints": constraints,
        "penalty": sanitize(actual),
    })


# ──────────────────────────────────────────────
# 5. ASF 归一化 (Python NSGA-III 方式)
# ──────────────────────────────────────────────
def normalize_objective_values(objective_matrix):
    """对齐 Python _normalize_objective_values"""
    obj = objective_matrix.copy()
    n_objectives = obj.shape[1]
    obj -= np.min(obj, axis=0)
    
    weights = np.eye(n_objectives)
    weights[weights == 0] = 1e6
    
    asf_value = np.max(np.einsum("nm,dm->dnm", obj, weights), axis=2)
    extreme_points = obj[np.argmin(asf_value, axis=1), :]
    
    if np.all(np.isfinite(extreme_points)) and np.linalg.matrix_rank(extreme_points) == len(extreme_points):
        intercepts_inv = np.linalg.solve(extreme_points, np.ones(n_objectives))
    else:
        intercepts = np.max(obj, axis=0)
        intercepts_inv = 1 / np.where(intercepts == 0, 1, intercepts)
    
    obj *= np.where(np.isfinite(intercepts_inv), intercepts_inv, 1)
    return obj


group_asf_norm = []

# Case 1: 2 目标
obj_2d = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0], [3.0, 2.0]])
norm_2d = normalize_objective_values(obj_2d)
group_asf_norm.append({
    "label": "2obj_4points",
    "original": sanitize(obj_2d),
    "normalized": sanitize(norm_2d),
})

# Case 2: 3 目标
obj_3d = np.array([[1.0, 5.0, 3.0], [3.0, 1.0, 5.0], [5.0, 3.0, 1.0]])
norm_3d = normalize_objective_values(obj_3d)
group_asf_norm.append({
    "label": "3obj_3points",
    "original": sanitize(obj_3d),
    "normalized": sanitize(norm_3d),
})


# ──────────────────────────────────────────────
# 6. 参考点关联
# ──────────────────────────────────────────────
def associate_with_reference_points(objective_matrix, reference_points):
    """对齐 Python _associate_individuals_with_reference_points"""
    ref_pts = np.array(reference_points)
    obj = np.array(objective_matrix)
    
    ref_norm_sq = np.linalg.norm(ref_pts, axis=1) ** 2
    perp_vecs = np.einsum("ni,pi,p,pm->npm", obj, ref_pts, 1/ref_norm_sq, ref_pts)
    dist = np.linalg.norm(obj[:, np.newaxis, :] - perp_vecs, axis=2)
    
    closest = np.argmin(dist, axis=1)
    min_dist = np.min(dist, axis=1)
    return closest.tolist(), min_dist.tolist()


group_assoc = []
ref_pts_2d = generate_reference_points(2, 3).tolist()  # [[0,1], [1/3,2/3], [2/3,1/3], [1,0]]
test_points = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.3, 0.7]]
closest, dists = associate_with_reference_points(test_points, ref_pts_2d)
group_assoc.append({
    "label": "2obj_div3",
    "points": test_points,
    "reference_points": ref_pts_2d,
    "closest_indices": closest,
    "distances": sanitize(dists),
})


# ──────────────────────────────────────────────
# 输出
# ──────────────────────────────────────────────
golden = {
    "das_dennis": group_das_dennis,
    "perpendicular_distance": group_perp_dist,
    "crowding_distance": group_crowding,
    "penalty": group_penalty,
    "asf_normalization": group_asf_norm,
    "reference_point_association": group_assoc,
}

with open("tests/nsga_deep_golden_values.json", "w") as f:
    json.dump(golden, f, indent=2)

# 统计
for key, group in golden.items():
    print(f"  {key}: {len(group)} cases")
print("Generated NSGA deep golden values")
