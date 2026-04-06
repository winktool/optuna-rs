#!/usr/bin/env python3
"""
Session 49 - 会话 1-4: NSGA-II 非支配排序验证
Python 基线值生成 - 快速非支配排序和拥挤距离计算

NSGA-II 核心算法 (from optuna.samplers.nsgaii._core):
1. 快速非支配排序 (Fast Non-dominated Sorting): O(n*m), m=目标数
2. 拥挤距离计算 (Crowding Distance): 边界解得 ∞，其他解基于目标值间距
3. 排序条件: 先比较 rank，rank 相同则比较 crowding_distance
"""

import json
import sys
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist

def fast_non_dominated_sort(objectives: np.ndarray) -> Tuple[List[int], List[List[int]]]:
    """
    快速非支配排序算法 (O(n^2 * m))
    
    Args:
        objectives: shape (n_solutions, n_objectives)
    
    Returns:
        ranks: shape (n_solutions,) - 每个解的 rank
        fronts: list of lists - 每个 front 中的解 indices
    """
    n_solutions = objectives.shape[0]
    ranks = np.zeros(n_solutions, dtype=int)
    fronts = []
    
    # 计算支配关系
    domination_count = np.zeros(n_solutions, dtype=int)  # 支配该解的解的个数
    dominated_solutions = [[] for _ in range(n_solutions)]  # 被该解支配的解列表
    
    for i in range(n_solutions):
        for j in range(i + 1, n_solutions):
            # 检查 i 是否支配 j, 或 j 是否支配 i
            # 最小化: 所有目标都更小则支配
            i_dominate_j = np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j])
            j_dominate_i = np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
            
            if i_dominate_j:
                dominated_solutions[i].append(j)
                domination_count[j] += 1
            elif j_dominate_i:
                dominated_solutions[j].append(i)
                domination_count[i] += 1
    
    # 第一个 front: 支配计数为 0 的解
    current_front = np.where(domination_count == 0)[0].tolist()
    
    rank_index = 0
    while current_front:
        fronts.append(current_front)
        for i in current_front:
            ranks[i] = rank_index
        
        # 找下一个 front
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        current_front = list(set(next_front))  # 去重
        rank_index += 1
    
    return ranks, fronts

def calculate_crowding_distance(objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """
    计算拥挤距离 (Crowding Distance)
    
    算法:
    1. 边界解: crowding_distance = ∞
    2. 其他解: 相邻解在各目标轴上的距离求和
    3. 如果所有目标值相同则距离 = 0
    """
    n_objectives = objectives.shape[1]
    crowding_distance = np.zeros(len(front_indices))
    
    if len(front_indices) <= 2:
        # 边界情况: 解数 <= 2 时全部标记为 ∞
        return np.full(len(front_indices), np.inf)
    
    front_objectives = objectives[front_indices]
    
    for m in range(n_objectives):
        # 按目标 m 排序
        sorted_indices = np.argsort(front_objectives[:, m])
        sorted_objs = front_objectives[sorted_indices, m]
        
        # 边界解
        crowding_distance[sorted_indices[0]] = np.inf
        crowding_distance[sorted_indices[-1]] = np.inf
        
        # 其他解
        obj_range = sorted_objs[-1] - sorted_objs[0]
        if obj_range > 0:
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                distance = (sorted_objs[i + 1] - sorted_objs[i - 1]) / obj_range
                crowding_distance[idx] += distance
    
    return crowding_distance

def generate_sorting_reference(n_tests: int = 15) -> dict:
    """生成快速非支配排序基线值"""
    results = {}
    
    test_configs = [
        # (n_solutions, n_objectives, seed, description)
        (5, 2, 42, "5sol_2obj"),
        (10, 2, 42, "10sol_2obj"),
        (10, 3, 42, "10sol_3obj"),
        (20, 2, 42, "20sol_2obj"),
        (20, 3, 42, "20sol_3obj"),
        (5, 4, 42, "5sol_4obj"),
        (15, 2, 42, "15sol_2obj"),
        (10, 2, 123, "10sol_2obj_seed123"),
        (10, 2, 999, "10sol_2obj_seed999"),
        (2, 2, 42, "2sol_2obj_edge"),
        (1, 2, 42, "1sol_2obj_edge"),
        (10, 1, 42, "10sol_1obj_degenerate"),
        # 特殊情况: 所有解相同
        (-1, 2, 42, "identical_solutions"),
    ]
    
    for n_sol, n_obj, seed, desc in test_configs:
        np.random.seed(seed)
        
        if n_sol == -1:
            # 所有解相同
            objectives = np.full((5, 2), 1.0)
        else:
            objectives = np.random.randn(n_sol, n_obj)
        
        try:
            ranks, fronts = fast_non_dominated_sort(objectives)
            
            # 计算每个 front 中每个解的拥挤距离
            crowding_distances_per_front = []
            for front in fronts:
                cd = calculate_crowding_distance(objectives, front)
                crowding_distances_per_front.append(cd.tolist())
            
            results[desc] = {
                "n_solutions": int(objectives.shape[0]),
                "n_objectives": int(n_obj),
                "seed": seed,
                "objectives": objectives.tolist(),
                "ranks": ranks.tolist(),
                "fronts": [[int(i) for i in front] for front in fronts],
                "n_fronts": len(fronts),
                "crowding_distances_per_front": crowding_distances_per_front,
                "unique_ranks": list(map(int, np.unique(ranks))),
            }
        except Exception as e:
            results[desc] = {
                "error": str(e),
                "n_solutions": n_sol,
                "n_objectives": n_obj,
            }
    
    return results

def generate_crowding_distance_reference() -> dict:
    """生成拥挤距离计算base eline"""
    results = {}
    
    np.random.seed(42)
    
    # Case 1: 2 目标, 10 解
    front = list(range(10))
    objectives = np.random.randn(10, 2)
    cd = calculate_crowding_distance(objectives, front)
    results["10sol_2obj_cd"] = {
        "front": front,
        "objectives": objectives.tolist(),
        "crowding_distance": cd.tolist(),
        "n_infinite": int(np.sum(np.isinf(cd))),  # 边界解数 (应为 4)
    }
    
    # Case 2: 3 目标, 5 解
    objectives = np.random.randn(5, 3)
    cd = calculate_crowding_distance(objectives, list(range(5)))
    results["5sol_3obj_cd"] = {
        "objectives": objectives.tolist(),
        "crowding_distance": cd.tolist(),
        "n_infinite": int(np.sum(np.isinf(cd))),
    }
    
    # Case 3: 边界情况 (2 解)
    objectives = np.array([[1.0, 2.0], [3.0, 4.0]])
    cd = calculate_crowding_distance(objectives, [0, 1])
    results["2sol_boundary"] = {
        "objectives": objectives.tolist(),
        "crowding_distance": cd.tolist(),
        "all_infinite": bool(np.all(np.isinf(cd))),
    }
    
    return results

def main():
    print("=" * 90)
    print("SESSION 49 - 会话 1-4: NSGA-II 非支配排序 Python 基线生成")
    print("=" * 90)
    print()
    
    print("📊 生成快速非支配排序基线值...")
    sorting_ref = generate_sorting_reference()
    
    print("📊 生成拥挤距离计算基线值...")
    cd_ref = generate_crowding_distance_reference()
    
    all_data = {
        "fast_non_dominated_sort": sorting_ref,
        "crowding_distance": cd_ref,
    }
    
    json_file = "session_49_nsgaii_sorting_baseline.json"
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2, allow_nan=True)
    
    print()
    print("✅ 完整基线值已保存为: " + json_file)
    print()
    
    # 摘要
    print("=" * 90)
    print("BASELINE SUMMARY - 快速非支配排序")
    print("=" * 90)
    print()
    
    valid_tests = [v for v in sorting_ref.values() if "error" not in v]
    print(f"✓ 有效测试案例: {len(valid_tests)}")
    
    for name, data in list(sorting_ref.items())[:5]:
        if "error" not in data:
            print(f"  {name:20} | fronts={data['n_fronts']:2} | solutions={data['n_solutions']:2}")
    
    print()
    print(f"📈 拥挤距离测试: {len(cd_ref)} 个配置")
    for name, data in cd_ref.items():
        if "all_infinite" in data:
            print(f"  {name:20} | n_infinite={data.get('n_infinite', data.get('all_infinite', 0))}")
    
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
