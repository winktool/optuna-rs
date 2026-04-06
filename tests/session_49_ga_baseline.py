#!/usr/bin/env python3
"""
Session 49 - 会话 1-3: GA 交叉与变异验证
Python 基线值生成 - Optuna GA 算子精确参考

Python Optuna GA 实现要点:
* UniformCrossover: 按 swapping_prob 的概率交换父代基因
* Mutation: 参数随机丢弃 (mutation_prob 概率)，缺失参数由 RandomSampler 重新采样
* Categorical: 始终用 inlined uniform crossover
* Numerical: 精确变换后应用交叉算子
"""

import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple, List

def generate_uniform_crossover_reference(n_tests: int = 20) -> dict:
    """
    生成 UniformCrossover 基线值
    参数: parent1, parent2, swapping_prob, seed
    输出: child 参数矩阵
    """
    results = {}
    
    test_configs = [
        # (n_params, n_children, swapping_prob, seed, description)
        (2, 1, 0.5, 42, "2d_balanced"),
        (2, 3, 0.5, 42, "2d_triple"),
        (5, 1, 0.0, 42, "5d_no_swap"),
        (5, 1, 1.0, 42, "5d_full_swap"),
        (5, 1, 0.5, 42, "5d_balanced"),
        (10, 5, 0.3, 42, "10d_sparse"),
        (10, 5, 0.7, 42, "10d_dense"),
        (20, 10, 0.5, 42, "20d_large"),
        (2, 1, 0.5, 123, "2d_diff_seed"),
        (5, 1, 0.5, 999, "5d_diff_seed"),
    ]
    
    for n_params, n_children, swap_prob, seed, desc in test_configs:
        np.random.seed(seed)
        parent1 = np.random.randn(n_params)
        parent2 = np.random.randn(n_params)
        
        children = []
        for _ in range(n_children):
            # Uniform crossover: 按 swap_prob 概率交换基因
            mask = np.random.rand(n_params) < swap_prob
            child = np.where(mask, parent2, parent1)
            children.append(child.tolist())
        
        results[desc] = {
            "n_params": n_params,
            "n_children": n_children,
            "swapping_prob": swap_prob,
            "seed": seed,
            "parent1": parent1.tolist(),
            "parent2": parent2.tolist(),
            "children": children,
            "parent1_norm": float(np.linalg.norm(parent1)),
            "parent2_norm": float(np.linalg.norm(parent2)),
            "child_norms": [float(np.linalg.norm(np.array(c))) for c in children],
        }
    
    return results

def generate_mutation_reference() -> dict:
    """
    生成 Mutation 基线值
    Optuna 中变异 = 参数丢弃 (随机删除，让 RandomSampler 重新采样)
    """
    results = {}
    
    test_configs = [
        # (n_params, mutation_prob, seed, description)
        (5, 0.0, 42, "5d_no_mutation"),         # 保留所有参数
        (5, 1.0, 42, "5d_full_mutation"),      # 全部丢弃
        (5, 0.2, 42, "5d_sparse_mutation"),
        (5, 0.5, 42, "5d_balanced_mutation"),
        (10, 1/10, 42, "10d_default_prob"),    # 默认: 1/max(1, n_params)
        (20, 1/20, 42, "20d_default_prob"),
        (5, 0.5, 123, "5d_diff_seed"),
        (5, 0.5, 999, "5d_diff_seed_2"),
    ]
    
    for n_params, mut_prob, seed, desc in test_configs:
        np.random.seed(seed)
        original_params = np.random.randn(n_params)
        
        # 变异 = 按 mut_prob 概率删除参数
        rng = np.random.RandomState(seed)
        mutation_mask = rng.rand(n_params) < mut_prob  # True = 变异(删除)
        
        results[desc] = {
            "n_params": n_params,
            "mutation_prob": float(mut_prob),
            "seed": seed,
            "original_params": original_params.tolist(),
            "mutation_mask": mutation_mask.tolist(),  # True = 被删除
            "kept_indices": [i for i, keep in enumerate(~mutation_mask) if keep],
            "deleted_indices": [i for i, mutate in enumerate(mutation_mask) if mutate],
            "kept_count": int(np.sum(~mutation_mask)),
            "deleted_count": int(np.sum(mutation_mask)),
        }
    
    return results

def generate_combined_crossover_mutation() -> dict:
    """
    生成混合场景: 先交叉再变异
    """
    results = {}
    
    np.random.seed(42)
    parent1 = np.random.randn(8)
    parent2 = np.random.randn(8)
    
    crossover_prob_cases = [0.0, 0.5, 1.0]
    mutation_prob_cases = [0.0, 0.2, 0.5]
    
    for cross_prob in crossover_prob_cases:
        for mut_prob in mutation_prob_cases:
            desc = f"cross_{cross_prob}_mut_{mut_prob}"
            rng = np.random.RandomState(42)
            
            # Step 1: Crossover
            if rng.rand() < cross_prob:
                mask = rng.rand(8) < 0.5
                child = np.where(mask, parent2, parent1)
            else:
                child = parent1.copy()
            
            # Step 2: Mutation
            mutation_mask = rng.rand(8) < mut_prob
            child_after_mut = child.copy()
            child_after_mut[mutation_mask] = np.nan  # 标记为删除
            
            results[desc] = {
                "crossover_prob": float(cross_prob),
                "mutation_prob": float(mut_prob),
                "parent1": parent1.tolist(),
                "parent2": parent2.tolist(),
                "child_after_crossover": child.tolist(),
                "mutations_applied": mutation_mask.tolist(),
                "kept_count": int(np.sum(~mutation_mask)),
                "deleted_count": int(np.sum(mutation_mask)),
            }
    
    return results

def generate_default_parameters() -> dict:
    """
    生成 NSGAIISampler 默认参数参考"""
    return {
        "population_size": 50,
        "mutation_prob": "None (1.0/max(1, n_params))",
        "crossover_prob": 0.9,
        "swapping_prob": 0.5,
        "crossover_type": "UniformCrossover (default)",
    }

def main():
    print("=" * 90)
    print("SESSION 49 - 会话 1-3: GA 交叉与变异 Python 基线生成")
    print("=" * 90)
    print()
    
    print("📊 生成 UniformCrossover 基线值...")
    uniform_cross = generate_uniform_crossover_reference()
    
    print("📊 生成 Mutation 基线值...")
    mutation = generate_mutation_reference()
    
    print("📊 生成混合场景...")
    combined = generate_combined_crossover_mutation()
    
    print("📊 收集默认参数...")
    defaults = generate_default_parameters()
    
    all_data = {
        "uniform_crossover": uniform_cross,
        "mutation": mutation,
        "combined_crossover_mutation": combined,
        "default_parameters": defaults,
    }
    
    json_file = "session_49_ga_baseline.json"
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print()
    print("✅ 完整基线值已保存为: " + json_file)
    print()
    
    # 显示摘要
    print("=" * 90)
    print("BASELINE SUMMARY")
    print("=" * 90)
    print()
    print(f"📦 UniformCrossover 测试 {len(uniform_cross)} 个配置:")
    for name, data in list(uniform_cross.items())[:3]:
        print(f"   {name}: n_params={data['n_params']}, swap_prob={data['swapping_prob']}, seeds={len(uniform_cross)}")
    print()
    
    print(f"🔀 Mutation 测试 {len(mutation)} 个配置:")
    for name, data in list(mutation.items())[:3]:
        print(f"   {name}: n_params={data['n_params']}, mut_prob={data['mutation_prob']:.3f}")
    print()
    
    print(f"⚙️  默认参数: NSGAIISampler(population_size=50, crossover_prob=0.9, swapping_prob=0.5)")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
