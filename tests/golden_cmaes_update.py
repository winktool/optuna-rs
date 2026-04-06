#!/usr/bin/env python3
"""
生成 CMA-ES update() 的金标准值。
使用 cmaes 库直接执行一步更新，记录 mean/sigma/p_sigma/p_c 的精确值。
"""
import numpy as np
from cmaes import CMA

# 2D Sphere function: f(x) = x_0^2 + x_1^2
def sphere(x):
    return sum(xi**2 for xi in x)

# 创建 CMA-ES 优化器，固定种子和初始条件
np.random.seed(42)
n = 2
mean0 = np.array([3.0, 3.0])
sigma0 = 1.5
optimizer = CMA(mean=mean0, sigma=sigma0, seed=42)

print(f"=== Initial State ===")
print(f"mean: {optimizer._mean.tolist()}")
print(f"sigma: {optimizer._sigma}")
print(f"p_sigma: {optimizer._p_sigma.tolist()}")
print(f"p_c: {optimizer._pc.tolist()}")
print(f"C diagonal: {np.diag(optimizer._C).tolist()}")
print(f"popsize: {optimizer.population_size}")
print(f"mu: {optimizer._mu}")
print(f"weights[:mu]: {optimizer._weights[:optimizer._mu].tolist()}")

# Generate and evaluate one generation
solutions = []
for _ in range(optimizer.population_size):
    x = optimizer.ask()
    val = sphere(x)
    solutions.append((x, val))
    print(f"  candidate: {x.tolist()}, value: {val}")

# Sort by value (same as CMA-ES does internally)
solutions.sort(key=lambda sv: sv[1])

# Tell optimizer  
optimizer.tell(solutions)

print(f"\n=== After 1st Update ===")
print(f"mean: {optimizer._mean.tolist()}")
print(f"sigma: {optimizer._sigma}")
print(f"p_sigma: {optimizer._p_sigma.tolist()}")
print(f"p_c: {optimizer._pc.tolist()}")
print(f"C diagonal: {np.diag(optimizer._C).tolist()}")
print(f"generation: {optimizer._g}")

# Second generation
solutions2 = []
for _ in range(optimizer.population_size):
    x = optimizer.ask()
    val = sphere(x)
    solutions2.append((x, val))

solutions2.sort(key=lambda sv: sv[1])
optimizer.tell(solutions2)

print(f"\n=== After 2nd Update ===")
print(f"mean: {optimizer._mean.tolist()}")
print(f"sigma: {optimizer._sigma}")
print(f"p_sigma: {optimizer._p_sigma.tolist()}")
print(f"p_c: {optimizer._pc.tolist()}")
print(f"C diagonal: {np.diag(optimizer._C).tolist()}")
print(f"generation: {optimizer._g}")
