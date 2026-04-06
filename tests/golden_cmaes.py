#!/usr/bin/env python3
"""
生成 CMA-ES 参数初始化的 Python 金标准值。
使用 cmaes 库（CyberAgentAILab/cmaes）作为参考。
"""
import numpy as np

# 手动计算 Hansen 2014 标准参数（与 cmaes 库和 Rust 实现对比）
for n in [2, 5, 10]:
    lam = 4 + int(3 * np.log(n))  # default popsize
    mu = lam // 2

    # Weights
    weights = np.array([np.log((lam + 1) / 2) - np.log(i + 1) for i in range(mu)])
    weights = weights / weights.sum()
    mu_eff = 1.0 / (weights ** 2).sum()

    # Adaptation parameters
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max((mu_eff - 1.0) / (n + 1.0), 0) ** 0.5 - 1.0 + c_sigma
    # Corrected: d_sigma = 1 + 2*max(sqrt((mu_eff-1)/(n+1))-1, 0) + c_sigma
    d_sigma2 = 1.0 + 2.0 * max(((mu_eff - 1.0) / (n + 1.0))**0.5 - 1.0, 0.0) + c_sigma

    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff), 1.0 - c1)
    chi_n = n ** 0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    print(f"\n=== n={n}, lambda={lam}, mu={mu} ===")
    print(f"weights = {weights.tolist()}")
    print(f"mu_eff = {mu_eff:.15e}")
    print(f"c_sigma = {c_sigma:.15e}")
    print(f"d_sigma = {d_sigma2:.15e}")
    print(f"c_c = {c_c:.15e}")
    print(f"c1 = {c1:.15e}")
    print(f"c_mu = {c_mu:.15e}")
    print(f"chi_n = {chi_n:.15e}")

# Also compute default popsize
for n in [1, 2, 3, 5, 10, 20, 50, 100]:
    lam = 4 + int(3 * np.log(n))
    print(f"n={n}: default_popsize={lam}")

print("\n=== CMA update test: 2D Sphere ===")
# Simple 2D test: mean=[5,5], sigma=1, minimize f(x)=x1^2+x2^2
n = 2
lam = 4 + int(3 * np.log(n))
mu = lam // 2

weights_raw = np.array([np.log((lam + 1) / 2) - np.log(i + 1) for i in range(mu)])
weights = weights_raw / weights_raw.sum()
mu_eff = 1.0 / (weights ** 2).sum()

c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
d_sigma = 1.0 + 2.0 * max(((mu_eff - 1.0) / (n + 1.0))**0.5 - 1.0, 0.0) + c_sigma
c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
c_mu = min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff), 1.0 - c1)
chi_n = n ** 0.5 * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

mean = np.array([5.0, 5.0])
sigma = 1.0

# Simulate known solutions (pre-determined, no randomness)
solutions = [
    (np.array([4.5, 4.2]), 4.5**2 + 4.2**2),
    (np.array([5.3, 4.8]), 5.3**2 + 4.8**2),
    (np.array([4.1, 5.5]), 4.1**2 + 5.5**2),
    (np.array([5.7, 5.1]), 5.7**2 + 5.1**2),
    (np.array([4.9, 4.0]), 4.9**2 + 4.0**2),
    (np.array([3.8, 5.9]), 3.8**2 + 5.9**2),
]

# Sort by value
solutions.sort(key=lambda x: x[1])
print(f"Sorted solutions:")
for s, v in solutions:
    print(f"  {s} -> {v:.4f}")

# Compute new mean (using top mu)
new_mean = np.zeros(n)
for i in range(mu):
    new_mean += weights[i] * solutions[i][0]

print(f"\nOld mean: {mean}")
print(f"New mean: {new_mean.tolist()}")
print(f"New mean exact: [{new_mean[0]:.15e}, {new_mean[1]:.15e}]")

# Mean displacement
mean_diff = (new_mean - mean) / sigma
print(f"mean_diff: [{mean_diff[0]:.15e}, {mean_diff[1]:.15e}]")
