#!/usr/bin/env python3
"""
生成 Python optuna 的金标准参考值，用于 Rust 交叉验证测试。
执行: python tests/golden_values_generator.py
"""
import json
import math
import numpy as np
from decimal import Decimal
from scipy import stats

print("=" * 70)
print("1. adjust_discrete_uniform_high 参考值")
print("=" * 70)

def adjust_discrete_uniform_high_py(low, high, step):
    """对齐 Python optuna 的 _adjust_discrete_uniform_high"""
    d_high = Decimal(str(high))
    d_low = Decimal(str(low))
    d_step = Decimal(str(step))
    d_r = d_high - d_low
    if d_r == Decimal("0"):
        return high
    d_mod = d_r % d_step
    if d_mod != Decimal("0"):
        adjusted = float(d_low + (d_r // d_step) * d_step)
        return adjusted
    return high

test_cases = [
    (0.0, 1.0, 0.3),
    (0.0, 1.0, 0.7),
    (0.0, 10.0, 3.0),
    (0.5, 5.5, 0.4),
    (0.1, 0.9, 0.15),
    (0.0, 1.0, 0.25),  # 整除
    (0.0, 1.0, 1.0),   # 整除
    (1.0, 10.0, 2.5),
    (0.0, 100.0, 7.0),
    (-5.0, 5.0, 3.0),
]

for low, high, step in test_cases:
    result = adjust_discrete_uniform_high_py(low, high, step)
    print(f"  ({low}, {high}, step={step}) → {result!r}")

print()
print("=" * 70)
print("2. Scott-Parzen 带宽参考值")
print("=" * 70)

def scott_bandwidth(counts):
    """
    计算 Scott's rule 带宽（对齐 PED-ANOVA 的 _ScottParzenEstimator）。
    """
    n_total = sum(counts)
    if n_total <= 1:
        return None
    
    # 非零 grid 点
    mus = []
    counts_nz = []
    for i, c in enumerate(counts):
        if c > 0:
            mus.append(float(i))
            counts_nz.append(float(c))
    
    if not mus:
        return None
    
    n = float(n_total)
    weights = [c / n for c in counts_nz]
    
    # 加权均值
    mean_est = sum(m * w for m, w in zip(mus, weights))
    
    # 加权标准差（样本方差，使用 n-1）
    var_est = sum((m - mean_est) ** 2 * c for m, c in zip(mus, counts_nz)) / max(n - 1, 1)
    sigma_est = math.sqrt(var_est)
    
    # IQR
    cum = []
    acc = 0.0
    for c in counts_nz:
        acc += c
        cum.append(acc)
    
    q25_target = n / 4.0
    q75_target = n * 3.0 / 4.0
    
    idx_q25 = 0
    for i, c in enumerate(cum):
        if c >= q25_target:
            idx_q25 = i
            break
    
    idx_q75 = len(mus) - 1
    for i, c in enumerate(cum):
        if c >= q75_target:
            idx_q75 = i
            break
    
    iqr = mus[min(idx_q75, len(mus) - 1)] - mus[idx_q25]
    
    if iqr > 0:
        sigma_choice = min(iqr / 1.34, sigma_est)
    else:
        sigma_choice = sigma_est
    
    h = 1.059 * sigma_choice * n ** (-0.2)
    sigma_min = 0.5 / 1.64
    return max(h, sigma_min)

# Test cases for Scott bandwidth
bandwidth_cases = [
    [0, 0, 5, 0, 3, 0, 2, 0, 0, 0],   # 稀疏
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],    # 均匀
    [10, 0, 0, 0, 0, 0, 0, 0, 0, 10],  # 双峰
    [0, 0, 0, 0, 20, 0, 0, 0, 0, 0],   # 单峰集中
    [1, 2, 3, 4, 5, 4, 3, 2, 1, 0],    # 类-正态
]

for i, counts in enumerate(bandwidth_cases):
    h = scott_bandwidth(counts)
    n_total = sum(counts)
    print(f"  Case {i}: counts={counts}, n={n_total}, bandwidth={h!r}")

print()
print("=" * 70)
print("3. Pearson 散度参考值")
print("=" * 70)

def pearson_divergence(pdf_p, pdf_q):
    """Pearson χ² 散度: Σ q(x) * ((p(x)/q(x)) - 1)²"""
    eps = 1e-12
    return sum((q + eps) * ((p + eps) / (q + eps) - 1.0) ** 2 
               for p, q in zip(pdf_p, pdf_q))

divergence_cases = [
    # (pdf_p, pdf_q)
    ([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]),   # 相同
    ([0.5, 0.3, 0.1, 0.05, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]),  # P 偏左
    ([0.05, 0.05, 0.1, 0.3, 0.5], [0.2, 0.2, 0.2, 0.2, 0.2]),  # P 偏右
    ([1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]),    # P 集中
    ([0.1, 0.2, 0.4, 0.2, 0.1], [0.3, 0.1, 0.2, 0.3, 0.1]),    # 不同形状
]

for i, (p, q) in enumerate(divergence_cases):
    d = pearson_divergence(p, q)
    print(f"  Case {i}: D_chi2 = {d:.15e}")

print()
print("=" * 70)
print("4. discretize_param 参考值")
print("=" * 70)

def discretize_param(values, low, high, n_steps, is_log=False):
    if abs(high - low) < 1e-14:
        return [0] * len(values), 1
    s_low = math.log(max(low, 1e-300)) if is_log else low
    s_high = math.log(max(high, 1e-300)) if is_log else high
    grids = [s_low + (s_high - s_low) * i / max(n_steps - 1, 1) for i in range(n_steps)]
    indices = []
    for v in values:
        sv = math.log(max(v, 1e-300)) if is_log else v
        best_idx = 0
        best_dist = float('inf')
        for idx, g in enumerate(grids):
            d = abs(sv - g)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        indices.append(best_idx)
    return indices, n_steps

disc_cases = [
    ([0.0, 0.25, 0.5, 0.75, 1.0], 0.0, 1.0, 5, False),
    ([0.1, 0.3, 0.7, 0.9], 0.0, 1.0, 10, False),
    ([1.0, 10.0, 100.0], 1.0, 100.0, 10, True),   # log domain
    ([-5.0, -2.5, 0.0, 2.5, 5.0], -5.0, 5.0, 11, False),
]

for i, (vals, lo, hi, ns, is_log) in enumerate(disc_cases):
    indices, n_actual = discretize_param(vals, lo, hi, ns, is_log)
    print(f"  Case {i}: indices={indices} (n_steps={n_actual})")

print()
print("=" * 70)
print("5. MedianErrorEvaluator 参考值")
print("=" * 70)

def median_error_evaluator(values, warm_up=4):
    """
    计算 MedianErrorEvaluator 的结果。
    对齐 Python: for i in range(warm_up, len(values)): 
        subset = sorted(values[:warm_up + i + 1 - warm_up])  
    错误——应该是 values[warm_up:warm_up+i+1-warm_up]
    实际 Python 代码:
        for i in range(1, len(values) - warm_up + 1):
            subset = values[warm_up:warm_up+i]
            med = np.median(subset)
    """
    results = []
    for i in range(1, len(values) - warm_up + 1):
        subset = sorted(values[warm_up:warm_up + i])
        n = len(subset)
        if n % 2 == 1:
            med = subset[n // 2]
        else:
            med = (subset[n // 2 - 1] + subset[n // 2]) / 2.0
        results.append(med)
    return results

median_values = [10.0, 8.0, 6.0, 4.0, 2.0, 9.0, 7.0, 5.0, 3.0, 1.0]
medians = median_error_evaluator(median_values, warm_up=4)
print(f"  values={median_values}")
print(f"  warm_up=4")
for i, m in enumerate(medians):
    step_idx = i + 1
    subset = sorted(median_values[4:4+step_idx])
    print(f"    step {step_idx}: subset={subset} → median={m:.15e}")

print()
print("=" * 70)
print("6. weighted_variance 参考值")  
print("=" * 70)

def weighted_variance(values, weights):
    """加权方差: Σ w_i * (x_i - μ)² / Σ w_i, 其中 μ = Σ w_i * x_i / Σ w_i"""
    w_sum = sum(weights)
    if w_sum < 1e-14:
        return 0.0
    mean = sum(v * w for v, w in zip(values, weights)) / w_sum
    var = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / w_sum
    return var

wv_cases = [
    ([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ([1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.2, 0.4, 0.2, 0.1]),
    ([10.0, 20.0, 30.0], [5.0, 3.0, 2.0]),
    ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),  # 零方差
    ([100.0], [1.0]),  # 单值
]

for i, (vals, ws) in enumerate(wv_cases):
    v = weighted_variance(vals, ws)
    w_sum = sum(ws)
    mean = sum(v_ * w for v_, w in zip(vals, ws)) / w_sum
    print(f"  Case {i}: values={vals}, weights={ws} → mean={mean:.15e}, var={v:.15e}")

print()
print("=" * 70)
print("7. quantile_filter 参考值")
print("=" * 70)

def quantile_filter_py(values, quantile, is_lower_better=True, min_n_top=2):
    n = len(values)
    if n == 0:
        return []
    losses = values[:] if is_lower_better else [-v for v in values]
    sorted_losses = sorted(losses)
    q_idx = max(min(int(math.ceil(n * quantile)), n), min_n_top)
    cutoff = sorted_losses[q_idx - 1]
    indices = [i for i in range(n) if losses[i] <= cutoff]
    if len(indices) < min_n_top:
        sorted_idx = sorted(range(n), key=lambda i: losses[i])
        indices = sorted_idx[:min(min_n_top, n)]
    return sorted(indices)

qf_cases = [
    ([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0], 0.1, True, 2),
    ([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0], 0.3, True, 2),
    ([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0], 0.5, True, 2),
    # maximize (lower is NOT better)
    ([5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0], 0.2, False, 2),
]

for i, (vals, q, ilb, mnt) in enumerate(qf_cases):
    indices = quantile_filter_py(vals, q, ilb, mnt)
    selected_vals = [vals[j] for j in indices]
    print(f"  Case {i}: quantile={q}, is_lower_better={ilb} → indices={indices}, values={selected_vals}")

print()
print("=" * 70)
print("8. EMMR β 计算参考值")
print("=" * 70)

def compute_beta(n_params, n_trials, delta=0.1):
    """β = 2 * ln(d * n² * π² / (6δ)) / 5"""
    d = float(n_params)
    n = float(n_trials)
    arg = d * n * n * math.pi ** 2 / (6.0 * delta)
    return 2.0 * math.log(arg) / 5.0

beta_cases = [
    (1, 10), (2, 20), (3, 30), (5, 50), (10, 100),
    (1, 100), (20, 200), (2, 5),
]

for np_, nt in beta_cases:
    b = compute_beta(np_, nt)
    print(f"  n_params={np_:3d}, n_trials={nt:4d} → β = {b:.15e}")

print()
print("=" * 70)
print("9. normal_pdf / normal_cdf 高精度参考值 (scipy)")
print("=" * 70)

test_points = [-5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
for x in test_points:
    pdf = stats.norm.pdf(x)
    cdf = stats.norm.cdf(x)
    print(f"  x={x:6.1f}: pdf={pdf:.15e}, cdf={cdf:.15e}")

print()
print("=" * 70)
print("10. CV Error 公式参考值")
print("=" * 70)

def cv_error(scores):
    """交叉验证误差 = sqrt(scale * var(scores)), scale = 1/k + 1/(k-1)"""
    k = len(scores)
    scale = 1.0 / k + 1.0 / (k - 1)
    mean = sum(scores) / k
    var = sum((s - mean) ** 2 for s in scores) / k
    return math.sqrt(scale * var)

cv_cases = [
    [0.8, 0.9, 1.0],
    [0.5, 0.6, 0.7, 0.8, 0.9],
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [0.95, 0.95, 0.95, 0.95, 0.95],
    [0.5, 1.5],
]

for scores in cv_cases:
    err = cv_error(scores)
    k = len(scores)
    scale = 1.0 / k + 1.0 / (k - 1)
    mean = sum(scores) / k
    var = sum((s - mean) ** 2 for s in scores) / k
    print(f"  k={k}: scores={scores}")
    print(f"       scale={scale:.15e}, mean={mean:.15e}, var={var:.15e}")
    print(f"       cv_error={err:.15e}")

print()
print("=" * 70)
print("11. IntDistribution adjust_int_uniform_high 参考值")
print("=" * 70)

int_cases = [
    (0, 10, 3),   # 10 % 3 = 1, 调整到 9
    (0, 10, 4),   # 10 % 4 = 2, 调整到 8
    (0, 10, 5),   # 10 / 5 = 2, 不调整
    (1, 10, 3),   # (10-1) % 3 = 0, 不调整
    (1, 10, 4),   # (10-1) % 4 = 1, 调整到 9
    (-5, 5, 3),   # (5-(-5)) % 3 = 1, 调整到 4
]

for low, high, step in int_cases:
    r = high - low
    n = r // step
    adjusted = low + n * step
    if adjusted != high:
        print(f"  ({low}, {high}, step={step}) → adjusted={adjusted}")
    else:
        print(f"  ({low}, {high}, step={step}) → unchanged={high}")

print()
print("=" * 70)
print("=== 完成。所有参考值可直接用于 Rust 测试 ===")
print("=" * 70)
