#!/usr/bin/env python3
"""
生成 GP search_space 归一化/反归一化的 Python 金标准值。
"""
import math
import numpy as np

def normalize_one_param(value, scale_type, bounds, step):
    if scale_type == 2:  # CATEGORICAL
        return value
    low, high = bounds[0] - 0.5 * step, bounds[1] + 0.5 * step
    if scale_type == 1:  # LOG
        low, high = math.log(low), math.log(high)
        value = math.log(value)
    if high == low:
        return 0.5
    return (value - low) / (high - low)

def unnormalize_one_param(value, scale_type, bounds, step):
    if scale_type == 2:  # CATEGORICAL
        return value
    low, high = bounds[0] - 0.5 * step, bounds[1] + 0.5 * step
    if scale_type == 1:  # LOG
        low, high = math.log(low), math.log(high)
    result = value * (high - low) + low
    if scale_type == 1:  # LOG
        result = math.exp(result)
    return result

print("=== GP normalize_param ===")
# Float [0, 10], step=0
for v in [0.0, 5.0, 10.0, 3.7]:
    n = normalize_one_param(v, 0, (0, 10), 0)
    print(f"Float[0,10] v={v}: normalized={n:.15e}")

# Float [0, 10], step=2
for v in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
    n = normalize_one_param(v, 0, (0, 10), 2)
    print(f"Float[0,10,step=2] v={v}: normalized={n:.15e}")

# Float log [1e-3, 1.0]
for v in [1e-3, 0.01, 0.1, 1.0]:
    n = normalize_one_param(v, 1, (1e-3, 1.0), 0)
    print(f"Float log[1e-3,1] v={v}: normalized={n:.15e}")

# Int [1, 10], step=1
for v in [1, 5, 10]:
    n = normalize_one_param(v, 0, (1, 10), 1)
    print(f"Int[1,10] v={v}: normalized={n:.15e}")

print("\n=== GP unnormalize_param ===")
# Float [0, 10]
for n in [0.0, 0.5, 1.0, 0.37]:
    u = unnormalize_one_param(n, 0, (0, 10), 0)
    print(f"Float[0,10] n={n}: unnormalized={u:.15e}")

# Int [1, 10] → unnormalize then round
for n in [0.0, 0.25, 0.5, 0.75, 1.0]:
    u = unnormalize_one_param(n, 0, (1, 10), 1)
    r = round(u)  # Python builtin round = banker's rounding
    cl = max(1, min(10, r))
    print(f"Int[1,10] n={n}: raw={u:.15e}, round={r}, clamp={cl}")

# Tricky: exact 0.5 boundary
# Int[1,10]: normalized=0.5 → unnormalized should round to even
n = 0.5
u = unnormalize_one_param(n, 0, (1, 10), 1)
print(f"\nBank round test: Int[1,10] n={n}: raw={u:.15e}, round={round(u)}")
# raw should be 5.5 → round(5.5) = 6 (banker's gives even)
# raw = 0.5 * (10.5 - 0.5) + 0.5 = 0.5 * 10 + 0.5 = 5.5
print(f"  raw=5.5 → Python round(5.5) = {round(5.5)}")

print("\n=== Roundtrip tests ===")
# normalize then unnormalize should recover original
for v in [0.0, 2.5, 5.0, 7.5, 10.0]:
    n = normalize_one_param(v, 0, (0, 10), 0)
    u = unnormalize_one_param(n, 0, (0, 10), 0)
    print(f"Float[0,10] v={v} → n={n:.6f} → u={u:.15e} (diff={abs(v-u):.2e})")

for v in [1e-3, 0.01, 0.1, 0.5, 1.0]:
    n = normalize_one_param(v, 1, (1e-3, 1.0), 0)
    u = unnormalize_one_param(n, 1, (1e-3, 1.0), 0)
    print(f"Float log[1e-3,1] v={v} → n={n:.6f} → u={u:.15e} (diff={abs(v-u):.2e})")
