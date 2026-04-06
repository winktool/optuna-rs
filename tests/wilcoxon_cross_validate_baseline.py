#!/usr/bin/env python3
"""
Generate cross-validation reference values for the Wilcoxon signed-rank test.
Uses scipy.stats.wilcoxon with the same parameters as Python optuna's WilcoxonPruner:
  - zero_method='zsplit'
  - method='auto' (default)
  - correction=False (scipy default)

Each test case records:
  - diff_values: the input differences
  - direction: "minimize" → alternative="greater", "maximize" → alternative="less"
  - p_value: the reference p-value from scipy
  - method_used: "exact" or "approx" (which method scipy auto-selected)
"""

import json
import sys
import numpy as np

try:
    from scipy.stats import wilcoxon
    import scipy
    print(f"scipy version: {scipy.__version__}", file=sys.stderr)
except ImportError:
    print("ERROR: scipy is required. pip install scipy", file=sys.stderr)
    sys.exit(1)


def compute_case(diff_values, alt):
    """Compute Wilcoxon p-value using scipy, matching optuna's parameters."""
    d = np.array(diff_values, dtype=np.float64)
    result = wilcoxon(d, alternative=alt, zero_method='zsplit')
    return float(result.pvalue)


def compute_case_with_method(diff_values, alt, method):
    """Compute with explicit method for checking auto-selection."""
    d = np.array(diff_values, dtype=np.float64)
    result = wilcoxon(d, alternative=alt, zero_method='zsplit', method=method)
    return float(result.pvalue)


def convert_for_json(v):
    """Convert special float values for JSON serialization."""
    if isinstance(v, float):
        if v == float('inf'):
            return "Infinity"
        elif v == float('-inf'):
            return "-Infinity"
        elif v != v:  # NaN
            return "NaN"
    return v


cases = {}

# ========================================================================
# 1. Exact distribution cases (small n, no ties, no zeros)
# ========================================================================

# Case 1a: n=5, all positive, minimize (alt='greater')
diff_1a = [1.0, 2.0, 3.0, 4.0, 5.0]
cases["exact_all_positive_minimize"] = {
    "diff": diff_1a,
    "direction": "minimize",
    "p_value": compute_case(diff_1a, "greater"),
    "p_exact": compute_case_with_method(diff_1a, "greater", "exact"),
    "p_approx": compute_case_with_method(diff_1a, "greater", "approx"),
}

# Case 1b: n=5, all positive, maximize (alt='less')
cases["exact_all_positive_maximize"] = {
    "diff": diff_1a,
    "direction": "maximize",
    "p_value": compute_case(diff_1a, "less"),
    "p_exact": compute_case_with_method(diff_1a, "less", "exact"),
    "p_approx": compute_case_with_method(diff_1a, "less", "approx"),
}

# Case 1c: n=3, all positive
diff_1c = [1.0, 2.0, 3.0]
cases["exact_n3_all_positive"] = {
    "diff": diff_1c,
    "direction": "minimize",
    "p_value": compute_case(diff_1c, "greater"),
}

# Case 1d: n=3, mixed signs
diff_1d = [1.0, -2.0, 3.0]
cases["exact_n3_mixed"] = {
    "diff": diff_1d,
    "direction": "minimize",
    "p_value": compute_case(diff_1d, "greater"),
}

# Case 1e: n=4, mixed signs
diff_1e = [1.0, -2.0, 3.0, 4.0]
cases["exact_n4_mixed"] = {
    "diff": diff_1e,
    "direction": "minimize",
    "p_value": compute_case(diff_1e, "greater"),
}

# Case 1f: n=10, no ties → should be exact
diff_1f = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
cases["exact_n10_alternating"] = {
    "diff": diff_1f,
    "direction": "minimize",
    "p_value": compute_case(diff_1f, "greater"),
    "p_exact": compute_case_with_method(diff_1f, "greater", "exact"),
    "p_approx": compute_case_with_method(diff_1f, "greater", "approx"),
}

# Case 1g: n=20, no ties → should be exact
diff_1g = [float(i) for i in range(1, 21)]  # 1..20
for i in [1, 3, 7, 11, 15]:
    diff_1g[i] = -diff_1g[i]  # negate some
cases["exact_n20_mixed"] = {
    "diff": diff_1g,
    "direction": "minimize",
    "p_value": compute_case(diff_1g, "greater"),
}

# Case 1h: all negative diffs → R+ = 0, p ≈ 1.0 for alt='greater'
diff_1h = [-1.0, -2.0, -3.0, -4.0, -5.0]
cases["exact_all_negative_minimize"] = {
    "diff": diff_1h,
    "direction": "minimize",
    "p_value": compute_case(diff_1h, "greater"),
}

# Case 1i: all negative diffs → p should be very small for alt='less' (maximize)
cases["exact_all_negative_maximize"] = {
    "diff": diff_1h,
    "direction": "maximize",
    "p_value": compute_case(diff_1h, "less"),
}

# ========================================================================
# 2. Normal approximation cases (ties or large n)
# ========================================================================

# Case 2a: n=10 with ties (0.5 appears twice)
diff_2a = [1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2]
cases["approx_n10_ties_minimize"] = {
    "diff": diff_2a,
    "direction": "minimize",
    "p_value": compute_case(diff_2a, "greater"),
}

# Case 2b: same data, maximize
cases["approx_n10_ties_maximize"] = {
    "diff": diff_2a,
    "direction": "maximize",
    "p_value": compute_case(diff_2a, "less"),
}

# Case 2c: with zeros (zsplit)
diff_2c = [1.0, 0.0, -1.0, 2.0, 0.0, -0.5, 1.5, 0.0, -2.0, 3.0]
cases["approx_with_zeros"] = {
    "diff": diff_2c,
    "direction": "minimize",
    "p_value": compute_case(diff_2c, "greater"),
}

# Case 2d: many ties
diff_2d = [1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, -2.0, 3.0, 0.0]
cases["approx_many_ties"] = {
    "diff": diff_2d,
    "direction": "minimize",
    "p_value": compute_case(diff_2d, "greater"),
}

# Case 2e: large n (60) → always approx
np.random.seed(42)
diff_2e = list(np.random.randn(60) + 0.5)  # slight positive bias
cases["approx_n60_random"] = {
    "diff": diff_2e,
    "direction": "minimize",
    "p_value": compute_case(diff_2e, "greater"),
}

# Case 2f: large n (60), maximize
cases["approx_n60_random_maximize"] = {
    "diff": diff_2e,
    "direction": "maximize",
    "p_value": compute_case(diff_2e, "less"),
}

# ========================================================================
# 3. Boundary / edge cases
# ========================================================================

# Case 3a: all zeros → p = 1.0
diff_3a = [0.0, 0.0, 0.0, 0.0, 0.0]
cases["edge_all_zeros"] = {
    "diff": diff_3a,
    "direction": "minimize",
    # scipy might raise a warning but should return NaN or 1.0
}
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases["edge_all_zeros"]["p_value"] = compute_case(diff_3a, "greater")
except Exception as e:
    cases["edge_all_zeros"]["p_value"] = 1.0
    cases["edge_all_zeros"]["error"] = str(e)

# Case 3b: n=2 (minimum)
diff_3b = [5.0, 3.0]
cases["edge_n2"] = {
    "diff": diff_3b,
    "direction": "minimize",
    "p_value": compute_case(diff_3b, "greater"),
}

# Case 3c: single value (n=1)
diff_3c = [5.0]
cases["edge_n1"] = {
    "diff": diff_3c,
    "direction": "minimize",
    "p_value": compute_case(diff_3c, "greater"),
}

# ========================================================================
# 4. Threshold boundary: n=50 vs n=51
# ========================================================================

# Case 4a: n=50, no ties → check if exact
diff_4a = [float(i) for i in range(1, 51)]
for i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
    diff_4a[i] = -diff_4a[i]
cases["boundary_n50_no_ties"] = {
    "diff": diff_4a,
    "direction": "minimize",
    "p_value": compute_case(diff_4a, "greater"),
    "p_exact": compute_case_with_method(diff_4a, "greater", "exact"),
    "p_approx": compute_case_with_method(diff_4a, "greater", "approx"),
}

# Case 4b: n=51, no ties → should be approx
diff_4b = [float(i) for i in range(1, 52)]
for i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    diff_4b[i] = -diff_4b[i]
cases["boundary_n51_no_ties"] = {
    "diff": diff_4b,
    "direction": "minimize",
    "p_value": compute_case(diff_4b, "greater"),
    "p_approx": compute_case_with_method(diff_4b, "greater", "approx"),
}

# Case 4c: n=25, no ties → check if exact  
diff_4c = [float(i) for i in range(1, 26)]
for i in [0, 5, 10, 15, 20]:
    diff_4c[i] = -diff_4c[i]
cases["boundary_n25_no_ties"] = {
    "diff": diff_4c,
    "direction": "minimize",
    "p_value": compute_case(diff_4c, "greater"),
    "p_exact": compute_case_with_method(diff_4c, "greater", "exact"),
    "p_approx": compute_case_with_method(diff_4c, "greater", "approx"),
}

# ========================================================================
# 5. Print method selection info (diagnostic)
# ========================================================================

for name, case in cases.items():
    if "p_exact" in case and "p_approx" in case:
        p_auto = case["p_value"]
        p_exact = case["p_exact"]
        p_approx = case["p_approx"]
        if abs(p_auto - p_exact) < 1e-15:
            method_sel = "exact"
        elif abs(p_auto - p_approx) < 1e-15:
            method_sel = "approx"
        else:
            method_sel = f"unknown (auto={p_auto}, exact={p_exact}, approx={p_approx})"
        case["method_selected"] = method_sel
        print(f"  {name}: auto={p_auto:.15e}, exact={p_exact:.15e}, approx={p_approx:.15e} → {method_sel}", file=sys.stderr)

# ========================================================================
# 6. Serialize
# ========================================================================

# Convert all float values
def deep_convert(obj):
    if isinstance(obj, dict):
        return {k: deep_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_convert(x) for x in obj]
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

output = deep_convert(cases)
print(json.dumps(output, indent=2))
