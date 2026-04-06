#!/usr/bin/env python3
"""生成 GP 模块的 golden values 用于 Rust 交叉验证。

覆盖: normalize/unnormalize, erfcx, erfinv, log_ndtr, standard_logei,
      default_log_prior, GP LML, constant liar 后验变化。
"""
import json
import math
import numpy as np

results = {}

# === 1. normalize_param golden values ===
# 对齐 Python search_space._normalize_one_param / _unnormalize_one_param
norm_cases = {}

# 1a: Float linear, no step
# normalize: (v - low) / (high - low)
for (val, low, high) in [(5.0, 0.0, 10.0), (0.0, 0.0, 10.0), (10.0, 0.0, 10.0),
                          (3.0, -5.0, 5.0), (-5.0, -5.0, 5.0)]:
    key = f"float_linear_{val}_{low}_{high}"
    norm = (val - low) / (high - low)
    unnorm = norm * (high - low) + low
    norm_cases[key] = {"normalized": norm, "unnormalized": unnorm}

# 1b: Float log, no step
for (val, low, high) in [(0.01, 0.001, 1.0), (0.001, 0.001, 1.0), (1.0, 0.001, 1.0)]:
    key = f"float_log_{val}_{low}_{high}"
    log_low = math.log(low)
    log_high = math.log(high)
    norm = (math.log(val) - log_low) / (log_high - log_low)
    unnorm = math.exp(norm * (log_high - log_low) + log_low)
    norm_cases[key] = {"normalized": norm, "unnormalized": unnorm}

# 1c: Float with step
for (val, low, high, step) in [(5.0, 0.0, 10.0, 2.0), (0.0, 0.0, 10.0, 2.0)]:
    key = f"float_step_{val}_{low}_{high}_{step}"
    adj_low = low - 0.5 * step
    adj_high = high + 0.5 * step
    norm = (val - adj_low) / (adj_high - adj_low)
    unnorm = norm * (adj_high - adj_low) + adj_low
    unnorm = max(low, min(high, unnorm))
    norm_cases[key] = {"normalized": norm, "unnormalized": unnorm}

# 1d: Int
for (val, low, high, step) in [(5, 0, 10, 1), (0, 0, 10, 1), (4, 0, 10, 2)]:
    key = f"int_{val}_{low}_{high}_{step}"
    adj_low = low - 0.5 * step
    adj_high = high + 0.5 * step
    norm = (val - adj_low) / (adj_high - adj_low)
    unnorm = norm * (adj_high - adj_low) + adj_low
    unnorm = max(low, min(high, round(unnorm)))
    norm_cases[key] = {"normalized": norm, "unnormalized": float(unnorm)}

results["normalize"] = norm_cases

# === 2. erfcx golden values ===
# erfcx(x) = exp(x²) * erfc(x)
from scipy.special import erfcx as scipy_erfcx
erfcx_cases = {}
for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
    erfcx_cases[str(x)] = float(scipy_erfcx(x))
results["erfcx"] = erfcx_cases

# === 3. erfinv golden values ===
from scipy.special import erfinv as scipy_erfinv
erfinv_cases = {}
for x in [-0.999, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.999]:
    erfinv_cases[str(x)] = float(scipy_erfinv(x))
results["erfinv"] = erfinv_cases

# === 4. log_ndtr golden values ===
from scipy.special import log_ndtr as scipy_log_ndtr
log_ndtr_cases = {}
for z in [-50.0, -30.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 8.0, 10.0]:
    log_ndtr_cases[str(z)] = float(scipy_log_ndtr(z))
results["log_ndtr"] = log_ndtr_cases

# === 5. standard_logei golden values ===
# standard_logei(z) = log(E_{x~N(0,1)}[max(0, x+z)])
# = log(z * Φ(z) + φ(z))
from scipy.stats import norm
def standard_logei_python(z):
    """对齐 Python acqf.standard_logei"""
    ei = z * norm.cdf(z) + norm.pdf(z)
    if ei <= 0:
        return -1e308
    return math.log(ei)

logei_cases = {}
for z in [-40.0, -30.0, -25.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0]:
    logei_cases[str(z)] = standard_logei_python(z)
results["standard_logei"] = logei_cases

# === 6. logEI(mean, var, f0) golden values ===
# logEI = standard_logei((mean - f0) / sigma) + log(sigma)
log_ei_full_cases = {}
for (mean, var, f0) in [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 1.0, 0.0),
                         (2.0, 0.5, 1.0), (0.5, 2.0, 0.0), (10.0, 0.01, 9.0),
                         (-5.0, 1.0, 0.0)]:
    sigma = math.sqrt(var)
    z = (mean - f0) / sigma
    lei = standard_logei_python(z) + math.log(sigma)
    log_ei_full_cases[f"{mean},{var},{f0}"] = lei
results["logei"] = log_ei_full_cases

# === 7. default_log_prior golden values ===
prior_cases = {}
for (inv_sq_ls, ks, nv) in [
    ([1.0, 1.0], 1.0, 1e-6),
    ([0.5, 2.0], 0.5, 0.01),
    ([0.1, 0.1, 0.1], 2.0, 0.001),
    ([10.0], 0.01, 1e-6),
]:
    ls_prior = sum(-(0.1/x + 0.1*x) for x in inv_sq_ls)
    ks_prior = math.log(ks) - ks
    nv_prior = 0.1 * math.log(nv) - 30.0 * nv
    total = ls_prior + ks_prior + nv_prior
    key = f"{inv_sq_ls}|{ks}|{nv}"
    prior_cases[key] = {"ls_prior": ls_prior, "ks_prior": ks_prior,
                        "nv_prior": nv_prior, "total": total}
results["log_prior"] = prior_cases

# === 8. GP LML golden values ===
# 使用已知核参数计算 LML
lml_cases = {}

# Case 8a: 1D, 3 points
X_1d = np.array([[0.0], [0.5], [1.0]])
y_1d = np.array([0.0, 1.0, 0.0])
inv_sq_ls_1d = np.array([1.0])
kernel_scale_1d = 1.0
noise_var_1d = 0.01

# 手动计算 Matern 5/2 核矩阵
def matern52_py(d2):
    if d2 < 1e-30:
        return 1.0
    sqrt5d = math.sqrt(5 * d2)
    return math.exp(-sqrt5d) * (5/3 * d2 + sqrt5d + 1)

K_1d = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        d2 = (X_1d[i,0] - X_1d[j,0])**2 * inv_sq_ls_1d[0]
        K_1d[i,j] = kernel_scale_1d * matern52_py(d2)
K_1d_noisy = K_1d + noise_var_1d * np.eye(3)
L_1d = np.linalg.cholesky(K_1d_noisy)
alpha_1d = np.linalg.solve(K_1d_noisy, y_1d)
logdet_1d = 2 * np.sum(np.log(np.diag(L_1d)))
quad_1d = y_1d @ alpha_1d
lml_1d = -0.5 * logdet_1d - 0.5 * quad_1d - 0.5 * 3 * math.log(2 * math.pi)

lml_cases["1d_3pts"] = {
    "X": X_1d.tolist(),
    "y": y_1d.tolist(),
    "inv_sq_ls": inv_sq_ls_1d.tolist(),
    "kernel_scale": kernel_scale_1d,
    "noise_var": noise_var_1d,
    "lml": float(lml_1d),
    "kernel_matrix_diag": [float(K_1d[i,i]) for i in range(3)],
}

# Case 8b: 2D, 4 points
X_2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y_2d = np.array([1.0, -1.0, -1.0, 1.0])
inv_sq_ls_2d = np.array([2.0, 0.5])
kernel_scale_2d = 1.5
noise_var_2d = 0.05

K_2d = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        d2 = sum((X_2d[i,k] - X_2d[j,k])**2 * inv_sq_ls_2d[k] for k in range(2))
        K_2d[i,j] = kernel_scale_2d * matern52_py(d2)
K_2d_noisy = K_2d + noise_var_2d * np.eye(4)
L_2d = np.linalg.cholesky(K_2d_noisy)
alpha_2d = np.linalg.solve(K_2d_noisy, y_2d)
logdet_2d = 2 * np.sum(np.log(np.diag(L_2d)))
quad_2d = y_2d @ alpha_2d
lml_2d = -0.5 * logdet_2d - 0.5 * quad_2d - 0.5 * 4 * math.log(2 * math.pi)

lml_cases["2d_4pts"] = {
    "X": X_2d.tolist(),
    "y": y_2d.tolist(),
    "inv_sq_ls": inv_sq_ls_2d.tolist(),
    "kernel_scale": kernel_scale_2d,
    "noise_var": noise_var_2d,
    "lml": float(lml_2d),
}

# 后验预测
posterior_cases = {}
x_pred = [0.25, 0.75]
k_star_2d = np.array([kernel_scale_2d * matern52_py(
    sum((x_pred[k] - X_2d[j,k])**2 * inv_sq_ls_2d[k] for k in range(2))
) for j in range(4)])
mean_pred = k_star_2d @ alpha_2d
v = np.linalg.solve(L_2d, k_star_2d)
var_pred = kernel_scale_2d - np.dot(v, v)
var_pred = max(0.0, var_pred)

posterior_cases["2d_at_025_075"] = {
    "x": x_pred,
    "mean": float(mean_pred),
    "var": float(var_pred),
    "k_star": k_star_2d.tolist(),
}

results["gp_lml"] = lml_cases
results["gp_posterior"] = posterior_cases

# === 9. Matern 5/2 核值 ===
matern_cases = {}
for d2 in [0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
    matern_cases[str(d2)] = matern52_py(d2)
results["matern52"] = matern_cases

# Save
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return 1e308 if obj > 0 else -1e308
        elif math.isnan(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        return sanitize(float(obj))
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    return obj

with open("/Users/lichangqing/Copilot/optuna/optuna-rs/tests/gp_deep_golden_values.json", "w") as f:
    json.dump(sanitize(results), f, indent=2)

print(f"Generated {len(results)} golden value groups")
for k, v in results.items():
    if isinstance(v, dict):
        print(f"  {k}: {len(v)} cases")
