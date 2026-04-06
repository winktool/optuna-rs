#!/usr/bin/env python3
"""生成 TPE Parzen Estimator 的 golden values 用于 Rust 交叉验证。"""
import json
import numpy as np
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.samplers._tpe.sampler import default_weights, default_gamma, hyperopt_default_gamma
from optuna.samplers._tpe._truncnorm import _ndtr_single, _log_ndtr_single, _log_gauss_mass, ppf, logpdf
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

results = {}

# === 1. default_weights golden values ===
weights_cases = {}
for n in [0, 1, 5, 10, 24, 25, 26, 30, 50, 100]:
    w = default_weights(n)
    weights_cases[str(n)] = w.tolist() if isinstance(w, np.ndarray) else list(w)
results["default_weights"] = weights_cases

# === 2. default_gamma golden values ===
gamma_cases = {}
for n in [0, 1, 5, 10, 20, 50, 100, 250, 300]:
    gamma_cases[str(n)] = default_gamma(n)
results["default_gamma"] = gamma_cases

# === 3. hyperopt_default_gamma golden values ===
hgamma_cases = {}
for n in [0, 1, 4, 16, 17, 25, 64, 100, 10000, 100000, 1000000]:
    hgamma_cases[str(n)] = hyperopt_default_gamma(n)
results["hyperopt_default_gamma"] = hgamma_cases

# === 4. ndtr golden values ===
ndtr_cases = {}
for x in [-40.0, -10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0]:
    ndtr_cases[str(x)] = _ndtr_single(x)
results["ndtr"] = ndtr_cases

# === 5. log_ndtr golden values ===
log_ndtr_cases = {}
for x in [-40.0, -25.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 5.0, 7.0, 10.0]:
    log_ndtr_cases[str(x)] = _log_ndtr_single(x)
results["log_ndtr"] = log_ndtr_cases

# === 6. log_gauss_mass golden values ===
lgm_cases = {}
for (a, b) in [(-10.0, 10.0), (-10.0, 0.0), (0.0, 10.0), (-2.0, 2.0), (-1.0, 1.0),
               (-5.0, -3.0), (3.0, 5.0), (-0.5, 0.5), (-20.0, -15.0), (1.0, 2.0)]:
    a_arr = np.array([a])
    b_arr = np.array([b])
    val = _log_gauss_mass(a_arr, b_arr)[0]
    lgm_cases[f"{a},{b}"] = float(val.real) if np.iscomplex(val) else float(val)
results["log_gauss_mass"] = lgm_cases

# === 7. ppf golden values ===
ppf_cases = {}
for (q, a, b) in [(0.0, -2.0, 2.0), (0.5, -2.0, 2.0), (1.0, -2.0, 2.0),
                   (0.1, -1.0, 1.0), (0.25, -1.0, 1.0), (0.5, -1.0, 1.0),
                   (0.75, -1.0, 1.0), (0.9, -1.0, 1.0),
                   (0.5, -10.0, 10.0), (0.5, 0.0, 5.0), (0.5, -5.0, 0.0),
                   (0.01, -3.0, 3.0), (0.99, -3.0, 3.0)]:
    val = ppf(np.array([q]), np.array([a]), np.array([b]))[0]
    ppf_cases[f"{q},{a},{b}"] = float(val)
results["ppf"] = ppf_cases

# === 8. logpdf golden values ===
# 注意: Rust's truncnorm::logpdf(x, a, b, loc, scale) 使用物理边界 a, b
# Python's logpdf(x, a_std, b_std, loc, scale) 使用标准化边界
# 为了交叉验证，传入 loc=0, scale=1 使两者一致
logpdf_cases = {}
for (x, a, b, loc, scale) in [
    (0.0, -1.0, 1.0, 0.0, 1.0),
    (0.5, -1.0, 1.0, 0.0, 1.0),
    (-0.5, -2.0, 2.0, 0.0, 1.0),
    (0.0, -3.0, 3.0, 0.0, 1.0),
    (2.0, -1.0, 1.0, 0.0, 1.0),  # out of range → -inf
    (1.0, -2.0, 2.0, 0.0, 1.0),
    (-1.0, -2.0, 2.0, 0.0, 1.0),
]:
    # 对于 Rust API: a, b 是物理边界，所以 a_std = (a-loc)/scale = a (when loc=0, scale=1)
    val = logpdf(np.array([x]), np.array([a]), np.array([b]), np.array([loc]), np.array([scale]))[0]
    logpdf_cases[f"{x},{a},{b},{loc},{scale}"] = float(val)
results["logpdf"] = logpdf_cases

# === 9. Parzen Estimator kernel parameters ===
pe_cases = {}

# Case 9a: univariate, 3 observations
search_space_9a = {"x": FloatDistribution(0.0, 10.0)}
obs_9a = {"x": np.array([2.0, 5.0, 8.0])}
params_9a = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func={},
)
pe_9a = _ParzenEstimator(obs_9a, search_space_9a, params_9a)
# Extract weights and per-param distributions
pe_9a_weights = pe_9a._mixture_distribution.weights.tolist()
pe_9a_dist = pe_9a._mixture_distribution.distributions[0]
pe_cases["univariate_3obs"] = {
    "weights": pe_9a_weights,
    "mus": pe_9a_dist.mu.tolist(),
    "sigmas": pe_9a_dist.sigma.tolist(),
    "low": pe_9a_dist.low,
    "high": pe_9a_dist.high,
}

# Case 9b: univariate, 0 observations
obs_9b = {"x": np.array([])}
pe_9b = _ParzenEstimator(obs_9b, search_space_9a, params_9a)
pe_9b_weights = pe_9b._mixture_distribution.weights.tolist()
pe_9b_dist = pe_9b._mixture_distribution.distributions[0]
pe_cases["univariate_0obs"] = {
    "weights": pe_9b_weights,
    "mus": pe_9b_dist.mu.tolist(),
    "sigmas": pe_9b_dist.sigma.tolist(),
    "low": pe_9b_dist.low,
    "high": pe_9b_dist.high,
}

# Case 9c: univariate with consider_endpoints=True
params_9c = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=True,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func={},
)
pe_9c = _ParzenEstimator(obs_9a, search_space_9a, params_9c)
pe_9c_dist = pe_9c._mixture_distribution.distributions[0]
pe_cases["univariate_3obs_endpoints"] = {
    "weights": pe_9c._mixture_distribution.weights.tolist(),
    "mus": pe_9c_dist.mu.tolist(),
    "sigmas": pe_9c_dist.sigma.tolist(),
    "low": pe_9c_dist.low,
    "high": pe_9c_dist.high,
}

# Case 9d: log-scale
# 注意: Python _BatchedTruncLogNormDistributions 存储原始 low/high
# 而 Rust 存储 log 变换后的 low/high，所以这里需要转换
search_space_9d = {"lr": FloatDistribution(0.001, 1.0, log=True)}
obs_9d = {"lr": np.array([0.01, 0.1])}
pe_9d = _ParzenEstimator(obs_9d, search_space_9d, params_9a)
pe_9d_dist = pe_9d._mixture_distribution.distributions[0]
import math
pe_cases["log_scale_2obs"] = {
    "weights": pe_9d._mixture_distribution.weights.tolist(),
    "mus": pe_9d_dist.mu.tolist(),
    "sigmas": pe_9d_dist.sigma.tolist(),
    "low": math.log(pe_9d_dist.low),     # ln(0.001) — 与 Rust 内部存储一致
    "high": math.log(pe_9d_dist.high),    # ln(1.0) — 与 Rust 内部存储一致
}

# Case 9e: categorical
search_space_9e = {"opt": CategoricalDistribution(["a", "b", "c"])}
obs_9e = {"opt": np.array([0.0, 0.0, 1.0])}
pe_9e = _ParzenEstimator(obs_9e, search_space_9e, params_9a)
pe_9e_dist = pe_9e._mixture_distribution.distributions[0]
pe_cases["categorical_3obs"] = {
    "weights": pe_9e._mixture_distribution.weights.tolist(),
    "cat_weights": pe_9e_dist.weights.tolist(),
}

# Case 9f: multivariate with 2D search space
search_space_9f = {
    "x": FloatDistribution(0.0, 10.0),
    "y": FloatDistribution(-5.0, 5.0),
}
obs_9f = {
    "x": np.array([2.0, 5.0, 8.0]),
    "y": np.array([-1.0, 0.0, 1.0]),
}
params_9f = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=True,
    categorical_distance_func={},
)
pe_9f = _ParzenEstimator(obs_9f, search_space_9f, params_9f)
pe_9f_x_dist = pe_9f._mixture_distribution.distributions[0]
pe_9f_y_dist = pe_9f._mixture_distribution.distributions[1]
pe_cases["multivariate_2d_3obs"] = {
    "weights": pe_9f._mixture_distribution.weights.tolist(),
    "x_mus": pe_9f_x_dist.mu.tolist(),
    "x_sigmas": pe_9f_x_dist.sigma.tolist(),
    "y_mus": pe_9f_y_dist.mu.tolist(),
    "y_sigmas": pe_9f_y_dist.sigma.tolist(),
}

# Case 9g: int with step
# 注意: Python _BatchedDiscreteTruncNormDistributions 存储原始 low/high
# 而 Rust 存储 step 扩展后的 low/high，所以这里需要转换
search_space_9g = {"n": IntDistribution(0, 10, step=2)}
obs_9g = {"n": np.array([2.0, 4.0, 6.0])}
pe_9g = _ParzenEstimator(obs_9g, search_space_9g, params_9a)
pe_9g_dist = pe_9g._mixture_distribution.distributions[0]
pe_cases["int_step2_3obs"] = {
    "weights": pe_9g._mixture_distribution.weights.tolist(),
    "mus": pe_9g_dist.mu.tolist(),
    "sigmas": pe_9g_dist.sigma.tolist(),
    "low": pe_9g_dist.low - pe_9g_dist.step / 2,   # 0 - 1 = -1 — 与 Rust 内部存储一致
    "high": pe_9g_dist.high + pe_9g_dist.step / 2,  # 10 + 1 = 11 — 与 Rust 内部存储一致
    "step": pe_9g_dist.step,
}

# Case 9h: predetermined weights
pe_9h = _ParzenEstimator(obs_9a, search_space_9a, params_9a, predetermined_weights=np.array([0.5, 0.3, 0.2]))
pe_cases["predetermined_weights"] = {
    "weights": pe_9h._mixture_distribution.weights.tolist(),
}

results["parzen_estimator"] = pe_cases

# === 10. log_pdf golden values for PE ===
logpdf_pe_cases = {}

# Use PE from 9a (3 obs, x ∈ [0, 10])
rng = np.random.RandomState(42)
sample_vals = pe_9a.sample(rng, 5)
transformed = pe_9a._transform(sample_vals)
log_pdf_vals = pe_9a._mixture_distribution.log_pdf(transformed)
logpdf_pe_cases["univariate_3obs"] = {
    "samples_x": sample_vals["x"].tolist(),
    "log_pdf": log_pdf_vals.tolist(),
}
results["parzen_estimator_logpdf"] = logpdf_pe_cases

# Save
import math

def sanitize(obj):
    """Replace inf/-inf/nan with JSON-safe values."""
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
    return obj

with open("/Users/lichangqing/Copilot/optuna/optuna-rs/tests/tpe_deep_golden_values.json", "w") as f:
    json.dump(sanitize(results), f, indent=2)

print(f"Generated {len(results)} golden value groups")
for k, v in results.items():
    if isinstance(v, dict):
        print(f"  {k}: {len(v)} cases")
