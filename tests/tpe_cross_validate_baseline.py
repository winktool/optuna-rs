#!/usr/bin/env python3
"""
Generate Python reference values for TPE cross-validation with Rust tests.
Covers: truncnorm math, parzen estimator kernels, log-pdf, weights, sampling statistics.

Run: python3 tests/tpe_cross_validate_baseline.py > tests/tpe_baseline_values.json
"""
import json, sys, math
sys.path.insert(0, "/Users/lichangqing/Copilot/optuna/optuna")

import numpy as np
from optuna.samplers._tpe._truncnorm import (
    _log_gauss_mass, _log_ndtr, _ndtr, logpdf as tn_logpdf, ppf as tn_ppf
)
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator, _ParzenEstimatorParameters
)
from optuna.samplers._tpe.sampler import default_gamma, default_weights
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

r = {}

# ═══════════════════════════════════════════════════════════════════════════
#  Section 1: truncnorm math — ndtr, log_ndtr, log_gauss_mass
# ═══════════════════════════════════════════════════════════════════════════

# 1a. ndtr (standard normal CDF) at known points
ndtr_inputs = [-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, -20.0, -40.0, 10.0]
for x in ndtr_inputs:
    r[f"ndtr_{x}"] = float(_ndtr(np.array([x]))[0])

# 1b. log_ndtr at known points (including extreme tails)
log_ndtr_inputs = [-40.0, -20.0, -10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0]
for x in log_ndtr_inputs:
    r[f"log_ndtr_{x}"] = float(_log_ndtr(np.array([x]))[0])

# 1c. log_gauss_mass at various intervals
lgm_cases = [
    (-10.0, 10.0),   # full range
    (-10.0, 0.0),    # left half
    (0.0, 10.0),     # right half
    (-1.0, 1.0),     # central ±1σ
    (-2.0, 2.0),     # central ±2σ
    (-3.0, -2.0),    # left tail
    (2.0, 3.0),      # right tail
    (-0.5, 0.5),     # narrow center
    (-30.0, -20.0),  # deep left tail
    (20.0, 30.0),    # deep right tail
    (-0.01, 0.01),   # very narrow center
]
for a, b in lgm_cases:
    val = float(_log_gauss_mass(np.array([a]), np.array([b]))[0])
    r[f"log_gauss_mass_{a}_{b}"] = val

# ═══════════════════════════════════════════════════════════════════════════
#  Section 2: truncnorm logpdf
# ═══════════════════════════════════════════════════════════════════════════

# 2a. Standard truncated normal logpdf
logpdf_cases = [
    # (x, a, b, loc, scale)
    (0.0, -1.0, 1.0, 0.0, 1.0),
    (0.5, -2.0, 2.0, 0.0, 1.0),
    (-0.5, -2.0, 2.0, 0.0, 1.0),
    (0.0, -10.0, 10.0, 0.0, 1.0),
    (5.0, 0.0, 10.0, 3.0, 2.0),
    (0.1, 0.0, 1.0, 0.5, 0.3),
    (0.99, 0.0, 1.0, 0.5, 0.3),
    # Out of range → -inf
    (2.0, -1.0, 1.0, 0.0, 1.0),
    (-2.0, -1.0, 1.0, 0.0, 1.0),
]
for i, (x, a, b, loc, scale) in enumerate(logpdf_cases):
    val = float(tn_logpdf(
        np.array([x]),
        a=np.array([(a - loc) / scale]),
        b=np.array([(b - loc) / scale]),
        loc=np.array([loc]),
        scale=np.array([scale]),
    )[0])
    r[f"tn_logpdf_{i}"] = val

# ═══════════════════════════════════════════════════════════════════════════
#  Section 3: truncnorm PPF (quantile function) 
# ═══════════════════════════════════════════════════════════════════════════

ppf_cases = [
    # (q, a, b)
    (0.0, -2.0, 2.0),
    (0.5, -2.0, 2.0),
    (1.0, -2.0, 2.0),
    (0.1, -1.0, 1.0),
    (0.25, -1.0, 1.0),
    (0.75, -1.0, 1.0),
    (0.9, -1.0, 1.0),
    (0.5, -10.0, 10.0),
    (0.01, -3.0, 3.0),
    (0.99, -3.0, 3.0),
    # Right-side (a >= 0)
    (0.5, 0.0, 5.0),
    (0.1, 1.0, 3.0),
    (0.9, 1.0, 3.0),
    # Left-side (b <= 0)
    (0.5, -5.0, 0.0),
    (0.1, -3.0, -1.0),
    (0.9, -3.0, -1.0),
    # Extreme quantiles
    (0.001, -5.0, 5.0),
    (0.999, -5.0, 5.0),
]
for i, (q, a, b) in enumerate(ppf_cases):
    val = float(tn_ppf(np.array([q]), np.array([a]), np.array([b]))[0])
    r[f"tn_ppf_{i}"] = val

# ═══════════════════════════════════════════════════════════════════════════
#  Section 4: Parzen Estimator — kernel computation
# ═══════════════════════════════════════════════════════════════════════════

def make_pe(obs_dict, search_space, **kwargs):
    """Helper to create a ParzenEstimator with default params."""
    params = _ParzenEstimatorParameters(
        prior_weight=kwargs.get("prior_weight", 1.0),
        consider_magic_clip=kwargs.get("consider_magic_clip", True),
        consider_endpoints=kwargs.get("consider_endpoints", False),
        weights=default_weights,
        multivariate=kwargs.get("multivariate", False),
        categorical_distance_func={},
    )
    arrays = {k: np.array(v) for k, v in obs_dict.items()}
    return _ParzenEstimator(arrays, search_space, params)

# 4a. Float [0, 10] with 3 observations → kernel centers and bandwidths
obs_x = [2.0, 5.0, 8.0]
ss_float = {"x": FloatDistribution(0.0, 10.0)}
pe = make_pe({"x": obs_x}, ss_float)
# Extract mus and sigmas from the internal distribution
dist_info = pe._mixture_distribution.distributions[0]
r["pe_float_mus"] = dist_info.mu.tolist()
r["pe_float_sigmas"] = dist_info.sigma.tolist()
r["pe_float_weights"] = pe._mixture_distribution.weights.tolist()

# 4b. Float [0, 10] with no observations
pe_empty = make_pe({"x": []}, ss_float)
dist_empty = pe_empty._mixture_distribution.distributions[0]
r["pe_float_empty_mus"] = dist_empty.mu.tolist()
r["pe_float_empty_sigmas"] = dist_empty.sigma.tolist()
r["pe_float_empty_weights"] = pe_empty._mixture_distribution.weights.tolist()

# 4c. Log scale [0.001, 1.0] with observations
ss_log = {"lr": FloatDistribution(0.001, 1.0, log=True)}
obs_lr = [0.01, 0.1]
pe_log = make_pe({"lr": obs_lr}, ss_log)
dist_log = pe_log._mixture_distribution.distributions[0]
r["pe_log_mus"] = dist_log.mu.tolist()
r["pe_log_sigmas"] = dist_log.sigma.tolist()

# 4d. Int with step: [0, 10] step=2
ss_int_step = {"n": IntDistribution(0, 10, step=2)}
obs_n = [2.0, 4.0, 6.0]
pe_int_step = make_pe({"n": obs_n}, ss_int_step)
dist_int = pe_int_step._mixture_distribution.distributions[0]
r["pe_int_step_mus"] = dist_int.mu.tolist()
r["pe_int_step_sigmas"] = dist_int.sigma.tolist()

# 4e. Int log step: [1, 100] log=True step=1
ss_int_log = {"x": IntDistribution(1, 100, log=True)}
obs_int_log = [10.0, 20.0, 50.0]
pe_int_log = make_pe({"x": obs_int_log}, ss_int_log)
dist_int_log = pe_int_log._mixture_distribution.distributions[0]
r["pe_int_log_mus"] = dist_int_log.mu.tolist()
r["pe_int_log_sigmas"] = dist_int_log.sigma.tolist()
r["pe_int_log_low"] = dist_int_log.low
r["pe_int_log_high"] = dist_int_log.high

# 4f. With consider_endpoints=True
pe_ep = make_pe({"x": obs_x}, ss_float, consider_endpoints=True)
dist_ep = pe_ep._mixture_distribution.distributions[0]
r["pe_endpoints_sigmas"] = dist_ep.sigma.tolist()

# 4g. Without magic clip
pe_nomc = make_pe({"x": obs_x}, ss_float, consider_magic_clip=False)
dist_nomc = pe_nomc._mixture_distribution.distributions[0]
r["pe_nomagicclip_sigmas"] = dist_nomc.sigma.tolist()

# 4h. Multivariate sigma computation
ss_multi = {"x": FloatDistribution(0.0, 10.0), "y": FloatDistribution(-5.0, 5.0)}
pe_multi = make_pe(
    {"x": [2.0, 5.0, 8.0], "y": [-1.0, 0.0, 3.0]},
    ss_multi, multivariate=True,
)
dist_mx = pe_multi._mixture_distribution.distributions[0]
dist_my = pe_multi._mixture_distribution.distributions[1]
r["pe_multi_x_sigmas"] = dist_mx.sigma.tolist()
r["pe_multi_y_sigmas"] = dist_my.sigma.tolist()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 5: Parzen Estimator — log_pdf computation
# ═══════════════════════════════════════════════════════════════════════════

# 5a. log_pdf for continuous float
samples_x = {"x": np.array([1.0, 3.0, 5.0, 7.0, 9.0])}
logpdf_vals = pe.log_pdf(samples_x).tolist()
r["pe_logpdf_float"] = logpdf_vals

# 5b. log_pdf for log-scale float  
samples_lr = {"lr": np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])}
logpdf_lr = pe_log.log_pdf(samples_lr).tolist()
r["pe_logpdf_log"] = logpdf_lr

# 5c. log_pdf for discrete int with step
samples_n = {"n": np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])}
logpdf_n = pe_int_step.log_pdf(samples_n).tolist()
r["pe_logpdf_int_step"] = logpdf_n

# 5d. log_pdf for discrete log-int
samples_intlog = {"x": np.array([1.0, 10.0, 20.0, 50.0, 100.0])}
logpdf_intlog = pe_int_log.log_pdf(samples_intlog).tolist()
r["pe_logpdf_int_log"] = logpdf_intlog

# ═══════════════════════════════════════════════════════════════════════════
#  Section 6: Categorical Parzen Estimator
# ═══════════════════════════════════════════════════════════════════════════

ss_cat = {"opt": CategoricalDistribution(["a", "b", "c"])}

# 6a. Empty observations → uniform prior
pe_cat_empty = make_pe({"opt": []}, ss_cat)
cat_d = pe_cat_empty._mixture_distribution.distributions[0]
r["pe_cat_empty_weights"] = cat_d.weights.tolist()

# 6b. With observations
pe_cat = make_pe({"opt": [0.0, 0.0, 1.0]}, ss_cat)
cat_d2 = pe_cat._mixture_distribution.distributions[0]
r["pe_cat_obs_weights"] = cat_d2.weights.tolist()  # n_kernels × n_choices

# 6c. log_pdf for categorical
cat_samples = {"opt": np.array([0.0, 1.0, 2.0])}
cat_logpdf = pe_cat.log_pdf(cat_samples).tolist()
r["pe_cat_logpdf"] = cat_logpdf

# ═══════════════════════════════════════════════════════════════════════════
#  Section 7: default_weights precise values
# ═══════════════════════════════════════════════════════════════════════════

for n in [0, 1, 5, 10, 24, 25, 26, 30, 50, 100]:
    r[f"default_weights_{n}"] = default_weights(n).tolist()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 8: default_gamma precise values
# ═══════════════════════════════════════════════════════════════════════════

for n in [0, 1, 5, 10, 20, 50, 100, 200, 250, 300, 1000]:
    r[f"default_gamma_{n}"] = default_gamma(n)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 9: Parzen Estimator with predetermined weights (MOTPE)
# ═══════════════════════════════════════════════════════════════════════════

# 9a. predetermined_weights override
pw = np.array([0.5, 0.3, 0.2])
pe_pw = _ParzenEstimator(
    {"x": np.array([2.0, 5.0, 8.0])},
    ss_float,
    _ParzenEstimatorParameters(1.0, True, False, default_weights, False, {}),
    predetermined_weights=pw,
)
r["pe_predetermined_weights"] = pe_pw._mixture_distribution.weights.tolist()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 10: Edge case — very close observations
# ═══════════════════════════════════════════════════════════════════════════

pe_close = make_pe({"x": [5.0, 5.0, 5.0]}, ss_float)
dist_close = pe_close._mixture_distribution.distributions[0]
r["pe_close_obs_mus"] = dist_close.mu.tolist()
r["pe_close_obs_sigmas"] = dist_close.sigma.tolist()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 11: Sigma clipping bounds
# ═══════════════════════════════════════════════════════════════════════════

# Single observation: magic_clip minsigma = (10-0) / min(100, 1+2) = 10/3 ≈ 3.333
pe_single = make_pe({"x": [5.0]}, ss_float)
dist_s = pe_single._mixture_distribution.distributions[0]
r["pe_single_obs_sigmas"] = dist_s.sigma.tolist()

# ═══════════════════════════════════════════════════════════════════════════
#  Section 12: ndtri_exp — inverse of log_ndtr
# ═══════════════════════════════════════════════════════════════════════════
from optuna.samplers._tpe._truncnorm import _ndtri_exp

ndtri_exp_inputs = [
    -0.001,   # near zero (flipped region)
    -0.01,    # boundary of flipped region
    -0.1,     # moderate near zero
    -0.5,     # log(0.6065)
    -0.6931471805599453,  # log(0.5)→x=0
    -1.0,
    -2.0,
    -5.0,
    -10.0,
    -20.0,
    -50.0,
    -100.0,
    -500.0,
    -1000.0,
]
for y in ndtri_exp_inputs:
    x = _ndtri_exp(np.array([y]))[0]
    r[f"ndtri_exp_{y}"] = float(x)

# ═══════════════════════════════════════════════════════════════════════════
#  Section 13: PPF at tighter tolerance — near-boundary quantiles
# ═══════════════════════════════════════════════════════════════════════════
ppf_tight_cases = [
    # (q, a, b)
    (0.5, -0.5, 0.5),
    (0.001, -0.5, 0.5),
    (0.999, -0.5, 0.5),
    (0.5, -20.0, -10.0),  # deep left tail
    (0.5, 10.0, 20.0),     # deep right tail
    (0.001, -20.0, 20.0),
    (0.999, -20.0, 20.0),
    (1e-6, -5.0, 5.0),     # extreme quantile
    (1.0 - 1e-6, -5.0, 5.0),
]
for i, (q, a, bb) in enumerate(ppf_tight_cases):
    v = tn_ppf(np.array([q]), a, bb)[0]
    r[f"tn_ppf_tight_{i}"] = float(v)

# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

# Convert any remaining numpy types
def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        if math.isnan(v):
            return "NaN"
        return v
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    if isinstance(obj, np.ndarray):
        return [convert(x) for x in obj.tolist()]
    if isinstance(obj, list):
        return [convert(x) for x in obj]
    return obj

r = {k: convert(v) for k, v in r.items()}
print(json.dumps(r, indent=2))
