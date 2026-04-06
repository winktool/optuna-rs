#!/usr/bin/env python3
"""
TPE ParzenEstimator log-pdf 的金标准值。
"""
import numpy as np

print("=== ParzenEstimator log_pdf ===")
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
from optuna.distributions import FloatDistribution

# Create a simple 1D parzen estimator
dist = FloatDistribution(0.0, 10.0)
from optuna.samplers._tpe.sampler import default_weights

params = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func={},
)

observations = {"x": np.array([2.0, 5.0, 8.0])}
pe = _ParzenEstimator(
    observations=observations,
    search_space={"x": dist},
    parameters=params,
)

# Evaluate log_pdf at some points
test_points = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 10.0])
log_pdf_vals = pe.log_pdf({"x": test_points})
print(f"Test points: {test_points.tolist()}")
print(f"Log PDF values: {log_pdf_vals.tolist()}")
for p, v in zip(test_points, log_pdf_vals):
    print(f"  x={p:.1f}: log_pdf={v:.10f}")

# Check internal mixture distribution
md = pe._mixture_distribution
print(f"\nMixture weights: {md.weights.tolist()}")
for i, d_list in enumerate(md.distributions):
    for d in d_list:
        print(f"  Dim {i}: type={type(d).__name__}, params={d}")

# Edge case: single observation
pe_single = _ParzenEstimator(
    observations={"x": np.array([5.0])},
    search_space={"x": dist},
    parameters=params,
)
log_pdf_single = pe_single.log_pdf({"x": np.array([3.0, 5.0, 7.0])})
print(f"\nSingle obs log_pdf at [3,5,7]: {log_pdf_single.tolist()}")
md_s = pe_single._mixture_distribution
print(f"Single obs weights: {md_s.weights.tolist()}")

# === TPE below/above split ===
print("\n=== TPE below/above split ===")
from optuna.samplers._tpe.sampler import _split_complete_trials_single_objective

import optuna
# Create trial objects
study = optuna.create_study(direction="minimize")
trials = []
values_list = [5.0, 1.0, 3.0, 2.0, 4.0, 0.5, 6.0, 1.5]
for i, v in enumerate(values_list):
    trial = optuna.trial.create_trial(
        params={"x": float(i)},
        distributions={"x": FloatDistribution(0.0, 10.0)},
        values=[v],
    )
    trials.append(trial)

# gamma(n) = ceil(0.25 * n)
import math
n = len(trials)
n_below = max(1, int(math.ceil(0.25 * n)))
print(f"n={n}, n_below={n_below}")

below, above = _split_complete_trials_single_objective(
    trials=trials,
    study=study,
    n_below=n_below,
)
print(f"below count: {len(below)}")
print(f"above count: {len(above)}")
print(f"below values: {[t.value for t in below]}")
print(f"above values: {[t.value for t in above]}")

# === Log distribution ===
print("\n=== Log Float Distribution ===")
log_dist = FloatDistribution(1e-5, 1.0, log=True)
log_obs = {"x": np.array([0.001, 0.01, 0.1])}
pe_log = _ParzenEstimator(
    observations=log_obs,
    search_space={"x": log_dist},
    parameters=params,
)
log_test = np.array([0.0001, 0.001, 0.01, 0.1, 0.5, 1.0])
log_pdf_vals = pe_log.log_pdf({"x": log_test})
for p, v in zip(log_test, log_pdf_vals):
    print(f"  x={p}: log_pdf={v:.10f}")

# === Integer Distribution ===
print("\n=== Integer Distribution ===")
from optuna.distributions import IntDistribution
int_dist = IntDistribution(0, 10)
int_obs = {"x": np.array([2.0, 5.0, 8.0])}
pe_int = _ParzenEstimator(
    observations=int_obs,
    search_space={"x": int_dist},
    parameters=params,
)
int_test = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
int_pdf_vals = pe_int.log_pdf({"x": int_test})
for p, v in zip(int_test, int_pdf_vals):
    print(f"  x={p:.0f}: log_pdf={v:.10f}")

# === Categorical Distribution ===
print("\n=== Categorical Distribution ===")
from optuna.distributions import CategoricalDistribution
cat_dist = CategoricalDistribution(["a", "b", "c", "d"])
cat_obs = {"x": np.array([0.0, 0.0, 1.0, 2.0])}
pe_cat = _ParzenEstimator(
    observations=cat_obs,
    search_space={"x": cat_dist},
    parameters=params,
)
cat_test = np.array([0.0, 1.0, 2.0, 3.0])
cat_pdf_vals = pe_cat.log_pdf({"x": cat_test})
for p, v in zip(cat_test, cat_pdf_vals):
    print(f"  x={int(p)}: log_pdf={v:.10f}")