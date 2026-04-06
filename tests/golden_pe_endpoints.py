#!/usr/bin/env python3
"""Generate ParzenEstimator sigma golden values for consider_endpoints and multivariate."""
import numpy as np
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    _ParzenEstimatorParameters,
)
from optuna.distributions import FloatDistribution
from optuna.samplers._tpe.sampler import default_weights

# ═══════════════════════════════════════════════════════
# 1. consider_endpoints=False (default)
# ═══════════════════════════════════════════════════════
print("=== consider_endpoints=False (default) ===")
params_no_endpoints = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func=None,
)

search_space = {"x": FloatDistribution(0.0, 10.0)}
observations = {"x": np.array([2.0, 5.0, 8.0])}

pe = _ParzenEstimator(observations, search_space, params_no_endpoints)
for param_name, dist in pe.distributions.items():
    print(f"  param={param_name}")
    print(f"  mus={dist._mus.tolist()}")
    print(f"  sigmas={dist._sigmas.tolist()}")
    print(f"  weights (pe.weights)={pe.weights.tolist()}")

# ═══════════════════════════════════════════════════════
# 2. consider_endpoints=True
# ═══════════════════════════════════════════════════════
print("\n=== consider_endpoints=True ===")
params_endpoints = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=True,
    weights=default_weights,
    multivariate=False,
    categorical_distance_func=None,
)

pe2 = _ParzenEstimator(observations, search_space, params_endpoints)
for param_name, dist in pe2.distributions.items():
    print(f"  param={param_name}")
    print(f"  mus={dist._mus.tolist()}")
    print(f"  sigmas={dist._sigmas.tolist()}")

# ═══════════════════════════════════════════════════════
# 3. consider_endpoints edge cases
# ═══════════════════════════════════════════════════════
print("\n=== consider_endpoints edge: 1 observation ===")
obs1 = {"x": np.array([5.0])}
pe3 = _ParzenEstimator(obs1, search_space, params_no_endpoints)
for param_name, dist in pe3.distributions.items():
    print(f"  mus={dist._mus.tolist()}")
    print(f"  sigmas={dist._sigmas.tolist()}")

pe4 = _ParzenEstimator(obs1, search_space, params_endpoints)
for param_name, dist in pe4.distributions.items():
    print(f"  endpoints=True: mus={dist._mus.tolist()}")
    print(f"  endpoints=True: sigmas={dist._sigmas.tolist()}")

# ═══════════════════════════════════════════════════════
# 4. Multivariate=True sigma
# ═══════════════════════════════════════════════════════
print("\n=== multivariate=True ===")
params_mv = _ParzenEstimatorParameters(
    prior_weight=1.0,
    consider_magic_clip=True,
    consider_endpoints=False,
    weights=default_weights,
    multivariate=True,
    categorical_distance_func=None,
)

search_space_2d = {
    "x": FloatDistribution(0.0, 10.0),
    "y": FloatDistribution(-5.0, 5.0),
}
observations_2d = {
    "x": np.array([2.0, 5.0, 8.0]),
    "y": np.array([-3.0, 0.0, 4.0]),
}

pe_mv = _ParzenEstimator(observations_2d, search_space_2d, params_mv)
for param_name, dist in pe_mv.distributions.items():
    print(f"  param={param_name}")
    print(f"  mus={dist._mus.tolist()}")
    print(f"  sigmas={dist._sigmas.tolist()}")

# ═══════════════════════════════════════════════════════
# 5. log_pdf with consider_endpoints=True
# ═══════════════════════════════════════════════════════
print("\n=== log_pdf with consider_endpoints ===")
test_points = {"x": np.array([0.0, 2.0, 5.0, 8.0, 10.0])}
logpdf_no = pe.log_pdf(test_points)
logpdf_yes = pe2.log_pdf(test_points)
print(f"  endpoints=False: {logpdf_no.tolist()}")
print(f"  endpoints=True:  {logpdf_yes.tolist()}")

# ═══════════════════════════════════════════════════════
# 6. Multivariate log_pdf
# ═══════════════════════════════════════════════════════
print("\n=== multivariate log_pdf ===")
test_mv = {
    "x": np.array([1.0, 5.0, 9.0]),
    "y": np.array([-4.0, 0.0, 3.0]),
}
logpdf_mv = pe_mv.log_pdf(test_mv)
print(f"  logpdf={logpdf_mv.tolist()}")
