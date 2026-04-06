#!/usr/bin/env python3
"""Generate GP golden values for Rust cross-validation tests.

Tests:
1. Matern52 kernel at various distances
2. GP posterior (mean, variance) with known data
3. GP log marginal likelihood
4. default_log_prior
5. Multivariate PE sigma computation
"""
import numpy as np
import math

# ═══════════════════════════════════════════════════════
# 1. Matern52 kernel golden values
# ═══════════════════════════════════════════════════════
def matern52_np(squared_distance):
    """Pure numpy matern52 (no torch)"""
    sqrt5d = np.sqrt(5.0 * squared_distance)
    exp_part = np.exp(-sqrt5d)
    return exp_part * ((5.0/3.0) * squared_distance + sqrt5d + 1.0)

test_distances_sq = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
print("=== Matern52 kernel values ===")
for d2 in test_distances_sq:
    val = matern52_np(d2)
    print(f"  d²={d2:8.2f} -> k={val:.16e}")

# ═══════════════════════════════════════════════════════
# 2. GP posterior with known training data
# ═══════════════════════════════════════════════════════
# Simple 1D case: X_train = [0.2, 0.5, 0.8], y_train = [1.0, -0.5, 0.3]
# kernel_scale=1.0, noise_var=0.01, inv_sq_lengthscale=1.0
X_train = np.array([[0.2], [0.5], [0.8]])
y_train = np.array([1.0, -0.5, 0.3])
inv_sq_ls = np.array([1.0])
kernel_scale = 1.0
noise_var = 0.01
is_categorical = np.array([False])

def build_kernel_matrix(X1, X2, inv_sq_ls, ks, is_cat):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            sqdist = 0.0
            for d in range(X1.shape[1]):
                if is_cat[d]:
                    sqdist += inv_sq_ls[d] * (1.0 if abs(X1[i,d] - X2[j,d]) > 0.5 else 0.0)
                else:
                    diff = X1[i,d] - X2[j,d]
                    sqdist += inv_sq_ls[d] * diff * diff
            K[i][j] = ks * matern52_np(sqdist)
    return K

K_train = build_kernel_matrix(X_train, X_train, inv_sq_ls, kernel_scale, is_categorical)
K_noise = K_train + noise_var * np.eye(3)
L = np.linalg.cholesky(K_noise)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Test points
x_test = np.array([[0.0], [0.2], [0.35], [0.5], [0.65], [0.8], [1.0]])
print("\n=== GP Posterior (mean, variance) ===")
print(f"X_train = {X_train.ravel().tolist()}, y_train = {y_train.tolist()}")
print(f"kernel_scale={kernel_scale}, noise_var={noise_var}, inv_sq_ls={inv_sq_ls.tolist()}")
for xt in x_test:
    k_star = build_kernel_matrix(xt.reshape(1,-1), X_train, inv_sq_ls, kernel_scale, is_categorical)[0]
    mean = np.dot(k_star, alpha)
    v = np.linalg.solve(L, k_star)
    var = max(0.0, kernel_scale - np.dot(v, v))
    print(f"  x={xt[0]:.2f} -> mean={mean:.16e}, var={var:.16e}")

# ═══════════════════════════════════════════════════════
# 3. Log marginal likelihood
# ═══════════════════════════════════════════════════════
n = len(y_train)
log_det = 2.0 * np.sum(np.log(np.diag(L)))
quad = np.dot(y_train, alpha)
lml = -0.5 * log_det - 0.5 * quad - 0.5 * n * np.log(2 * np.pi)
print(f"\n=== Log Marginal Likelihood ===")
print(f"  LML = {lml:.16e}")

# ═══════════════════════════════════════════════════════
# 4. default_log_prior
# ═══════════════════════════════════════════════════════
def default_log_prior(inv_sq_ls, ks, nv):
    ls_prior = sum(-(0.1 / x + 0.1 * x) for x in inv_sq_ls)
    ks_prior = math.log(ks) - ks  # Gamma(2,1)
    nv_prior = 0.1 * math.log(nv) - 30.0 * nv  # Gamma(1.1, 30)
    return ls_prior + ks_prior + nv_prior

test_params = [
    ([1.0], 1.0, 0.01),
    ([0.5, 2.0], 1.5, 0.001),
    ([10.0], 0.1, 0.1),
    ([0.1, 0.1, 0.1], 5.0, 0.05),
]
print(f"\n=== default_log_prior ===")
for ils, ks, nv in test_params:
    lp = default_log_prior(ils, ks, nv)
    print(f"  inv_sq_ls={ils}, ks={ks}, nv={nv} -> {lp:.16e}")

# ═══════════════════════════════════════════════════════
# 5. 2D GP posterior
# ═══════════════════════════════════════════════════════
X2d = np.array([[0.1, 0.3], [0.4, 0.6], [0.7, 0.2], [0.9, 0.8]])
y2d = np.array([0.5, -1.0, 0.8, -0.3])
inv_sq_ls_2d = np.array([2.0, 0.5])
ks2d = 1.5
nv2d = 0.02
is_cat_2d = np.array([False, False])

K2d = build_kernel_matrix(X2d, X2d, inv_sq_ls_2d, ks2d, is_cat_2d)
K2d_noise = K2d + nv2d * np.eye(4)
L2d = np.linalg.cholesky(K2d_noise)
alpha2d = np.linalg.solve(L2d.T, np.linalg.solve(L2d, y2d))

x2d_test = np.array([[0.0, 0.0], [0.5, 0.5], [0.25, 0.45], [1.0, 1.0]])
print(f"\n=== 2D GP Posterior ===")
print(f"X_train = {X2d.tolist()}, y_train = {y2d.tolist()}")
print(f"ks={ks2d}, nv={nv2d}, inv_sq_ls={inv_sq_ls_2d.tolist()}")
for xt in x2d_test:
    k_star = build_kernel_matrix(xt.reshape(1,-1), X2d, inv_sq_ls_2d, ks2d, is_cat_2d)[0]
    mean = np.dot(k_star, alpha2d)
    v = np.linalg.solve(L2d, k_star)
    var = max(0.0, ks2d - np.dot(v, v))
    print(f"  x={xt.tolist()} -> mean={mean:.16e}, var={var:.16e}")

# 2D LML
n2d = len(y2d)
log_det_2d = 2.0 * np.sum(np.log(np.diag(L2d)))
quad_2d = np.dot(y2d, alpha2d)
lml_2d = -0.5 * log_det_2d - 0.5 * quad_2d - 0.5 * n2d * np.log(2 * np.pi)
print(f"  LML_2d = {lml_2d:.16e}")

# ═══════════════════════════════════════════════════════
# 6. Multivariate PE sigma computation
# ═══════════════════════════════════════════════════════
print(f"\n=== Multivariate PE sigma ===")
# sigma = SIGMA0_MAGNITUDE * n_mus^(-1/(n_params+4)) * (high - low)
# SIGMA0_MAGNITUDE = 0.2
SIGMA0_MAGNITUDE = 0.2
test_cases = [
    (3, 2, 0.0, 10.0),   # 3 obs, 2 params, range [0,10]
    (5, 1, 0.0, 1.0),    # 5 obs, 1 param, range [0,1]
    (10, 3, -5.0, 5.0),  # 10 obs, 3 params, range [-5,5]
    (1, 1, 0.0, 1.0),    # 1 obs, 1 param
    (100, 4, 0.0, 1.0),  # 100 obs, 4 params
]
for n_mus, n_params, low, high in test_cases:
    sigma = SIGMA0_MAGNITUDE * max(1, n_mus) ** (-1.0 / (n_params + 4.0)) * (high - low)
    print(f"  n_mus={n_mus}, n_params={n_params}, range=[{low},{high}] -> sigma={sigma:.16e}")

# ═══════════════════════════════════════════════════════
# 7. GP with categorical parameter  
# ═══════════════════════════════════════════════════════
X_cat = np.array([[0.2, 0.0], [0.5, 1.0], [0.8, 0.0]])
y_cat = np.array([1.0, -0.5, 0.3])
inv_sq_ls_cat = np.array([1.0, 1.0])
is_cat_mix = np.array([False, True])

K_cat = build_kernel_matrix(X_cat, X_cat, inv_sq_ls_cat, 1.0, is_cat_mix)
K_cat_noise = K_cat + 0.01 * np.eye(3)
L_cat = np.linalg.cholesky(K_cat_noise)
alpha_cat = np.linalg.solve(L_cat.T, np.linalg.solve(L_cat, y_cat))

x_cat_test = np.array([[0.3, 0.0], [0.3, 1.0]])
print(f"\n=== GP with mixed categorical ===")
for xt in x_cat_test:
    k_star = build_kernel_matrix(xt.reshape(1,-1), X_cat, inv_sq_ls_cat, 1.0, is_cat_mix)[0]
    mean = np.dot(k_star, alpha_cat)
    v = np.linalg.solve(L_cat, k_star)
    var = max(0.0, 1.0 - np.dot(v, v))
    print(f"  x={xt.tolist()} -> mean={mean:.16e}, var={var:.16e}")
