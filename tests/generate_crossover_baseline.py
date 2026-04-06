#!/usr/bin/env python3
"""Generate statistical baselines for crossover operators."""
import numpy as np
import json

results = {}
N = 50000

# 1. UniformCrossover — swap ratio for different probs
rng = np.random.RandomState(42)
for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
    parents = np.array([[0.0]*5, [1.0]*5])
    swap_counts = 0
    for _ in range(N):
        masks = (rng.rand(5) >= prob).astype(int)
        child = parents[masks, range(5)]
        swap_counts += child.sum()
    results[f'uniform_p{prob}_swap_ratio'] = swap_counts / (N * 5)

# 2. BLXAlpha range
rng = np.random.RandomState(42)
p0, p1, alpha = 0.3, 0.7, 0.5
d = p1 - p0
expected_low = p0 - alpha * d
expected_high = p1 + alpha * d
children = []
for _ in range(N):
    r = rng.rand()
    c = (expected_high - expected_low) * r + expected_low
    children.append(c)
children = np.array(children)
results['blx_mean'] = float(children.mean())
results['blx_std'] = float(children.std())
results['blx_expected_low'] = expected_low
results['blx_expected_high'] = expected_high

# 3. SBX eta=2 (with bounds correction for [0,1] space)
rng = np.random.RandomState(42)
children_sbx = []
for _ in range(N):
    xs_min, xs_max = 0.3, 0.7
    xs_diff = 0.4
    eta = 2.0
    beta1 = 1 + 2*(xs_min - 0)/xs_diff
    beta2 = 1 + 2*(1 - xs_max)/xs_diff
    alpha1 = 2 - beta1**(-(eta+1))
    alpha2 = 2 - beta2**(-(eta+1))
    u = rng.rand()
    if u <= 1/alpha1:
        betaq1 = (u * alpha1)**(1/(eta+1))
    else:
        betaq1 = (1/(2 - u*alpha1))**(1/(eta+1))
    if u <= 1/alpha2:
        betaq2 = (u * alpha2)**(1/(eta+1))
    else:
        betaq2 = (1/(2 - u*alpha2))**(1/(eta+1))
    c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)
    c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)
    child = c1 if rng.rand() < 0.5 else c2
    children_sbx.append(float(np.clip(child, 0, 1)))
children_sbx = np.array(children_sbx)
results['sbx_eta2_mean'] = float(children_sbx.mean())
results['sbx_eta2_std'] = float(children_sbx.std())

# SBX eta=20
rng = np.random.RandomState(42)
children_sbx20 = []
for _ in range(N):
    xs_min, xs_max = 0.3, 0.7
    xs_diff = 0.4
    eta = 20.0
    beta1 = 1 + 2*(xs_min)/xs_diff
    beta2 = 1 + 2*(1 - xs_max)/xs_diff
    alpha1 = 2 - beta1**(-(eta+1))
    alpha2 = 2 - beta2**(-(eta+1))
    u = rng.rand()
    if u <= 1/alpha1:
        betaq1 = (u * alpha1)**(1/(eta+1))
    else:
        betaq1 = (1/(2 - u*alpha1))**(1/(eta+1))
    if u <= 1/alpha2:
        betaq2 = (u * alpha2)**(1/(eta+1))
    else:
        betaq2 = (1/(2 - u*alpha2))**(1/(eta+1))
    c1 = 0.5 * ((xs_min + xs_max) - betaq1 * xs_diff)
    c2 = 0.5 * ((xs_min + xs_max) + betaq2 * xs_diff)
    child = c1 if rng.rand() < 0.5 else c2
    children_sbx20.append(float(np.clip(child, 0, 1)))
children_sbx20 = np.array(children_sbx20)
results['sbx_eta20_mean'] = float(children_sbx20.mean())
results['sbx_eta20_std'] = float(children_sbx20.std())

# 4. SPX centroid and spread
rng = np.random.RandomState(42)
p0 = np.array([0.3, 0.3])
p1 = np.array([0.7, 0.7])
p2 = np.array([0.5, 0.2])
parents = np.array([p0, p1, p2])
centroid = parents.mean(axis=0)
children_spx = []
for _ in range(N):
    n = 2
    G = centroid.copy()
    rs = np.power(rng.rand(n), 1 / (np.arange(n) + 1))
    epsilon = np.sqrt(len(p0) + 2)
    xks = [G + epsilon * (pk - G) for pk in parents]
    ck = 0
    for k in range(1, 3):
        ck = rs[k-1] * (xks[k-1] - xks[k] + ck)
    child = xks[-1] + ck
    child = np.clip(child, 0, 1)
    children_spx.append(child.tolist())
children_spx = np.array(children_spx)
results['spx_mean_x'] = float(children_spx[:,0].mean())
results['spx_mean_y'] = float(children_spx[:,1].mean())
results['spx_std_x'] = float(children_spx[:,0].std())
results['spx_std_y'] = float(children_spx[:,1].std())
results['spx_centroid'] = centroid.tolist()

# 5. UNDX
rng = np.random.RandomState(42)
p0 = np.array([0.3, 0.3])
p1 = np.array([0.7, 0.7])
p2 = np.array([0.5, 0.2])
sigma_xi = 0.5
n = 2
sigma_eta = 0.35 / np.sqrt(n)
midpoint = (p0 + p1) / 2
children_undx = []
for _ in range(N):
    xp = midpoint.copy()
    d = p0 - p1
    d_norm = max(np.linalg.norm(d), 1e-10)
    e12 = d / d_norm
    v13 = p2 - p0
    v_orth = v13 - np.dot(v13, e12) * e12
    D = np.linalg.norm(v_orth)
    xi = rng.normal(0, sigma_xi**2)
    etas = rng.normal(0, sigma_eta**2, size=n)
    basis_matrix = np.eye(n)
    if np.count_nonzero(e12) != 0:
        basis_matrix[0] = e12
    Q, _ = np.linalg.qr(basis_matrix.T)
    es = Q.T[1:]
    child = xp + xi * d
    if n > 1:
        three = np.zeros(n)
        for i in range(n-1):
            three += etas[i] * es[i]
        three *= D
        child += three
    child = np.clip(child, 0, 1)
    children_undx.append(child.tolist())
children_undx = np.array(children_undx)
results['undx_mean_x'] = float(children_undx[:,0].mean())
results['undx_mean_y'] = float(children_undx[:,1].mean())
results['undx_std_x'] = float(children_undx[:,0].std())
results['undx_std_y'] = float(children_undx[:,1].std())
results['undx_midpoint'] = midpoint.tolist()

# 6. VSBX eta=20
rng = np.random.RandomState(42)
p0 = np.array([0.3, 0.3])
p1 = np.array([0.7, 0.7])
eta = 20.0
children_vsbx = []
for _ in range(N):
    eps = 1e-10
    us = rng.rand(2)
    beta_1 = np.power(1 / np.maximum(2*us, eps), 1/(eta+1))
    beta_2 = np.power(1 / np.maximum(2*(1-us), eps), 1/(eta+1))
    u_1 = rng.rand()
    if u_1 <= 0.5:
        c1 = 0.5 * ((1+beta_1)*p0 + (1-beta_2)*p1)
    else:
        c1 = 0.5 * ((1-beta_1)*p0 + (1+beta_2)*p1)
    u_2 = rng.rand()
    if u_2 <= 0.5:
        c2 = 0.5 * ((3-beta_1)*p0 - (1-beta_2)*p1)
    else:
        c2 = 0.5 * (-(1-beta_1)*p0 + (3-beta_2)*p1)
    child1, child2 = [], []
    for c1_i, c2_i, x1_i, x2_i in zip(c1, c2, p0, p1):
        if rng.rand() < 0.5:
            if rng.rand() >= 0.5:
                child1.append(c1_i)
                child2.append(c2_i)
            else:
                child1.append(c2_i)
                child2.append(c1_i)
        else:
            if rng.rand() >= 0.5:
                child1.append(x1_i)
                child2.append(x2_i)
            else:
                child1.append(x2_i)
                child2.append(x1_i)
    child_params = child1 if rng.rand() < 0.5 else child2
    child_params = np.clip(child_params, 0, 1)
    children_vsbx.append(list(child_params))
children_vsbx = np.array(children_vsbx)
results['vsbx_eta20_mean_x'] = float(children_vsbx[:,0].mean())
results['vsbx_eta20_mean_y'] = float(children_vsbx[:,1].mean())
results['vsbx_eta20_std_x'] = float(children_vsbx[:,0].std())
results['vsbx_eta20_std_y'] = float(children_vsbx[:,1].std())

with open('/Users/lichangqing/Copilot/optuna/optuna-rs/tests/crossover_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)
print("OK — baseline saved")
for k, v in sorted(results.items()):
    print(f"  {k}: {v}")
