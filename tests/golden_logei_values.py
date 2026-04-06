#!/usr/bin/env python3
"""Generate logEI and log_ndtr golden values for Rust cross-validation."""
import torch
import numpy as np
import math

# ═══════════════════════════════════════════════════════
# 1. log_ndtr golden values
# ═══════════════════════════════════════════════════════
print("=== log_ndtr golden values ===")
test_z = [-50.0, -30.0, -26.0, -10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0]
for z in test_z:
    zt = torch.tensor(z, dtype=torch.float64)
    val = torch.special.log_ndtr(zt).item()
    print(f"  z={z:8.1f} -> log_ndtr={val:.16e}")

# ═══════════════════════════════════════════════════════
# 2. logEI golden values (standard_logei)
# ═══════════════════════════════════════════════════════
# optuna._gp.acqf.standard_logei(z) implementation
from optuna._gp import acqf as acqf_module

print("\n=== standard_logei golden values ===")
z_values = [-30.0, -25.0, -10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0]
for z in z_values:
    zt = torch.tensor([z], dtype=torch.float64)
    val = acqf_module.standard_logei(zt).item()
    print(f"  z={z:8.1f} -> standard_logei={val:.16e}")

# ═══════════════════════════════════════════════════════
# 3. logEI with specific (mean, var, f0)
# ═══════════════════════════════════════════════════════
print("\n=== logEI(mean, var, f0) golden values ===")
test_cases = [
    (1.0, 0.25, 0.5),    # z = (1-0.5)/0.5 = 1
    (0.5, 1.0, 0.5),     # z = 0
    (-1.0, 0.5, 0.0),    # z = -1/sqrt(0.5) = -1.414
    (0.0, 0.01, 5.0),    # z = -5/0.1 = -50
    (10.0, 4.0, 5.0),    # z = 5/2 = 2.5
    (0.1, 0.001, 0.1),   # z = 0 (mean == f0)
]
for mean, var, f0 in test_cases:
    sigma = math.sqrt(var)
    z = (mean - f0) / sigma
    zt = torch.tensor([z], dtype=torch.float64)
    logei_std = acqf_module.standard_logei(zt).item()
    logei = logei_std + math.log(sigma)
    print(f"  mean={mean:6.2f}, var={var:.3f}, f0={f0:.1f}, z={z:.6f} -> logEI={logei:.16e}")

# ═══════════════════════════════════════════════════════
# 4. erfcx golden values
# ═══════════════════════════════════════════════════════
print("\n=== erfcx golden values ===")
erfcx_x = [0.0, 0.5, 1.0, 5.0, 10.0, 25.0, 30.0, 50.0]
for x in erfcx_x:
    xt = torch.tensor(x, dtype=torch.float64)
    val = torch.special.erfcx(xt).item()
    print(f"  x={x:6.1f} -> erfcx={val:.16e}")
