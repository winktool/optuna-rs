"""QMC 和 RandomSampler 的 Python 参考值生成。"""
import numpy as np
from scipy.stats.qmc import Halton, Sobol

print("=" * 60)
print("1. Van der Corput 序列 (Halton 1D base=2)")
print("=" * 60)
h2 = Halton(1, scramble=False)
pts = h2.random(10)
for i, p in enumerate(pts):
    print(f"  index={i}: {p[0]:.15e}")

print()
print("=" * 60)
print("2. Halton 3D (base 2,3,5), scramble=False")
print("=" * 60)
h3 = Halton(3, scramble=False)
pts3 = h3.random(8)
for i, row in enumerate(pts3):
    print(f"  index={i}: [{row[0]:.15e}, {row[1]:.15e}, {row[2]:.15e}]")

print()
print("=" * 60)
print("3. Sobol 2D, scramble=False")
print("=" * 60)
sob = Sobol(2, scramble=False)
pts_s = sob.random(8)
for i, row in enumerate(pts_s):
    print(f"  index={i}: [{row[0]:.15e}, {row[1]:.15e}]")

print()
print("=" * 60)
print("4. Sobol 3D, scramble=False, first 8 points")
print("=" * 60)
sob3 = Sobol(3, scramble=False)
pts_s3 = sob3.random(8)
for i, row in enumerate(pts_s3):
    print(f"  index={i}: [{row[0]:.15e}, {row[1]:.15e}, {row[2]:.15e}]")

print()
print("=" * 60)
print("5. RandomSampler 逻辑验证 (SearchSpaceTransform)")
print("=" * 60)
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import FloatDistribution, IntDistribution

# Float [0, 10], no step, no log
fd = FloatDistribution(0.0, 10.0)
trans = _SearchSpaceTransform({"x": fd})
print(f"  Float [0,10] bounds = {trans.bounds}")

# Float log [0.001, 1.0]
fd_log = FloatDistribution(0.001, 1.0, log=True)
trans_log = _SearchSpaceTransform({"x": fd_log})
print(f"  Float log [0.001,1.0] bounds = {trans_log.bounds}")

# Float step [0, 1, step=0.25]
fd_step = FloatDistribution(0.0, 1.0, step=0.25)
trans_step = _SearchSpaceTransform({"x": fd_step})
print(f"  Float step [0,1,step=0.25] bounds = {trans_step.bounds}")

# Int [1, 10]
id = IntDistribution(1, 10)
trans_int = _SearchSpaceTransform({"n": id})
print(f"  Int [1,10] bounds = {trans_int.bounds}")

# Int log [1, 100]
id_log = IntDistribution(1, 100, log=True)
trans_int_log = _SearchSpaceTransform({"n": id_log})
print(f"  Int log [1,100] bounds = {trans_int_log.bounds}")

print()
print("=" * 60)
print("6. Halton 低差异性检验 (与随机序列对比)")
print("=" * 60)
# Halton 100 点在 [0,1]^2 的 L∞-star 差异应小于随机
h2d = Halton(2, scramble=False)
pts_h = h2d.random(100)
# 分成 10x10 网格计数
grid = np.zeros((10, 10))
for p in pts_h:
    i = min(int(p[0] * 10), 9)
    j = min(int(p[1] * 10), 9)
    grid[i][j] += 1
print(f"  Halton 100pts in 10x10 grid: min={grid.min()}, max={grid.max()}, mean={grid.mean()}")

print()
print("Done!")
