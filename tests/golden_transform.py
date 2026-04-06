"""SearchSpaceTransform untransform 精确参考值生成。"""
import numpy as np
import math
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import FloatDistribution, IntDistribution

print("=== Float untransform ===")
fd = FloatDistribution(0.0, 10.0)
t = _SearchSpaceTransform({"x": fd})
r = t.untransform(np.array([5.0]))
print(f"Float [0,10] encoded=5.0 -> {r['x']:.15e}")

fd2 = FloatDistribution(0.001, 1.0, log=True)
t2 = _SearchSpaceTransform({"x": fd2})
r2 = t2.untransform(np.array([math.log(0.01)]))
print(f"Float log encoded=ln(0.01) -> {r2['x']:.15e}")

fd3 = FloatDistribution(0.0, 1.0, step=0.25)
t3 = _SearchSpaceTransform({"x": fd3})
for enc in [0.3, 0.6, 0.88, 0.13, -0.1]:
    r3 = t3.untransform(np.array([enc]))
    print(f"Float step encoded={enc} -> {r3['x']:.15e}")

print()
print("=== Int untransform ===")
id1 = IntDistribution(1, 10)
t6 = _SearchSpaceTransform({"n": id1})
for enc in [5.5, 5.4, 1.2, 9.8, 0.5, 10.5]:
    r = t6.untransform(np.array([enc]))
    print(f"Int [1,10] encoded={enc} -> {r['n']}")

id2 = IntDistribution(1, 100, log=True)
t8 = _SearchSpaceTransform({"n": id2})
r8 = t8.untransform(np.array([math.log(50)]))
print(f"Int log encoded=ln(50) -> {r8['n']}")

id3 = IntDistribution(0, 12, step=3)
t9 = _SearchSpaceTransform({"n": id3})
for enc in [5.0, 7.0, 1.0, 11.0]:
    r = t9.untransform(np.array([enc]))
    print(f"Int step=3 [0,12] encoded={enc} -> {r['n']}")

print()
print("=== transform_0_1 ===")
fd_01 = FloatDistribution(0.0, 10.0)
t11 = _SearchSpaceTransform({"x": fd_01}, transform_0_1=True)
for enc in [0.0, 0.5, 1.0]:
    r = t11.untransform(np.array([enc]))
    print(f"Float [0,10] 0_1 encoded={enc} -> {r['x']:.15e}")

print()
print(f"nextafter(10.0, 9.0) = {np.nextafter(10.0, 9.0):.20e}")
print(f"nextafter(1.0, 0.0) = {np.nextafter(1.0, 0.0):.20e}")
