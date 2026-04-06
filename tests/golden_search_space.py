#!/usr/bin/env python3
"""
生成 IntersectionSearchSpace 和 GroupDecomposedSearchSpace 的 Python 金标准值。
用于交叉验证 Rust 实现的精确性。
"""
import json
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.search_space.intersection import IntersectionSearchSpace as _IntersectionSearchSpace
from optuna.search_space.group_decomposed import _SearchSpaceGroup, _GroupDecomposedSearchSpace
import optuna

# ========== SearchSpaceGroup 分裂测试 ==========
print("=== SearchSpaceGroup add_distributions ===")

# Case 1: {x,y,z} + {x,y} → 2 groups: {x,y}, {z}
g1 = _SearchSpaceGroup()
g1.add_distributions({"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1), "z": FloatDistribution(0, 1)})
print(f"After {{x,y,z}}: {len(g1.search_spaces)} groups, keys={[sorted(s.keys()) for s in g1.search_spaces]}")

g1.add_distributions({"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1)})
print(f"After {{x,y}}: {len(g1.search_spaces)} groups, keys={[sorted(s.keys()) for s in g1.search_spaces]}")

# Case 2: {a,b,c} + {b,c,d} + {a,d}
g2 = _SearchSpaceGroup()
g2.add_distributions({"a": FloatDistribution(0, 1), "b": FloatDistribution(0, 1), "c": FloatDistribution(0, 1)})
print(f"\nAfter {{a,b,c}}: {len(g2.search_spaces)} groups")

g2.add_distributions({"b": FloatDistribution(0, 1), "c": FloatDistribution(0, 1), "d": FloatDistribution(0, 1)})
print(f"After {{b,c,d}}: {len(g2.search_spaces)} groups, keys={[sorted(s.keys()) for s in g2.search_spaces]}")

g2.add_distributions({"a": FloatDistribution(0, 1), "d": FloatDistribution(0, 1)})
print(f"After {{a,d}}: {len(g2.search_spaces)} groups, keys={[sorted(s.keys()) for s in g2.search_spaces]}")

# Case 3: Progressive split {x,y} → {x,y} → {x} → {z} → {x,z}
g3 = _SearchSpaceGroup()
g3.add_distributions({"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1)})
print(f"\nProgressive: After {{x,y}}: {len(g3.search_spaces)} groups")

g3.add_distributions({"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1)})
print(f"After {{x,y}} again: {len(g3.search_spaces)} groups")

g3.add_distributions({"x": FloatDistribution(0, 1)})
print(f"After {{x}}: {len(g3.search_spaces)} groups, keys={[sorted(s.keys()) for s in g3.search_spaces]}")

g3.add_distributions({"z": FloatDistribution(0, 1)})
print(f"After {{z}}: {len(g3.search_spaces)} groups, keys={[sorted(s.keys()) for s in g3.search_spaces]}")

g3.add_distributions({"x": FloatDistribution(0, 1), "z": FloatDistribution(0, 1)})
print(f"After {{x,z}}: {len(g3.search_spaces)} groups, keys={[sorted(s.keys()) for s in g3.search_spaces]}")

# ========== IntersectionSearchSpace 测试 ==========
print("\n=== IntersectionSearchSpace ===")

study = optuna.create_study()

# Trial 0: {x, y, z}
trial = study.ask()
trial.suggest_float("x", 0, 1)
trial.suggest_float("y", 0, 1)
trial.suggest_float("z", 0, 1)
study.tell(trial, 1.0)

iss = _IntersectionSearchSpace()
result = iss.calculate(study)
print(f"After trial(x,y,z): {sorted(result.keys())}")

# Trial 1: {x, y}
trial = study.ask()
trial.suggest_float("x", 0, 1)
trial.suggest_float("y", 0, 1)
study.tell(trial, 2.0)

result = iss.calculate(study)
print(f"After trial(x,y): {sorted(result.keys())}")

# Trial 2: {x, w}
trial = study.ask()
trial.suggest_float("x", 0, 1)
trial.suggest_float("w", 0, 1)
study.tell(trial, 3.0)

result = iss.calculate(study)
print(f"After trial(x,w): {sorted(result.keys())}")

# ========== GroupDecomposedSearchSpace 测试 ==========
print("\n=== GroupDecomposedSearchSpace ===")

study2 = optuna.create_study()

# Trial 0: {x, y}
trial = study2.ask()
trial.suggest_float("x", 0, 1)
trial.suggest_int("y", 0, 10)
study2.tell(trial, 1.0)

# Trial 1: {x, z}
trial = study2.ask()
trial.suggest_float("x", 0, 1)
trial.suggest_categorical("z", ["a", "b", "c"])
study2.tell(trial, 2.0)

gdss = _GroupDecomposedSearchSpace()
result = gdss.calculate(study2)
groups = [sorted(s.keys()) for s in result.search_spaces]
print(f"After trial(x,y) + trial(x,z): {len(groups)} groups, keys={sorted(groups)}")

# Trial 2: {y, z}
trial = study2.ask()
trial.suggest_int("y", 0, 10)
trial.suggest_categorical("z", ["a", "b", "c"])
study2.tell(trial, 3.0)

result = gdss.calculate(study2)
groups = [sorted(s.keys()) for s in result.search_spaces]
print(f"After + trial(y,z): {len(groups)} groups, keys={sorted(groups)}")

# ========== Int distribution intersection conflict ==========
print("\n=== Distribution conflict ===")
study3 = optuna.create_study()

trial = study3.ask()
trial.suggest_float("x", 0.0, 1.0)
study3.tell(trial, 1.0)

trial = study3.ask()
trial.suggest_float("x", 0.0, 10.0)  # Different range!
study3.tell(trial, 2.0)

iss2 = _IntersectionSearchSpace()
result = iss2.calculate(study3)
print(f"Same name, different range: {sorted(result.keys())}")  # Should be empty - conflict removes

print("\n=== All golden values generated ===")
