#!/usr/bin/env python3
"""
MOTPE 权重计算的金标准值。
验证 _calculate_weights_below_for_multi_objective 的 HV 贡献权重。
"""
import numpy as np
import optuna
from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective
from optuna.study import StudyDirection

print("=== MOTPE HV Contribution Weights ===")

# Case 1: 2D minimize, 4 feasible trials on Pareto front
study = optuna.create_study(directions=["minimize", "minimize"])
trials = []
pareto_values = [[1.0, 4.0], [2.0, 2.0], [3.0, 1.5], [4.0, 1.0]]
for i, vals in enumerate(pareto_values):
    trial = optuna.trial.create_trial(
        params={"x": float(i)},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        values=vals,
    )
    trials.append(trial)

weights = _calculate_weights_below_for_multi_objective(study, trials, None)
print(f"Case 1 (4 Pareto): weights = {weights.tolist()}")

# Case 2: 2D, mix of Pareto and dominated
study2 = optuna.create_study(directions=["minimize", "minimize"])
trials2 = []
values2 = [[1.0, 4.0], [2.0, 2.0], [3.0, 3.0], [4.0, 1.0]]  # [3,3] dominated
for i, vals in enumerate(values2):
    trial = optuna.trial.create_trial(
        params={"x": float(i)},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        values=vals,
    )
    trials2.append(trial)

weights2 = _calculate_weights_below_for_multi_objective(study2, trials2, None)
print(f"Case 2 (3 Pareto + 1 dominated): weights = {weights2.tolist()}")

# Case 3: Single trial
study3 = optuna.create_study(directions=["minimize", "minimize"])
trial_single = optuna.trial.create_trial(
    params={"x": 1.0},
    distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
    values=[2.0, 3.0],
)
weights3 = _calculate_weights_below_for_multi_objective(study3, [trial_single], None)
print(f"Case 3 (single): weights = {weights3.tolist()}")

# Case 4: 3D minimize
study4 = optuna.create_study(directions=["minimize", "minimize", "minimize"])
trials4 = []
pareto_3d = [[1.0, 4.0, 3.0], [2.0, 1.0, 4.0], [3.0, 3.0, 1.0], [4.0, 2.0, 2.0]]
for i, vals in enumerate(pareto_3d):
    trial = optuna.trial.create_trial(
        params={"x": float(i)},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        values=vals,
    )
    trials4.append(trial)

weights4 = _calculate_weights_below_for_multi_objective(study4, trials4, None)
print(f"Case 4 (3D Pareto): weights = {weights4.tolist()}")

# Case 5: Maximize direction
study5 = optuna.create_study(directions=["maximize", "maximize"])
trials5 = []
max_values = [[4.0, 1.0], [2.0, 2.0], [1.0, 4.0]]
for i, vals in enumerate(max_values):
    trial = optuna.trial.create_trial(
        params={"x": float(i)},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        values=vals,
    )
    trials5.append(trial)

weights5 = _calculate_weights_below_for_multi_objective(study5, trials5, None)
print(f"Case 5 (2D maximize): weights = {weights5.tolist()}")
