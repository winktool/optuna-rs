#!/usr/bin/env python3
"""
生成 NSGA-II 拥挤距离的金标准值。
"""
from collections import defaultdict
import numpy as np
from optuna.trial import FrozenTrial, TrialState
from datetime import datetime

def make_trial(number, values):
    return FrozenTrial(
        number=number, trial_id=number, state=TrialState.COMPLETE,
        value=None, values=values,
        datetime_start=datetime.now(), datetime_complete=datetime.now(),
        params={}, distributions={}, user_attrs={}, system_attrs={},
        intermediate_values={},
    )

# Import the actual crowding distance function
from optuna.samplers.nsgaii._elite_population_selection_strategy import _calc_crowding_distance

print("=== Crowding Distance ===")

# Case 1: Simple 2D
pop1 = [make_trial(0, [1.0, 4.0]), make_trial(1, [2.0, 3.0]),
        make_trial(2, [3.0, 2.0]), make_trial(3, [4.0, 1.0])]
cd1 = _calc_crowding_distance([t for t in pop1])
print(f"2D simple: {dict(cd1)}")

# Case 2: With extreme values
pop2 = [make_trial(0, [0.0, 10.0]), make_trial(1, [5.0, 5.0]),
        make_trial(2, [10.0, 0.0])]
cd2 = _calc_crowding_distance([t for t in pop2])
print(f"2D extreme: {dict(cd2)}")

# Case 3: All same values
pop3 = [make_trial(0, [1.0, 1.0]), make_trial(1, [1.0, 1.0]),
        make_trial(2, [1.0, 1.0])]
cd3 = _calc_crowding_distance([t for t in pop3])
print(f"All same: {dict(cd3)}")

# Case 4: Single trial
pop4 = [make_trial(0, [1.0, 2.0])]
cd4 = _calc_crowding_distance([t for t in pop4])
print(f"Single: {dict(cd4)}")

# Case 5: 3D
pop5 = [make_trial(0, [1.0, 3.0, 5.0]), make_trial(1, [2.0, 2.0, 3.0]),
        make_trial(2, [3.0, 1.0, 1.0])]
cd5 = _calc_crowding_distance([t for t in pop5])
print(f"3D: {dict(cd5)}")

# Case 6: inf handling
pop6 = [make_trial(0, [1.0, 4.0]), make_trial(1, [float('inf'), 2.0]),
        make_trial(2, [3.0, 1.0])]
cd6 = _calc_crowding_distance([t for t in pop6])
print(f"Inf handling: {dict(cd6)}")
