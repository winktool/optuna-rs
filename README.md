# optuna-rs

A Rust port of [Optuna](https://github.com/optuna/optuna), the hyperparameter optimization framework. Provides automatic hyperparameter search with single and multi-objective optimization, pruning, and a variety of sampling algorithms.

## Features

- **Single and multi-objective optimization** with Pareto front analysis
- **9 built-in samplers**: Random, TPE, Grid, QMC (Halton), CMA-ES, NSGA-II, NSGA-III, BruteForce, PartialFixed
- **3 pruners**: Median, Percentile, Nop
- **Pluggable storage** via trait (in-memory included)
- **Ask-and-tell interface** for external optimization loops
- Define-by-run API matching Optuna's Python interface

## Quick Start

```rust
use optuna::{create_study, RandomSampler, Sampler, StudyDirection};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let study = create_study(
        None,              // storage (default: in-memory)
        Some(sampler),     // sampler
        None,              // pruner
        Some("my-study"),  // study name
        Some(StudyDirection::Minimize),
        None,              // directions (for multi-objective)
        false,             // load_if_exists
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
            let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
            Ok(x * x + y * y)
        },
        Some(100),  // n_trials
        None,       // timeout
        None,       // callbacks
    ).unwrap();

    println!("Best value: {}", study.best_value().unwrap());
    println!("Best params: {:?}", study.best_params().unwrap());
}
```

## Multi-Objective Optimization

```rust
use optuna::{create_study, NSGAIISampler, Sampler, StudyDirection};
use std::sync::Arc;

let directions = vec![StudyDirection::Minimize, StudyDirection::Minimize];
let sampler: Arc<dyn Sampler> = Arc::new(NSGAIISampler::new(
    directions.clone(), None, None, None, None, Some(42),
));

let study = create_study(
    None, Some(sampler), None, None, None, Some(directions), false,
).unwrap();

study.optimize_multi(
    |trial| {
        let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
        Ok(vec![x, 1.0 - x])  // two conflicting objectives
    },
    Some(100), None, None,
).unwrap();

let pareto_front = study.best_trials().unwrap();
println!("Pareto front size: {}", pareto_front.len());
```

## Samplers

| Sampler | Use Case |
|---------|----------|
| `RandomSampler` | Baseline, no assumptions about the objective |
| `TpeSampler` | General-purpose Bayesian optimization |
| `GridSampler` | Exhaustive search over discrete parameters |
| `QmcSampler` | Low-discrepancy sampling (Halton sequences) |
| `CmaEsSampler` | Continuous optimization with covariance adaptation |
| `NSGAIISampler` | Multi-objective optimization (2-3 objectives) |
| `NSGAIIISampler` | Many-objective optimization (3+ objectives) |
| `BruteForceSampler` | Enumerate all discrete parameter combinations |
| `PartialFixedSampler` | Fix some parameters, optimize the rest |

## Parameter Types

- `suggest_float(name, low, high, log, step)` — continuous or stepped floats
- `suggest_int(name, low, high, log, step)` — integers with optional step
- `suggest_categorical(name, choices)` — categorical choices (strings, numbers, bools)

## License

This project is licensed under the [MIT License](LICENSE).

This project is derived from [Optuna](https://github.com/optuna/optuna) by Preferred Networks, Inc. See [NOTICE](NOTICE) for the original license.
