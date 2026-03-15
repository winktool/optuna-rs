# optuna-rs

A Rust port of [Optuna](https://github.com/optuna/optuna), the hyperparameter optimization framework. Provides automatic hyperparameter search with single and multi-objective optimization, pruning, and a variety of sampling algorithms.

## Test Status

| Feature Set | Tests Passing |
|------------|---------------|
| Default | 419 |
| Visualization | 447 |
| All Features | 489 |

## Features

- **Single and multi-objective optimization** with Pareto front analysis
- **10 built-in samplers**: Random, TPE, GP, Grid, QMC (Halton), CMA-ES, NSGA-II, NSGA-III, BruteForce, PartialFixed
- **9 pruners**: Median, Percentile, SuccessiveHalving, Hyperband, Patient, Threshold, Wilcoxon, Nop
- **3 importance evaluators**: fANOVA, MDI (Random Forest), PED-ANOVA
- **5 terminators**: MaxTrials, NoImprovement, BestValueStagnation, RegretBound (GP), EMMR (GP)
- **3 storage backends**: InMemory, JournalFile (append-only log), RDB (SQLite/PostgreSQL/MySQL via SeaORM)
- **Artifacts module**: FileSystemArtifactStore, upload/download, metadata management
- **CLI binary** (optional `cli` feature): create-study, delete-study, studies, trials, best-trial, best-trials, tell
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
    directions.clone(), None, None, None, None, None, Some(42), None, None, None, None,
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
| `GpSampler` | Gaussian Process with Matern 5/2 kernel + logEI |

## Storage Backends

| Backend | Description | Feature |
|---------|-------------|---------|
| `InMemoryStorage` | Default, non-persistent | (default) |
| `JournalFileStorage` | Append-only JSON log with replay recovery | (default) |
| `RdbStorage` | SQLite/PostgreSQL/MySQL via SeaORM | `rdb` |

## CLI

Enable with `--features cli`:

```bash
# 创建研究
cargo run --features cli -- create-study --storage journal.log --study-name my_study --direction minimize

# 列出研究
cargo run --features cli -- studies --storage journal.log

# 列出试验
cargo run --features cli -- trials --storage journal.log --study-name my_study --format json

# 获取最佳试验
cargo run --features cli -- best-trial --storage journal.log --study-name my_study
```

## Parameter Types

- `suggest_float(name, low, high, log, step)` — continuous or stepped floats
- `suggest_int(name, low, high, log, step)` — integers with optional step
- `suggest_categorical(name, choices)` — categorical choices (strings, numbers, bools)

## License

This project is licensed under the [MIT License](LICENSE).

This project is derived from [Optuna](https://github.com/optuna/optuna) by Preferred Networks, Inc. See [NOTICE](NOTICE) for the original license.
