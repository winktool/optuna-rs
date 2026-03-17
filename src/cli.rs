//! 命令行接口 (CLI) 模块。
//!
//! 对应 Python `optuna.cli`。
//!
//! 提供通过命令行管理 Optuna study 和 trial 的接口：
//! - `create-study` — 创建新研究
//! - `delete-study` — 删除研究
//! - `studies` — 列出所有研究
//! - `trials` — 列出试验
//! - `best-trial` — 获取最佳试验
//! - `best-trials` — 获取 Pareto 前沿试验（多目标）
//! - `tell` — 报告试验结果
//!
//! ## 使用方式
//! 需要启用 `cli` feature:
//! ```bash
//! cargo run --features cli -- create-study --storage sqlite:///study.db --study-name my_study --direction minimize
//! ```

#[cfg(feature = "cli")]
pub mod commands {
    use clap::{Parser, Subcommand};
    use std::sync::Arc;

    use crate::storage::Storage;
    use crate::study::StudyDirection;
    use crate::trial::TrialState;

    /// Optuna Rust CLI — 超参数优化工具
    #[derive(Parser)]
    #[command(name = "optuna", version, about = "Optuna hyperparameter optimization CLI")]
    pub struct Cli {
        #[command(subcommand)]
        pub command: Commands,
    }

    #[derive(Subcommand)]
    pub enum Commands {
        /// 创建新研究
        CreateStudy {
            /// 存储 URL（如 sqlite:///study.db 或 journal 文件路径）
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: Option<String>,
            /// 优化方向（minimize 或 maximize）
            #[arg(long)]
            direction: Option<String>,
            /// 多目标方向（逗号分隔）
            #[arg(long)]
            directions: Option<String>,
            /// 如果已存在则跳过
            #[arg(long, default_value_t = false)]
            skip_if_exists: bool,
        },
        /// 删除研究
        DeleteStudy {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
        },
        /// 列出所有研究
        Studies {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 输出格式: table, json
            #[arg(short, long, default_value = "table")]
            format: String,
        },
        /// 列出研究中的试验
        Trials {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 输出格式: table, json
            #[arg(short, long, default_value = "table")]
            format: String,
        },
        /// 获取最佳试验
        BestTrial {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 输出格式: table, json
            #[arg(short, long, default_value = "table")]
            format: String,
        },
        /// 获取 Pareto 前沿最佳试验（多目标）
        BestTrials {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 输出格式: table, json
            #[arg(short, long, default_value = "table")]
            format: String,
        },
        /// 报告试验结果
        Tell {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 试验编号
            #[arg(long)]
            trial_number: i64,
            /// 目标值（逗号分隔）
            #[arg(long)]
            values: Option<String>,
            /// 试验状态: complete, pruned, fail
            #[arg(long, default_value = "complete")]
            state: String,
            /// 如果已完成则跳过
            #[arg(long, default_value_t = false)]
            skip_if_finished: bool,
        },
        /// 列出所有研究名
        StudyNames {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
        },
        /// 设置研究的用户属性
        StudySetUserAttr {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 属性键
            #[arg(short, long)]
            key: String,
            /// 属性值
            #[arg(long)]
            value: String,
        },
        /// 创建新试验并建议参数（实验性）
        Ask {
            /// 存储 URL
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
            /// 研究名称
            #[arg(long)]
            study_name: String,
            /// 搜索空间（JSON 格式：{"name": {"type": "float", ...}}）
            #[arg(long)]
            search_space: Option<String>,
            /// 采样器类名 (如 TPESampler, RandomSampler, CmaEsSampler 等)
            #[arg(long)]
            sampler: Option<String>,
            /// 采样器初始化参数 (JSON 格式, 如 {"seed": 42})
            #[arg(long)]
            sampler_kwargs: Option<String>,
            /// 输出格式: json, table, yaml
            #[arg(short, long, default_value = "json")]
            format: String,
        },
        /// 升级 RDB 存储的 schema 版本
        ///
        /// 对应 Python `optuna storage upgrade`。
        StorageUpgrade {
            /// 存储 URL (必须是 RDB URL)
            #[arg(long, env = "OPTUNA_STORAGE")]
            storage: String,
        },
    }

    /// 解析方向字符串为 StudyDirection
    fn parse_direction(s: &str) -> Result<StudyDirection, String> {
        match s.to_lowercase().as_str() {
            "minimize" | "min" => Ok(StudyDirection::Minimize),
            "maximize" | "max" => Ok(StudyDirection::Maximize),
            _ => Err(format!("invalid direction: {s} (expected minimize/maximize)")),
        }
    }

    /// 从 URL 创建存储。
    ///
    /// 支持的 URL 格式:
    /// - `sqlite://...`, `postgres://...`, `mysql://...` → RdbStorage (需要 rdb feature)
    /// - 文件路径 (如 `journal.log`) → JournalFileStorage
    fn create_storage(url: &str) -> Result<Arc<dyn Storage>, String> {
        // RDB 存储 URL 模式
        #[cfg(feature = "rdb")]
        {
            let is_db_url = url.starts_with("sqlite:")
                || url.starts_with("postgres:")
                || url.starts_with("mysql:");
            if is_db_url {
                let storage = crate::storage::RdbStorage::new(url)
                    .map_err(|e| format!("failed to connect to database: {e}"))?;
                return Ok(Arc::new(storage));
            }
        }

        // 默认: JournalFileStorage
        let storage = crate::storage::JournalFileStorage::new(url)
            .map_err(|e| format!("failed to open journal file: {e}"))?;
        Ok(Arc::new(storage))
    }

    /// 执行 CLI 命令。
    pub fn run(cli: Cli) -> Result<(), String> {
        match cli.command {
            Commands::CreateStudy {
                storage,
                study_name,
                direction,
                directions,
                skip_if_exists,
            } => {
                let store = create_storage(&storage)?;
                let dirs = if let Some(ds) = directions {
                    ds.split(',')
                        .map(|s| parse_direction(s.trim()))
                        .collect::<Result<Vec<_>, _>>()?
                } else if let Some(d) = direction {
                    vec![parse_direction(&d)?]
                } else {
                    vec![StudyDirection::Minimize]
                };

                match store.create_new_study(&dirs, study_name.as_deref()) {
                    Ok(study_id) => {
                        let name = store.get_study_name_from_id(study_id)
                            .unwrap_or_else(|_| format!("study_{study_id}"));
                        println!("Created study: {name} (id={study_id})");
                    }
                    Err(crate::error::OptunaError::DuplicatedStudyError(_)) if skip_if_exists => {
                        if let Some(name) = &study_name {
                            let id = store.get_study_id_from_name(name).map_err(|e| e.to_string())?;
                            println!("Study already exists: {name} (id={id})");
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                }
                Ok(())
            }

            Commands::DeleteStudy { storage, study_name } => {
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                store.delete_study(study_id).map_err(|e| e.to_string())?;
                println!("Deleted study: {study_name}");
                Ok(())
            }

            Commands::Studies { storage, format } => {
                let store = create_storage(&storage)?;
                let studies = store.get_all_studies().map_err(|e| e.to_string())?;

                if format == "json" {
                    let json: Vec<serde_json::Value> = studies.iter().map(|s| {
                        serde_json::json!({
                            "study_id": s.study_id,
                            "study_name": s.study_name,
                            "directions": s.directions.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>(),
                        })
                    }).collect();
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                } else {
                    println!("{:<10} {:<30} {}", "ID", "Name", "Directions");
                    println!("{}", "-".repeat(60));
                    for s in &studies {
                        let dirs: Vec<String> = s.directions.iter().map(|d| format!("{d:?}")).collect();
                        println!("{:<10} {:<30} {}", s.study_id, s.study_name, dirs.join(", "));
                    }
                }
                Ok(())
            }

            Commands::Trials { storage, study_name, format } => {
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                let trials = store.get_all_trials(study_id, None).map_err(|e| e.to_string())?;

                if format == "json" {
                    let json: Vec<serde_json::Value> = trials.iter().map(|t| {
                        serde_json::json!({
                            "number": t.number,
                            "state": format!("{:?}", t.state),
                            "values": t.values,
                            "params": t.params,
                        })
                    }).collect();
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                } else {
                    println!("{:<8} {:<12} {:<20} {}", "Number", "State", "Values", "Params");
                    println!("{}", "-".repeat(70));
                    for t in &trials {
                        let vals = t.values.as_ref()
                            .map(|v| format!("{v:?}"))
                            .unwrap_or_else(|| "None".to_string());
                        let params: String = t.params.iter()
                            .map(|(k, v)| format!("{k}={v:?}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        println!("{:<8} {:<12} {:<20} {}", t.number, format!("{:?}", t.state), vals, params);
                    }
                }
                Ok(())
            }

            Commands::BestTrial { storage, study_name, format } => {
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                let directions = store.get_study_directions(study_id).map_err(|e| e.to_string())?;
                let trials = store.get_all_trials(study_id, Some(&[TrialState::Complete]))
                    .map_err(|e| e.to_string())?;

                if trials.is_empty() {
                    return Err("no completed trials".to_string());
                }

                let is_minimize = directions.first() == Some(&StudyDirection::Minimize);
                let best = trials.iter().min_by(|a, b| {
                    let va = a.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::NAN);
                    let vb = b.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::NAN);
                    if is_minimize {
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    }
                }).unwrap();

                if format == "json" {
                    let json = serde_json::json!({
                        "number": best.number,
                        "values": best.values,
                        "params": best.params,
                        "state": format!("{:?}", best.state),
                    });
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                } else {
                    println!("Best trial #{}", best.number);
                    println!("  Values: {:?}", best.values);
                    println!("  Params:");
                    for (k, v) in &best.params {
                        println!("    {k}: {v:?}");
                    }
                }
                Ok(())
            }

            Commands::BestTrials { storage, study_name, format } => {
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                let directions = store.get_study_directions(study_id).map_err(|e| e.to_string())?;
                let trials = store.get_all_trials(study_id, Some(&[TrialState::Complete]))
                    .map_err(|e| e.to_string())?;

                if trials.is_empty() {
                    return Err("no completed trials".to_string());
                }

                // Pareto 前沿
                let pareto = crate::multi_objective::get_pareto_front_trials(&trials, &directions);

                if format == "json" {
                    let json: Vec<serde_json::Value> = pareto.iter().map(|t| {
                        serde_json::json!({
                            "number": t.number,
                            "values": t.values,
                            "params": t.params,
                        })
                    }).collect();
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                } else {
                    println!("Pareto front ({} trials):", pareto.len());
                    for t in &pareto {
                        println!("  Trial #{}: values={:?}", t.number, t.values);
                    }
                }
                Ok(())
            }

            Commands::Tell {
                storage,
                study_name,
                trial_number,
                values,
                state,
                skip_if_finished,
            } => {
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                let trial_id = store.get_trial_id_from_study_id_trial_number(study_id, trial_number)
                    .map_err(|e| e.to_string())?;

                let trial_state = match state.to_lowercase().as_str() {
                    "complete" => TrialState::Complete,
                    "pruned" => TrialState::Pruned,
                    "fail" | "failed" => TrialState::Fail,
                    _ => return Err(format!("invalid state: {state}")),
                };

                let vals: Option<Vec<f64>> = values.map(|s| {
                    s.split(',')
                        .map(|v| v.trim().parse::<f64>().unwrap_or(f64::NAN))
                        .collect()
                });

                if skip_if_finished {
                    let trial = store.get_trial(trial_id).map_err(|e| e.to_string())?;
                    if trial.state == TrialState::Complete || trial.state == TrialState::Pruned || trial.state == TrialState::Fail {
                        println!("Trial #{trial_number} already finished (state={:?}), skipping.", trial.state);
                        return Ok(());
                    }
                }

                // 对齐 Python: 通过 Study.tell() 走完整验证流程
                // 包括 NaN 检查、values 数量验证、after_trial 回调等
                let study = crate::study::load_study(&study_name, store, None, None)
                    .map_err(|e| e.to_string())?;
                study.tell_with_options(
                    trial_id,
                    trial_state,
                    vals.as_deref(),
                    false, // skip_if_finished 已在上面处理
                ).map_err(|e| e.to_string())?;

                println!("Trial #{trial_number} → {state}");
                Ok(())
            }

            Commands::StudyNames { storage } => {
                let store = create_storage(&storage)?;
                let studies = store.get_all_studies().map_err(|e| e.to_string())?;
                for s in &studies {
                    println!("{}", s.study_name);
                }
                Ok(())
            }

            Commands::StudySetUserAttr { storage, study_name, key, value } => {
                // 加载研究并设置用户属性
                let store = create_storage(&storage)?;
                let study = crate::study::load_study(&study_name, store, None, None)
                    .map_err(|e| e.to_string())?;
                study.set_user_attr(&key, serde_json::Value::String(value))
                    .map_err(|e| e.to_string())?;
                println!("Attribute successfully set.");
                Ok(())
            }

            Commands::Ask { storage, study_name, search_space, sampler, sampler_kwargs, format } => {
                // 验证: sampler_kwargs 需要配合 sampler 使用
                if sampler.is_none() && sampler_kwargs.is_some() {
                    return Err("`--sampler-kwargs` is set without `--sampler`. Please specify `--sampler` as well or omit `--sampler-kwargs`.".to_string());
                }

                // 先创建存储并获取研究方向，以便正确创建采样器
                let store = create_storage(&storage)?;
                let study_id = store.get_study_id_from_name(&study_name).map_err(|e| e.to_string())?;
                let directions = store.get_study_directions(study_id).map_err(|e| e.to_string())?;
                let direction = directions[0];

                // 根据 --sampler 创建采样器
                let custom_sampler: Option<Arc<dyn crate::samplers::Sampler>> = if let Some(ref sampler_name) = sampler {
                    let kwargs_json: serde_json::Value = sampler_kwargs
                        .as_deref()
                        .map(|s| serde_json::from_str(s).map_err(|e| format!("invalid --sampler-kwargs JSON: {e}")))
                        .transpose()?
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let seed = kwargs_json.get("seed").and_then(|v| v.as_u64());
                    let sampler_obj: Arc<dyn crate::samplers::Sampler> = match sampler_name.as_str() {
                        "TPESampler" | "TpeSampler" => {
                            let mut builder = crate::samplers::TpeSamplerBuilder::new(direction);
                            if let Some(s) = seed { builder = builder.seed(s); }
                            if let Some(n) = kwargs_json.get("n_startup_trials").and_then(|v| v.as_u64()) {
                                builder = builder.n_startup_trials(n as usize);
                            }
                            if let Some(n) = kwargs_json.get("n_ei_candidates").and_then(|v| v.as_u64()) {
                                builder = builder.n_ei_candidates(n as usize);
                            }
                            if let Some(m) = kwargs_json.get("multivariate").and_then(|v| v.as_bool()) {
                                builder = builder.multivariate(m);
                            }
                            if let Some(g) = kwargs_json.get("group").and_then(|v| v.as_bool()) {
                                builder = builder.group(g);
                            }
                            if let Some(cl) = kwargs_json.get("constant_liar").and_then(|v| v.as_bool()) {
                                builder = builder.constant_liar(cl);
                            }
                            Arc::new(builder.build())
                        }
                        "RandomSampler" => {
                            Arc::new(crate::samplers::RandomSampler::new(seed))
                        }
                        "CmaEsSampler" => {
                            Arc::new(crate::samplers::CmaEsSampler::new(
                                direction,
                                kwargs_json.get("sigma0").and_then(|v| v.as_f64()),
                                kwargs_json.get("n_startup_trials").and_then(|v| v.as_u64()).map(|v| v as usize),
                                kwargs_json.get("popsize").and_then(|v| v.as_u64()).map(|v| v as usize),
                                None, // independent_sampler
                                seed,
                                None, // x0
                                false, false, false, false, None,
                            ))
                        }
                        "GPSampler" | "GpSampler" => {
                            Arc::new(crate::samplers::GpSampler::new(
                                seed,
                                None, // direction——由 study 的 direction 决定
                                kwargs_json.get("n_startup_trials").and_then(|v| v.as_u64()).map(|v| v as usize),
                                kwargs_json.get("deterministic_objective").and_then(|v| v.as_bool()).unwrap_or(false),
                                None,
                                None,
                            ))
                        }
                        "QMCSampler" | "QmcSampler" => {
                            Arc::new(crate::samplers::QmcSampler::new(
                                kwargs_json.get("qmc_type").and_then(|v| v.as_str()).and_then(|s| match s {
                                    "halton" => Some(crate::samplers::qmc::QmcType::Halton),
                                    "sobol" => Some(crate::samplers::qmc::QmcType::Sobol),
                                    _ => None,
                                }),
                                kwargs_json.get("scramble").and_then(|v| v.as_bool()),
                                seed,
                                None,
                                None,
                                None,
                            ))
                        }
                        "BruteForceSampler" => {
                            Arc::new(crate::samplers::BruteForceSampler::new(seed, false))
                        }
                        _ => return Err(format!("unknown sampler: {sampler_name}. Available: TPESampler, RandomSampler, CmaEsSampler, GPSampler, QMCSampler, BruteForceSampler")),
                    };
                    Some(sampler_obj)
                } else {
                    None
                };

                // 加载研究（使用同一个 store）
                let study = crate::study::load_study(&study_name, store, custom_sampler, None)
                    .map_err(|e| e.to_string())?;

                // 解析搜索空间 JSON（可选）
                let distributions: indexmap::IndexMap<String, crate::distributions::Distribution> =
                    if let Some(ss_json) = search_space {
                        serde_json::from_str(&ss_json)
                            .map_err(|e| format!("invalid search_space JSON: {e}"))?
                    } else {
                        indexmap::IndexMap::new()
                    };

                let fixed = if distributions.is_empty() { None } else { Some(distributions) };
                let trial = study.ask(fixed.as_ref()).map_err(|e| e.to_string())?;
                let params = trial.params();

                if format == "json" {
                    let json = serde_json::json!({
                        "number": trial.number(),
                        "params": params,
                    });
                    println!("{}", serde_json::to_string_pretty(&json).unwrap());
                } else if format == "yaml" {
                    // YAML 输出: 使用简单的 key: value 格式
                    println!("number: {}", trial.number());
                    println!("params:");
                    for (k, v) in &params {
                        println!("  {k}: {v:?}");
                    }
                } else {
                    println!("Trial #{}", trial.number());
                    for (k, v) in &params {
                        println!("  {k}: {v:?}");
                    }
                }
                Ok(())
            }

            Commands::StorageUpgrade { storage } => {
                // 对应 Python `optuna storage upgrade`
                // 需要 RDB 存储
                #[cfg(feature = "rdb")]
                {
                    let is_db_url = storage.starts_with("sqlite:")
                        || storage.starts_with("postgres:")
                        || storage.starts_with("mysql:");
                    if !is_db_url {
                        return Err("storage upgrade only supports RDB storage URLs (sqlite://, postgres://, mysql://)".to_string());
                    }
                    let rdb_storage = crate::storage::RdbStorage::new(&storage)
                        .map_err(|e| format!("failed to connect to database: {e}"))?;
                    // RDB 存储创建时已自动迁移，通知用户
                    println!("Storage is up-to-date.");
                    let _ = rdb_storage;
                    Ok(())
                }
                #[cfg(not(feature = "rdb"))]
                {
                    let _ = storage;
                    Err("storage upgrade requires the 'rdb' feature to be enabled".to_string())
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::parse_direction;
        use crate::study::StudyDirection;

        /// 对齐 Python: parse_direction "minimize"
        #[test]
        fn test_parse_direction_minimize() {
            assert_eq!(parse_direction("minimize").unwrap(), StudyDirection::Minimize);
            assert_eq!(parse_direction("min").unwrap(), StudyDirection::Minimize);
            assert_eq!(parse_direction("MINIMIZE").unwrap(), StudyDirection::Minimize);
        }

        /// 对齐 Python: parse_direction "maximize"
        #[test]
        fn test_parse_direction_maximize() {
            assert_eq!(parse_direction("maximize").unwrap(), StudyDirection::Maximize);
            assert_eq!(parse_direction("max").unwrap(), StudyDirection::Maximize);
        }

        /// 对齐 Python: parse_direction 无效输入报错
        #[test]
        fn test_parse_direction_invalid() {
            assert!(parse_direction("unknown").is_err());
            assert!(parse_direction("").is_err());
        }

        /// 对齐 Python: 大小写不敏感
        #[test]
        fn test_parse_direction_case_insensitive() {
            assert_eq!(parse_direction("Minimize").unwrap(), StudyDirection::Minimize);
            assert_eq!(parse_direction("MAXIMIZE").unwrap(), StudyDirection::Maximize);
            assert_eq!(parse_direction("Min").unwrap(), StudyDirection::Minimize);
            assert_eq!(parse_direction("Max").unwrap(), StudyDirection::Maximize);
        }

        /// 对齐 Python: CLI 参数解析 create-study
        #[test]
        fn test_cli_parse_create_study() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "create-study",
                "--storage", "sqlite:///test.db",
                "--study-name", "test_study",
                "--direction", "minimize",
            ]);
            match cli.command {
                Commands::CreateStudy { study_name, direction, .. } => {
                    assert_eq!(study_name.as_deref(), Some("test_study"));
                    assert_eq!(direction.as_deref(), Some("minimize"));
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: CLI 参数解析 delete-study
        #[test]
        fn test_cli_parse_delete_study() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "delete-study",
                "--storage", "sqlite:///test.db",
                "--study-name", "my_study",
            ]);
            match cli.command {
                Commands::DeleteStudy { study_name, .. } => {
                    assert_eq!(study_name, "my_study");
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: CLI 参数解析 tell
        #[test]
        fn test_cli_parse_tell() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "tell",
                "--storage", "sqlite:///test.db",
                "--study-name", "test",
                "--trial-number", "0",
                "--values", "1.0,2.0",
                "--state", "complete",
            ]);
            match cli.command {
                Commands::Tell { values, state, trial_number, .. } => {
                    assert_eq!(values.as_deref(), Some("1.0,2.0"));
                    assert_eq!(state, "complete");
                    assert_eq!(trial_number, 0);
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: CLI 参数解析 ask
        #[test]
        fn test_cli_parse_ask() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "ask",
                "--storage", "sqlite:///test.db",
                "--study-name", "test",
                "--sampler", "TPESampler",
                "--sampler-kwargs", r#"{"seed": 42}"#,
            ]);
            match cli.command {
                Commands::Ask { sampler, sampler_kwargs, .. } => {
                    assert_eq!(sampler.as_deref(), Some("TPESampler"));
                    assert!(sampler_kwargs.is_some());
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: CLI 参数解析 studies json 格式
        #[test]
        fn test_cli_parse_studies_json() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "studies",
                "--storage", "sqlite:///test.db",
                "-f", "json",
            ]);
            match cli.command {
                Commands::Studies { format, .. } => {
                    assert_eq!(format, "json");
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: CLI 默认方向
        #[test]
        fn test_cli_default_direction() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "create-study",
                "--storage", "sqlite:///test.db",
            ]);
            match cli.command {
                Commands::CreateStudy { direction, directions, .. } => {
                    assert!(direction.is_none());
                    assert!(directions.is_none());
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: best-trial 命令解析
        #[test]
        fn test_cli_parse_best_trial() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "best-trial",
                "--storage", "sqlite:///test.db",
                "--study-name", "my_study",
                "-f", "json",
            ]);
            match cli.command {
                Commands::BestTrial { study_name, format, .. } => {
                    assert_eq!(study_name, "my_study");
                    assert_eq!(format, "json");
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: best-trials 命令解析
        #[test]
        fn test_cli_parse_best_trials() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "best-trials",
                "--storage", "sqlite:///test.db",
                "--study-name", "my_study",
            ]);
            match cli.command {
                Commands::BestTrials { study_name, .. } => {
                    assert_eq!(study_name, "my_study");
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: trials 命令解析
        #[test]
        fn test_cli_parse_trials() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "trials",
                "--storage", "sqlite:///test.db",
                "--study-name", "my_study",
                "-f", "table",
            ]);
            match cli.command {
                Commands::Trials { study_name, format, .. } => {
                    assert_eq!(study_name, "my_study");
                    assert_eq!(format, "table");
                }
                _ => panic!("wrong command"),
            }
        }

        /// 对齐 Python: storage-upgrade 命令解析
        #[test]
        fn test_cli_parse_storage_upgrade() {
            use clap::Parser;
            let cli = Cli::parse_from(&[
                "optuna", "storage-upgrade",
                "--storage", "sqlite:///test.db",
            ]);
            match cli.command {
                Commands::StorageUpgrade { storage, .. } => {
                    assert_eq!(storage, "sqlite:///test.db");
                }
                _ => panic!("wrong command"),
            }
        }
    }
}
