//! RDB 持久化存储 — 使用 SeaORM 实现
//!
//! 对应 Python `optuna.storages.RDBStorage`。
//! 支持 SQLite / PostgreSQL / MySQL 三种数据库后端。
//!
//! # 用法
//! ```ignore
//! use optuna_rs::storage::RdbStorage;
//!
//! // SQLite
//! let storage = RdbStorage::new("sqlite:///path/to/db.sqlite3").unwrap();
//!
//! // PostgreSQL
//! let storage = RdbStorage::new("postgres://user:pass@host/db").unwrap();
//! ```

pub mod entity;

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectOptions, Database, DatabaseConnection,
    EntityTrait, IntoActiveModel, PaginatorTrait, QueryFilter, QueryOrder, Set,
    ActiveValue,
};

use entity::TrialValueType;

use crate::distributions::Distribution;
use crate::error::{OptunaError, Result};
use crate::study::{FrozenStudy, StudyDirection};
use crate::storage::Storage;
use crate::trial::{FrozenTrial, TrialState};

/// RDB 持久化存储 — 使用 SeaORM 实现。
///
/// 对应 Python `optuna.storages.RDBStorage`。
/// 支持 SQLite / PostgreSQL / MySQL。
pub struct RdbStorage {
    /// SeaORM 数据库连接
    db: DatabaseConnection,
    /// Tokio 运行时（用于同步接口调用异步 SeaORM）
    runtime: Arc<tokio::runtime::Runtime>,
    /// 全局锁（保护多个操作的原子性）
    lock: Mutex<()>,
}

impl RdbStorage {
    /// 创建 RDB 存储并初始化数据库表。
    ///
    /// # 参数
    /// * `url` - 数据库连接字符串
    ///   - SQLite: `sqlite:///path/to/db.sqlite3` 或 `sqlite::memory:`
    ///   - PostgreSQL: `postgres://user:pass@host/db`
    ///   - MySQL: `mysql://user:pass@host/db`
    pub fn new(url: &str) -> Result<Self> {
        // 创建 Tokio 运行时
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| OptunaError::StorageInternalError(format!("tokio 运行时创建失败: {e}")))?;

        // 连接数据库
        let db = runtime.block_on(async {
            let mut opts = ConnectOptions::new(url.to_string());
            opts.max_connections(10)
                .min_connections(1)
                .sqlx_logging(false);
            Database::connect(opts).await
        }).map_err(|e| OptunaError::StorageInternalError(format!("数据库连接失败: {e}")))?;

        let storage = Self {
            db,
            runtime: Arc::new(runtime),
            lock: Mutex::new(()),
        };

        // 初始化数据库表
        storage.create_tables()?;

        Ok(storage)
    }

    /// 创建所有数据库表（如不存在则建表）
    fn create_tables(&self) -> Result<()> {
        use sea_orm::sea_query::*;
        use sea_orm::ConnectionTrait;

        self.runtime.block_on(async {
            let backend = self.db.get_database_backend();

            // studies 表
            let stmt = Table::create()
                .table(entity::study::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::study::Column::StudyId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::study::Column::StudyName).string_len(512).not_null().unique_key())
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // study_directions 表
            let stmt = Table::create()
                .table(entity::study_direction::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::study_direction::Column::StudyDirectionId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::study_direction::Column::Direction).integer().not_null())
                .col(ColumnDef::new(entity::study_direction::Column::StudyId).big_integer().not_null())
                .col(ColumnDef::new(entity::study_direction::Column::Objective).integer().not_null())
                .foreign_key(ForeignKey::create().from(entity::study_direction::Entity, entity::study_direction::Column::StudyId).to(entity::study::Entity, entity::study::Column::StudyId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // study_user_attributes 表
            let stmt = Table::create()
                .table(entity::study_user_attr::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::study_user_attr::Column::StudyUserAttributeId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::study_user_attr::Column::StudyId).big_integer().not_null())
                .col(ColumnDef::new(entity::study_user_attr::Column::Key).string_len(512).not_null())
                .col(ColumnDef::new(entity::study_user_attr::Column::ValueJson).text().not_null())
                .foreign_key(ForeignKey::create().from(entity::study_user_attr::Entity, entity::study_user_attr::Column::StudyId).to(entity::study::Entity, entity::study::Column::StudyId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // study_system_attributes 表
            let stmt = Table::create()
                .table(entity::study_system_attr::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::study_system_attr::Column::StudySystemAttributeId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::study_system_attr::Column::StudyId).big_integer().not_null())
                .col(ColumnDef::new(entity::study_system_attr::Column::Key).string_len(512).not_null())
                .col(ColumnDef::new(entity::study_system_attr::Column::ValueJson).text().not_null())
                .foreign_key(ForeignKey::create().from(entity::study_system_attr::Entity, entity::study_system_attr::Column::StudyId).to(entity::study::Entity, entity::study::Column::StudyId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trials 表
            let stmt = Table::create()
                .table(entity::trial::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial::Column::TrialId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial::Column::Number).big_integer().not_null())
                .col(ColumnDef::new(entity::trial::Column::StudyId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial::Column::State).integer().not_null())
                .col(ColumnDef::new(entity::trial::Column::DatetimeStart).string_len(64).null())
                .col(ColumnDef::new(entity::trial::Column::DatetimeComplete).string_len(64).null())
                .foreign_key(ForeignKey::create().from(entity::trial::Entity, entity::trial::Column::StudyId).to(entity::study::Entity, entity::study::Column::StudyId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trial_params 表
            let stmt = Table::create()
                .table(entity::trial_param::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial_param::Column::ParamId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial_param::Column::TrialId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_param::Column::ParamName).string_len(512).not_null())
                .col(ColumnDef::new(entity::trial_param::Column::ParamValue).double().not_null())
                .col(ColumnDef::new(entity::trial_param::Column::DistributionJson).text().not_null())
                .foreign_key(ForeignKey::create().from(entity::trial_param::Entity, entity::trial_param::Column::TrialId).to(entity::trial::Entity, entity::trial::Column::TrialId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trial_values 表
            let stmt = Table::create()
                .table(entity::trial_value::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial_value::Column::TrialValueId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial_value::Column::TrialId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_value::Column::Objective).integer().not_null())
                .col(ColumnDef::new(entity::trial_value::Column::Value).double().not_null())
                .col(ColumnDef::new(entity::trial_value::Column::ValueType).integer().not_null())
                .foreign_key(ForeignKey::create().from(entity::trial_value::Entity, entity::trial_value::Column::TrialId).to(entity::trial::Entity, entity::trial::Column::TrialId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trial_intermediate_values 表
            let stmt = Table::create()
                .table(entity::trial_intermediate_value::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial_intermediate_value::Column::TrialIntermediateValueId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial_intermediate_value::Column::TrialId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_intermediate_value::Column::Step).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_intermediate_value::Column::IntermediateValue).double().not_null())
                .col(ColumnDef::new(entity::trial_intermediate_value::Column::IntermediateValueType).integer().not_null())
                .foreign_key(ForeignKey::create().from(entity::trial_intermediate_value::Entity, entity::trial_intermediate_value::Column::TrialId).to(entity::trial::Entity, entity::trial::Column::TrialId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trial_user_attributes 表
            let stmt = Table::create()
                .table(entity::trial_user_attr::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial_user_attr::Column::TrialUserAttributeId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial_user_attr::Column::TrialId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_user_attr::Column::Key).string_len(512).not_null())
                .col(ColumnDef::new(entity::trial_user_attr::Column::ValueJson).text().not_null())
                .foreign_key(ForeignKey::create().from(entity::trial_user_attr::Entity, entity::trial_user_attr::Column::TrialId).to(entity::trial::Entity, entity::trial::Column::TrialId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            // trial_system_attributes 表
            let stmt = Table::create()
                .table(entity::trial_system_attr::Entity)
                .if_not_exists()
                .col(ColumnDef::new(entity::trial_system_attr::Column::TrialSystemAttributeId).big_integer().not_null().auto_increment().primary_key())
                .col(ColumnDef::new(entity::trial_system_attr::Column::TrialId).big_integer().not_null())
                .col(ColumnDef::new(entity::trial_system_attr::Column::Key).string_len(512).not_null())
                .col(ColumnDef::new(entity::trial_system_attr::Column::ValueJson).text().not_null())
                .foreign_key(ForeignKey::create().from(entity::trial_system_attr::Entity, entity::trial_system_attr::Column::TrialId).to(entity::trial::Entity, entity::trial::Column::TrialId))
                .to_owned();
            self.db.execute(backend.build(&stmt)).await?;

            Ok::<(), sea_orm::DbErr>(())
        }).map_err(|e| OptunaError::StorageInternalError(format!("建表失败: {e}")))?;

        Ok(())
    }

    // ── 辅助函数 ──────────────────────────────────────────────────────

    /// 将数据库时间字符串解析为 DateTime<Utc>
    fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
        // 兼容 ISO 8601 和 Python datetime 格式
        s.parse::<DateTime<Utc>>().ok()
            .or_else(|| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
                .ok()
                .map(|dt| dt.and_utc()))
    }

    /// 将 DateTime<Utc> 转为存储字符串
    fn format_datetime(dt: &DateTime<Utc>) -> String {
        dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()
    }

    /// 将数据库 state(i32) 转为 TrialState
    fn i32_to_trial_state(v: i32) -> Result<TrialState> {
        match v {
            0 => Ok(TrialState::Running),
            1 => Ok(TrialState::Complete),
            2 => Ok(TrialState::Pruned),
            3 => Ok(TrialState::Fail),
            4 => Ok(TrialState::Waiting),
            _ => Err(OptunaError::StorageInternalError(format!("未知 TrialState: {v}"))),
        }
    }

    /// 将 StudyDirection 转为 i32
    fn direction_to_i32(d: StudyDirection) -> i32 {
        match d {
            StudyDirection::NotSet => 0,
            StudyDirection::Minimize => 1,
            StudyDirection::Maximize => 2,
        }
    }

    /// 将 i32 转为 StudyDirection
    fn i32_to_direction(v: i32) -> StudyDirection {
        match v {
            1 => StudyDirection::Minimize,
            2 => StudyDirection::Maximize,
            _ => StudyDirection::NotSet,
        }
    }

    /// 从数据库加载一个完整的 FrozenTrial
    fn load_frozen_trial(&self, trial_model: &entity::trial::Model) -> Result<FrozenTrial> {
        let trial_id = trial_model.trial_id;

        self.runtime.block_on(async {
            // 加载参数
            let params_models = entity::trial_param::Entity::find()
                .filter(entity::trial_param::Column::TrialId.eq(trial_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut params = HashMap::new();
            let mut distributions = HashMap::new();
            for pm in &params_models {
                let dist = crate::distributions::json_to_distribution(&pm.distribution_json)
                    .map_err(|e| OptunaError::StorageInternalError(
                        format!("分布反序列化失败: {e}")))?;
                let param_val = dist.to_external_repr(pm.param_value)?;
                params.insert(pm.param_name.clone(), param_val);
                distributions.insert(pm.param_name.clone(), dist);
            }

            // 加载目标值
            let value_models = entity::trial_value::Entity::find()
                .filter(entity::trial_value::Column::TrialId.eq(trial_id))
                .order_by_asc(entity::trial_value::Column::Objective)
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let values = if value_models.is_empty() {
                None
            } else {
                let vals: Vec<f64> = value_models.iter().map(|vm| {
                    let vtype = TrialValueType::from_i32(vm.value_type)
                        .unwrap_or(TrialValueType::Finite);
                    vtype.to_f64(vm.value)
                }).collect();
                Some(vals)
            };

            // 加载中间值
            let iv_models = entity::trial_intermediate_value::Entity::find()
                .filter(entity::trial_intermediate_value::Column::TrialId.eq(trial_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut intermediate_values = HashMap::new();
            for iv in &iv_models {
                let vtype = TrialValueType::from_i32(iv.intermediate_value_type)
                    .unwrap_or(TrialValueType::Finite);
                intermediate_values.insert(iv.step, vtype.to_f64(iv.intermediate_value));
            }

            // 加载用户属性
            let ua_models = entity::trial_user_attr::Entity::find()
                .filter(entity::trial_user_attr::Column::TrialId.eq(trial_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut user_attrs = HashMap::new();
            for ua in &ua_models {
                let val: serde_json::Value = serde_json::from_str(&ua.value_json)
                    .unwrap_or(serde_json::Value::Null);
                user_attrs.insert(ua.key.clone(), val);
            }

            // 加载系统属性
            let sa_models = entity::trial_system_attr::Entity::find()
                .filter(entity::trial_system_attr::Column::TrialId.eq(trial_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut system_attrs = HashMap::new();
            for sa in &sa_models {
                let val: serde_json::Value = serde_json::from_str(&sa.value_json)
                    .unwrap_or(serde_json::Value::Null);
                system_attrs.insert(sa.key.clone(), val);
            }

            // 解析时间
            let datetime_start = trial_model.datetime_start.as_ref()
                .and_then(|s| Self::parse_datetime(s));
            let datetime_complete = trial_model.datetime_complete.as_ref()
                .and_then(|s| Self::parse_datetime(s));

            let state = Self::i32_to_trial_state(trial_model.state)?;

            Ok(FrozenTrial {
                number: trial_model.number,
                state,
                values,
                datetime_start,
                datetime_complete,
                params,
                distributions,
                user_attrs,
                system_attrs,
                intermediate_values,
                trial_id: trial_model.trial_id,
            })
        })
    }
}

// ════════════════════════════════════════════════════════════════════════
// Storage trait 实现 — 完整 CRUD
// ════════════════════════════════════════════════════════════════════════

impl Storage for RdbStorage {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let _guard = self.lock.lock();

        // 生成唯一名称（如未提供）
        let name = study_name
            .map(String::from)
            .unwrap_or_else(|| format!("no-name-{}", uuid::Uuid::new_v4()));

        self.runtime.block_on(async {
            // 检查重名
            let existing = entity::study::Entity::find()
                .filter(entity::study::Column::StudyName.eq(&name))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if existing.is_some() {
                return Err(OptunaError::DuplicatedStudyError(format!(
                    "study '{}' already exists", name
                )));
            }

            // 插入 study
            let model = entity::study::ActiveModel {
                study_id: ActiveValue::NotSet,
                study_name: Set(name),
            };
            let result = entity::study::Entity::insert(model)
                .exec(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            let study_id = result.last_insert_id;

            // 插入 directions
            let dirs = if directions.is_empty() {
                vec![StudyDirection::Minimize]
            } else {
                directions.to_vec()
            };
            for (i, &dir) in dirs.iter().enumerate() {
                let dir_model = entity::study_direction::ActiveModel {
                    study_direction_id: ActiveValue::NotSet,
                    direction: Set(Self::direction_to_i32(dir)),
                    study_id: Set(study_id),
                    objective: Set(i as i32),
                };
                entity::study_direction::Entity::insert(dir_model)
                    .exec(&self.db)
                    .await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(study_id)
        })
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        let _guard = self.lock.lock();

        self.runtime.block_on(async {
            // 先删除关联数据（trials 及其子表），再删除 study
            let trials = entity::trial::Entity::find()
                .filter(entity::trial::Column::StudyId.eq(study_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            for t in &trials {
                let tid = t.trial_id;
                entity::trial_param::Entity::delete_many()
                    .filter(entity::trial_param::Column::TrialId.eq(tid))
                    .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                entity::trial_value::Entity::delete_many()
                    .filter(entity::trial_value::Column::TrialId.eq(tid))
                    .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                entity::trial_intermediate_value::Entity::delete_many()
                    .filter(entity::trial_intermediate_value::Column::TrialId.eq(tid))
                    .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                entity::trial_user_attr::Entity::delete_many()
                    .filter(entity::trial_user_attr::Column::TrialId.eq(tid))
                    .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                entity::trial_system_attr::Entity::delete_many()
                    .filter(entity::trial_system_attr::Column::TrialId.eq(tid))
                    .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            entity::trial::Entity::delete_many()
                .filter(entity::trial::Column::StudyId.eq(study_id))
                .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            entity::study_direction::Entity::delete_many()
                .filter(entity::study_direction::Column::StudyId.eq(study_id))
                .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            entity::study_user_attr::Entity::delete_many()
                .filter(entity::study_user_attr::Column::StudyId.eq(study_id))
                .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            entity::study_system_attr::Entity::delete_many()
                .filter(entity::study_system_attr::Column::StudyId.eq(study_id))
                .exec(&self.db).await.map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            entity::study::Entity::delete_by_id(study_id)
                .exec(&self.db).await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            Ok(())
        })
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let _guard = self.lock.lock();
        let json_str = serde_json::to_string(&value)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

        self.runtime.block_on(async {
            // 查找已有记录
            let existing = entity::study_user_attr::Entity::find()
                .filter(entity::study_user_attr::Column::StudyId.eq(study_id))
                .filter(entity::study_user_attr::Column::Key.eq(key))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                // 更新
                let mut am: entity::study_user_attr::ActiveModel = existing.into_active_model();
                am.value_json = Set(json_str);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                // 插入
                let am = entity::study_user_attr::ActiveModel {
                    study_user_attribute_id: ActiveValue::NotSet,
                    study_id: Set(study_id),
                    key: Set(key.to_string()),
                    value_json: Set(json_str),
                };
                entity::study_user_attr::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let _guard = self.lock.lock();
        let json_str = serde_json::to_string(&value)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

        self.runtime.block_on(async {
            let existing = entity::study_system_attr::Entity::find()
                .filter(entity::study_system_attr::Column::StudyId.eq(study_id))
                .filter(entity::study_system_attr::Column::Key.eq(key))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                let mut am: entity::study_system_attr::ActiveModel = existing.into_active_model();
                am.value_json = Set(json_str);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                let am = entity::study_system_attr::ActiveModel {
                    study_system_attribute_id: ActiveValue::NotSet,
                    study_id: Set(study_id),
                    key: Set(key.to_string()),
                    value_json: Set(json_str),
                };
                entity::study_system_attr::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        self.runtime.block_on(async {
            let model = entity::study::Entity::find()
                .filter(entity::study::Column::StudyName.eq(study_name))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?
                .ok_or_else(|| OptunaError::ValueError(
                    format!("study '{}' not found", study_name)))?;
            Ok(model.study_id)
        })
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        self.runtime.block_on(async {
            let model = entity::study::Entity::find_by_id(study_id)
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?
                .ok_or_else(|| OptunaError::ValueError(
                    format!("study_id {} not found", study_id)))?;
            Ok(model.study_name)
        })
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        self.runtime.block_on(async {
            let dirs = entity::study_direction::Entity::find()
                .filter(entity::study_direction::Column::StudyId.eq(study_id))
                .order_by_asc(entity::study_direction::Column::Objective)
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            Ok(dirs.iter().map(|d| Self::i32_to_direction(d.direction)).collect())
        })
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        self.runtime.block_on(async {
            let attrs = entity::study_user_attr::Entity::find()
                .filter(entity::study_user_attr::Column::StudyId.eq(study_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut map = HashMap::new();
            for a in &attrs {
                let val: serde_json::Value = serde_json::from_str(&a.value_json)
                    .unwrap_or(serde_json::Value::Null);
                map.insert(a.key.clone(), val);
            }
            Ok(map)
        })
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        self.runtime.block_on(async {
            let attrs = entity::study_system_attr::Entity::find()
                .filter(entity::study_system_attr::Column::StudyId.eq(study_id))
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut map = HashMap::new();
            for a in &attrs {
                let val: serde_json::Value = serde_json::from_str(&a.value_json)
                    .unwrap_or(serde_json::Value::Null);
                map.insert(a.key.clone(), val);
            }
            Ok(map)
        })
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        self.runtime.block_on(async {
            let studies = entity::study::Entity::find()
                .all(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            let mut result = Vec::new();
            for s in &studies {
                let dirs = entity::study_direction::Entity::find()
                    .filter(entity::study_direction::Column::StudyId.eq(s.study_id))
                    .order_by_asc(entity::study_direction::Column::Objective)
                    .all(&self.db)
                    .await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

                let user_attrs = entity::study_user_attr::Entity::find()
                    .filter(entity::study_user_attr::Column::StudyId.eq(s.study_id))
                    .all(&self.db)
                    .await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

                let system_attrs = entity::study_system_attr::Entity::find()
                    .filter(entity::study_system_attr::Column::StudyId.eq(s.study_id))
                    .all(&self.db)
                    .await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

                let n_trials = entity::trial::Entity::find()
                    .filter(entity::trial::Column::StudyId.eq(s.study_id))
                    .count(&self.db)
                    .await
                    .map_err(|e: sea_orm::DbErr| OptunaError::StorageInternalError(e.to_string()))? as usize;

                let mut ua_map = HashMap::new();
                for a in &user_attrs {
                    let val: serde_json::Value = serde_json::from_str(&a.value_json)
                        .unwrap_or(serde_json::Value::Null);
                    ua_map.insert(a.key.clone(), val);
                }

                let mut sa_map = HashMap::new();
                for a in &system_attrs {
                    let val: serde_json::Value = serde_json::from_str(&a.value_json)
                        .unwrap_or(serde_json::Value::Null);
                    sa_map.insert(a.key.clone(), val);
                }

                let _ = n_trials; // 用于未来扩展

                result.push(FrozenStudy {
                    study_name: s.study_name.clone(),
                    study_id: s.study_id,
                    directions: dirs.iter().map(|d| Self::i32_to_direction(d.direction)).collect(),
                    user_attrs: ua_map,
                    system_attrs: sa_map,
                });
            }
            Ok(result)
        })
    }

    // ── Trial CRUD ───────────────────────────────────────────────────

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        let _guard = self.lock.lock();

        self.runtime.block_on(async {
            // 计算新 trial 编号
            let trial_count = entity::trial::Entity::find()
                .filter(entity::trial::Column::StudyId.eq(study_id))
                .count(&self.db)
                .await
                .map_err(|e: sea_orm::DbErr| OptunaError::StorageInternalError(e.to_string()))? as i64;

            let now = Utc::now();

            let (state, datetime_start) = if let Some(tmpl) = template_trial {
                (tmpl.state as i32, tmpl.datetime_start.as_ref().map(Self::format_datetime))
            } else {
                (TrialState::Running as i32, Some(Self::format_datetime(&now)))
            };

            let trial_model = entity::trial::ActiveModel {
                trial_id: ActiveValue::NotSet,
                number: Set(trial_count),
                study_id: Set(study_id),
                state: Set(state),
                datetime_start: Set(datetime_start),
                datetime_complete: Set(None),
            };

            let result = entity::trial::Entity::insert(trial_model)
                .exec(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            let trial_id = result.last_insert_id;

            // 如有模板，复制参数/属性/值
            if let Some(tmpl) = template_trial {
                // 复制参数
                for (name, val) in &tmpl.params {
                    if let Some(dist) = tmpl.distributions.get(name) {
                        let internal = dist.to_internal_repr(val)?;
                        let dist_json = crate::distributions::distribution_to_json(dist)?;
                        let pm = entity::trial_param::ActiveModel {
                            param_id: ActiveValue::NotSet,
                            trial_id: Set(trial_id),
                            param_name: Set(name.clone()),
                            param_value: Set(internal),
                            distribution_json: Set(dist_json),
                        };
                        entity::trial_param::Entity::insert(pm)
                            .exec(&self.db).await
                            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                    }
                }

                // 复制目标值
                if let Some(vals) = &tmpl.values {
                    for (i, &v) in vals.iter().enumerate() {
                        let vtype = TrialValueType::from_f64(v);
                        let stored = if vtype == TrialValueType::Finite { v } else { 0.0 };
                        let vm = entity::trial_value::ActiveModel {
                            trial_value_id: ActiveValue::NotSet,
                            trial_id: Set(trial_id),
                            objective: Set(i as i32),
                            value: Set(stored),
                            value_type: Set(vtype as i32),
                        };
                        entity::trial_value::Entity::insert(vm)
                            .exec(&self.db).await
                            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                    }
                }

                // 复制中间值
                for (&step, &iv) in &tmpl.intermediate_values {
                    let vtype = TrialValueType::from_f64(iv);
                    let stored = if vtype == TrialValueType::Finite { iv } else { 0.0 };
                    let ivm = entity::trial_intermediate_value::ActiveModel {
                        trial_intermediate_value_id: ActiveValue::NotSet,
                        trial_id: Set(trial_id),
                        step: Set(step),
                        intermediate_value: Set(stored),
                        intermediate_value_type: Set(vtype as i32),
                    };
                    entity::trial_intermediate_value::Entity::insert(ivm)
                        .exec(&self.db).await
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                }

                // 复制用户属性
                for (k, v) in &tmpl.user_attrs {
                    let json_str = serde_json::to_string(v)
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                    let ua = entity::trial_user_attr::ActiveModel {
                        trial_user_attribute_id: ActiveValue::NotSet,
                        trial_id: Set(trial_id),
                        key: Set(k.clone()),
                        value_json: Set(json_str),
                    };
                    entity::trial_user_attr::Entity::insert(ua)
                        .exec(&self.db).await
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                }

                // 复制系统属性
                for (k, v) in &tmpl.system_attrs {
                    let json_str = serde_json::to_string(v)
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                    let sa = entity::trial_system_attr::ActiveModel {
                        trial_system_attribute_id: ActiveValue::NotSet,
                        trial_id: Set(trial_id),
                        key: Set(k.clone()),
                        value_json: Set(json_str),
                    };
                    entity::trial_system_attr::Entity::insert(sa)
                        .exec(&self.db).await
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                }
            }

            Ok(trial_id)
        })
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        let _guard = self.lock.lock();
        let dist_json = crate::distributions::distribution_to_json(distribution)?;

        self.runtime.block_on(async {
            // 检查是否已存在此参数
            let existing = entity::trial_param::Entity::find()
                .filter(entity::trial_param::Column::TrialId.eq(trial_id))
                .filter(entity::trial_param::Column::ParamName.eq(param_name))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                let mut am: entity::trial_param::ActiveModel = existing.into_active_model();
                am.param_value = Set(param_value_internal);
                am.distribution_json = Set(dist_json);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                let am = entity::trial_param::ActiveModel {
                    param_id: ActiveValue::NotSet,
                    trial_id: Set(trial_id),
                    param_name: Set(param_name.to_string()),
                    param_value: Set(param_value_internal),
                    distribution_json: Set(dist_json),
                };
                entity::trial_param::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        let _guard = self.lock.lock();

        self.runtime.block_on(async {
            // 获取当前试验
            let trial_model = entity::trial::Entity::find_by_id(trial_id)
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?;

            let current_state = Self::i32_to_trial_state(trial_model.state)?;

            // 已完成的试验不可更新
            if current_state.is_finished() {
                return Err(OptunaError::UpdateFinishedTrialError(format!(
                    "trial {} already finished (state={:?})", trial_id, current_state
                )));
            }

            // RUNNING → RUNNING 被静默拒绝（用于重复 ask）
            if state == TrialState::Running && current_state == TrialState::Running {
                return Ok(false);
            }

            let now = Utc::now();

            // 更新 trial 状态
            let mut am: entity::trial::ActiveModel = trial_model.into_active_model();
            am.state = Set(state as i32);

            // 设置 datetime_start（WAITING → RUNNING）
            if current_state == TrialState::Waiting && state == TrialState::Running {
                am.datetime_start = Set(Some(Self::format_datetime(&now)));
            }

            // 终态设置 datetime_complete
            if state.is_finished() {
                am.datetime_complete = Set(Some(Self::format_datetime(&now)));
            }

            am.update(&self.db).await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            // 设置目标值
            if let Some(vals) = values {
                // 先删除旧值
                entity::trial_value::Entity::delete_many()
                    .filter(entity::trial_value::Column::TrialId.eq(trial_id))
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

                for (i, &v) in vals.iter().enumerate() {
                    let vtype = TrialValueType::from_f64(v);
                    let stored = if vtype == TrialValueType::Finite { v } else { 0.0 };
                    let vm = entity::trial_value::ActiveModel {
                        trial_value_id: ActiveValue::NotSet,
                        trial_id: Set(trial_id),
                        objective: Set(i as i32),
                        value: Set(stored),
                        value_type: Set(vtype as i32),
                    };
                    entity::trial_value::Entity::insert(vm)
                        .exec(&self.db).await
                        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                }
            }

            Ok(true)
        })
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        let _guard = self.lock.lock();

        self.runtime.block_on(async {
            let vtype = TrialValueType::from_f64(intermediate_value);
            let stored = if vtype == TrialValueType::Finite { intermediate_value } else { 0.0 };

            let existing = entity::trial_intermediate_value::Entity::find()
                .filter(entity::trial_intermediate_value::Column::TrialId.eq(trial_id))
                .filter(entity::trial_intermediate_value::Column::Step.eq(step))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                let mut am: entity::trial_intermediate_value::ActiveModel = existing.into_active_model();
                am.intermediate_value = Set(stored);
                am.intermediate_value_type = Set(vtype as i32);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                let am = entity::trial_intermediate_value::ActiveModel {
                    trial_intermediate_value_id: ActiveValue::NotSet,
                    trial_id: Set(trial_id),
                    step: Set(step),
                    intermediate_value: Set(stored),
                    intermediate_value_type: Set(vtype as i32),
                };
                entity::trial_intermediate_value::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let _guard = self.lock.lock();
        let json_str = serde_json::to_string(&value)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

        self.runtime.block_on(async {
            let existing = entity::trial_user_attr::Entity::find()
                .filter(entity::trial_user_attr::Column::TrialId.eq(trial_id))
                .filter(entity::trial_user_attr::Column::Key.eq(key))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                let mut am: entity::trial_user_attr::ActiveModel = existing.into_active_model();
                am.value_json = Set(json_str);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                let am = entity::trial_user_attr::ActiveModel {
                    trial_user_attribute_id: ActiveValue::NotSet,
                    trial_id: Set(trial_id),
                    key: Set(key.to_string()),
                    value_json: Set(json_str),
                };
                entity::trial_user_attr::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let _guard = self.lock.lock();
        let json_str = serde_json::to_string(&value)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

        self.runtime.block_on(async {
            let existing = entity::trial_system_attr::Entity::find()
                .filter(entity::trial_system_attr::Column::TrialId.eq(trial_id))
                .filter(entity::trial_system_attr::Column::Key.eq(key))
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

            if let Some(existing) = existing {
                let mut am: entity::trial_system_attr::ActiveModel = existing.into_active_model();
                am.value_json = Set(json_str);
                am.update(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            } else {
                let am = entity::trial_system_attr::ActiveModel {
                    trial_system_attribute_id: ActiveValue::NotSet,
                    trial_id: Set(trial_id),
                    key: Set(key.to_string()),
                    value_json: Set(json_str),
                };
                entity::trial_system_attr::Entity::insert(am)
                    .exec(&self.db).await
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            }

            Ok(())
        })
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let trial_model = self.runtime.block_on(async {
            entity::trial::Entity::find_by_id(trial_id)
                .one(&self.db)
                .await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))
        })?.ok_or_else(|| OptunaError::ValueError(
            format!("trial {} not found", trial_id)))?;

        self.load_frozen_trial(&trial_model)
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let trial_models = self.runtime.block_on(async {
            let mut query = entity::trial::Entity::find()
                .filter(entity::trial::Column::StudyId.eq(study_id))
                .order_by_asc(entity::trial::Column::Number);

            if let Some(states) = states {
                let state_ints: Vec<i32> = states.iter().map(|s| *s as i32).collect();
                query = query.filter(entity::trial::Column::State.is_in(state_ints));
            }

            query.all(&self.db).await
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))
        })?;

        let mut trials = Vec::with_capacity(trial_models.len());
        for tm in &trial_models {
            trials.push(self.load_frozen_trial(tm)?);
        }
        Ok(trials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{FloatDistribution, Distribution, ParamValue};
    use crate::study::StudyDirection;
    use crate::trial::TrialState;

    /// 创建内存 SQLite 存储用于测试
    fn test_storage() -> RdbStorage {
        RdbStorage::new("sqlite::memory:").unwrap()
    }

    #[test]
    fn test_rdb_create_study() {
        let storage = test_storage();
        let id = storage.create_new_study(
            &[StudyDirection::Minimize],
            Some("test_study"),
        ).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn test_rdb_duplicate_study_name() {
        let storage = test_storage();
        storage.create_new_study(&[StudyDirection::Minimize], Some("dup")).unwrap();
        let result = storage.create_new_study(&[StudyDirection::Minimize], Some("dup"));
        assert!(result.is_err());
    }

    #[test]
    fn test_rdb_get_study_by_name() {
        let storage = test_storage();
        let id = storage.create_new_study(
            &[StudyDirection::Maximize],
            Some("my_study"),
        ).unwrap();
        let found = storage.get_study_id_from_name("my_study").unwrap();
        assert_eq!(id, found);
    }

    #[test]
    fn test_rdb_study_directions() {
        let storage = test_storage();
        let id = storage.create_new_study(
            &[StudyDirection::Minimize, StudyDirection::Maximize],
            Some("multi"),
        ).unwrap();
        let dirs = storage.get_study_directions(id).unwrap();
        assert_eq!(dirs.len(), 2);
        assert_eq!(dirs[0], StudyDirection::Minimize);
        assert_eq!(dirs[1], StudyDirection::Maximize);
    }

    #[test]
    fn test_rdb_study_user_attrs() {
        let storage = test_storage();
        let id = storage.create_new_study(&[StudyDirection::Minimize], Some("attrs")).unwrap();
        storage.set_study_user_attr(id, "key1", serde_json::json!("value1")).unwrap();
        storage.set_study_user_attr(id, "key2", serde_json::json!(42)).unwrap();
        let attrs = storage.get_study_user_attrs(id).unwrap();
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs["key1"], serde_json::json!("value1"));
        assert_eq!(attrs["key2"], serde_json::json!(42));
    }

    #[test]
    fn test_rdb_create_trial() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("trial_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();
        let trial = storage.get_trial(trial_id).unwrap();
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state, TrialState::Running);
    }

    #[test]
    fn test_rdb_set_trial_param() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("param_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();

        let dist = Distribution::FloatDistribution(FloatDistribution {
            low: 0.0, high: 1.0, log: false, step: None,
        });
        storage.set_trial_param(trial_id, "x", 0.5, &dist).unwrap();

        let trial = storage.get_trial(trial_id).unwrap();
        assert!(trial.params.contains_key("x"));
    }

    #[test]
    fn test_rdb_set_trial_state_values() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("state_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();

        let ok = storage.set_trial_state_values(trial_id, TrialState::Complete, Some(&[1.23])).unwrap();
        assert!(ok);

        let trial = storage.get_trial(trial_id).unwrap();
        assert_eq!(trial.state, TrialState::Complete);
        assert_eq!(trial.values.unwrap()[0], 1.23);
    }

    #[test]
    fn test_rdb_intermediate_values() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("iv_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();

        storage.set_trial_intermediate_value(trial_id, 0, 0.1).unwrap();
        storage.set_trial_intermediate_value(trial_id, 1, 0.2).unwrap();

        let trial = storage.get_trial(trial_id).unwrap();
        assert_eq!(trial.intermediate_values.len(), 2);
        assert!((trial.intermediate_values[&0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_rdb_delete_study() {
        let storage = test_storage();
        let id = storage.create_new_study(&[StudyDirection::Minimize], Some("del")).unwrap();
        storage.create_new_trial(id, None).unwrap();
        storage.delete_study(id).unwrap();
        let studies = storage.get_all_studies().unwrap();
        assert!(studies.is_empty());
    }

    #[test]
    fn test_rdb_get_all_trials_with_state_filter() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("filter_test")).unwrap();
        let t1 = storage.create_new_trial(study_id, None).unwrap();
        let t2 = storage.create_new_trial(study_id, None).unwrap();
        storage.set_trial_state_values(t1, TrialState::Complete, Some(&[1.0])).unwrap();
        storage.set_trial_state_values(t2, TrialState::Fail, None).unwrap();

        let complete = storage.get_all_trials(study_id, Some(&[TrialState::Complete])).unwrap();
        assert_eq!(complete.len(), 1);
        let failed = storage.get_all_trials(study_id, Some(&[TrialState::Fail])).unwrap();
        assert_eq!(failed.len(), 1);
        let all = storage.get_all_trials(study_id, None).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_rdb_get_all_studies() {
        let storage = test_storage();
        storage.create_new_study(&[StudyDirection::Minimize], Some("s1")).unwrap();
        storage.create_new_study(&[StudyDirection::Maximize], Some("s2")).unwrap();
        let studies = storage.get_all_studies().unwrap();
        assert_eq!(studies.len(), 2);
    }

    #[test]
    fn test_rdb_value_types_inf() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("inf_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();
        storage.set_trial_state_values(trial_id, TrialState::Complete, Some(&[f64::INFINITY])).unwrap();
        let trial = storage.get_trial(trial_id).unwrap();
        assert!(trial.values.unwrap()[0].is_infinite());
    }

    #[test]
    fn test_rdb_trial_attrs() {
        let storage = test_storage();
        let study_id = storage.create_new_study(&[StudyDirection::Minimize], Some("attr_test")).unwrap();
        let trial_id = storage.create_new_trial(study_id, None).unwrap();
        storage.set_trial_user_attr(trial_id, "ua1", serde_json::json!("val")).unwrap();
        storage.set_trial_system_attr(trial_id, "sa1", serde_json::json!(123)).unwrap();
        let trial = storage.get_trial(trial_id).unwrap();
        assert_eq!(trial.user_attrs["ua1"], serde_json::json!("val"));
        assert_eq!(trial.system_attrs["sa1"], serde_json::json!(123));
    }
}
