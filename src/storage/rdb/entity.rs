//! SeaORM 实体定义 — 对应 Python optuna.storages._rdb.models
//!
//! 数据库表结构完全对齐 Python 版本的 Schema Version 12。
//! 包含: studies, study_directions, study_user_attributes, study_system_attributes,
//!       trials, trial_params, trial_values, trial_intermediate_values,
//!       trial_user_attributes, trial_system_attributes

use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════════════
// 常量 — 对应 Python SCHEMA_VERSION / MAX_INDEXED_STRING_LENGTH 等
// ════════════════════════════════════════════════════════════════════════

/// 数据库 schema 版本号
pub const SCHEMA_VERSION: i32 = 12;
/// 索引字符串最大长度
pub const MAX_INDEXED_STRING_LENGTH: usize = 512;
/// IEEE 754 双精度浮点位数
pub const FLOAT_PRECISION: u32 = 53;

// ════════════════════════════════════════════════════════════════════════
// 值类型枚举 — 对应 Python TrialValueType
// ════════════════════════════════════════════════════════════════════════

/// trial_values / trial_intermediate_values 中的值类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialValueType {
    /// 有限浮点数
    Finite = 1,
    /// 正无穷
    InfPos = 2,
    /// 负无穷
    InfNeg = 3,
    /// NaN（仅用于 intermediate values）
    Nan = 4,
}

impl TrialValueType {
    /// 从 f64 推断值类型
    pub fn from_f64(v: f64) -> Self {
        if v.is_nan() {
            Self::Nan
        } else if v.is_infinite() {
            if v > 0.0 { Self::InfPos } else { Self::InfNeg }
        } else {
            Self::Finite
        }
    }

    /// 将值类型和存储值还原为 f64
    pub fn to_f64(self, stored: f64) -> f64 {
        match self {
            Self::Finite => stored,
            Self::InfPos => f64::INFINITY,
            Self::InfNeg => f64::NEG_INFINITY,
            Self::Nan => f64::NAN,
        }
    }

    /// 从数据库整型转换
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            1 => Some(Self::Finite),
            2 => Some(Self::InfPos),
            3 => Some(Self::InfNeg),
            4 => Some(Self::Nan),
            _ => None,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// 表 1: studies — 研究表
// ════════════════════════════════════════════════════════════════════════
pub mod study {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "studies")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub study_id: i64,
        #[sea_orm(unique, indexed)]
        pub study_name: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(has_many = "super::study_direction::Entity")]
        StudyDirections,
        #[sea_orm(has_many = "super::study_user_attr::Entity")]
        UserAttrs,
        #[sea_orm(has_many = "super::study_system_attr::Entity")]
        SystemAttrs,
        #[sea_orm(has_many = "super::trial::Entity")]
        Trials,
    }

    impl Related<super::study_direction::Entity> for Entity {
        fn to() -> RelationDef { Relation::StudyDirections.def() }
    }
    impl Related<super::study_user_attr::Entity> for Entity {
        fn to() -> RelationDef { Relation::UserAttrs.def() }
    }
    impl Related<super::study_system_attr::Entity> for Entity {
        fn to() -> RelationDef { Relation::SystemAttrs.def() }
    }
    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trials.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 2: study_directions — 研究方向表
// ════════════════════════════════════════════════════════════════════════
pub mod study_direction {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "study_directions")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub study_direction_id: i64,
        /// 优化方向: 0=NotSet, 1=Minimize, 2=Maximize
        pub direction: i32,
        /// 所属研究 ID
        pub study_id: i64,
        /// 目标函数索引（多目标时 0, 1, 2, ...）
        pub objective: i32,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::study::Entity",
            from = "Column::StudyId",
            to = "super::study::Column::StudyId"
        )]
        Study,
    }

    impl Related<super::study::Entity> for Entity {
        fn to() -> RelationDef { Relation::Study.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 3: study_user_attributes — 研究用户属性表
// ════════════════════════════════════════════════════════════════════════
pub mod study_user_attr {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "study_user_attributes")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub study_user_attribute_id: i64,
        pub study_id: i64,
        #[sea_orm(indexed)]
        pub key: String,
        #[sea_orm(column_type = "Text")]
        pub value_json: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::study::Entity",
            from = "Column::StudyId",
            to = "super::study::Column::StudyId"
        )]
        Study,
    }

    impl Related<super::study::Entity> for Entity {
        fn to() -> RelationDef { Relation::Study.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 4: study_system_attributes — 研究系统属性表
// ════════════════════════════════════════════════════════════════════════
pub mod study_system_attr {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "study_system_attributes")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub study_system_attribute_id: i64,
        pub study_id: i64,
        #[sea_orm(indexed)]
        pub key: String,
        #[sea_orm(column_type = "Text")]
        pub value_json: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::study::Entity",
            from = "Column::StudyId",
            to = "super::study::Column::StudyId"
        )]
        Study,
    }

    impl Related<super::study::Entity> for Entity {
        fn to() -> RelationDef { Relation::Study.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 5: trials — 试验表
// ════════════════════════════════════════════════════════════════════════
pub mod trial {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "trials")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub trial_id: i64,
        /// 试验编号（在研究内递增）
        pub number: i64,
        /// 所属研究 ID
        #[sea_orm(indexed)]
        pub study_id: i64,
        /// 试验状态: 0=Running, 1=Complete, 2=Pruned, 3=Fail, 4=Waiting
        #[sea_orm(indexed)]
        pub state: i32,
        /// 开始时间
        pub datetime_start: Option<String>,
        /// 完成时间
        pub datetime_complete: Option<String>,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::study::Entity",
            from = "Column::StudyId",
            to = "super::study::Column::StudyId"
        )]
        Study,
        #[sea_orm(has_many = "super::trial_param::Entity")]
        Params,
        #[sea_orm(has_many = "super::trial_value::Entity")]
        Values,
        #[sea_orm(has_many = "super::trial_intermediate_value::Entity")]
        IntermediateValues,
        #[sea_orm(has_many = "super::trial_user_attr::Entity")]
        UserAttrs,
        #[sea_orm(has_many = "super::trial_system_attr::Entity")]
        SystemAttrs,
    }

    impl Related<super::study::Entity> for Entity {
        fn to() -> RelationDef { Relation::Study.def() }
    }
    impl Related<super::trial_param::Entity> for Entity {
        fn to() -> RelationDef { Relation::Params.def() }
    }
    impl Related<super::trial_value::Entity> for Entity {
        fn to() -> RelationDef { Relation::Values.def() }
    }
    impl Related<super::trial_intermediate_value::Entity> for Entity {
        fn to() -> RelationDef { Relation::IntermediateValues.def() }
    }
    impl Related<super::trial_user_attr::Entity> for Entity {
        fn to() -> RelationDef { Relation::UserAttrs.def() }
    }
    impl Related<super::trial_system_attr::Entity> for Entity {
        fn to() -> RelationDef { Relation::SystemAttrs.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 6: trial_params — 试验参数表
// ════════════════════════════════════════════════════════════════════════
pub mod trial_param {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "trial_params")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub param_id: i64,
        pub trial_id: i64,
        #[sea_orm(indexed)]
        pub param_name: String,
        /// 内部表示值（IEEE 754 双精度）
        pub param_value: f64,
        /// 分布的 JSON 序列化
        #[sea_orm(column_type = "Text")]
        pub distribution_json: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::trial::Entity",
            from = "Column::TrialId",
            to = "super::trial::Column::TrialId"
        )]
        Trial,
    }

    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trial.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 7: trial_values — 试验目标值表
// ════════════════════════════════════════════════════════════════════════
pub mod trial_value {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "trial_values")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub trial_value_id: i64,
        pub trial_id: i64,
        /// 目标函数索引
        pub objective: i32,
        /// 目标值（有限值时存储实际值，无穷/NaN 时存储 0.0）
        pub value: f64,
        /// 值类型: 1=Finite, 2=InfPos, 3=InfNeg
        pub value_type: i32,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::trial::Entity",
            from = "Column::TrialId",
            to = "super::trial::Column::TrialId"
        )]
        Trial,
    }

    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trial.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 8: trial_intermediate_values — 试验中间值表
// ════════════════════════════════════════════════════════════════════════
pub mod trial_intermediate_value {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
    #[sea_orm(table_name = "trial_intermediate_values")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub trial_intermediate_value_id: i64,
        pub trial_id: i64,
        /// 步骤编号
        pub step: i64,
        /// 中间值
        pub intermediate_value: f64,
        /// 值类型: 1=Finite, 2=InfPos, 3=InfNeg, 4=NaN
        pub intermediate_value_type: i32,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::trial::Entity",
            from = "Column::TrialId",
            to = "super::trial::Column::TrialId"
        )]
        Trial,
    }

    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trial.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 9: trial_user_attributes — 试验用户属性表
// ════════════════════════════════════════════════════════════════════════
pub mod trial_user_attr {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "trial_user_attributes")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub trial_user_attribute_id: i64,
        pub trial_id: i64,
        #[sea_orm(indexed)]
        pub key: String,
        #[sea_orm(column_type = "Text")]
        pub value_json: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::trial::Entity",
            from = "Column::TrialId",
            to = "super::trial::Column::TrialId"
        )]
        Trial,
    }

    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trial.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}

// ════════════════════════════════════════════════════════════════════════
// 表 10: trial_system_attributes — 试验系统属性表
// ════════════════════════════════════════════════════════════════════════
pub mod trial_system_attr {
    use sea_orm::entity::prelude::*;

    #[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
    #[sea_orm(table_name = "trial_system_attributes")]
    pub struct Model {
        #[sea_orm(primary_key)]
        pub trial_system_attribute_id: i64,
        pub trial_id: i64,
        #[sea_orm(indexed)]
        pub key: String,
        #[sea_orm(column_type = "Text")]
        pub value_json: String,
    }

    #[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
    pub enum Relation {
        #[sea_orm(
            belongs_to = "super::trial::Entity",
            from = "Column::TrialId",
            to = "super::trial::Column::TrialId"
        )]
        Trial,
    }

    impl Related<super::trial::Entity> for Entity {
        fn to() -> RelationDef { Relation::Trial.def() }
    }

    impl ActiveModelBehavior for ActiveModel {}
}
