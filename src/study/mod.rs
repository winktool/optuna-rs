mod direction;
mod frozen;
mod core;
#[cfg(feature = "dataframe")]
pub mod dataframe;

pub use core::Study;
pub use direction::StudyDirection;
pub use frozen::FrozenStudy;

use std::sync::Arc;

use crate::error::{OptunaError, Result};
use crate::pruners::{MedianPruner, Pruner};
use crate::samplers::{Sampler, TpeSamplerBuilder};
use crate::storage::{InMemoryStorage, Storage};

/// 创建新的研究。
///
/// 对应 Python `optuna.create_study()`。
///
/// # 参数
/// * `storage` - 存储后端。`None` 使用 `InMemoryStorage`。
/// * `sampler` - 采样器。`None` 使用 `TpeSampler`（与 Python 默认一致）。
/// * `pruner` - 剪枝器。`None` 使用 `MedianPruner`（与 Python 默认一致）。
/// * `study_name` - 研究名称。`None` 自动生成。
/// * `direction` - 单目标方向。与 `directions` 互斥。
/// * `directions` - 多目标方向。与 `direction` 互斥。
/// * `load_if_exists` - `true` 则加载已有研究而非报错。
#[allow(clippy::too_many_arguments)]
pub fn create_study(
    storage: Option<Arc<dyn Storage>>,
    sampler: Option<Arc<dyn Sampler>>,
    pruner: Option<Arc<dyn Pruner>>,
    study_name: Option<&str>,
    direction: Option<StudyDirection>,
    directions: Option<Vec<StudyDirection>>,
    load_if_exists: bool,
) -> Result<Study> {
    // 不允许同时指定 direction 和 directions
    if direction.is_some() && directions.is_some() {
        return Err(OptunaError::ValueError(
            "specify either `direction` or `directions`, not both".into(),
        ));
    }

    // 解析优化方向
    let dirs = if let Some(d) = direction {
        vec![d]
    } else if let Some(ds) = directions {
        ds
    } else {
        vec![StudyDirection::Minimize]
    };

    // 默认使用内存存储
    let storage = storage.unwrap_or_else(|| Arc::new(InMemoryStorage::new()));

    // 创建或加载研究
    let study_id = match storage.create_new_study(&dirs, study_name) {
        Ok(id) => id,
        Err(OptunaError::DuplicatedStudyError(_)) if load_if_exists => {
            let name = study_name.ok_or_else(|| {
                OptunaError::ValueError(
                    "load_if_exists requires a study_name".into(),
                )
            })?;
            storage.get_study_id_from_name(name)?
        }
        Err(e) => return Err(e),
    };

    let name = storage.get_study_name_from_id(study_id)?;
    let stored_dirs = storage.get_study_directions(study_id)?;

    // 与 Python 对齐：默认采样器为 TpeSampler，默认剪枝器为 MedianPruner
    let sampler =
        sampler.unwrap_or_else(|| Arc::new(TpeSamplerBuilder::new(dirs[0]).build()));
    let pruner = pruner.unwrap_or_else(|| {
        Arc::new(MedianPruner::new(5, 0, 1, 1, dirs[0]))
    });

    Ok(Study::new(
        name,
        study_id,
        storage,
        stored_dirs,
        sampler,
        pruner,
    ))
}

/// 加载已存在的研究。
///
/// 对应 Python `optuna.load_study()`。
///
/// # 参数
/// * `study_name` - 要加载的研究名称
/// * `storage` - 存储后端
/// * `sampler` - 采样器（可选）
/// * `pruner` - 剪枝器（可选）
pub fn load_study(
    study_name: &str,
    storage: Arc<dyn Storage>,
    sampler: Option<Arc<dyn Sampler>>,
    pruner: Option<Arc<dyn Pruner>>,
) -> Result<Study> {
    // 通过名称查找研究 ID
    let study_id = storage.get_study_id_from_name(study_name)?;
    let name = storage.get_study_name_from_id(study_id)?;
    let dirs = storage.get_study_directions(study_id)?;

    // 使用默认或指定的采样器/剪枝器
    let sampler =
        sampler.unwrap_or_else(|| Arc::new(TpeSamplerBuilder::new(dirs[0]).build()));
    let pruner = pruner.unwrap_or_else(|| {
        Arc::new(MedianPruner::new(5, 0, 1, 1, dirs[0]))
    });

    Ok(Study::new(name, study_id, storage, dirs, sampler, pruner))
}

/// 删除研究。
///
/// 对应 Python `optuna.delete_study()`。
///
/// # 参数
/// * `study_name` - 要删除的研究名称
/// * `storage` - 存储后端
pub fn delete_study(study_name: &str, storage: &dyn Storage) -> Result<()> {
    let study_id = storage.get_study_id_from_name(study_name)?;
    storage.delete_study(study_id)
}

/// 复制研究到同一存储或不同存储。
///
/// 对应 Python `optuna.copy_study()`。
///
/// # 参数
/// * `from_study_name` - 源研究名称
/// * `from_storage` - 源存储
/// * `to_storage` - 目标存储
/// * `to_study_name` - 目标研究名称（可选，默认与源同名）
pub fn copy_study(
    from_study_name: &str,
    from_storage: &dyn Storage,
    to_storage: &dyn Storage,
    to_study_name: Option<&str>,
) -> Result<()> {
    // 获取源研究信息
    let from_id = from_storage.get_study_id_from_name(from_study_name)?;
    let directions = from_storage.get_study_directions(from_id)?;
    let to_name = to_study_name.unwrap_or(from_study_name);

    // 在目标存储创建新研究
    let to_id = to_storage.create_new_study(&directions, Some(to_name))?;

    // 复制用户属性
    let user_attrs = from_storage.get_study_user_attrs(from_id)?;
    for (key, value) in user_attrs {
        to_storage.set_study_user_attr(to_id, &key, value)?;
    }

    // 复制系统属性
    let system_attrs = from_storage.get_study_system_attrs(from_id)?;
    for (key, value) in system_attrs {
        to_storage.set_study_system_attr(to_id, &key, value)?;
    }

    // 复制所有试验
    let trials = from_storage.get_all_trials(from_id, None)?;
    for trial in &trials {
        to_storage.create_new_trial(to_id, Some(trial))?;
    }

    Ok(())
}

/// 获取所有研究名称。
///
/// 对应 Python `optuna.get_all_study_names()`。
pub fn get_all_study_names(storage: &dyn Storage) -> Result<Vec<String>> {
    let studies = storage.get_all_studies()?;
    Ok(studies.into_iter().map(|s| s.study_name).collect())
}

/// 获取所有研究摘要。
///
/// 对应 Python `optuna.get_all_study_summaries()`。
pub fn get_all_study_summaries(storage: &dyn Storage) -> Result<Vec<FrozenStudy>> {
    storage.get_all_studies()
}
