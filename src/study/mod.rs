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
use crate::samplers::{NSGAIISamplerBuilder, Sampler, TpeSamplerBuilder};
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
        if ds.is_empty() {
            return Err(OptunaError::ValueError(
                "directions must have at least one element".into(),
            ));
        }
        ds
    } else {
        vec![StudyDirection::Minimize]
    };

    // 对齐 Python: 禁止使用 NOT_SET 方向
    if dirs.iter().any(|d| matches!(d, StudyDirection::NotSet)) {
        return Err(OptunaError::ValueError(
            "StudyDirection must be either MINIMIZE or MAXIMIZE.".into(),
        ));
    }

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

    // 对齐 Python: 多目标默认 NSGAIISampler，单目标默认 TPESampler
    let sampler = sampler.unwrap_or_else(|| {
        if dirs.len() > 1 {
            Arc::new(NSGAIISamplerBuilder::new(dirs.clone()).build())
        } else {
            Arc::new(TpeSamplerBuilder::new(dirs[0]).build())
        }
    });
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

    // 对齐 Python: 多目标时默认切换到 NSGAIISampler
    let sampler = sampler.unwrap_or_else(|| {
        if dirs.len() > 1 {
            Arc::new(NSGAIISamplerBuilder::new(dirs.to_vec()).build())
        } else {
            Arc::new(TpeSamplerBuilder::new(dirs[0]).build())
        }
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// 对齐 Python: create_study 默认参数
    #[test]
    fn test_create_study_default() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        // 默认方向为 Minimize
        assert_eq!(study.directions(), &[StudyDirection::Minimize]);
    }

    /// 对齐 Python: direction 和 directions 互斥
    #[test]
    fn test_create_study_direction_conflict() {
        let result = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize),
            Some(vec![StudyDirection::Maximize]),
            false,
        );
        assert!(result.is_err());
    }

    /// 对齐 Python: load_if_exists
    #[test]
    fn test_create_study_load_if_exists() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let s1 = create_study(
            Some(Arc::clone(&storage)),
            None, None, Some("my_study"),
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        // 重复创建同名 study 报错
        let err = create_study(
            Some(Arc::clone(&storage)),
            None, None, Some("my_study"),
            Some(StudyDirection::Minimize), None, false,
        );
        assert!(err.is_err());
        // load_if_exists=true 则加载
        let s2 = create_study(
            Some(Arc::clone(&storage)),
            None, None, Some("my_study"),
            Some(StudyDirection::Minimize), None, true,
        ).unwrap();
        assert_eq!(s1.study_id(), s2.study_id());
    }

    /// 对齐 Python: load_study
    #[test]
    fn test_load_study() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let s1 = create_study(
            Some(Arc::clone(&storage)),
            None, None, Some("test_load"),
            Some(StudyDirection::Maximize), None, false,
        ).unwrap();
        let s2 = load_study("test_load", Arc::clone(&storage), None, None).unwrap();
        assert_eq!(s1.study_id(), s2.study_id());
    }

    /// 对齐 Python: load_study 未找到报错
    #[test]
    fn test_load_study_not_found() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let result = load_study("nonexistent", storage, None, None);
        assert!(result.is_err());
    }

    /// 对齐 Python: delete_study
    #[test]
    fn test_delete_study() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let _ = create_study(
            Some(Arc::clone(&storage)),
            None, None, Some("to_delete"),
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        delete_study("to_delete", &*storage).unwrap();
        // 再次删除应报错
        assert!(delete_study("to_delete", &*storage).is_err());
    }

    /// 对齐 Python: copy_study
    #[test]
    fn test_copy_study() {
        let from_storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let to_storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let study = create_study(
            Some(Arc::clone(&from_storage)),
            None, None, Some("source"),
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        // 添加试验
        study.optimize(|trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(x * x)
        }, Some(3), None, None).unwrap();
        // 复制
        copy_study("source", &*from_storage, &*to_storage, Some("target")).unwrap();
        // 验证目标
        let target = load_study("target", Arc::clone(&to_storage), None, None).unwrap();
        assert_eq!(target.trials().unwrap().len(), 3);
    }

    /// 对齐 Python: get_all_study_names
    #[test]
    fn test_get_all_study_names() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let _ = create_study(
            Some(Arc::clone(&storage)), None, None, Some("s1"),
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let _ = create_study(
            Some(Arc::clone(&storage)), None, None, Some("s2"),
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let names = get_all_study_names(&*storage).unwrap();
        assert!(names.contains(&"s1".to_string()));
        assert!(names.contains(&"s2".to_string()));
    }

    /// 对齐 Python: create_study 多目标
    #[test]
    fn test_create_study_multi_objective() {
        let study = create_study(
            None, None, None, None, None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Maximize]),
            false,
        ).unwrap();
        assert_eq!(study.directions().len(), 2);
    }

    /// 对齐 Python: create_study 空方向列表报 ValueError
    #[test]
    fn test_create_study_empty_directions_error() {
        let result = create_study(
            None, None, None, None, None,
            Some(vec![]),
            false,
        );
        assert!(result.is_err());
    }

    /// 对齐 Python: create_study 同时指定 direction 和 directions 报错
    #[test]
    fn test_create_study_both_direction_directions_error() {
        let result = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize),
            Some(vec![StudyDirection::Minimize]),
            false,
        );
        assert!(result.is_err());
    }
}
