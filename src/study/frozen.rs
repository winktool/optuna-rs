use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::study::StudyDirection;

/// An immutable snapshot of a study's metadata.
///
/// Returned by `Storage::get_all_studies()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenStudy {
    pub study_id: i64,
    pub study_name: String,
    pub directions: Vec<StudyDirection>,
    pub user_attrs: HashMap<String, serde_json::Value>,
    pub system_attrs: HashMap<String, serde_json::Value>,
}

impl FrozenStudy {
    /// 对齐 Python `FrozenStudy.direction` property:
    /// 单目标时返回唯一方向，多目标时报 RuntimeError。
    pub fn direction(&self) -> crate::error::Result<StudyDirection> {
        if self.directions.len() != 1 {
            return Err(crate::error::OptunaError::RuntimeError(
                "This attribute is not available during multi-objective optimization.".into(),
            ));
        }
        Ok(self.directions[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 对齐 Python: FrozenStudy 基本构造
    #[test]
    fn test_frozen_study_basic() {
        let fs = FrozenStudy {
            study_id: 1,
            study_name: "test_study".to_string(),
            directions: vec![StudyDirection::Minimize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        assert_eq!(fs.study_id, 1);
        assert_eq!(fs.study_name, "test_study");
        assert_eq!(fs.directions.len(), 1);
    }

    /// 对齐 Python: FrozenStudy Clone
    #[test]
    fn test_frozen_study_clone() {
        let fs = FrozenStudy {
            study_id: 2,
            study_name: "cloned".to_string(),
            directions: vec![StudyDirection::Minimize, StudyDirection::Maximize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        let fs2 = fs.clone();
        assert_eq!(fs2.study_id, 2);
        assert_eq!(fs2.directions.len(), 2);
    }

    /// 对齐 Python: FrozenStudy 多目标方向
    #[test]
    fn test_frozen_study_multi_objective() {
        let fs = FrozenStudy {
            study_id: 3,
            study_name: "multi".to_string(),
            directions: vec![
                StudyDirection::Minimize,
                StudyDirection::Maximize,
                StudyDirection::Minimize,
            ],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        assert_eq!(fs.directions[0], StudyDirection::Minimize);
        assert_eq!(fs.directions[1], StudyDirection::Maximize);
        assert_eq!(fs.directions[2], StudyDirection::Minimize);
    }

    /// 对齐 Python: FrozenStudy 带 user_attrs
    #[test]
    fn test_frozen_study_with_attrs() {
        let mut ua = HashMap::new();
        ua.insert("key".to_string(), serde_json::json!("value"));
        let fs = FrozenStudy {
            study_id: 4,
            study_name: "attrs".to_string(),
            directions: vec![StudyDirection::Minimize],
            user_attrs: ua,
            system_attrs: HashMap::new(),
        };
        assert_eq!(fs.user_attrs["key"], serde_json::json!("value"));
    }

    /// 对齐 Python: FrozenStudy serde 序列化/反序列化
    #[test]
    fn test_frozen_study_serde_roundtrip() {
        let fs = FrozenStudy {
            study_id: 5,
            study_name: "serde_test".to_string(),
            directions: vec![StudyDirection::Maximize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        let json = serde_json::to_string(&fs).unwrap();
        let fs2: FrozenStudy = serde_json::from_str(&json).unwrap();
        assert_eq!(fs2.study_id, 5);
        assert_eq!(fs2.study_name, "serde_test");
    }

    /// 对齐 Python: FrozenStudy.direction() 单目标时返回方向
    #[test]
    fn test_frozen_study_direction_single() {
        let fs = FrozenStudy {
            study_id: 6,
            study_name: "single_dir".to_string(),
            directions: vec![StudyDirection::Minimize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        assert_eq!(fs.direction().unwrap(), StudyDirection::Minimize);
    }

    /// 对齐 Python: FrozenStudy.direction() 多目标时报 RuntimeError
    #[test]
    fn test_frozen_study_direction_multi_error() {
        let fs = FrozenStudy {
            study_id: 7,
            study_name: "multi_dir".to_string(),
            directions: vec![StudyDirection::Minimize, StudyDirection::Maximize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        let err = fs.direction().unwrap_err();
        assert!(matches!(err, crate::error::OptunaError::RuntimeError(_)));
    }

    /// 对齐 Python: FrozenStudy PartialEq
    #[test]
    fn test_frozen_study_partial_eq() {
        let fs1 = FrozenStudy {
            study_id: 8,
            study_name: "eq".to_string(),
            directions: vec![StudyDirection::Minimize],
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
        };
        let fs2 = fs1.clone();
        assert_eq!(fs1, fs2);
    }
}
