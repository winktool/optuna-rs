use serde::{Deserialize, Serialize};

/// The optimization direction of a study.
///
/// Corresponds to Python `optuna.study.StudyDirection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StudyDirection {
    /// Direction has not been set yet.
    NotSet = 0,
    /// Minimize the objective value.
    Minimize = 1,
    /// Maximize the objective value.
    Maximize = 2,
}

impl std::fmt::Display for StudyDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSet => write!(f, "NOT_SET"),
            Self::Minimize => write!(f, "MINIMIZE"),
            Self::Maximize => write!(f, "MAXIMIZE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", StudyDirection::NotSet), "NOT_SET");
        assert_eq!(format!("{}", StudyDirection::Minimize), "MINIMIZE");
        assert_eq!(format!("{}", StudyDirection::Maximize), "MAXIMIZE");
    }

    #[test]
    fn test_equality() {
        assert_eq!(StudyDirection::Minimize, StudyDirection::Minimize);
        assert_ne!(StudyDirection::Minimize, StudyDirection::Maximize);
    }

    /// 对齐 Python: IntEnum 数值 NOT_SET=0, MINIMIZE=1, MAXIMIZE=2
    #[test]
    fn test_repr_values() {
        assert_eq!(StudyDirection::NotSet as u8, 0);
        assert_eq!(StudyDirection::Minimize as u8, 1);
        assert_eq!(StudyDirection::Maximize as u8, 2);
    }

    /// 对齐 Python: 序列化/反序列化一致性
    #[test]
    fn test_serde_roundtrip() {
        for dir in [StudyDirection::NotSet, StudyDirection::Minimize, StudyDirection::Maximize] {
            let json = serde_json::to_string(&dir).unwrap();
            let deser: StudyDirection = serde_json::from_str(&json).unwrap();
            assert_eq!(dir, deser);
        }
    }

    /// 对齐 Python: Hash trait 可作为 HashMap 键
    #[test]
    fn test_hash_as_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(StudyDirection::Minimize, "min");
        map.insert(StudyDirection::Maximize, "max");
        map.insert(StudyDirection::NotSet, "not_set");
        assert_eq!(map[&StudyDirection::Minimize], "min");
        assert_eq!(map[&StudyDirection::Maximize], "max");
        assert_eq!(map[&StudyDirection::NotSet], "not_set");
        assert_eq!(map.len(), 3);
    }

    /// 对齐 Python: Copy/Clone 值语义
    #[test]
    fn test_clone_and_copy() {
        let d = StudyDirection::Maximize;
        let d2 = d; // Copy
        let d3 = d.clone();
        assert_eq!(d, d2);
        assert_eq!(d, d3);
    }

    /// 对齐 Python: Debug 格式用于日志
    #[test]
    fn test_debug_format() {
        assert_eq!(format!("{:?}", StudyDirection::Minimize), "Minimize");
        assert_eq!(format!("{:?}", StudyDirection::Maximize), "Maximize");
        assert_eq!(format!("{:?}", StudyDirection::NotSet), "NotSet");
    }

    /// 对齐 Python: 所有变体不相等
    #[test]
    fn test_all_variants_distinct() {
        let variants = [StudyDirection::NotSet, StudyDirection::Minimize, StudyDirection::Maximize];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
