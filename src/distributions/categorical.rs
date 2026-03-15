use serde::{Deserialize, Serialize};

use crate::error::{OptunaError, Result};

/// A value that can appear as a categorical choice.
///
/// Corresponds to Python `CategoricalChoiceType = Union[None, bool, int, float, str]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CategoricalChoice {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl PartialEq for CategoricalChoice {
    /// NaN-safe 比较。
    /// 对应 Python `_categorical_choice_equal(v1, v2)`:
    /// `math.isnan(v1) and math.isnan(v2)` → True
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => {
                // NaN == NaN → true (对应 Python 行为)
                if a.is_nan() && b.is_nan() {
                    return true;
                }
                a == b
            }
            (Self::Str(a), Self::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CategoricalChoice {}

impl std::fmt::Display for CategoricalChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(s) => write!(f, "{s}"),
        }
    }
}

/// A distribution over categorical values.
///
/// Corresponds to Python `optuna.distributions.CategoricalDistribution`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoricalDistribution {
    pub choices: Vec<CategoricalChoice>,
}

impl CategoricalDistribution {
    /// Create a new `CategoricalDistribution` with validation.
    pub fn new(choices: Vec<CategoricalChoice>) -> Result<Self> {
        if choices.is_empty() {
            return Err(OptunaError::InvalidDistribution(
                "choices must not be empty".into(),
            ));
        }
        Ok(Self { choices })
    }

    /// Check if `value` (an index as f64) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        let index = value as usize;
        index < self.choices.len() && (index as f64 - value).abs() < 1e-8
    }

    /// Convert an external value to internal representation (the index as f64).
    pub fn to_internal_repr(&self, value: &CategoricalChoice) -> Result<f64> {
        self.choices
            .iter()
            .position(|c| c == value)
            .map(|i| i as f64)
            .ok_or_else(|| {
                OptunaError::ValueError(format!("value {value} not found in choices"))
            })
    }

    /// Convert an internal representation (index as f64) back to an external value.
    pub fn to_external_repr(&self, value: f64) -> Result<CategoricalChoice> {
        let index = value as usize;
        self.choices.get(index).cloned().ok_or_else(|| {
            OptunaError::ValueError(format!("index {index} out of range for choices"))
        })
    }

    /// True if this distribution contains exactly one choice.
    pub fn single(&self) -> bool {
        self.choices.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_choices_rejected() {
        assert!(CategoricalDistribution::new(vec![]).is_err());
    }

    #[test]
    fn test_valid_distribution() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ])
        .unwrap();
        assert_eq!(d.choices.len(), 2);
    }

    #[test]
    fn test_contains() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
        ])
        .unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(1.0));
        assert!(d.contains(2.0));
        assert!(!d.contains(3.0));
        assert!(!d.contains(-1.0));
        assert!(!d.contains(0.5));
    }

    #[test]
    fn test_internal_external_roundtrip() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("x".into()),
            CategoricalChoice::Int(42),
            CategoricalChoice::None,
        ])
        .unwrap();

        let internal = d
            .to_internal_repr(&CategoricalChoice::Int(42))
            .unwrap();
        assert_eq!(internal, 1.0);

        let external = d.to_external_repr(1.0).unwrap();
        assert_eq!(external, CategoricalChoice::Int(42));
    }

    #[test]
    fn test_single() {
        assert!(
            CategoricalDistribution::new(vec![CategoricalChoice::Bool(true)])
                .unwrap()
                .single()
        );
        assert!(
            !CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
            ])
            .unwrap()
            .single()
        );
    }

    /// 对应 Python `_categorical_choice_equal` 中 NaN == NaN 的行为。
    /// Python: `math.isnan(v1) and math.isnan(v2)` → True
    #[test]
    fn test_nan_equality() {
        let nan1 = CategoricalChoice::Float(f64::NAN);
        let nan2 = CategoricalChoice::Float(f64::NAN);
        assert_eq!(nan1, nan2, "NaN should equal NaN for categorical choices");
    }

    /// NaN 在分类选项中应可被正确索引
    #[test]
    fn test_nan_to_internal_repr() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Float(f64::NAN),
            CategoricalChoice::Float(1.0),
        ])
        .unwrap();
        let idx = d.to_internal_repr(&CategoricalChoice::Float(f64::NAN)).unwrap();
        assert_eq!(idx, 0.0, "NaN should be found at index 0");
    }

    /// NaN 不应与普通浮点数相等
    #[test]
    fn test_nan_not_equal_to_number() {
        assert_ne!(CategoricalChoice::Float(f64::NAN), CategoricalChoice::Float(1.0));
        assert_ne!(CategoricalChoice::Float(f64::NAN), CategoricalChoice::Float(0.0));
    }

    /// None 选项测试
    #[test]
    fn test_none_choice() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::None,
            CategoricalChoice::Str("x".into()),
        ])
        .unwrap();
        let idx = d.to_internal_repr(&CategoricalChoice::None).unwrap();
        assert_eq!(idx, 0.0);
        let ext = d.to_external_repr(0.0).unwrap();
        assert_eq!(ext, CategoricalChoice::None);
    }

    /// Bool 选项测试
    #[test]
    fn test_bool_choice() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Bool(true),
            CategoricalChoice::Bool(false),
        ])
        .unwrap();
        assert_eq!(d.to_internal_repr(&CategoricalChoice::Bool(false)).unwrap(), 1.0);
    }

    /// to_internal_repr 找不到值时应报错
    #[test]
    fn test_to_internal_repr_not_found() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
        ])
        .unwrap();
        assert!(d.to_internal_repr(&CategoricalChoice::Str("z".into())).is_err());
    }

    /// to_external_repr 索引越界应报错
    #[test]
    fn test_to_external_repr_out_of_range() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Int(1),
        ])
        .unwrap();
        assert!(d.to_external_repr(5.0).is_err());
    }

    // ========================================================================
    // Python 交叉验证测试
    // ========================================================================

    /// Python 交叉验证: CategoricalDistribution(['a','b','c'])
    /// Python: to_internal('a')=0, to_internal('c')=2, to_external(0)='a'
    ///         single=false, contains(0)=true, contains(3)=false
    #[test]
    fn test_python_cross_categorical() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
        ]).unwrap();

        assert_eq!(
            d.to_internal_repr(&CategoricalChoice::Str("a".into())).unwrap(),
            0.0, "Python: to_internal('a')=0"
        );
        assert_eq!(
            d.to_internal_repr(&CategoricalChoice::Str("c".into())).unwrap(),
            2.0, "Python: to_internal('c')=2"
        );
        let ext = d.to_external_repr(0.0).unwrap();
        assert_eq!(ext, CategoricalChoice::Str("a".into()), "Python: to_external(0)='a'");
        assert!(!d.single(), "Python: single=false");
        assert!(d.contains(0.0),  "Python: contains(0)=true");
        assert!(!d.contains(3.0), "Python: contains(3)=false");
    }

    /// Python 交叉验证: CategoricalDistribution([42]) → single=true
    #[test]
    fn test_python_cross_categorical_one() {
        let d = CategoricalDistribution::new(vec![CategoricalChoice::Int(42)]).unwrap();
        assert!(d.single(), "Python: single=true");
    }
}
