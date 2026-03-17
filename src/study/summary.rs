use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// 研究摘要。
///
/// 对应 Python `optuna.study.StudySummary`。
/// 包含研究的基本属性和聚合结果（best_trial、n_trials、datetime_start 等）。
#[derive(Debug, Clone)]
pub struct StudySummary {
    pub study_name: String,
    pub directions: Vec<StudyDirection>,
    pub best_trial: Option<FrozenTrial>,
    pub user_attrs: HashMap<String, serde_json::Value>,
    pub system_attrs: HashMap<String, serde_json::Value>,
    pub n_trials: usize,
    pub datetime_start: Option<DateTime<Utc>>,
    pub study_id: i64,
}

impl StudySummary {
    /// 对齐 Python `StudySummary.direction` property:
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

impl PartialEq for StudySummary {
    fn eq(&self, other: &Self) -> bool {
        self.study_id == other.study_id
            && self.study_name == other.study_name
            && self.directions == other.directions
            && self.n_trials == other.n_trials
    }
}

impl Eq for StudySummary {}

impl PartialOrd for StudySummary {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StudySummary {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.study_id.cmp(&other.study_id)
    }
}

/// 从存储中构建所有研究摘要。
///
/// 对应 Python `optuna.get_all_study_summaries()`。
pub fn build_study_summaries(
    storage: &dyn crate::storage::Storage,
    include_best_trial: bool,
) -> crate::error::Result<Vec<StudySummary>> {
    let frozen_studies = storage.get_all_studies()?;
    let mut summaries = Vec::with_capacity(frozen_studies.len());

    for s in frozen_studies {
        let all_trials = storage.get_all_trials(s.study_id, None)?;
        let n_trials = all_trials.len();

        // 计算 best_trial（仅单目标 + include_best_trial）
        let best_trial = if include_best_trial && s.directions.len() == 1 {
            let completed: Vec<&FrozenTrial> = all_trials
                .iter()
                .filter(|t| t.state == TrialState::Complete)
                .collect();
            if completed.is_empty() {
                None
            } else {
                let is_maximize = s.directions[0] == StudyDirection::Maximize;
                let best = completed
                    .into_iter()
                    .filter(|t| {
                        t.values.as_ref().map_or(false, |v| {
                            !v.is_empty() && !v[0].is_nan()
                        })
                    })
                    .reduce(|a, b| {
                        let va = a.values.as_ref().unwrap()[0];
                        let vb = b.values.as_ref().unwrap()[0];
                        if is_maximize {
                            if vb > va { b } else { a }
                        } else {
                            if vb < va { b } else { a }
                        }
                    });
                best.cloned()
            }
        } else {
            None
        };

        // 计算 datetime_start（最早 trial 的开始时间）
        let datetime_start = all_trials
            .iter()
            .filter_map(|t| t.datetime_start)
            .min();

        summaries.push(StudySummary {
            study_name: s.study_name,
            directions: s.directions,
            best_trial,
            user_attrs: s.user_attrs,
            system_attrs: s.system_attrs,
            n_trials,
            datetime_start,
            study_id: s.study_id,
        });
    }

    Ok(summaries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use crate::storage::Storage;

    #[test]
    fn test_study_summary_basic() {
        let storage = InMemoryStorage::new();
        let _study_id = storage
            .create_new_study(&[StudyDirection::Minimize], Some("test"))
            .unwrap();

        let summaries = build_study_summaries(&storage, true).unwrap();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].study_name, "test");
        assert_eq!(summaries[0].n_trials, 0);
        assert!(summaries[0].best_trial.is_none());
        assert!(summaries[0].datetime_start.is_none());
    }

    #[test]
    fn test_study_summary_with_trials() {
        let storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study(&[StudyDirection::Minimize], Some("s1"))
            .unwrap();

        // 创建 3 个 trial，完成 2 个
        let t0 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t1 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[0.5]))
            .unwrap();

        let _t2 = storage.create_new_trial(study_id, None).unwrap();
        // t2 stays Running

        let summaries = build_study_summaries(&storage, true).unwrap();
        assert_eq!(summaries[0].n_trials, 3);
        let best = summaries[0].best_trial.as_ref().unwrap();
        assert_eq!(best.values.as_ref().unwrap()[0], 0.5);
        assert!(summaries[0].datetime_start.is_some());
    }

    #[test]
    fn test_study_summary_maximize() {
        let storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study(&[StudyDirection::Maximize], Some("max"))
            .unwrap();

        let t0 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t1 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[5.0]))
            .unwrap();

        let summaries = build_study_summaries(&storage, true).unwrap();
        let best = summaries[0].best_trial.as_ref().unwrap();
        assert_eq!(best.values.as_ref().unwrap()[0], 5.0);
    }

    #[test]
    fn test_study_summary_multi_objective_no_best() {
        let storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study(
                &[StudyDirection::Minimize, StudyDirection::Maximize],
                Some("mo"),
            )
            .unwrap();

        let t0 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0, 2.0]))
            .unwrap();

        let summaries = build_study_summaries(&storage, true).unwrap();
        // 多目标不计算 best_trial
        assert!(summaries[0].best_trial.is_none());
        assert_eq!(summaries[0].n_trials, 1);
    }

    #[test]
    fn test_study_summary_exclude_best_trial() {
        let storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study(&[StudyDirection::Minimize], Some("no_best"))
            .unwrap();

        let t0 = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let summaries = build_study_summaries(&storage, false).unwrap();
        assert!(summaries[0].best_trial.is_none());
        assert_eq!(summaries[0].n_trials, 1);
    }

    #[test]
    fn test_study_summary_ordering() {
        let s1 = StudySummary {
            study_name: "a".into(),
            directions: vec![StudyDirection::Minimize],
            best_trial: None,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            n_trials: 0,
            datetime_start: None,
            study_id: 2,
        };
        let s2 = StudySummary {
            study_name: "b".into(),
            directions: vec![StudyDirection::Minimize],
            best_trial: None,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            n_trials: 0,
            datetime_start: None,
            study_id: 1,
        };
        assert!(s2 < s1);
        assert!(s1 > s2);
    }

    #[test]
    fn test_study_summary_direction_single() {
        let s = StudySummary {
            study_name: "test".into(),
            directions: vec![StudyDirection::Minimize],
            best_trial: None,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            n_trials: 0,
            datetime_start: None,
            study_id: 1,
        };
        assert_eq!(s.direction().unwrap(), StudyDirection::Minimize);
    }

    #[test]
    fn test_study_summary_direction_multi_error() {
        let s = StudySummary {
            study_name: "test".into(),
            directions: vec![StudyDirection::Minimize, StudyDirection::Maximize],
            best_trial: None,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            n_trials: 0,
            datetime_start: None,
            study_id: 1,
        };
        assert!(s.direction().is_err());
    }
}
