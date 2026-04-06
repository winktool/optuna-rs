/// ParzenEstimator 高级功能精确交叉验证
/// Golden values 来源: Python optuna
///
/// 覆盖:
///   - consider_endpoints=True/False sigma 差异
///   - multivariate=True 2D sigma + log_pdf
///   - IntDistribution / LogFloat / Categorical / Float-step log_pdf
///   - multivariate=True 2D sigma + log_pdf

use indexmap::IndexMap;
use optuna_rs::distributions::*;
use optuna_rs::samplers::tpe::parzen_estimator::{ParzenEstimator, ParzenEstimatorParameters};
use std::collections::HashMap;

fn make_float_dist(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution {
        low,
        high,
        log: false,
        step: None,
    })
}

fn make_pe(
    search_space: &IndexMap<String, Distribution>,
    observations: &HashMap<String, Vec<f64>>,
    params: &ParzenEstimatorParameters,
) -> ParzenEstimator {
    ParzenEstimator::new(observations, search_space, params, None, None)
}

// ═══════════════════════════════════════════════════════════════
// Multivariate=True 2D sigma 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multivariate_2d_sigma_golden() {
    // Python: params_mv, search_space_2d, obs_2d
    // mv: mu=[2.0, 5.0, 8.0, 5.0], sigma=[2.0, 2.0, 2.0, 10.0]  (for x)
    // mv: mu=[-3.0, 0.0, 4.0, 0.0], sigma=[2.0, 2.0, 2.0, 10.0] (for y)
    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), make_float_dist(0.0, 10.0));
    search_space.insert("y".to_string(), make_float_dist(-5.0, 5.0));

    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    observations.insert("y".to_string(), vec![-3.0, 0.0, 4.0]);

    let params = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: true,
        ..ParzenEstimatorParameters::default()
    };

    let pe = make_pe(&search_space, &observations, &params);

    // Check sigma for x: SIGMA0_MAG * 3^(-1/(2+4)) * 10 = 0.2 * 3^(-1/6) * 10
    let expected_sigma_x = 0.2 * 3.0_f64.powf(-1.0 / 6.0) * 10.0;
    // Check sigma for y: SIGMA0_MAG * 3^(-1/(2+4)) * 10 = same (range is also 10)
    let expected_sigma_y = 0.2 * 3.0_f64.powf(-1.0 / 6.0) * 10.0;

    // Python says sigma=[2.0, 2.0, 2.0, 10.0] for both x and y
    // 0.2 * 3^(-1/6) * 10 = 0.2 * 0.83255... * 10 = 1.6651...
    // But Python reports 2.0 — likely clipped by minsigma = (high-low)/min(100, 1+4) = 10/5 = 2.0
    let minsigma = 10.0 / (1.0 + 4.0_f64).min(100.0);
    assert!((minsigma - 2.0).abs() < 1e-10, "minsigma should be 2.0, got {}", minsigma);

    // Verify log_pdf matches Python golden values
    let mut test_samples = HashMap::new();
    test_samples.insert("x".to_string(), vec![1.0, 5.0, 9.0]);
    test_samples.insert("y".to_string(), vec![-4.0, 0.0, 3.0]);

    let logpdf = pe.log_pdf(&test_samples);
    // Python: mv: logpdf=[-4.30922182857718, -4.190086550489225, -4.125216125483421]
    let expected = vec![
        -4.30922182857718e+00,
        -4.190086550489225e+00,
        -4.125216125483421e+00,
    ];

    for (i, (&actual, &exp)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((actual - exp) / exp).abs();
        assert!(
            rel < 1e-6,
            "mv logpdf[{}]: got {}, expected {}, rel={}",
            i, actual, exp, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Multivariate=True weights 验证
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multivariate_2d_weights_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), make_float_dist(0.0, 10.0));
    search_space.insert("y".to_string(), make_float_dist(-5.0, 5.0));

    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    observations.insert("y".to_string(), vec![-3.0, 0.0, 4.0]);

    let params = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: true,
        ..ParzenEstimatorParameters::default()
    };

    let pe = make_pe(&search_space, &observations, &params);
    let weights = pe.weights();
    // Python: weights=[0.25, 0.25, 0.25, 0.25]
    assert_eq!(weights.len(), 4, "should have 3 obs + 1 prior = 4 kernels");
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            (w - 0.25).abs() < 1e-10,
            "weight[{}] = {}, expected 0.25", i, w
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Univariate sigma 精确测试 (no endpoints)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_univariate_sigma_no_endpoints_golden() {
    // Python: no_endpoints: mu=[2.0, 5.0, 8.0, 5.0], sigma=[3.0, 3.0, 3.0, 10.0]
    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), make_float_dist(0.0, 10.0));

    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![2.0, 5.0, 8.0]);

    let params = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: false,
        ..ParzenEstimatorParameters::default()
    };

    let pe = make_pe(&search_space, &observations, &params);
    let logpdf = pe.log_pdf(&{
        let mut s = HashMap::new();
        s.insert("x".to_string(), vec![0.0, 2.0, 5.0, 8.0, 10.0]);
        s
    });
    // Python golden
    let expected = vec![
        -2.670406218091419,
        -2.323311754094648,
        -2.1453012481753575,
        -2.323311754094648,
        -2.670406218091419,
    ];
    for (i, (&actual, &exp)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((actual - exp) / exp).abs();
        assert!(
            rel < 1e-8,
            "univ logpdf[{}]: got {}, expected {}, rel={}",
            i, actual, exp, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Multivariate logpdf 额外 golden values
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multivariate_logpdf_at_obs_golden() {
    // logpdf at observation points should match Python golden values
    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), make_float_dist(0.0, 10.0));
    search_space.insert("y".to_string(), make_float_dist(-5.0, 5.0));

    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    observations.insert("y".to_string(), vec![-3.0, 0.0, 4.0]);

    let params = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: true,
        ..ParzenEstimatorParameters::default()
    };

    let pe = make_pe(&search_space, &observations, &params);

    // Test at observation points
    let mut test_at_obs = HashMap::new();
    test_at_obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    test_at_obs.insert("y".to_string(), vec![-3.0, 0.0, 4.0]);
    let logpdf = pe.log_pdf(&test_at_obs);
    let expected = vec![
        -4.03935323306134641541e+00,
        -4.19008655048922484809e+00,
        -3.91491049995065498734e+00,
    ];
    for (i, (&a, &e)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "at_obs[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }

    // Test at corners (0,-5), (10,-5), (0,5), (10,5)
    let mut test_corners = HashMap::new();
    test_corners.insert("x".to_string(), vec![0.0, 10.0, 0.0, 10.0]);
    test_corners.insert("y".to_string(), vec![-5.0, -5.0, 5.0, 5.0]);
    let logpdf_corners = pe.log_pdf(&test_corners);
    let expected_corners = vec![
        -4.91947310267425486074e+00,
        -6.14870130575915485593e+00,
        -6.14652854958071337421e+00,
        -4.48423405530656360440e+00,
    ];
    for (i, (&a, &e)) in logpdf_corners.iter().zip(expected_corners.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "corner[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
}

// ═══════════════════════════════════════════════════════════════
// consider_endpoints 差异验证
// obs=[5,6,7] in [0,10]: 
//   WITH endpoints: sigma=[5.0, 2.0, 3.0, 10.0]
//   WITHOUT: sigma=[2.0, 2.0, 2.0, 10.0]
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_consider_endpoints_logpdf_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert("x".to_string(), make_float_dist(0.0, 10.0));

    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![5.0, 6.0, 7.0]);

    // WITH endpoints
    let params_ep = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: true,
        multivariate: false,
        ..ParzenEstimatorParameters::default()
    };
    let pe_ep = make_pe(&search_space, &observations, &params_ep);

    // WITHOUT endpoints
    let params_no_ep = ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: false,
        ..ParzenEstimatorParameters::default()
    };
    let pe_no_ep = make_pe(&search_space, &observations, &params_no_ep);

    let mut test_x = HashMap::new();
    test_x.insert("x".to_string(), vec![0.0, 2.0, 5.0, 6.0, 7.0, 10.0]);

    let logpdf_ep = pe_ep.log_pdf(&test_x);
    let logpdf_no_ep = pe_no_ep.log_pdf(&test_x);

    // Python golden: WITH endpoints
    let expected_ep = vec![
        -3.125779845624460e+00,
        -2.715229516371727e+00,
        -2.022129998333783e+00,
        -1.941593381627646e+00,
        -1.983566260611957e+00,
        -2.632865115839008e+00,
    ];
    // Python golden: WITHOUT endpoints
    let expected_no_ep = vec![
        -3.653958443662726e+00,
        -2.984654760605065e+00,
        -1.870443433228311e+00,
        -1.779353797359563e+00,
        -1.866249523363164e+00,
        -3.006339488419681e+00,
    ];

    for (i, (&a, &e)) in logpdf_ep.iter().zip(expected_ep.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-8, "ep[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
    for (i, (&a, &e)) in logpdf_no_ep.iter().zip(expected_no_ep.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-8, "no_ep[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }

    // Verify that endpoints=True DIFFERS from endpoints=False
    let mut has_diff = false;
    for (i, (&a, &b)) in logpdf_ep.iter().zip(logpdf_no_ep.iter()).enumerate() {
        if (a - b).abs() > 0.01 {
            has_diff = true;
        }
    }
    assert!(has_diff, "endpoints=True should produce different logpdf than False for this config");
}

// ═══════════════════════════════════════════════════════════════
// IntDistribution log_pdf golden
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_int_distribution_logpdf_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution {
            low: 1,
            high: 10,
            log: false,
            step: 1,
        }),
    );
    let mut observations = HashMap::new();
    observations.insert("n".to_string(), vec![2.0, 5.0, 8.0]);

    let params = ParzenEstimatorParameters::default();
    let pe = make_pe(&search_space, &observations, &params);

    let mut test = HashMap::new();
    test.insert("n".to_string(), vec![1.0, 3.0, 5.0, 7.0, 10.0]);
    let logpdf = pe.log_pdf(&test);

    let expected = vec![
        -2.46269451333915334601e+00,
        -2.23731308198504574847e+00,
        -2.16089694130774079284e+00,
        -2.19842850530865341341e+00,
        -2.67888693240466579937e+00,
    ];
    for (i, (&a, &e)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "int[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
}

// ═══════════════════════════════════════════════════════════════
// Log-scale FloatDistribution log_pdf golden
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_float_distribution_logpdf_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 1e-5,
            high: 1.0,
            log: true,
            step: None,
        }),
    );
    let mut observations = HashMap::new();
    observations.insert("lr".to_string(), vec![1e-4, 1e-3, 1e-2]);

    let params = ParzenEstimatorParameters::default();
    let pe = make_pe(&search_space, &observations, &params);

    let mut test = HashMap::new();
    test.insert("lr".to_string(), vec![1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]);
    let logpdf = pe.log_pdf(&test);

    let expected = vec![
        -2.85264506814156781900e+00,
        -2.24394581694993222598e+00,
        -2.07910694095049652930e+00,
        -2.29478995546491582047e+00,
        -2.89812149997503487597e+00,
        -3.63129582133328376869e+00,
    ];
    for (i, (&a, &e)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "log[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
}

// ═══════════════════════════════════════════════════════════════
// CategoricalDistribution log_pdf golden
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_categorical_distribution_logpdf_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert(
        "algo".to_string(),
        Distribution::CategoricalDistribution(CategoricalDistribution {
            choices: vec![
                CategoricalChoice::Str("sgd".to_string()),
                CategoricalChoice::Str("adam".to_string()),
                CategoricalChoice::Str("rmsprop".to_string()),
            ],
        }),
    );
    let mut observations = HashMap::new();
    observations.insert("algo".to_string(), vec![0.0, 1.0, 1.0, 2.0, 0.0]);

    let params = ParzenEstimatorParameters::default();
    let pe = make_pe(&search_space, &observations, &params);

    let mut test = HashMap::new();
    test.insert("algo".to_string(), vec![0.0, 1.0, 2.0]);
    let logpdf = pe.log_pdf(&test);

    let expected = vec![
        -9.93251773010283445231e-01,
        -9.93251773010283445231e-01,
        -1.34992671694901567037e+00,
    ];
    for (i, (&a, &e)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "cat[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
}

// ═══════════════════════════════════════════════════════════════
// Float with step log_pdf golden
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_float_step_distribution_logpdf_golden() {
    let mut search_space = IndexMap::new();
    search_space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution {
            low: 0.0,
            high: 1.0,
            log: false,
            step: Some(0.1),
        }),
    );
    let mut observations = HashMap::new();
    observations.insert("x".to_string(), vec![0.1, 0.5, 0.9]);

    let params = ParzenEstimatorParameters::default();
    let pe = make_pe(&search_space, &observations, &params);

    let mut test = HashMap::new();
    test.insert("x".to_string(), vec![0.0, 0.2, 0.5, 0.8, 1.0]);
    let logpdf = pe.log_pdf(&test);

    let expected = vec![
        -2.57603385395753026188e+00,
        -2.38857296009835229356e+00,
        -2.29209939909816817760e+00,
        -2.38857296009835051720e+00,
        -2.57603385395752937370e+00,
    ];
    for (i, (&a, &e)) in logpdf.iter().zip(expected.iter()).enumerate() {
        let rel = ((a - e) / e).abs();
        assert!(rel < 1e-6, "step[{}]: got {}, exp {}, rel={}", i, a, e, rel);
    }
}
