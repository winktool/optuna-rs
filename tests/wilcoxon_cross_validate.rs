/// Wilcoxon 符号秩检验 — 交叉验证测试
///
/// 使用 scipy.stats.wilcoxon (v1.17.0) 生成的精确参考值验证 Rust 实现。
/// 参数对齐 Python optuna 的 WilcoxonPruner:
///   zero_method='zsplit', method='auto', correction=False (scipy 默认)
///
/// 参考值来源: tests/wilcoxon_baseline_values.json
///   生成脚本: tests/wilcoxon_cross_validate_baseline.py

use optuna_rs::pruners::wilcoxon_signed_rank_test;
use optuna_rs::study::StudyDirection;

const TOL: f64 = 1e-10;

// ============================================================
// 精确分布测试 (n ≤ 50, method='auto' → exact)
// ============================================================

/// n=5, 全正: R+ = 15, p = 1/32 = 0.03125
#[test]
fn cv_exact_all_positive_minimize() {
    let diff = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.03125).abs() < TOL, "scipy=0.03125, rust={p}");
}

/// n=5, 全正, maximize: p = 1.0
#[test]
fn cv_exact_all_positive_maximize() {
    let diff = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Maximize);
    assert!((p - 1.0).abs() < TOL, "scipy=1.0, rust={p}");
}

/// n=3, 全正: p = 0.125
#[test]
fn cv_exact_n3_all_positive() {
    let diff = vec![1.0, 2.0, 3.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.125).abs() < TOL, "scipy=0.125, rust={p}");
}

/// n=3, 混合: [1, -2, 3], p = 0.375
#[test]
fn cv_exact_n3_mixed() {
    let diff = vec![1.0, -2.0, 3.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.375).abs() < TOL, "scipy=0.375, rust={p}");
}

/// n=4, 混合: [1, -2, 3, 4], p = 0.1875
#[test]
fn cv_exact_n4_mixed() {
    let diff = vec![1.0, -2.0, 3.0, 4.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.1875).abs() < TOL, "scipy=0.1875, rust={p}");
}

/// n=10, 交替正负, 无并列: p = 0.615234375
#[test]
fn cv_exact_n10_alternating() {
    let diff = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.615234375).abs() < TOL, "scipy=0.615234375, rust={p}");
}

/// n=20, 混合: p = 0.0085906982421875
#[test]
fn cv_exact_n20_mixed() {
    let mut diff: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    for &i in &[1, 3, 7, 11, 15] {
        diff[i] = -diff[i];
    }
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!(
        (p - 0.0085906982421875).abs() < TOL,
        "scipy=0.0085906982421875, rust={p}"
    );
}

/// n=5, 全负, minimize → p = 1.0
#[test]
fn cv_exact_all_negative_minimize() {
    let diff = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 1.0).abs() < TOL, "scipy=1.0, rust={p}");
}

/// n=5, 全负, maximize → p = 0.03125
#[test]
fn cv_exact_all_negative_maximize() {
    let diff = vec![-1.0, -2.0, -3.0, -4.0, -5.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Maximize);
    assert!((p - 0.03125).abs() < TOL, "scipy=0.03125, rust={p}");
}

// ============================================================
// 精确分布 + 并列/零值 (n ≤ 50, scipy 仍用精确分布)
// ============================================================

/// n=10 有并列 (0.5出现2次), minimize: p = 0.0439453125
#[test]
fn cv_exact_n10_ties_minimize() {
    let diff = vec![1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!(
        (p - 0.0439453125).abs() < TOL,
        "scipy=0.0439453125, rust={p}"
    );
}

/// n=10 有并列, maximize: p = 0.9599609375
#[test]
fn cv_exact_n10_ties_maximize() {
    let diff = vec![1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Maximize);
    assert!(
        (p - 0.9599609375).abs() < TOL,
        "scipy=0.9599609375, rust={p}"
    );
}

/// n=10 含零值 (zsplit): p = 0.2734375
#[test]
fn cv_exact_with_zeros() {
    let diff = vec![1.0, 0.0, -1.0, 2.0, 0.0, -0.5, 1.5, 0.0, -2.0, 3.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.2734375).abs() < TOL, "scipy=0.2734375, rust={p}");
}

/// n=10 大量并列 + 零值: p = 0.171875
#[test]
fn cv_exact_many_ties() {
    let diff = vec![1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, -2.0, 3.0, 0.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.171875).abs() < TOL, "scipy=0.171875, rust={p}");
}

// ============================================================
// 边界测试
// ============================================================

/// 全零 → p = 1.0
#[test]
fn cv_edge_all_zeros() {
    let diff = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 1.0).abs() < TOL, "scipy=1.0, rust={p}");
}

/// n=2: p = 0.25
#[test]
fn cv_edge_n2() {
    let diff = vec![5.0, 3.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.25).abs() < TOL, "scipy=0.25, rust={p}");
}

/// n=1: p = 0.5
#[test]
fn cv_edge_n1() {
    let diff = vec![5.0];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!((p - 0.5).abs() < TOL, "scipy=0.5, rust={p}");
}

// ============================================================
// n=50 精确边界
// ============================================================

/// n=50, 无并列 → exact: p = 2.5605460131750135e-05
#[test]
fn cv_boundary_n50_exact() {
    let mut diff: Vec<f64> = (1..=50).map(|i| i as f64).collect();
    for &i in &[0, 5, 10, 15, 20, 25, 30, 35, 40, 45] {
        diff[i] = -diff[i];
    }
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!(
        (p - 2.5605460131750135e-05).abs() < 1e-15,
        "scipy=2.5605460131750135e-05, rust={p}"
    );
}

/// n=25, 无并列 → exact: p = 0.0013924241065979004
#[test]
fn cv_boundary_n25_exact() {
    let mut diff: Vec<f64> = (1..=25).map(|i| i as f64).collect();
    for &i in &[0, 5, 10, 15, 20] {
        diff[i] = -diff[i];
    }
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!(
        (p - 0.0013924241065979004).abs() < 1e-15,
        "scipy=0.0013924241065979004, rust={p}"
    );
}

// ============================================================
// n > 50 正态近似
// ============================================================

/// n=51, 无并列 → approx: p ≈ 0.00020481777690820052
#[test]
fn cv_boundary_n51_approx() {
    let mut diff: Vec<f64> = (1..=51).map(|i| i as f64).collect();
    for &i in &[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] {
        diff[i] = -diff[i];
    }
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    // 正态近似允许较大容差 (libm::erfc 精度 ~1e-15, 但近似方法本身的差异)
    assert!(
        (p - 0.00020481777690820052).abs() < 1e-6,
        "scipy=0.00020481777690820052, rust={p}"
    );
}

/// n=60 随机数据, minimize → approx: p ≈ 0.0033736061011727493
#[test]
fn cv_approx_n60_random_minimize() {
    let diff = vec![
        0.9967141530112327, 0.36173569882881534, 1.1476885381006925,
        2.0230298564080256, 0.26584662527666403, 0.2658630430508194,
        2.0792128155073915, 1.267434729152909, 0.03052561406504789,
        1.0425600435859645, 0.036582307187537744, 0.034270246429743134,
        0.7419622715660341, -1.413280244657798, -1.2249178325130328,
        -0.062287529240972694, -0.5128311203344238, 0.8142473325952739,
        -0.4080240755212111, -0.9123037013352917, 1.965648768921554,
        0.27422369951346437, 0.5675282046879239, -0.9247481862134568,
        -0.04438272452518266, 0.6109225897098661, -0.6509935774223028,
        0.875698018345672, -0.10063868991880498, 0.20830625020672322,
        -0.10170661222939692, 2.352278184508938, 0.4865027752620661,
        -0.5577109289558999, 1.3225449121031891, -0.7208436499710222,
        0.7088635950047554, -1.4596701238797756, -0.8281860488984305,
        0.6968612358691235, 1.2384665799954104, 0.6713682811899705,
        0.38435171761175946, 0.1988963044107112, -0.9785219903674274,
        -0.21984420839470864, 0.0393612290402125, 1.5571222262189157,
        0.8436182895684614, -1.263040155362734, 0.824083969394795,
        0.11491771958368346, -0.17692200030595873, 1.1116762888408678,
        1.530999522495951, 1.4312801191161986, -0.33921752322263854,
        0.19078762414878542, 0.831263431403564, 1.4755451271223592,
    ];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
    assert!(
        (p - 0.0033736061011727493).abs() < 1e-6,
        "scipy=0.0033736061011727493, rust={p}"
    );
}

/// n=60 随机数据, maximize → approx: p ≈ 0.9966263938988272
#[test]
fn cv_approx_n60_random_maximize() {
    let diff = vec![
        0.9967141530112327, 0.36173569882881534, 1.1476885381006925,
        2.0230298564080256, 0.26584662527666403, 0.2658630430508194,
        2.0792128155073915, 1.267434729152909, 0.03052561406504789,
        1.0425600435859645, 0.036582307187537744, 0.034270246429743134,
        0.7419622715660341, -1.413280244657798, -1.2249178325130328,
        -0.062287529240972694, -0.5128311203344238, 0.8142473325952739,
        -0.4080240755212111, -0.9123037013352917, 1.965648768921554,
        0.27422369951346437, 0.5675282046879239, -0.9247481862134568,
        -0.04438272452518266, 0.6109225897098661, -0.6509935774223028,
        0.875698018345672, -0.10063868991880498, 0.20830625020672322,
        -0.10170661222939692, 2.352278184508938, 0.4865027752620661,
        -0.5577109289558999, 1.3225449121031891, -0.7208436499710222,
        0.7088635950047554, -1.4596701238797756, -0.8281860488984305,
        0.6968612358691235, 1.2384665799954104, 0.6713682811899705,
        0.38435171761175946, 0.1988963044107112, -0.9785219903674274,
        -0.21984420839470864, 0.0393612290402125, 1.5571222262189157,
        0.8436182895684614, -1.263040155362734, 0.824083969394795,
        0.11491771958368346, -0.17692200030595873, 1.1116762888408678,
        1.530999522495951, 1.4312801191161986, -0.33921752322263854,
        0.19078762414878542, 0.831263431403564, 1.4755451271223592,
    ];
    let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Maximize);
    assert!(
        (p - 0.9966263938988272).abs() < 1e-6,
        "scipy=0.9966263938988272, rust={p}"
    );
}
