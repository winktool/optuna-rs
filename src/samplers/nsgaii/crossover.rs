/// Trait for crossover operators used in evolutionary algorithms.
pub trait Crossover: Send + Sync {
    /// Number of parent solutions required.
    fn n_parents(&self) -> usize;

    /// Perform crossover on parent vectors (in \[0,1\] space) to produce a child.
    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64>;
}

/// Re-export `Rng` for `&mut dyn Rng` usage.
pub use rand::Rng;

/// 从 dyn Rng 生成 [0, 1) 范围的 f64。
/// rand 0.10 的 `random::<f64>()` 不是 object-safe，需要手动转换。
#[inline]
pub(crate) fn rng_f64(rng: &mut dyn Rng) -> f64 {
    // 标准方法：取 53 位尾数
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// 从 dyn Rng 生成随机 bool。
#[inline]
pub(crate) fn rng_bool(rng: &mut dyn Rng) -> bool {
    rng.next_u32() & 1 == 1
}

/// Uniform crossover: randomly selects each gene from one of the two parents.
pub struct UniformCrossover {
    /// Probability of swapping each gene.
    pub swapping_prob: f64,
}

impl UniformCrossover {
    pub fn new(swapping_prob: Option<f64>) -> Self {
        Self {
            swapping_prob: swapping_prob.unwrap_or(0.5),
        }
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for UniformCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                if rng_f64(rng) < self.swapping_prob {
                    p1
                } else {
                    p0
                }
            })
            .collect()
    }
}

/// BLX-alpha crossover: samples from an expanded range between parents.
pub struct BLXAlphaCrossover {
    pub alpha: f64,
}

impl BLXAlphaCrossover {
    pub fn new(alpha: Option<f64>) -> Self {
        Self {
            alpha: alpha.unwrap_or(0.5),
        }
    }
}

impl Default for BLXAlphaCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for BLXAlphaCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                let lo = p0.min(p1);
                let hi = p0.max(p1);
                let d = hi - lo;
                let lower = lo - self.alpha * d;
                let upper = hi + self.alpha * d;
                let v: f64 = rng_f64(rng) * (upper - lower) + lower;
                v.clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// 模拟二进制交叉 (SBX - Simulated Binary Crossover)。
///
/// 对应 Python `optuna.samplers.nsgaii.SBXCrossover`。
/// 使用分布指数 eta 控制子代与父代的距离。
pub struct SBXCrossover {
    /// 分布指数 (eta)。值越大，子代越接近父代。
    pub eta: f64,
}

impl SBXCrossover {
    pub fn new(eta: Option<f64>) -> Self {
        Self {
            eta: eta.unwrap_or(2.0),
        }
    }
}

impl Default for SBXCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for SBXCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                if (p0 - p1).abs() < 1e-14 {
                    return p0;
                }

                let u: f64 = rng_f64(rng);
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (self.eta + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta + 1.0))
                };

                let c = if rng_bool(rng) {
                    0.5 * ((1.0 + beta) * p0 + (1.0 - beta) * p1)
                } else {
                    0.5 * ((1.0 - beta) * p0 + (1.0 + beta) * p1)
                };

                c.clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// 单纯形交叉 (SPX - Simplex Crossover)。
///
/// 对应 Python `optuna.samplers.nsgaii.SPXCrossover`。
/// 需要 3 个父代，基于质心扩展生成子代。
pub struct SPXCrossover {
    /// 扩展率。`None` 表示使用默认值 sqrt(n_params + 2)。
    pub epsilon: Option<f64>,
}

impl SPXCrossover {
    /// 创建新的 SPX 交叉算子。
    ///
    /// # 参数
    /// * `epsilon` - 扩展率。`None` 表示自动计算 sqrt(n_params + 2)。
    pub fn new(epsilon: Option<f64>) -> Self {
        Self { epsilon }
    }
}

impl Crossover for SPXCrossover {
    fn n_parents(&self) -> usize {
        3 // SPX 始终需要 3 个父代
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        let n_parents = 3;
        let n_params = parents[0].len();

        // 确定扩展率 epsilon
        let epsilon = self.epsilon.unwrap_or_else(|| ((n_params + 2) as f64).sqrt());

        // 步骤1：计算质心 G = (1/3) * Σ P_i
        let mut g = vec![0.0; n_params];
        for parent in parents.iter().take(n_parents) {
            for (j, &val) in parent.iter().enumerate() {
                g[j] += val / n_parents as f64;
            }
        }

        // 步骤2：生成随机权重 r_s = U(0,1)^(1/(k-1))，其中 k = n_parents - 1
        let mut r_s = Vec::with_capacity(n_parents - 1);
        for _ in 0..(n_parents - 1) {
            let u: f64 = rng_f64(rng);
            r_s.push(u.powf(1.0 / (n_parents as f64 - 2.0).max(1.0)));
        }

        // 步骤3：计算扩展顶点 X_k = G + epsilon * (P_k - G)
        let mut x: Vec<Vec<f64>> = Vec::with_capacity(n_parents);
        for parent in parents.iter().take(n_parents) {
            let xi: Vec<f64> = (0..n_params)
                .map(|j| g[j] + epsilon * (parent[j] - g[j]))
                .collect();
            x.push(xi);
        }

        // 步骤4：累计偏移 c_k = r_s[k-1] * (X_{k-1} - X_k + c_{k-1})
        let mut c = vec![0.0; n_params];
        for k in 1..n_parents {
            for j in 0..n_params {
                c[j] = r_s[k - 1] * (x[k - 1][j] - x[k][j] + c[j]);
            }
        }

        // 步骤5：子代 = X_{last} + c
        let child: Vec<f64> = (0..n_params)
            .map(|j| (x[n_parents - 1][j] + c[j]).clamp(0.0, 1.0))
            .collect();

        child
    }
}

/// UNDX 交叉 (Unimodal Normal Distribution Crossover)。
///
/// 对应 Python `optuna.samplers.nsgaii.UNDXCrossover`。
/// 需要 3 个父代，沿主搜索线和正交方向采样。
pub struct UNDXCrossover {
    /// 主搜索方向上的标准差
    pub sigma_xi: f64,
    /// 正交方向上的标准差。`None` 表示使用 0.35 / sqrt(n_params)。
    pub sigma_eta: Option<f64>,
}

impl UNDXCrossover {
    /// 创建新的 UNDX 交叉算子。
    ///
    /// # 参数
    /// * `sigma_xi` - 主搜索方向标准差（默认 0.5）。
    /// * `sigma_eta` - 正交方向标准差。`None` 表示 0.35 / sqrt(n_params)。
    pub fn new(sigma_xi: f64, sigma_eta: Option<f64>) -> Self {
        Self {
            sigma_xi,
            sigma_eta,
        }
    }
}

impl Default for UNDXCrossover {
    fn default() -> Self {
        Self::new(0.5, None)
    }
}

impl Crossover for UNDXCrossover {
    fn n_parents(&self) -> usize {
        3 // UNDX 始终需要 3 个父代
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        let n = parents[0].len(); // 参数维度

        // 步骤1：主搜索线 (PSL) 设置
        // 归一化向量 e_12 = (P2 - P1) / ||P2 - P1||
        let d: Vec<f64> = (0..n).map(|j| parents[0][j] - parents[1][j]).collect();
        let d_norm = d.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        let e12: Vec<f64> = d.iter().map(|x| x / d_norm).collect();

        // 中点 x_p = (P1 + P2) / 2
        let x_p: Vec<f64> = (0..n).map(|j| (parents[0][j] + parents[1][j]) / 2.0).collect();

        // 步骤2：计算 P3 到 PSL 的距离
        let v13: Vec<f64> = (0..n).map(|j| parents[2][j] - parents[0][j]).collect();
        let v13_dot_e12: f64 = v13.iter().zip(e12.iter()).map(|(a, b)| a * b).sum();
        let v_orth: Vec<f64> = (0..n).map(|j| v13[j] - v13_dot_e12 * e12[j]).collect();
        let dist = v_orth.iter().map(|x| x * x).sum::<f64>().sqrt();

        // 步骤3：确定 sigma_eta
        let sigma_eta = self.sigma_eta.unwrap_or_else(|| 0.35 / (n as f64).sqrt());

        // 步骤4：沿主搜索方向采样
        let xi: f64 = normal_sample(rng) * self.sigma_xi;

        // 步骤5：生成子代
        let mut child: Vec<f64> = (0..n).map(|j| x_p[j] + xi * d[j]).collect();

        // 步骤6：沿正交方向添加扰动
        if n > 1 {
            // 生成正交基向量（使用 Gram-Schmidt 正交化）
            let basis = orthonormal_basis(&e12, n);
            for basis_vec in &basis {
                let eta: f64 = normal_sample(rng) * sigma_eta;
                for j in 0..n {
                    child[j] += dist * eta * basis_vec[j];
                }
            }
        }

        // 裁剪到 [0, 1]
        child.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }
}

/// 生成标准正态分布随机数（Box-Muller 变换）
fn normal_sample(rng: &mut dyn Rng) -> f64 {
    let u1: f64 = rng_f64(rng).max(1e-300); // 避免 log(0)
    let u2: f64 = rng_f64(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// 计算与给定向量正交的基向量（Gram-Schmidt 正交化）。
fn orthonormal_basis(v: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();

    // 生成 n-1 个与 v 正交的基向量
    // 使用所有标准基向量作为候选，确保能生成足够的正交向量
    for i in 0..n {
        // 起始向量：标准基向量 e_i
        let mut w = vec![0.0; n];
        w[i] = 1.0;

        // 从 v 正交化
        let dot: f64 = w.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        for j in 0..n {
            w[j] -= dot * v[j];
        }

        // 从之前的基向量正交化
        for prev in &basis {
            let dot: f64 = w.iter().zip(prev.iter()).map(|(a, b): (&f64, &f64)| a * b).sum();
            for j in 0..n {
                w[j] -= dot * prev[j];
            }
        }

        // 归一化
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for j in 0..n {
                w[j] /= norm;
            }
            basis.push(w);
        }

        // 只需要 n-1 个基向量
        if basis.len() >= n - 1 {
            break;
        }
    }

    basis
}

/// VSBX 交叉 (Vector Simulated Binary Crossover)。
///
/// 对应 Python `optuna.samplers.nsgaii.VSBXCrossover`。
/// SBX 的向量化变体，生成两个候选子代后随机选择一个。
pub struct VSBXCrossover {
    /// 分布指数。`None` 表示多目标用 20.0，单目标用 2.0。
    pub eta: Option<f64>,
    /// 均匀交叉概率
    pub uniform_crossover_prob: f64,
    /// 使用子代基因的概率
    pub use_child_gene_prob: f64,
}

impl VSBXCrossover {
    /// 创建新的 VSBX 交叉算子。
    ///
    /// # 参数
    /// * `eta` - 分布指数。`None` 表示自动选择。
    /// * `uniform_crossover_prob` - 均匀交叉概率 [0.0, 1.0]。
    /// * `use_child_gene_prob` - 使用子代基因概率 (0.0, 1.0]。
    pub fn new(
        eta: Option<f64>,
        uniform_crossover_prob: f64,
        use_child_gene_prob: f64,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&uniform_crossover_prob),
            "uniform_crossover_prob 必须在 [0.0, 1.0] 范围内"
        );
        assert!(
            use_child_gene_prob > 0.0 && use_child_gene_prob <= 1.0,
            "use_child_gene_prob 必须在 (0.0, 1.0] 范围内"
        );
        Self {
            eta,
            uniform_crossover_prob,
            use_child_gene_prob,
        }
    }
}

impl Default for VSBXCrossover {
    fn default() -> Self {
        Self::new(None, 0.5, 0.5)
    }
}

impl Crossover for VSBXCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn Rng) -> Vec<f64> {
        let n = parents[0].len();

        // 确定 eta（默认：多目标 20.0，保守默认）
        let eta = self.eta.unwrap_or(20.0);
        let eps = 1e-10; // 数值稳定性

        // 为每个维度生成 beta 值
        let mut child1 = vec![0.0; n];
        let mut child2 = vec![0.0; n];

        for i in 0..n {
            let p0 = parents[0][i];
            let p1 = parents[1][i];

            let u: f64 = rng_f64(rng);

            // 计算 beta_1 和 beta_2
            let beta1 = (1.0 / (2.0 * u).max(eps)).powf(1.0 / (eta + 1.0));
            let beta2 =
                (1.0 / (2.0 * (1.0 - u)).max(eps)).powf(1.0 / (eta + 1.0));

            // 生成两个候选子代
            let u1: f64 = rng_f64(rng);
            if u1 <= 0.5 {
                child1[i] = 0.5 * ((1.0 + beta1) * p0 + (1.0 - beta2) * p1);
            } else {
                child1[i] = 0.5 * ((1.0 - beta1) * p0 + (1.0 + beta2) * p1);
            }

            let u2: f64 = rng_f64(rng);
            if u2 <= 0.5 {
                child2[i] = 0.5 * ((3.0 - beta1) * p0 - (1.0 - beta2) * p1);
            } else {
                child2[i] = 0.5 * (-(1.0 - beta1) * p0 + (3.0 - beta2) * p1);
            }

            // 基因选择逻辑
            let r1: f64 = rng_f64(rng);
            if r1 >= self.use_child_gene_prob {
                // 使用父代基因
                child1[i] = p0;
                child2[i] = p1;
            } else {
                // 交叉概率选择
                let r2: f64 = rng_f64(rng);
                if r2 < self.uniform_crossover_prob {
                    std::mem::swap(&mut child1[i], &mut child2[i]);
                }
            }
        }

        // 随机选择一个子代返回
        let r3: f64 = rng_f64(rng);
        let chosen = if r3 < 0.5 { &child1 } else { &child2 };
        chosen.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_uniform_crossover() {
        let cx = UniformCrossover::new(Some(0.5));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.0, 0.0, 0.0, 0.0];
        let p1 = vec![1.0, 1.0, 1.0, 1.0];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 4);
        // Each gene should be 0.0 or 1.0
        for &v in &child {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_blx_alpha_crossover() {
        let cx = BLXAlphaCrossover::new(Some(0.5));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.3];
        let p1 = vec![0.7, 0.7];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_sbx_crossover() {
        let cx = SBXCrossover::new(Some(2.0));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.3];
        let p1 = vec![0.7, 0.7];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_sbx_identical_parents() {
        let cx = SBXCrossover::new(Some(2.0));
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.5, 0.5];
        let p1 = vec![0.5, 0.5];
        let child = cx.crossover(&[p0, p1], &mut rng);
        for &v in &child {
            assert!((v - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spx_crossover() {
        // SPX 需要 3 个父代
        let cx = SPXCrossover::new(None);
        assert_eq!(cx.n_parents(), 3);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.2, 0.3, 0.4];
        let p1 = vec![0.5, 0.6, 0.7];
        let p2 = vec![0.3, 0.4, 0.5];
        let child = cx.crossover(&[p0, p1, p2], &mut rng);
        assert_eq!(child.len(), 3);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v), "SPX 子代值 {} 超出 [0,1] 范围", v);
        }
    }

    #[test]
    fn test_spx_with_custom_epsilon() {
        // 使用自定义 epsilon
        let cx = SPXCrossover::new(Some(1.5));
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let p0 = vec![0.3, 0.4];
        let p1 = vec![0.6, 0.5];
        let p2 = vec![0.4, 0.7];
        let child = cx.crossover(&[p0, p1, p2], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_undx_crossover() {
        // UNDX 需要 3 个父代
        let cx = UNDXCrossover::default();
        assert_eq!(cx.n_parents(), 3);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.2, 0.3, 0.4, 0.5];
        let p1 = vec![0.5, 0.6, 0.7, 0.8];
        let p2 = vec![0.3, 0.4, 0.5, 0.6];
        let child = cx.crossover(&[p0, p1, p2], &mut rng);
        assert_eq!(child.len(), 4);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v), "UNDX 子代值 {} 超出 [0,1] 范围", v);
        }
    }

    #[test]
    fn test_undx_1d() {
        // 1 维情况下不需要正交方向
        let cx = UNDXCrossover::new(0.5, None);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3];
        let p1 = vec![0.7];
        let p2 = vec![0.5];
        let child = cx.crossover(&[p0, p1, p2], &mut rng);
        assert_eq!(child.len(), 1);
        assert!((0.0..=1.0).contains(&child[0]));
    }

    #[test]
    fn test_vsbx_crossover() {
        // VSBX 需要 2 个父代
        let cx = VSBXCrossover::default();
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.2, 0.3, 0.4];
        let p1 = vec![0.7, 0.8, 0.9];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 3);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v), "VSBX 子代值 {} 超出 [0,1] 范围", v);
        }
    }

    #[test]
    fn test_vsbx_with_custom_eta() {
        // 使用自定义 eta
        let cx = VSBXCrossover::new(Some(20.0), 0.5, 0.5);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.4];
        let p1 = vec![0.6, 0.7];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_orthonormal_basis() {
        // 验证正交基的正交性和归一化
        let v = vec![1.0, 0.0, 0.0];
        let basis = orthonormal_basis(&v, 3);
        assert_eq!(basis.len(), 2); // 3维空间中应有 2 个正交向量

        // 验证正交性
        for b in &basis {
            let dot: f64 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
            assert!(dot.abs() < 1e-10, "基向量应与 v 正交");

            // 验证归一化
            let norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "基向量应已归一化");
        }

        // 验证基向量之间互相正交
        if basis.len() >= 2 {
            let dot: f64 = basis[0].iter().zip(basis[1].iter()).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-10, "基向量之间应互相正交");
        }
    }
}
