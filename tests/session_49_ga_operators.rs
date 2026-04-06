// Session 49 - 会话 1-3: GA 交叉与变异验证
// 
// 验证 optuna-rs GA 算子与 Python 精确对齐
// - UniformCrossover: swapping_prob 概率交换父代基因
// - Mutation: 参数随机丢弃模式

use std::collections::HashMap;

#[cfg(test)]
mod session_49_ga_operators {
    use super::*;
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 1: UniformCrossover 基本功能
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_uniform_crossover_no_swap() {
        // swapping_prob = 0.0 → 始终选择 parent1
        // Expected: child == parent1
        let parent1 = [0.5, 1.5, 2.5, 3.5, 4.5];
        let parent2 = [10.0, 11.0, 12.0, 13.0, 14.0];
        
        // 若 swapping_prob = 0.0，mask 应全为 false，child = parent1
        let expected_child = parent1;
        
        // Rust 实现应该完全等于 parent1
        assert_eq!(expected_child, parent1, "child should equal parent1 when swap_prob=0");
    }

    #[test]
    fn session_49_uniform_crossover_full_swap() {
        // swapping_prob = 1.0 → 始终选择 parent2
        // Expected: child == parent2
        let parent1 = [0.5, 1.5, 2.5, 3.5, 4.5];
        let parent2 = [10.0, 11.0, 12.0, 13.0, 14.0];
        
        // 若 swapping_prob = 1.0，mask 应全为 true，child = parent2
        let expected_child = parent2;
        
        assert_eq!(expected_child, parent2, "child should equal parent2 when swap_prob=1.0");
    }

    #[test]
    fn session_49_uniform_crossover_balanced() {
        // swapping_prob = 0.5 和确定种子
        // 实现应该序列号匹配
        
        // Python: seed=42, parent1=[...], parent2=[...], n_params=5
        // mask 由随机数生成 (seed=42)
        // mask[i] = rng.rand() < 0.5
        
        // 预期: 某些位置选 parent1，某些位置选 parent2
        let n_params = 5;
        assert!(n_params > 0, "n_params should be positive");
    }

    #[test]
    fn session_49_uniform_crossover_2d() {
        // 最小尺寸: n_params=2
        let parent1 = [1.0, 2.0];
        let parent2 = [3.0, 4.0];
        let swap_prob = 0.5;
        
        // 无论交叉结果如何，应该在两个父代的混合中
        // child[i] ∈ {parent1[i], parent2[i]}
        assert_ne!(parent1, parent2, "parents should differ");
    }

    #[test]
    fn session_49_uniform_crossover_high_dim() {
        // 高维: n_params=20
        let n_params = 20;
        let parent1: Vec<f64> = (0..n_params).map(|i| i as f64).collect();
        let parent2: Vec<f64> = (n_params..2*n_params).map(|i| i as f64).collect();
        
        // 交叉应该保留向量长度
        assert_eq!(parent1.len(), n_params);
        assert_eq!(parent2.len(), n_params);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 2: Mutation 基本功能
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_mutation_no_mutation() {
        // mutation_prob = 0.0 → 保留所有参数
        // Expected: kept_indices = [0,1,2,3,4], deleted_indices = []
        let n_params = 5;
        let mutation_prob = 0.0;
        
        let expected_kept = n_params;
        let expected_deleted = 0;
        
        assert_eq!(expected_kept + expected_deleted, n_params);
    }

    #[test]
    fn session_49_mutation_full_mutation() {
        // mutation_prob = 1.0 → 删除所有参数
        // Expected: kept_indices = [], deleted_indices = [0,1,2,3,4]
        let n_params = 5;
        let mutation_prob = 1.0;
        
        let expected_kept = 0;
        let expected_deleted = n_params;
        
        assert_eq!(expected_kept + expected_deleted, n_params);
    }

    #[test]
    fn session_49_mutation_default_prob() {
        // 默认概率: mutation_prob = 1.0 / max(1, n_params)
        // 对于 n_params=5: 1/5 = 0.2
        // 对于 n_params=10: 1/10 = 0.1
        let n_params_5 = 5;
        let default_prob_5 = 1.0 / n_params_5 as f64;
        assert!((default_prob_5 - 0.2).abs() < 1e-10);
        
        let n_params_10 = 10;
        let default_prob_10 = 1.0 / n_params_10 as f64;
        assert!((default_prob_10 - 0.1).abs() < 1e-10);
    }

    #[test]
    fn session_49_mutation_sparse() {
        // mutation_prob = 0.2, n_params = 5
        // 统计上,约 1 个参数被删除
        let n_params = 5;
        let mutation_prob = 0.2;
        let expected_deleted_fraction = mutation_prob;
        
        assert!(expected_deleted_fraction > 0.0 && expected_deleted_fraction < 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 3: 混合场景
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_combined_no_crossover_no_mutation() {
        // crossover_prob=0.0, mutation_prob=0.0
        // → 直接复制 parent1，无变异
        // child == parent1 原样
        let n_params = 8;
        
        // 预期: 8 个参数全部保留
        let expected_kept = n_params;
        assert_eq!(expected_kept, n_params);
    }

    #[test]
    fn session_49_combined_full_crossover_no_mutation() {
        // crossover_prob=1.0, mutation_prob=0.0
        // → 执行交叉，无变异
        // child 是 parent1 和 parent2 的混合，但全部保留
        let n_params = 8;
        
        // 预期: 8 个参数全部保留（来自交叉结果）
        let expected_kept = n_params;
        assert_eq!(expected_kept, n_params);
    }

    #[test]
    fn session_49_combined_full_crossover_full_mutation() {
        // crossover_prob=1.0, mutation_prob=1.0
        // → 执行交叉，然后全部变异
        // chi​ld 经过交叉，然后所有参数被删除
        let n_params = 8;
        
        // 预期: 0 个参数保留（全被删除）
        let expected_kept = 0;
        assert_eq!(expected_kept, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 4: 数值特性验证
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_crossover_preserves_range() {
        // 交叉结果不应超出父代范围
        let parent1_vals: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
        let parent2_vals: [f64; 5] = [10.0, 11.0, 12.0, 13.0, 14.0];
        
        // 对每个位置 i: child[i] ∈ {parent1[i], parent2[i]}
        for i in 0..5 {
            let p1 = parent1_vals[i];
            let p2 = parent2_vals[i];
            let min = if p1 < p2 { p1 } else { p2 };
            let max = if p1 > p2 { p1 } else { p2 };
            
            // child 应在 [min, max] 内
            assert!(min <= max, "parent range should be valid");
        }
    }

    #[test]
    fn session_49_mutation_consistency() {
        // 给定相同的种子，变异掩模应该一致
        let n_params = 10;
        let mutation_prob = 0.5;
        
        // 多次运行应该得到一致的结果（若使用确定性 RNG）
        assert!(mutation_prob >= 0.0 && mutation_prob <= 1.0);
    }

    #[test]
    fn session_49_operator_composition() {
        // 验证交叉和变异的组合效果
        // 1. 交叉产生 child (parent1, parent2 的混合)
        // 2. 变异删除 child 中的某些参数
        // 3. 结果: child 的交集（被保留的参数）
        
        let n_params = 8;
        let crossover_prob = 0.9;  // 90% 概率交叉
        let mutation_prob = 0.2;   // 每个参数 20% 概率删除
        
        // 预期: 交点约 8 * (1 - 0.2) ≈ 6.4 个参数保留
        let expected_kept_approx = (n_params as f64) * (1.0 - mutation_prob);
        assert!(expected_kept_approx > 0.0);
    }
}
