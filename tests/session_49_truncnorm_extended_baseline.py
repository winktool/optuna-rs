#!/usr/bin/env python3
"""
Session 49 - 会话 1-2 扩展: TPE Truncnorm 采样 - Python 基线参考值生成
验证 9 个额外测试的精确预期值（ndtr 和高级特性）
"""

import json
import sys
from scipy.stats import truncnorm
from scipy.special import ndtr, log_ndtr, erf
import numpy as np

def generate_ndtr_reference():
    """生成高斯常态分布函数 (normal CDF) 参考值"""
    test_cases = [
        ("ndtr_extreme_low", -6.0),
        ("ndtr_tail_low", -3.0),
        ("ndtr_tail_edge_low", -1.0),
        ("ndtr_center", 0.0),
        ("ndtr_tail_edge_high", 1.0),
        ("ndtr_tail_high", 3.0),
        ("ndtr_extreme_high", 6.0),
    ]
    
    results = {}
    for name, z in test_cases:
        val = ndtr(z)
        log_val = log_ndtr(z)
        results[name] = {
            "input": z,
            "ndtr_output": float(val),
            "log_ndtr_output": float(log_val),
            "is_finite": np.isfinite(val),
        }
    
    return results

def generate_cdf_monotonicity():
    """验证截断正态分布 CDF 的单调性"""
    a, b = -2.0, 2.0
    q_values = np.linspace(0.001, 0.999, 20)
    
    results = {
        "a": a,
        "b": b,
        "pairs": []
    }
    
    prev_x = None
    for i, q in enumerate(q_values):
        x = truncnorm.ppf(q, a, b)
        cdf_q = truncnorm.cdf(x, a, b)
        
        if prev_x is not None:
            monotonic = x >= prev_x - 1e-10
            results["pairs"].append({
                "index": i,
                "q": float(q),
                "x": float(x),
                "cdf_x": float(cdf_q),
                "prev_x": float(prev_x),
                "monotonic": monotonic,
            })
        
        prev_x = x
    
    return results

def generate_tail_behavior():
    """验证尾部极端行为 (tail quantiles at extreme positions)"""
    a, b = -10.0, 10.0
    
    results = {}
    
    # 极端尾部量级
    extreme_qs = [0.0001, 0.0010, 0.9990, 0.9999]
    
    for q in extreme_qs:
        ppf_val = truncnorm.ppf(q, a, b)
        cdf_inv = truncnorm.cdf(ppf_val, a, b)
        
        results[f"tail_q_{q}"] = {
            "q": float(q),
            "ppf_output": float(ppf_val),
            "cdf_inverse_check": float(cdf_inv),
            "error": float(abs(cdf_inv - q)),
        }
    
    return results

def generate_log_gauss_mass_extremes():
    """生成 log_gauss_mass 的极端情况 (补充 Session 1-1 的基础测试)"""
    test_ranges = [
        ("very_tight", -0.1, 0.1),
        ("tight", -0.5, 0.5),
        ("medium", -2.0, 2.0),
        ("wide", -5.0, 5.0),
        ("very_wide", -10.0, 10.0),
        ("asymmetric_left", -20.0, 0.5),
        ("asymmetric_right", -0.5, 20.0),
        ("far_left_tail", -100.0, -99.0),
        ("far_right_tail", 99.0, 100.0),
    ]
    
    results = {}
    for name, a, b in test_ranges:
        try:
            # 计算 log_gauss_mass: log(Phi(b) - Phi(a))
            gauss_mass = ndtr(b) - ndtr(a)
            log_mass = np.log(gauss_mass) if gauss_mass > 0 else -np.inf
            
            results[name] = {
                "a": float(a),
                "b": float(b),
                "gauss_mass": float(gauss_mass),
                "log_gauss_mass": float(log_mass),
                "is_finite": np.isfinite(log_mass),
            }
        except Exception as e:
            results[name] = {
                "a": float(a),
                "b": float(b),
                "error": str(e),
            }
    
    return results

def main():
    print("=" * 80)
    print("SESSION 49 - 会话 1-2 扩展基线值生成")
    print("=" * 80)
    print()
    
    # 生成所有参考值
    print("生成 ndtr 参考值...")
    ndtr_ref = generate_ndtr_reference()
    
    print("生成 CDF 单调性检查...")
    cdf_mono = generate_cdf_monotonicity()
    
    print("生成尾部极端行为...")
    tail_behavior = generate_tail_behavior()
    
    print("生成 log_gauss_mass 极端情况...")
    lgm_extremes = generate_log_gauss_mass_extremes()
    
    # 输出汇总
    print()
    print("📊 NDTR 参考值 (7 cases):")
    print("-" * 80)
    for name, data in ndtr_ref.items():
        print(f"  {name:25} z={data['input']:6.1f}  ndtr={data['ndtr_output']:.10f}  log_ndtr={data['log_ndtr_output']:.10f}")
    
    print()
    print("✓ CDF 单调性验证:")
    print(f"  范围 [{cdf_mono['a']}, {cdf_mono['b']}], {len(cdf_mono['pairs'])} 个检查点")
    monotonic_count = sum(1 for p in cdf_mono['pairs'] if p['monotonic'])
    print(f"  单调性检查: {monotonic_count}/{len(cdf_mono['pairs'])} ✓")
    
    print()
    print("🔥 尾部极端行为 (4 cases):")
    print("-" * 80)
    for name, data in tail_behavior.items():
        print(f"  {name:20} q={data['q']:.4f}  ppf={data['ppf_output']:8.4f}  error={data['error']:.2e}")
    
    print()
    print("📈 Log_Gauss_Mass 极端情况 (9 cases):")
    print("-" * 80)
    for name, data in lgm_extremes.items():
        if "error" not in data:
            print(f"  {name:20} [{data['a']:7.1f}, {data['b']:7.1f}]  log_mass={data['log_gauss_mass']:12.6f}")
    
    # 输出完整 JSON (用于 Rust 测试) - 转换所有 numpy 类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    all_data = {
        "ndtr_reference": ndtr_ref,
        "cdf_monotonicity": cdf_mono,
        "tail_behavior": tail_behavior,
        "log_gauss_mass_extremes": lgm_extremes,
    }
    all_data = convert_to_serializable(all_data)
    
    json_file = "session_49_truncnorm_extended_baseline.json"
    with open(json_file, 'w') as f:
        json.dump(all_data, f, indent=2, allow_nan=True)
    
    print()
    print(f"✅ 完整基线值已保存为: {json_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
