#!/usr/bin/env python3
"""
Golden test script for CMA-ES Learning Rate Adaptation.
Generates deterministic reference values for Rust cross-validation.
"""
import numpy as np
from cmaes import CMA

def test_lr_adapt_2d():
    """2D LR adaptation test with deterministic solutions."""
    mean = np.array([0.5, 0.5])
    sigma = 0.3
    opt = CMA(mean=mean, sigma=sigma, seed=42, lr_adapt=True)
    
    print(f"=== 2D LR Adaptation Test ===")
    print(f"n_dim = {opt.dim}")
    print(f"popsize = {opt.population_size}")
    print(f"Initial sigma = {opt._sigma}")
    print(f"Initial eta_mean = {opt._eta_mean}")
    print(f"Initial eta_Sigma = {opt._eta_Sigma}")
    print(f"Initial Emean = {opt._Emean.flatten().tolist()}")
    print(f"Initial ESigma = {opt._ESigma.flatten().tolist()[:4]}")
    print(f"Initial Vmean = {opt._Vmean}")
    print(f"Initial VSigma = {opt._VSigma}")
    
    # Run 3 generations with deterministic solutions
    # Generation 1: use fixed solutions
    solutions_gen1 = [
        (np.array([0.6, 0.4]), 1.0),
        (np.array([0.3, 0.7]), 2.0),
        (np.array([0.7, 0.3]), 0.5),
        (np.array([0.4, 0.6]), 1.5),
        (np.array([0.5, 0.5]), 3.0),
        (np.array([0.8, 0.2]), 0.8),
    ]
    
    opt.tell(solutions_gen1)
    
    print(f"\n--- After Generation 1 ---")
    print(f"mean = {opt.mean.tolist()}")
    print(f"sigma = {opt._sigma:.15e}")
    print(f"eta_mean = {opt._eta_mean:.15e}")
    print(f"eta_Sigma = {opt._eta_Sigma:.15e}")
    print(f"Emean = {opt._Emean.flatten().tolist()}")
    print(f"Vmean = {opt._Vmean:.15e}")
    print(f"VSigma = {opt._VSigma:.15e}")
    print(f"C_diag = [{opt._C[0,0]:.15e}, {opt._C[1,1]:.15e}]")
    print(f"C_offdiag = {opt._C[0,1]:.15e}")
    print(f"p_sigma = {opt._p_sigma.tolist()}")
    print(f"p_c = {opt._pc.tolist()}")
    
    # Generation 2
    solutions_gen2 = [
        (np.array([0.55, 0.45]), 0.3),
        (np.array([0.65, 0.35]), 0.4),
        (np.array([0.45, 0.55]), 0.6),
        (np.array([0.75, 0.25]), 0.2),
        (np.array([0.35, 0.65]), 1.0),
        (np.array([0.50, 0.50]), 0.7),
    ]
    
    opt.tell(solutions_gen2)
    
    print(f"\n--- After Generation 2 ---")
    print(f"mean = {opt.mean.tolist()}")
    print(f"sigma = {opt._sigma:.15e}")
    print(f"eta_mean = {opt._eta_mean:.15e}")
    print(f"eta_Sigma = {opt._eta_Sigma:.15e}")
    print(f"Vmean = {opt._Vmean:.15e}")
    print(f"VSigma = {opt._VSigma:.15e}")
    print(f"C_diag = [{opt._C[0,0]:.15e}, {opt._C[1,1]:.15e}]")
    
    # Generation 3
    solutions_gen3 = [
        (np.array([0.60, 0.40]), 0.1),
        (np.array([0.70, 0.30]), 0.15),
        (np.array([0.50, 0.50]), 0.5),
        (np.array([0.80, 0.20]), 0.12),
        (np.array([0.40, 0.60]), 0.8),
        (np.array([0.55, 0.45]), 0.3),
    ]
    
    opt.tell(solutions_gen3)
    
    print(f"\n--- After Generation 3 ---")
    print(f"mean = {opt.mean.tolist()}")
    print(f"sigma = {opt._sigma:.15e}")
    print(f"eta_mean = {opt._eta_mean:.15e}")
    print(f"eta_Sigma = {opt._eta_Sigma:.15e}")


def test_should_stop_criteria():
    """Test should_stop() behavior."""
    mean = np.array([0.5, 0.5])
    sigma = 0.3
    opt = CMA(mean=mean, sigma=sigma, seed=42)
    
    print(f"\n=== should_stop Test ===")
    print(f"funhist_term = {opt._funhist_term}")
    print(f"tolx = {opt._tolx:.15e}")
    print(f"tolxup = {opt._tolxup:.15e}")
    print(f"tolfun = {opt._tolfun:.15e}")
    print(f"tolconditioncov = {opt._tolconditioncov:.15e}")
    print(f"Initial should_stop = {opt.should_stop()}")
    
    # After one generation, should not stop
    solutions = [
        (np.array([0.6, 0.4]), 1.0),
        (np.array([0.3, 0.7]), 2.0),
        (np.array([0.7, 0.3]), 0.5),
        (np.array([0.4, 0.6]), 1.5),
        (np.array([0.5, 0.5]), 3.0),
        (np.array([0.8, 0.2]), 0.8),
    ]
    opt.tell(solutions)
    print(f"After gen 1: should_stop = {opt.should_stop()}")


if __name__ == "__main__":
    test_lr_adapt_2d()
    test_should_stop_criteria()
