#!/usr/bin/env python3
"""
Golden test: eigendecomposition cross-validation.
Generates deterministic reference eigenvalues/eigenvectors
for Rust cross-validation of the improved eigensolver.
"""
import numpy as np

def test_eigen_3x3():
    """3x3 symmetric matrix eigendecomposition."""
    C = np.array([
        [2.0, 1.0, 0.5],
        [1.0, 3.0, 0.8],
        [0.5, 0.8, 1.5],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    print("=== 3x3 Symmetric Matrix ===")
    print(f"eigenvalues = {eigenvalues.tolist()}")
    # eigenvectors are columns of the returned matrix
    for i in range(3):
        print(f"eigenvector[{i}] = {eigenvectors[:, i].tolist()}")

def test_eigen_ill_conditioned():
    """Ill-conditioned matrix to stress test the solver."""
    C = np.array([
        [1e4, 1.0, 0.0],
        [1.0, 1.0, 0.5],
        [0.0, 0.5, 1e-4],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    print("\n=== Ill-conditioned Matrix (cond ~ 1e8) ===")
    print(f"eigenvalues = {eigenvalues.tolist()}")
    print(f"condition number = {eigenvalues[-1]/eigenvalues[0]:.6e}")

def test_eigen_5x5():
    """5x5 matrix from actual CMA-ES evolution."""
    np.random.seed(123)
    A = np.random.randn(5, 5)
    C = A @ A.T + 0.1 * np.eye(5)  # positive definite
    # Force symmetry
    C = (C + C.T) / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    print("\n=== 5x5 PD Matrix ===")
    print(f"C = {C.tolist()}")
    print(f"eigenvalues = {eigenvalues.tolist()}")
    for i in range(5):
        print(f"eigenvector[{i}] = {eigenvectors[:, i].tolist()}")

def test_eigen_repeated():
    """Matrix with repeated eigenvalues."""
    C = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])
    eigenvalues, _ = np.linalg.eigh(C)
    print("\n=== Repeated Eigenvalues ===")
    print(f"eigenvalues = {eigenvalues.tolist()}")

def test_eigen_cmaes_after_update():
    """Eigenvalues of C after a CMA-ES update step.
    This tests the actual use case."""
    from cmaes import CMA
    opt = CMA(mean=np.zeros(4), sigma=1.0, seed=42)
    
    # Run 3 generations
    for _ in range(3):
        solutions = []
        for _ in range(opt.population_size):
            x = opt.ask()
            val = np.sum(x**2)  # sphere function
            solutions.append((x, val))
        opt.tell(solutions)
    
    C = opt._C
    eigenvalues, _ = np.linalg.eigh(C)
    print("\n=== CMA-ES C after 3 gens (4D sphere) ===")
    print(f"eigenvalues = {eigenvalues.tolist()}")
    print(f"C = {C.tolist()}")
    
    # Verify C = B D B^T
    B, D = opt._eigen_decomposition()
    C_reconstructed = B @ np.diag(D**2) @ B.T
    max_err = np.max(np.abs(C - C_reconstructed))
    print(f"Reconstruction error = {max_err:.2e}")

if __name__ == "__main__":
    test_eigen_3x3()
    test_eigen_ill_conditioned()
    test_eigen_5x5()
    test_eigen_repeated()
    test_eigen_cmaes_after_update()
