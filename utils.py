"""
Utility functions for Quantum Entanglement Loss (QEL).

Implements:
- Von Neumann entropy computation
- Top-k eigenvalue approximation via power iteration
- Density matrix construction
"""

import torch
import torch.nn.functional as F


def l2_normalize_rows(H: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize each row of H to unit norm.

    Required for valid density matrix interpretation where Tr(rho) = 1.

    Args:
        H: Node embeddings of shape (n, d).
        eps: Small constant for numerical stability.

    Returns:
        Row-normalized embeddings of shape (n, d).
    """
    return F.normalize(H, p=2, dim=-1, eps=eps)


def compute_density_matrix(H: torch.Tensor) -> torch.Tensor:
    """Construct the density matrix rho = (1/n) H H^T.

    H must be row-normalized so that Tr(rho) = 1.

    Args:
        H: Row-normalized node embeddings of shape (n, d).

    Returns:
        Density matrix of shape (n, n).
    """
    n = H.size(0)
    return (1.0 / n) * (H @ H.t())


def power_iteration_topk(
    rho: torch.Tensor,
    k: int = 16,
    num_iters: int = 20,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Compute top-k eigenvalues of a symmetric PSD matrix via power iteration
    with deflation.

    Args:
        rho: Symmetric PSD matrix of shape (n, n).
        k: Number of top eigenvalues to compute.
        num_iters: Maximum power iterations per eigenvalue.
        tol: Convergence tolerance.

    Returns:
        Top-k eigenvalues as a 1-D tensor of shape (k,), sorted descending.
    """
    n = rho.size(0)
    k = min(k, n)
    device = rho.device
    dtype = rho.dtype

    eigenvalues = []
    residual = rho.clone()

    for _ in range(k):
        # Random initialization
        v = torch.randn(n, 1, device=device, dtype=dtype)
        v = v / v.norm()

        for _ in range(num_iters):
            v_new = residual @ v
            eigenvalue = v.t() @ v_new  # Rayleigh quotient
            v_new_norm = v_new.norm()
            if v_new_norm < 1e-12:
                break
            v_new = v_new / v_new_norm

            # Check convergence
            if (v_new - v).norm() < tol:
                v = v_new
                break
            v = v_new

        lam = (v.t() @ residual @ v).squeeze()
        lam = torch.clamp(lam, min=0.0)  # Ensure non-negative
        eigenvalues.append(lam)

        # Deflation: remove contribution of found eigenvector
        residual = residual - lam * (v @ v.t())

    return torch.stack(eigenvalues)


def von_neumann_entropy(
    H: torch.Tensor,
    k: int = 16,
    num_iters: int = 20,
    eps: float = 1e-8,
    use_full_eigen: bool = False,
) -> torch.Tensor:
    """Compute the von Neumann entropy S(rho) = -sum_i lambda_i log(lambda_i).

    Uses top-k eigenvalue approximation for efficiency on large graphs.

    Args:
        H: Row-normalized node embeddings of shape (n, d).
        k: Number of top eigenvalues for approximation.
        num_iters: Power iteration steps.
        eps: Small constant to avoid log(0).
        use_full_eigen: If True, use full eigendecomposition (exact but O(n^3)).

    Returns:
        Scalar entropy value.
    """
    rho = compute_density_matrix(H)
    n = rho.size(0)

    if use_full_eigen or n <= 2 * k:
        # Full eigendecomposition for small graphs
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
    else:
        eigenvalues = power_iteration_topk(rho, k=k, num_iters=num_iters)

    # S(rho) = -sum lambda_i * log(lambda_i)
    eigenvalues = eigenvalues + eps  # Numerical stability
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))

    return entropy


def quantum_entanglement_loss(
    H: torch.Tensor,
    alpha: float = 0.1,
    k: int = 16,
    num_iters: int = 20,
) -> torch.Tensor:
    """Compute the Quantum Entanglement Loss.

    QEL = alpha * S(rho), where S is von Neumann entropy of the density matrix
    constructed from row-normalized embeddings.

    Args:
        H: Node embeddings of shape (n, d). Will be L2-normalized internally.
        alpha: Weight for the QEL term.
        k: Number of top eigenvalues for approximation.
        num_iters: Power iteration steps.

    Returns:
        Scalar QEL loss value.
    """
    H_norm = l2_normalize_rows(H)
    entropy = von_neumann_entropy(H_norm, k=k, num_iters=num_iters)
    return alpha * entropy
