"""
Utility functions for Quantum Entanglement Loss (QEL).

Implements:
- Von Neumann entropy computation
- Top-k eigenvalue approximation via power iteration
- Density matrix construction
- Node sampling for large graphs
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


def topk_eigenvalues(
    rho: torch.Tensor,
    k: int = 16,
    num_iters: int = 20,
) -> torch.Tensor:
    """Compute the top-k eigenvalues of a symmetric PSD matrix.

    Implements the paper's stated approach (Section 2.5): "we compute via
    eigendecomposition of the top-k subspace to maintain gradient flow
    through small eigenvalues". We compute the full eigenspectrum via
    torch.linalg.eigvalsh (numerically stable, fully differentiable) and
    extract the top-k subspace.

    Complexity note: Paper Table 1 lists O(t*k*n^2) for iterative top-k
    methods (power iteration, Lanczos). This implementation is O(n^3) via
    full eigvalsh. We chose eigvalsh because torch.lobpcg's backward pass
    performs an internal Cholesky factorization that fails on near-
    degenerate spectra (common early in training when embeddings are
    near-random). For the graph sizes used in the paper (n <= ~3000 for
    standard benchmarks, n <= 500 per graph for batched LRGB), eigvalsh
    has a smaller constant factor and is faster in practice than iterative
    methods with small k.

    Args:
        rho: Symmetric PSD matrix of shape (n, n).
        k: Number of top eigenvalues to compute.
        num_iters: Unused; retained for API compatibility.

    Returns:
        Top-k eigenvalues as a 1-D tensor of shape (k,), sorted descending.
    """
    n = rho.size(0)
    k = min(k, n)

    # torch.linalg.eigvalsh is not implemented on MPS; temporarily move to CPU
    # for the eigendecomposition, then move results back. Gradient flow is
    # preserved through the .to() calls.
    orig_device = rho.device
    if orig_device.type == "mps":
        rho_cpu = rho.to("cpu")
        eigenvalues = torch.linalg.eigvalsh(rho_cpu)
        eigenvalues = eigenvalues.to(orig_device)
    else:
        eigenvalues = torch.linalg.eigvalsh(rho)

    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    return eigenvalues.flip(0)[:k]


def adaptive_k(n: int, k_override: int = None) -> int:
    """Select k based on graph size per paper Section 2.5 guidance.

    k=8-16 for n<500, k=16-32 for 500<=n<5000, k=32-64 for n>=5000.

    Args:
        n: Number of nodes.
        k_override: If provided, use this value instead.

    Returns:
        Selected k value.
    """
    if k_override is not None:
        return min(k_override, n)
    if n < 500:
        return min(16, n)
    elif n < 5000:
        return min(32, n)
    else:
        return min(64, n)


def von_neumann_entropy(
    H: torch.Tensor,
    k: int = None,
    num_iters: int = 20,
    eps: float = 1e-8,
    use_full_eigen: bool = False,
) -> torch.Tensor:
    """Compute the von Neumann entropy S(rho) = -sum_i lambda_i log(lambda_i).

    Uses top-k eigenvalue approximation for efficiency on large graphs.

    Args:
        H: Row-normalized node embeddings of shape (n, d).
        k: Number of top eigenvalues for approximation. If None, uses adaptive k.
        num_iters: Power iteration steps.
        eps: Small constant to avoid log(0).
        use_full_eigen: If True, use full eigendecomposition (exact but O(n^3)).

    Returns:
        Scalar entropy value.
    """
    rho = compute_density_matrix(H)
    n = rho.size(0)

    # Adaptive k selection (Fix #8)
    k_val = adaptive_k(n, k)

    if use_full_eigen:
        # Full eigenspectrum requested
        eigenvalues = torch.linalg.eigvalsh(rho)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
    else:
        # Top-k subspace eigendecomposition (paper Section 2.5)
        eigenvalues = topk_eigenvalues(rho, k=k_val, num_iters=num_iters)

    # S(rho) = -sum lambda_i * log(lambda_i)
    eigenvalues = eigenvalues + eps  # Numerical stability (paper: eps=1e-8)
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))

    return entropy


def quantum_entanglement_loss(
    H: torch.Tensor,
    alpha: float = 0.1,
    k: int = None,
    num_iters: int = 20,
    node_sample_size: int = None,
) -> torch.Tensor:
    """Compute the Quantum Entanglement Loss.

    QEL = alpha * S(rho), where S is von Neumann entropy of the density matrix
    constructed from row-normalized embeddings.

    Args:
        H: Node embeddings of shape (n, d). Will be L2-normalized internally.
        alpha: Weight for the QEL term.
        k: Number of top eigenvalues for approximation. None for adaptive.
        num_iters: Power iteration steps.
        node_sample_size: If set, randomly sample this many nodes for QEL
            computation (paper Section 2.5: 256 for large LRGB graphs).

    Returns:
        Scalar QEL loss value.
    """
    H_norm = l2_normalize_rows(H)

    # Fix #7: Node sampling for large graphs
    if node_sample_size is not None and H_norm.size(0) > node_sample_size:
        indices = torch.randperm(H_norm.size(0), device=H_norm.device)[:node_sample_size]
        H_norm = H_norm[indices]

    entropy = von_neumann_entropy(H_norm, k=k, num_iters=num_iters)
    return alpha * entropy
