"""
Quantum-Inspired Graph Neural Network (QGNN) with Quantum Entanglement Loss.

Architecture:
    H^(l+1) = sigma( A_norm @ H^(l) @ W^(l)  +  beta * rho^(l) @ H^(l) @ W_V^(l) )
                      [Local Aggregation]          [Global Correlation]

where rho^(l) = (1/n) H^(l) (H^(l))^T is the density matrix at layer l,
followed by L2 row normalization.

Reference:
    "Quantum-Enhanced Learning: Leveraging Von Neumann Entropy for Enhanced
     Graph Neural Network Performance"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, degree

from utils import l2_normalize_rows, compute_density_matrix, quantum_entanglement_loss


class QGNNLayer(nn.Module):
    """Single QGNN layer combining local GCN aggregation with global
    quantum-inspired correlation.

    Implements Equation (5) from the paper:
        H^(l+1) = sigma(A_norm H^(l) W^(l) + beta * rho^(l) H^(l) W_V^(l))

    Since rho = (1/n) H H^T, the global term expands to:
        beta * (1/n) * H H^T H W_V

    This avoids explicitly constructing V in R^(n x d_{l+1}), making the layer
    applicable to graphs of any size.
    """

    def __init__(self, in_channels: int, out_channels: int, beta: float = 0.1):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        # Learnable projection for the global correlation term
        self.W_V = nn.Linear(in_channels, out_channels, bias=False)
        self.beta = beta

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (n, in_channels).
            edge_index: Edge indices of shape (2, num_edges).

        Returns:
            Updated node embeddings of shape (n, out_channels).
        """
        # Local aggregation: A_norm @ H @ W
        z_local = self.gcn(x, edge_index)

        # Global correlation: beta * (1/n) * H @ H^T @ H @ W_V
        n = x.size(0)
        H_norm = l2_normalize_rows(x)
        # Efficient computation: first H^T @ H (d x d), then H @ (H^T H) @ W_V
        # This is O(n*d^2) instead of O(n^2*d) when d << n
        HtH = H_norm.t() @ H_norm  # (d, d)
        global_feat = H_norm @ HtH  # (n, d)
        z_global = self.beta * (1.0 / n) * self.W_V(global_feat)

        # Combined output with ReLU activation
        out = F.relu(z_local + z_global)

        # L2 row normalization (essential for density matrix interpretation)
        out = l2_normalize_rows(out)

        return out


class QGNN(nn.Module):
    """Quantum-Inspired Graph Neural Network.

    A multi-layer GNN that combines standard message-passing with
    quantum-inspired global correlation, regularized by Quantum
    Entanglement Loss (QEL).

    Default configuration: 2-layer GCN with hidden dimension 16,
    alpha=0.1, beta=0.1, dropout=0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        out_channels: int = None,
        num_layers: int = 2,
        alpha: float = 0.1,
        beta: float = 0.1,
        dropout: float = 0.2,
        k: int = 16,
        task: str = "node_classification",
    ):
        """
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer dimension.
            out_channels: Output dimension (number of classes or regression targets).
            num_layers: Number of QGNN layers.
            alpha: Weight for QEL regularization.
            beta: Strength of global correlation in each layer.
            dropout: Dropout rate between layers.
            k: Number of top eigenvalues for entropy approximation.
            task: One of 'node_classification', 'graph_classification',
                  'graph_regression', 'link_prediction', 'multilabel'.
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.dropout = dropout
        self.task = task
        self.num_layers = num_layers

        # Build QGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            self.layers.append(QGNNLayer(in_ch, out_ch, beta=beta))

        # Task-specific output head
        if out_channels is not None:
            self.classifier = nn.Linear(hidden_channels, out_channels)
        else:
            self.classifier = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> tuple:
        """Forward pass.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            batch: Batch assignment vector for graph-level tasks.

        Returns:
            Tuple of (logits/predictions, qel_loss).
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Store final embeddings for QEL computation
        final_embeddings = h

        # Compute QEL on final-layer embeddings
        if self.training:
            if batch is not None:
                # For batched graphs, compute QEL per graph and average
                qel = self._batched_qel(final_embeddings, batch)
            else:
                qel = quantum_entanglement_loss(
                    final_embeddings, alpha=self.alpha, k=self.k
                )
        else:
            qel = torch.tensor(0.0, device=x.device)

        # Task-specific output
        if self.task in ("graph_classification", "graph_regression"):
            # Global mean pooling
            from torch_geometric.nn import global_mean_pool
            h = global_mean_pool(final_embeddings, batch)

        if self.classifier is not None:
            out = self.classifier(h)
        else:
            out = h

        return out, qel

    def _batched_qel(
        self, H: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute QEL for a batch of graphs, averaging across graphs."""
        unique_graphs = batch.unique()
        total_qel = torch.tensor(0.0, device=H.device)

        for g in unique_graphs:
            mask = batch == g
            H_g = H[mask]
            total_qel = total_qel + quantum_entanglement_loss(
                H_g, alpha=self.alpha, k=min(self.k, H_g.size(0))
            )

        return total_qel / len(unique_graphs)


class QGNNNodeClassifier(QGNN):
    """QGNN for node classification tasks (e.g., Cora, Citeseer)."""

    def __init__(self, in_channels, num_classes, **kwargs):
        kwargs.setdefault("hidden_channels", 16)
        kwargs.setdefault("task", "node_classification")
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            **kwargs,
        )


class QGNNGraphClassifier(QGNN):
    """QGNN for graph classification tasks (e.g., Peptides-func)."""

    def __init__(self, in_channels, num_classes, **kwargs):
        kwargs.setdefault("hidden_channels", 16)
        kwargs.setdefault("task", "graph_classification")
        super().__init__(
            in_channels=in_channels,
            out_channels=num_classes,
            **kwargs,
        )


class QGNNGraphRegressor(QGNN):
    """QGNN for graph regression tasks (e.g., Peptides-struct)."""

    def __init__(self, in_channels, out_channels=1, **kwargs):
        kwargs.setdefault("hidden_channels", 16)
        kwargs.setdefault("task", "graph_regression")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
