"""
Quantum-Inspired Graph Neural Network (QGNN) with Quantum Entanglement Loss.

Architecture (Equation 5):
    H^(l+1) = sigma( A_norm @ H^(l) @ W^(l)  +  beta * rho^(l) @ V^(l) )
                      [Local Aggregation]          [Global Correlation]

where rho^(l) = (1/n) H^(l) (H^(l))^T is the density matrix at layer l,
V^(l) is a learnable projection matrix, followed by L2 row normalization.

For fixed-size graphs (Cora, Citeseer): V is a standalone nn.Parameter(n, d_out)
    matching the paper exactly.
For variable-size graphs (LRGB, PPI): V = H @ W_V where W_V is learnable,
    since V cannot be a fixed parameter when n varies.

Reference:
    "Quantum-Enhanced Learning: Leveraging Von Neumann Entropy for Enhanced
     Graph Neural Network Performance"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from utils import l2_normalize_rows, quantum_entanglement_loss


class QGNNLayer(nn.Module):
    """Single QGNN layer combining local GCN aggregation with global
    quantum-inspired correlation.

    Implements Equation (5) from the paper:
        H^(l+1) = sigma(A_norm H^(l) W^(l) + beta * rho^(l) V^(l))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        beta: float = 0.1,
        fixed_n: int = None,
    ):
        """
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            beta: Strength of global correlation.
            fixed_n: If set, use a standalone V parameter of shape (n, out_channels)
                matching the paper exactly. If None, parameterize V = H @ W_V.
        """
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.beta = beta
        self.fixed_n = fixed_n

        if fixed_n is not None:
            # Paper Eq. 5: V^(l) in R^(n x d_{l+1}) as independent learnable param
            self.V = nn.Parameter(torch.empty(fixed_n, out_channels))
            nn.init.xavier_uniform_(self.V)
            self.W_V = None
        else:
            # Variable-size graphs: V = H @ W_V
            self.V = None
            self.W_V = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (n, in_channels).
            edge_index: Edge indices of shape (2, num_edges).

        Returns:
            Updated node embeddings of shape (n, out_channels).
        """
        # Paper Algorithm 1 uses the same H^(l) for both Z_local and rho,
        # and Eq. 1 requires ||H_i||_2 = 1 for rho to be a valid density
        # matrix. For l >= 1 the input is already unit-norm (from the previous
        # layer's line-12 normalization); for l = 0 we normalize here.
        H = l2_normalize_rows(x)

        # Local aggregation: A_norm @ H @ W
        z_local = self.gcn(H, edge_index)

        # Global correlation: beta * rho^(l) @ V^(l)
        n = H.size(0)

        if self.V is not None:
            # Fixed-n: rho @ V directly (paper Eq. 5 exact)
            rho = (1.0 / n) * (H @ H.t())
            z_global = self.beta * (rho @ self.V)
        else:
            # Variable-n: V = H @ W_V, so rho @ V = rho @ H @ W_V
            # Efficient: rho @ H = (1/n) H (H^T H), compute via associativity
            HtH = H.t() @ H  # (d, d)
            global_feat = H @ HtH  # (n, d) = O(n*d^2) instead of O(n^2*d)
            z_global = self.beta * (1.0 / n) * self.W_V(global_feat)

        # Combined output with ReLU activation
        out = F.relu(z_local + z_global)

        # L2 row normalization (Algorithm 1, line 12)
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
        k: int = None,
        task: str = "node_classification",
        fixed_n: int = None,
        node_sample_size: int = None,
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
            k: Number of top eigenvalues for entropy approximation. None = adaptive.
            task: One of 'node_classification', 'graph_classification',
                  'graph_regression', 'link_prediction', 'multilabel'.
            fixed_n: If set, use standalone V parameters (for fixed-size graphs).
            node_sample_size: If set, subsample nodes for QEL on large graphs
                (paper Section 2.5: 256 for PascalVOC-SP/COCO-SP).
        """
        super().__init__()
        self.alpha = alpha
        self.k = k
        self.dropout = dropout
        self.task = task
        self.num_layers = num_layers
        self.node_sample_size = node_sample_size

        # Build QGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            self.layers.append(QGNNLayer(in_ch, out_ch, beta=beta, fixed_n=fixed_n))

        # Task-specific output head
        if task == "link_prediction":
            # Edge scoring via bilinear form for link prediction (Fix #5)
            self.edge_scorer = nn.Bilinear(hidden_channels, hidden_channels, 1)
            self.classifier = None
        elif out_channels is not None:
            self.classifier = nn.Linear(hidden_channels, out_channels)
            self.edge_scorer = None
        else:
            self.classifier = None
            self.edge_scorer = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        edge_label_index: torch.Tensor = None,
    ) -> tuple:
        """Forward pass.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            batch: Batch assignment vector for graph-level tasks.
            edge_label_index: Edge indices to score for link prediction (2, E_pred).

        Returns:
            Tuple of (logits/predictions, qel_loss).
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if i < self.num_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        final_embeddings = h

        # Compute QEL on final-layer embeddings
        if self.training:
            if batch is not None:
                qel = self._batched_qel(final_embeddings, batch)
            else:
                qel = quantum_entanglement_loss(
                    final_embeddings,
                    alpha=self.alpha,
                    k=self.k,
                    node_sample_size=self.node_sample_size,
                )
        else:
            qel = torch.tensor(0.0, device=x.device)

        # Task-specific output
        if self.task == "link_prediction" and edge_label_index is not None:
            # Score edges via bilinear function (Fix #5)
            src = final_embeddings[edge_label_index[0]]
            dst = final_embeddings[edge_label_index[1]]
            out = self.edge_scorer(src, dst)
        elif self.task in ("graph_classification", "graph_regression", "multilabel"):
            h_pool = global_mean_pool(final_embeddings, batch)
            out = self.classifier(h_pool) if self.classifier else h_pool
        else:
            out = self.classifier(final_embeddings) if self.classifier else final_embeddings

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
                H_g,
                alpha=self.alpha,
                k=self.k,
                node_sample_size=self.node_sample_size,
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
    """QGNN for graph classification / multi-label tasks (e.g., Peptides-func)."""

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
