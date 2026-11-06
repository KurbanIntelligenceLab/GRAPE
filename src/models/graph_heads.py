"""
H3 — Graph-based Task Head Baselines: ML-GCN and ADD-GCN

Both heads are drop-in replacements for GNNTaskHead (GAT-based).
They receive sim_scores (B, K, M) and output class logits (B, num_classes).

ML-GCN: Chen et al., "Multi-label image recognition with graph convolutional
         networks," CVPR 2019. Standard 2-layer GCN with symmetric normalisation.

ADD-GCN: Ye et al., "Attention-driven dynamic graph convolutional network for
          multi-label image recognition," ECCV 2020. Combines a static
          co-occurrence graph with a per-sample dynamic graph whose edges are
          cosine similarities of the current node features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared utility ────────────────────────────────────────────────────────────

def _build_sym_norm_adj(edge_index: torch.Tensor,
                        edge_weight: torch.Tensor,
                        num_nodes: int) -> torch.Tensor:
    """Build dense D^{-1/2} A D^{-1/2} (with self-loops) from COO format."""
    device = edge_index.device
    A = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.numel() > 0:
        A[edge_index[0], edge_index[1]] = edge_weight
    A = A + torch.eye(num_nodes, device=device)          # self-loops
    D = A.sum(dim=1).clamp(min=1e-6)
    D_inv_sqrt = D.pow(-0.5)
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)  # (K, K)


def _build_row_norm_adj(edge_index: torch.Tensor,
                        edge_weight: torch.Tensor,
                        num_nodes: int) -> torch.Tensor:
    """Build dense D^{-1} A (with self-loops) from COO format (for ADD-GCN static)."""
    device = edge_index.device
    A = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.numel() > 0:
        A[edge_index[0], edge_index[1]] = edge_weight
    A = A + torch.eye(num_nodes, device=device)
    D = A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return A / D                                          # (K, K)


# ── ML-GCN ────────────────────────────────────────────────────────────────────

class MLGCNTaskHead(nn.Module):
    """
    ML-GCN task head adapted for prototype similarity scores.

    Architecture:
      sim_scores (B, K, M) [M = prototype features per concept node]
        → GCN Layer 1: LeakyReLU( A_hat @ s @ W1 )   (B, K, hidden_dim)
        → GCN Layer 2: LeakyReLU( A_hat @ h @ W2 )   (B, K, hidden_dim)
        → Flatten + Linear                             (B, num_classes)

    A_hat = D^{-1/2} (A + I) D^{-1/2}  (symmetric normalised, fixed per graph)
    """

    def __init__(
        self,
        num_concepts: int,
        num_prototypes: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        self.W1 = nn.Linear(num_prototypes, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.LayerNorm(num_concepts * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(num_concepts * hidden_dim, num_classes),
        )

        self.register_buffer("adj", None)   # (K, K) symmetric norm adj

    def set_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.adj = _build_sym_norm_adj(edge_index, edge_weight, self.num_concepts)

    def forward(self, sim_scores: torch.Tensor) -> torch.Tensor:
        """sim_scores: (B, K, M) → logits (B, num_classes)"""
        assert self.adj is not None, "Call set_graph() before forward()"
        B, K, M = sim_scores.shape

        # Layer 1
        h = self.W1(sim_scores)                           # (B, K, hidden)
        h = torch.einsum("kj,bjd->bkd", self.adj, h)     # GCN propagation
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.dropout(h)

        # Layer 2
        h = self.W2(h)                                    # (B, K, hidden)
        h = torch.einsum("kj,bjd->bkd", self.adj, h)
        h = F.leaky_relu(h, negative_slope=0.2)

        return self.readout(h.reshape(B, -1))


# ── ADD-GCN ───────────────────────────────────────────────────────────────────

class ADDGCNTaskHead(nn.Module):
    """
    ADD-GCN (Adaptive Dynamic) task head adapted for prototype similarity scores.

    At each layer, edge weights combine:
      - Static graph A_s: from co-occurrence statistics (fixed per training set)
      - Dynamic graph A_d: per-sample cosine similarity between node features

    Combined adjacency: A = α * A_s + (1-α) * A_d

    The dynamic graph lets the model adapt message passing to the actual visual
    content of each image — high co-occurrence concepts that are visually
    dissimilar in a given image get down-weighted.
    """

    def __init__(
        self,
        num_concepts: int,
        num_prototypes: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        alpha: float = 0.5,    # balance static vs dynamic graph
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.alpha = alpha

        self.W1 = nn.Linear(num_prototypes, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.LayerNorm(num_concepts * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(num_concepts * hidden_dim, num_classes),
        )

        self.register_buffer("adj_static", None)  # (K, K) row-norm static adj

    def set_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.adj_static = _build_row_norm_adj(edge_index, edge_weight, self.num_concepts)

    @staticmethod
    def _dynamic_adj(h: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample dynamic adjacency from node features.

        A_d[b, k, k'] = softmax_k'( cosine_sim(h[b,k], h[b,k']) )

        Returns: (B, K, K) row-stochastic matrix
        """
        h_norm = F.normalize(h, dim=-1)                     # (B, K, D)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))     # (B, K, K)
        return F.softmax(sim, dim=-1)

    def _propagate(self, h: torch.Tensor) -> torch.Tensor:
        """Combined GCN step: A_combined @ h"""
        adj_d = self._dynamic_adj(h)                        # (B, K, K)
        h_static  = torch.einsum("kj,bjd->bkd", self.adj_static, h)
        h_dynamic = torch.bmm(adj_d, h)
        return self.alpha * h_static + (1.0 - self.alpha) * h_dynamic

    def forward(self, sim_scores: torch.Tensor) -> torch.Tensor:
        """sim_scores: (B, K, M) → logits (B, num_classes)"""
        assert self.adj_static is not None, "Call set_graph() before forward()"
        B, K, M = sim_scores.shape

        # Layer 1
        h = self.W1(sim_scores)                             # (B, K, hidden)
        h = self._propagate(h)
        h = F.leaky_relu(h, negative_slope=0.2)
        h = self.dropout(h)

        # Layer 2
        h = self.W2(h)                                      # (B, K, hidden)
        h = self._propagate(h)
        h = F.leaky_relu(h, negative_slope=0.2)

        return self.readout(h.reshape(B, -1))
