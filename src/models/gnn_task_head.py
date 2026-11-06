"""
Improvement A: GNN Task Head
Replaces the original linear task head H with a 2-layer Graph Attention
Network (GAT) that models anatomical concept co-occurrence relationships.

Why GAT over GCN?
  - Attention weights allow the model to learn WHICH neighbors matter most
    for each concept, rather than treating all edges equally
  - More expressive than GCN for heterogeneous concept graphs

Architecture:
  similarity scores s_km (K x M per image)
    └─► Node features: aggregate M scores per concept → (K, M)
          └─► GAT Layer 1: multi-head attention over concept graph
                └─► GAT Layer 2: refinement
                      └─► Linear readout → class logits

The concept graph (edge_index, edge_weight) is built once from training
label statistics using src/utils/graph_builder.py and fixed during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention Network layer.

    Computes: h'_i = σ( Σ_{j∈N(i)} α_ij * W * h_j )
    where α_ij are softmax-normalized attention coefficients.

    We implement this without torch_geometric dependency
    for portability — uses sparse tensor operations instead.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.head_dim = out_features

        # Linear transform for node features (one per head)
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)

        # Attention vector: scores each (h_i || h_j) pair
        self.a = nn.Parameter(torch.empty(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        Args:
            x:                node features (K, in_features) or (B, K, in_features)
            edge_index:       (2, E) edge indices
            edge_weight:      (E,) optional edge weights
            return_attention: if True, also return attention weights (E, heads)

        Returns:
            If return_attention=False: updated node features tensor
            If return_attention=True:  (node_features, alpha) where alpha is (E, heads)
        """
        batched = x.dim() == 3
        if batched:
            B, K, _ = x.shape
            results = [
                self._forward_single(x[b], edge_index, edge_weight,
                                     return_attention=return_attention)
                for b in range(B)
            ]
            if return_attention:
                feats  = torch.stack([r[0] for r in results])
                alphas = torch.stack([r[1] for r in results])
                return feats, alphas
            return torch.stack(results)
        return self._forward_single(x, edge_index, edge_weight,
                                    return_attention=return_attention)

    def _forward_single(self, x, edge_index, edge_weight, return_attention=False):
        K = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Linear transform: (K, num_heads, head_dim)
        Wh = self.W(x).view(K, self.num_heads, self.head_dim)

        # Attention coefficients for each edge
        # e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
        Wh_src = Wh[src]                                     # (E, heads, D)
        Wh_dst = Wh[dst]                                     # (E, heads, D)
        cat = torch.cat([Wh_src, Wh_dst], dim=-1)            # (E, heads, 2D)
        e = self.leaky_relu((cat * self.a).sum(dim=-1))      # (E, heads)

        # Optional: scale by edge weight
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)

        # Softmax per destination node and head
        alpha = self._sparse_softmax(e, dst, K)              # (E, heads)
        alpha = self.dropout(alpha)

        # Aggregate: h'_i = Σ_j α_ij * Wh_j
        out = torch.zeros(K, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(Wh_dst),
                         alpha.unsqueeze(-1) * Wh_dst)       # (K, heads, D)

        if self.concat:
            feat = F.elu(out.view(K, self.num_heads * self.head_dim))
        else:
            feat = F.elu(out.mean(dim=1))                    # (K, D)

        if return_attention:
            return feat, alpha.detach()
        return feat

    @staticmethod
    def _sparse_softmax(e: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Per-node softmax over incoming edge attention scores."""
        # Numerical stability: subtract max per destination node
        e_max = torch.zeros(num_nodes, e.shape[1], device=e.device)
        e_max.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(e), e, reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[dst])

        e_sum = torch.zeros(num_nodes, e.shape[1], device=e.device)
        e_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(e_exp), e_exp)

        return e_exp / (e_sum[dst] + 1e-8)


class GNNTaskHead(nn.Module):
    """
    2-layer GAT task head that replaces the linear task head H.

    Takes similarity scores s ∈ R^(B, K, M) as node features,
    applies two GAT layers over the concept graph,
    and produces class logits.

    Drop-in replacement for nn.Linear in the original CSR pipeline.
    """

    def __init__(
        self,
        num_concepts: int,        # K
        num_prototypes: int,      # M — input features per node
        num_classes: int,         # output classes
        hidden_dim: int = 64,     # GAT hidden dimension
        num_heads: int = 4,       # attention heads in layer 1
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        # Layer 1: M → hidden_dim (multi-head, concat)
        self.gat1 = GATLayer(
            in_features=num_prototypes,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        # Layer 2: num_heads*hidden_dim → hidden_dim (single-head, mean)
        self.gat2 = GATLayer(
            in_features=num_heads * hidden_dim,
            out_features=hidden_dim,
            num_heads=1,
            dropout=dropout,
            concat=False,
        )

        # Readout: flatten all node features → class logits
        self.readout = nn.Sequential(
            nn.LayerNorm(num_concepts * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(num_concepts * hidden_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

        # Graph structure — registered as buffers (not parameters)
        # Set via set_graph() before training
        self.register_buffer("edge_index", None)
        self.register_buffer("edge_weight", None)

    def set_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """
        Register the concept co-occurrence graph.
        Call this once after building the graph from training labels.

        Args:
            edge_index:  (2, E)
            edge_weight: (E,)
        """
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(self, s: torch.Tensor, return_attention: bool = False):
        """
        Args:
            s:                similarity scores (B, K, M)
            return_attention: if True, return (logits, attn1, attn2) where
                              attn1/attn2 are (B, E, heads) from layers 1 and 2

        Returns:
            logits (B, num_classes), or (logits, attn1, attn2) if return_attention
        """
        assert self.edge_index is not None, "Call set_graph() before forward()"

        B, K, M = s.shape
        x = s                                                # (B, K, M)

        if return_attention:
            x, attn1 = self.gat1(x, self.edge_index, self.edge_weight,
                                  return_attention=True)     # (B, K, heads*hidden)
            x, attn2 = self.gat2(x, self.edge_index, self.edge_weight,
                                  return_attention=True)     # (B, K, hidden)
            x = x.view(B, K * x.shape[-1])
            return self.readout(x), attn1, attn2
        else:
            x = self.gat1(x, self.edge_index, self.edge_weight)
            x = self.gat2(x, self.edge_index, self.edge_weight)
            x = x.view(B, K * x.shape[-1])
            return self.readout(x)
