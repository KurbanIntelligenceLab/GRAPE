"""
Graph Builder — Improvement A
Constructs a concept co-occurrence graph from training label statistics.

The graph is used by the GNN task head to model anatomical relationships
between concepts (e.g. enlarged heart and lung fluid often co-occur in
pulmonary edema).

Nodes:  one per concept
Edges:  connect concept pairs whose co-occurrence probability > threshold τ
Weights: conditional probability P(concept_k=1 | concept_k'=1)
"""

import torch
import numpy as np


def build_cooccurrence_graph(
    concept_labels: torch.Tensor,
    threshold: float = 0.1,
    self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a concept co-occurrence graph from binary concept labels.

    Args:
        concept_labels: binary label matrix (N, K) — N samples, K concepts
        threshold:      minimum conditional probability to add an edge
        self_loops:     whether to add self-loops (recommended for GNNs)

    Returns:
        edge_index: (2, E) — COO format edge indices
        edge_weight: (E,) — edge weights = conditional probabilities
    """
    N, K = concept_labels.shape
    labels = concept_labels.float()

    # Co-occurrence count: C[k, k'] = number of samples where both k and k' are 1
    cooccur = torch.mm(labels.T, labels)                     # (K, K)

    # Marginal counts: how many times each concept appears
    marginals = labels.sum(dim=0)                            # (K,)

    # Conditional probability: P(k | k') = C[k,k'] / count(k')
    # Avoid division by zero for concepts with zero occurrences
    denom = marginals.unsqueeze(0).clamp(min=1)              # (1, K)
    cond_prob = cooccur / denom                              # (K, K)

    # Zero out diagonal (self-loop added separately below)
    cond_prob.fill_diagonal_(0)

    # Threshold: only keep edges with strong co-occurrence signal
    edge_mask = cond_prob > threshold                        # (K, K) bool

    # Build COO edge list
    src, dst = edge_mask.nonzero(as_tuple=True)              # each (E,)
    weights = cond_prob[src, dst]                            # (E,)

    # Add self-loops
    if self_loops:
        self_idx = torch.arange(K, device=concept_labels.device)
        src = torch.cat([src, self_idx])
        dst = torch.cat([dst, self_idx])
        weights = torch.cat([weights, torch.ones(K, device=weights.device)])

    edge_index = torch.stack([src, dst], dim=0)              # (2, E)

    return edge_index, weights


def normalize_edge_weights(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Row-normalize edge weights: for each source node, weights sum to 1.
    This is the symmetric normalization used in GCN.

    W_norm[i,j] = W[i,j] / Σ_j W[i,j]

    Args:
        edge_index:  (2, E)
        edge_weight: (E,)
        num_nodes:   K

    Returns:
        normalized edge weights (E,)
    """
    src = edge_index[0]
    # Sum of weights per source node
    deg = torch.zeros(num_nodes, device=edge_weight.device)
    deg.scatter_add_(0, src, edge_weight)
    # Normalize
    row_sum = deg[src].clamp(min=1e-8)
    return edge_weight / row_sum


def build_graph_from_numpy(
    concept_labels: np.ndarray,
    threshold: float = 0.1,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for numpy label arrays."""
    labels_tensor = torch.from_numpy(concept_labels).float().to(device)
    edge_index, edge_weight = build_cooccurrence_graph(labels_tensor, threshold=threshold)
    edge_weight = normalize_edge_weights(edge_index, edge_weight, labels_tensor.shape[1])
    return edge_index, edge_weight
