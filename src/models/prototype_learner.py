"""
Stage 3: Prototype Learner
Learns M concept prototypes per concept {p_km} and the projector P
using a multi-prototype contrastive loss.

Key ideas from the paper:
  - Single local concept vector v_k is non-generalizable to new images
  - Contrastive learning pulls intra-concept vectors together (compactness)
    and pushes inter-concept vectors apart (separation)
  - M prototypes per concept handle multi-modal visual appearance of findings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLearner(nn.Module):
    """
    Learns K*M concept prototypes in the projected feature space.

    Prototypes are stored as a learnable parameter matrix and updated
    via the contrastive loss during Stage 3 training.
    """

    def __init__(
        self,
        num_concepts: int,       # K
        num_prototypes: int,     # M — prototypes per concept
        proto_dim: int,          # dimension of each prototype vector
        lambda_scale: float = 10.0,   # λ: contrastive scaling factor
        gamma_scale: float = 5.0,     # γ: prototype assignment sharpness
        margin: float = 0.1,          # δ: inter-concept margin
    ):
        super().__init__()
        self.K = num_concepts
        self.M = num_prototypes
        self.proto_dim = proto_dim
        self.lambda_scale = lambda_scale
        self.gamma_scale = gamma_scale
        self.margin = margin

        # Learnable prototype matrix: (K, M, D)
        # Initialized with random unit vectors
        protos = torch.randn(num_concepts, num_prototypes, proto_dim)
        protos = F.normalize(protos, p=2, dim=-1)
        self.prototypes = nn.Parameter(protos)

    @property
    def normalized_prototypes(self) -> torch.Tensor:
        """Always return L2-normalized prototypes. Shape: (K, M, D)"""
        return F.normalize(self.prototypes, p=2, dim=-1)

    def assignment_distribution(self, v_prime: torch.Tensor, concept_idx: int) -> torch.Tensor:
        """
        Compute soft assignment of projected vector v' to the M prototypes
        of a given concept using scaled softmax.

        q_m(v') = softmax_m( γ * <p_km, v'> )

        Args:
            v_prime:     projected concept vector (D,) or (B, D)
            concept_idx: which concept's prototypes to use

        Returns:
            q: assignment weights over M prototypes (M,) or (B, M)
        """
        protos_k = self.normalized_prototypes[concept_idx]   # (M, D)
        if v_prime.dim() == 1:
            sim = torch.mv(protos_k, v_prime)                # (M,)
        else:
            sim = torch.mm(v_prime, protos_k.T)              # (B, M)
        return F.softmax(self.gamma_scale * sim, dim=-1)

    def concept_similarity(self, v_prime: torch.Tensor, concept_idx: int) -> torch.Tensor:
        """
        Compute similarity of v' with concept c_k using all M prototypes.

        sim_k(v') = Σ_m q_m(v') * <p_km, v'>

        Args:
            v_prime:     projected concept vector (D,) or (B, D)
            concept_idx: target concept index

        Returns:
            scalar or (B,) similarity score
        """
        protos_k = self.normalized_prototypes[concept_idx]   # (M, D)
        if v_prime.dim() == 1:
            dot = torch.mv(protos_k, v_prime)                # (M,)
        else:
            dot = torch.mm(v_prime, protos_k.T)              # (B, M)

        q = self.assignment_distribution(v_prime, concept_idx)  # (M,) or (B,M)
        return (q * dot).sum(dim=-1)                         # scalar or (B,)

    def contrastive_loss(self, v_prime: torch.Tensor, concept_labels: torch.Tensor) -> torch.Tensor:
        """
        Multi-prototype contrastive loss (Equation 9 from paper).

        For each concept vector v'_k^i, maximize its similarity to its own
        concept prototypes while minimizing similarity to other concepts.

        L = -log( exp(λ*(sim_k̃(v') + δ)) / Σ_k exp(λ*sim_k(v')) )

        Args:
            v_prime:       projected concept vectors (B, K, D)
                           v_prime[b, k] is the projected vector for concept k, image b
            concept_labels: binary concept labels (B, K)
                            used to identify the positive concept for each vector

        Returns:
            mean contrastive loss (scalar)
        """
        B, K, D = v_prime.shape
        protos = self.normalized_prototypes                   # (K, M, D)
        total_loss = 0.0
        count = 0

        for k in range(K):
            vk = v_prime[:, k, :]                            # (B, D) — vectors for concept k

            # Compute sim_k'(v'_k) for all concepts k' in K
            # Shape: (B, K) — similarity of each image's concept-k vector to all K concepts
            all_sims = torch.zeros(B, K, device=v_prime.device)
            for kp in range(K):
                all_sims[:, kp] = self.concept_similarity(vk, kp)

            # Add margin δ to the positive concept (concept k itself)
            all_sims_with_margin = all_sims.clone()
            all_sims_with_margin[:, k] = all_sims[:, k] + self.margin

            # Contrastive loss: numerator = positive concept, denominator = all concepts
            numerator = torch.exp(self.lambda_scale * all_sims_with_margin[:, k])
            denominator = torch.exp(self.lambda_scale * all_sims).sum(dim=-1)
            loss_k = -torch.log(numerator / (denominator + 1e-8))

            # Only average over images where concept k is present
            # (vectors for absent concepts are less meaningful)
            mask = concept_labels[:, k].bool()
            if mask.sum() > 0:
                total_loss += loss_k[mask].mean()
                count += 1

        return total_loss / max(count, 1)

    def get_similarity_scores(self, f_prime: torch.Tensor) -> torch.Tensor:
        """
        Inference: compute similarity scores s_km for all concepts and prototypes
        given a projected feature map.

        s_km = max_{h,w} <p_km, P(f(h,w))>

        Args:
            f_prime: projected feature map (B, D, H, W)

        Returns:
            s: similarity scores (B, K, M)
        """
        B, D, H, W = f_prime.shape
        protos = self.normalized_prototypes                   # (K, M, D)

        # Flatten spatial dims: (B, H*W, D)
        patches = f_prime.permute(0, 2, 3, 1).reshape(B, H * W, D)

        # Cosine similarity: (B, H*W, D) x (K*M, D)^T → (B, H*W, K*M)
        protos_flat = protos.view(self.K * self.M, D)
        sim = torch.matmul(patches, protos_flat.T)            # (B, H*W, K*M)

        # Max over spatial positions
        sim_max = sim.max(dim=1).values                       # (B, K*M)
        return sim_max.view(B, self.K, self.M)                # (B, K, M)

    def get_similarity_maps(self, f_prime: torch.Tensor) -> torch.Tensor:
        """
        Inference: compute 2D similarity maps S_km(h,w) for visualization.

        Args:
            f_prime: projected feature map (B, D, H, W)

        Returns:
            maps: similarity maps (B, K, M, H, W)
        """
        B, D, H, W = f_prime.shape
        protos = self.normalized_prototypes                   # (K, M, D)

        patches = f_prime.permute(0, 2, 3, 1).reshape(B, H * W, D)
        protos_flat = protos.view(self.K * self.M, D)

        sim = torch.matmul(patches, protos_flat.T)            # (B, H*W, K*M)
        return sim.view(B, H, W, self.K, self.M).permute(0, 3, 4, 1, 2)  # (B, K, M, H, W)

    def initialize_from_vectors(self, concept_vectors: torch.Tensor):
        """
        Initialize prototypes from a set of local concept vectors using
        k-means-style random selection (one per M slots per concept).

        Args:
            concept_vectors: (N, K, D) — local concept vectors from Stage 2
        """
        N, K, D = concept_vectors.shape
        with torch.no_grad():
            for k in range(K):
                vk = concept_vectors[:, k, :]                # (N, D)
                # Random subset of size M (or all if N < M)
                idx = torch.randperm(N)[:self.M]
                selected = vk[idx]                           # (M, D)
                if selected.shape[0] < self.M:
                    # Pad by repeating if N < M
                    repeat = (self.M + selected.shape[0] - 1) // selected.shape[0]
                    selected = selected.repeat(repeat, 1)[:self.M]
                self.prototypes.data[k] = F.normalize(selected, p=2, dim=-1)
