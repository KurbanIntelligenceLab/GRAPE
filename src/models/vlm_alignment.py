"""
Improvement C: Medical VLM Alignment
Anchors visual concept prototypes to text embeddings from a frozen
Medical Vision-Language Model (BioViL-T or MedCLIP).

This enables open-vocabulary concept addition at test time:
  → write a clinical text description → encode → project → add to atlas
  → no retraining required

Architecture addition:
  Frozen VLM text encoder
    └─► t_k ∈ R^D_text  (concept text embedding)
          └─► Learnable projection W: R^D_text → R^D_visual
                └─► t̂_k ∈ R^D_visual  (text anchor in visual space)
                      └─► AlignmentLoss: minimize (1 - cos(p_km, t̂_k))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Concept text descriptions for each dataset
# These are fed to the VLM text encoder to generate t_k embeddings
# ──────────────────────────────────────────────────────────────────────────────

CONCEPT_DESCRIPTIONS = {
    # TBX11K: 3 TB finding categories from COCO annotations
    "tbx11k": {
        "active_tuberculosis": (
            "Active pulmonary tuberculosis with cavitary lesions, nodular opacities, and "
            "upper-lobe infiltrates, often with signs of consolidation on chest X-ray."
        ),
        "obsolete_pulmonary_tb": (
            "Healed or inactive tuberculosis with calcified granulomas, fibrous scarring, "
            "and volume loss in the upper lobes on chest X-ray."
        ),
        "pulmonary_tuberculosis": (
            "Pulmonary tuberculosis presenting as patchy or confluent consolidation, "
            "tree-in-bud opacities, or miliary nodules on chest X-ray."
        ),
    },
    "vindrcxr": {
        "aortic_enlargement": (
            "Widening of the aortic contour on chest X-ray, suggesting aortic aneurysm or dilation."
        ),
        "atelectasis": (
            "Linear or plate-like opacities representing collapsed lung segments, often at the "
            "lung bases or adjacent to pleural effusions."
        ),
        "calcification": (
            "Focal areas of increased density within the lung or mediastinum, representing "
            "calcium deposits from prior granulomatous infection."
        ),
    },
    "isic": {
        "pigment_network": (
            "A reticular pattern of pigmented lines forming a network structure over a skin lesion, "
            "visible on dermoscopy."
        ),
        "globules": (
            "Round to oval structures of various sizes and colors within or at the periphery "
            "of a skin lesion under dermoscopic examination."
        ),
        "streaks": (
            "Irregular linear projections at the periphery of a melanocytic lesion, "
            "appearing as radial lines or pseudopods on dermoscopy."
        ),
    },
    # NIH ChestX-ray14 concept descriptions
    "nih": {
        "Atelectasis": (
            "Partial or complete collapse of lung tissue visible as linear or plate-like opacities, "
            "often at the lung bases with volume loss and shift of adjacent structures on chest X-ray."
        ),
        "Cardiomegaly": (
            "Enlarged cardiac silhouette with cardiothoracic ratio greater than 0.5 on frontal chest X-ray, "
            "indicating cardiac enlargement or pericardial effusion."
        ),
        "Consolidation": (
            "Homogeneous airspace opacity replacing normal lung aeration, with air bronchograms visible, "
            "consistent with pneumonia, hemorrhage, or other alveolar filling process."
        ),
        "Edema": (
            "Bilateral perihilar haziness and Kerley B lines indicating pulmonary edema, with possible "
            "pleural effusions and upper lobe vascular redistribution on chest X-ray."
        ),
        "Effusion": (
            "Fluid accumulation in the pleural space causing blunting of costophrenic angle and "
            "meniscus-shaped opacity at the lung base, visible on upright chest X-ray."
        ),
        "Emphysema": (
            "Hyperinflated lungs with flattened diaphragms, increased anteroposterior diameter, and "
            "hyperlucent lung fields due to destruction of alveolar walls on chest X-ray."
        ),
        "Fibrosis": (
            "Reticular or reticulonodular opacities in the lung parenchyma indicating fibrotic change, "
            "often with volume loss and honeycombing in advanced stages on chest X-ray."
        ),
        "Hernia": (
            "Herniation of abdominal contents through the diaphragm visible as bowel loops or soft tissue "
            "density above the diaphragm level on chest X-ray."
        ),
        "Infiltration": (
            "Ill-defined hazy opacity in the lung parenchyma without complete airspace consolidation, "
            "representing inflammatory or infectious infiltrate on chest X-ray."
        ),
        "Mass": (
            "Discrete pulmonary opacity greater than 3 cm in diameter with well-defined or irregular "
            "margins, requiring further evaluation for malignancy on chest X-ray."
        ),
        "Nodule": (
            "Discrete rounded opacity less than 3 cm in diameter within the lung parenchyma, "
            "with well-defined or irregular margins on chest X-ray."
        ),
        "Pleural_Thickening": (
            "Thickening of the pleural lining visible as an irregular soft tissue density along the "
            "chest wall, often from prior inflammation or asbestos exposure on chest X-ray."
        ),
        "Pneumonia": (
            "Lobar or segmental consolidation with air bronchograms and possible associated pleural "
            "effusion, consistent with bacterial pneumonia on chest X-ray."
        ),
        "Pneumothorax": (
            "Visible visceral pleural line with absent lung markings beyond it, indicating air in the "
            "pleural space and partial or complete lung collapse on chest X-ray."
        ),
    },
}


class TextProjection(nn.Module):
    """
    Learnable linear projection from VLM text embedding space
    to visual prototype space.

    W: R^D_text → R^D_visual
    """

    def __init__(self, text_dim: int, visual_dim: int):
        super().__init__()
        self.projection = nn.Linear(text_dim, visual_dim, bias=False)
        # Initialize near identity-like mapping
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: text embeddings (K, D_text) or (D_text,)

        Returns:
            projected and L2-normalized anchors (K, D_visual) or (D_visual,)
        """
        projected = self.projection(t)
        return F.normalize(projected, p=2, dim=-1)


class AlignmentLoss(nn.Module):
    """
    Cross-modal alignment loss that pulls visual prototypes toward their
    corresponding text embedding anchors.

    L_align = Σ_k Σ_m (1 - cos(p_km, W·t_k))

    Added to Stage 3 training loss:
        L_total = L_con-m + λ_align * L_align
    """

    def __init__(self, lambda_align: float = 0.1):
        super().__init__()
        self.lambda_align = lambda_align

    def forward(
        self,
        prototypes: torch.Tensor,
        text_anchors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prototypes:   visual prototypes (K, M, D_visual), L2-normalized
            text_anchors: projected text embeddings (K, D_visual), L2-normalized

        Returns:
            weighted alignment loss (scalar)
        """
        K, M, D = prototypes.shape

        # Expand text anchors to match prototypes: (K, 1, D) → (K, M, D)
        anchors = text_anchors.unsqueeze(1).expand(K, M, D)

        # Cosine similarity between each prototype and its text anchor
        # Both are L2-normalized, so cos_sim = dot product
        cos_sim = (prototypes * anchors).sum(dim=-1)          # (K, M)

        # Loss: 1 - cos_sim (minimize to maximize alignment)
        loss = (1.0 - cos_sim).mean()

        return self.lambda_align * loss


class VLMAligner(nn.Module):
    """
    Full VLM alignment module.

    Handles:
      1. Loading / caching text embeddings for all concepts
      2. Projecting text embeddings to visual space via W
      3. Computing alignment loss during training
      4. Zero-shot prototype generation for new concepts at test time
    """

    def __init__(
        self,
        text_dim: int,
        visual_dim: int,
        lambda_align: float = 0.1,
        vlm_model_name: str = "microsoft/BiomedVLP-BioViL-T",
    ):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.vlm_model_name = vlm_model_name

        self.text_projection = TextProjection(text_dim, visual_dim)
        self.alignment_loss = AlignmentLoss(lambda_align)

        # Cache for text embeddings (set by encode_concepts)
        self.register_buffer("text_embeddings", None)    # (K, D_text)
        self.concept_names = []

    def encode_concepts(self, concept_descriptions: dict[str, str]) -> torch.Tensor:
        """
        Encode concept text descriptions using the frozen VLM text encoder.
        Results are cached in self.text_embeddings.

        This method requires the transformers library and a GPU for speed.
        Falls back to random embeddings if the VLM is unavailable (for testing).

        Args:
            concept_descriptions: dict mapping concept_name → text description

        Returns:
            text_embeddings: (K, D_text)
        """
        self.concept_names = list(concept_descriptions.keys())
        texts = list(concept_descriptions.values())

        try:
            embeddings = self._encode_with_biovil(texts)
        except Exception as e:
            print(f"VLM encoding failed ({e}), using random embeddings for testing.")
            embeddings = F.normalize(
                torch.randn(len(texts), self.text_dim), p=2, dim=-1
            )

        self.text_embeddings = embeddings
        return embeddings

    def _encode_with_biovil(self, texts: list[str]) -> torch.Tensor:
        """
        Encode texts using BioViL-T from HuggingFace.
        Model is loaded frozen — no gradients through the VLM.
        """
        from transformers import AutoTokenizer, AutoModel

        # Uses the default HuggingFace cache (honors the HF_HOME / HF_HUB_CACHE env vars).
        tokenizer = AutoTokenizer.from_pretrained(self.vlm_model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.vlm_model_name, trust_remote_code=True)
        model.eval()

        # Move to same device as this module
        device = next(self.text_projection.parameters()).device
        model = model.to(device)

        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                ).to(device)
                outputs = model(**inputs)
                # Use [CLS] token embedding as sentence representation
                emb = outputs.last_hidden_state[:, 0, :]     # (1, D)
                embeddings.append(emb.squeeze(0))

        return F.normalize(torch.stack(embeddings), p=2, dim=-1)  # (K, D_text)

    def get_text_anchors(self) -> torch.Tensor:
        """
        Project cached text embeddings to visual space.

        Returns:
            anchors: (K, D_visual) L2-normalized text anchors in visual space
        """
        assert self.text_embeddings is not None, "Call encode_concepts() first."
        return self.text_projection(self.text_embeddings)    # (K, D_visual)

    def compute_alignment_loss(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between visual prototypes and text anchors.

        Args:
            prototypes: (K, M, D_visual) L2-normalized

        Returns:
            scalar loss
        """
        anchors = self.get_text_anchors()                    # (K, D_visual)
        return self.alignment_loss(prototypes, anchors)

    def zero_shot_prototype(self, description: str) -> torch.Tensor:
        """
        Generate a prototype for a NEW concept at test time using only
        its text description. No retraining required.

        Args:
            description: clinical text description of the new concept

        Returns:
            prototype: (D_visual,) — can be directly added to the atlas
        """
        try:
            emb = self._encode_with_biovil([description])    # (1, D_text)
        except Exception:
            emb = F.normalize(torch.randn(1, self.text_dim), p=2, dim=-1)

        with torch.no_grad():
            prototype = self.text_projection(emb.squeeze(0)) # (D_visual,)

        return prototype
