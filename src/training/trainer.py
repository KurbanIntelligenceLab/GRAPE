"""
Multi-stage training loop for CSR++.

Stages:
  1 — Train concept model (backbone + CAM heads) with BCE loss
  2 — Generate local concept vectors {v_k^i} for all training images (no training)
  3 — Train projector P + prototypes {p_km} with contrastive + alignment loss
  4 — Train task head (GNN or linear) with cross-entropy loss

Each stage is encapsulated in its own method and can be run independently,
allowing checkpointing and resuming mid-pipeline.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from ..models.csr_baseline import CSRModel
from ..training.losses import ConceptBCELoss, PrototypeLoss, ClassificationLoss
from ..utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from ..utils.metrics import macro_f1, pointing_game


class CSRTrainer:
    """
    Orchestrates all four training stages and evaluation.

    A100 optimizations:
      - BF16 mixed precision (native A100 support, no overflow issues)
      - torch.compile() on the feature extractor for Stage 1 speedup
      - Gradient scaling for stable mixed-precision training

    Usage:
        trainer = CSRTrainer(model, config, device)
        trainer.run_stage1(train_loader, val_loader)
        trainer.run_stage2(train_loader)
        trainer.run_stage3(train_loader)
        trainer.run_stage4(train_loader, val_loader)
        trainer.evaluate(test_loader)
    """

    def __init__(
        self,
        model: CSRModel,
        config: dict,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
        compile_model: bool = False,   # torch.compile — disable if causing issues
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        # BF16 AMP: A100 has native BF16 support (better than FP16 for training stability)
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.scaler = GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        if compile_model and hasattr(torch, "compile"):
            print("  Compiling feature extractor with torch.compile...")
            self.model.concept_model.feature_extractor = torch.compile(
                self.model.concept_model.feature_extractor, mode="reduce-overhead"
            )
        if self.use_amp:
            print(f"  Mixed precision: {self.amp_dtype} (AMP enabled)")
        else:
            print("  Mixed precision: disabled (CPU mode)")
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.concept_bce = ConceptBCELoss()
        self.proto_loss = PrototypeLoss(
            lambda_align=config.get("vlm", {}).get("lambda_align", 0.1),
            use_vlm=config.get("improvements", {}).get("use_vlm", True),
        )
        self.cls_loss = ClassificationLoss(
            label_smoothing=config.get("training", {}).get("label_smoothing", 0.0)
        )

        self.history = {
            "stage1_train_loss": [], "stage1_val_f1": [],
            "stage3_train_loss": [],
            "stage4_train_loss": [], "stage4_val_f1": [],
        }

    # ── Stage 1: Concept Model ────────────────────────────────────────────────

    def run_stage1(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Train backbone + CAM heads to predict concept presence via BCE loss.
        Freezes the task head and prototype learner during this stage.
        """
        print("\n" + "=" * 60)
        print("STAGE 1: Concept Model Training")
        print("=" * 60)

        cfg = self.config["training"]
        params = self.model.get_stage_parameters(stage=1)
        optimizer = AdamW(params, lr=cfg["stage1_lr"], weight_decay=cfg["stage1_weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["stage1_epochs"])

        for epoch in range(cfg["stage1_epochs"]):
            self.model.train()
            epoch_loss = self._train_stage1_epoch(train_loader, optimizer)
            scheduler.step()
            self.history["stage1_train_loss"].append(epoch_loss)

            val_info = ""
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_f1 = self._eval_concept_model(val_loader)
                self.history["stage1_val_f1"].append(val_f1)
                val_info = f"  val_concept_f1={val_f1:.4f}"

            print(f"  Epoch {epoch+1:3d}/{cfg['stage1_epochs']}  "
                  f"loss={epoch_loss:.4f}{val_info}")

        self._save_checkpoint("stage1")
        print("Stage 1 complete.\n")

    def _train_stage1_epoch(self, loader: DataLoader, optimizer) -> float:
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device)
            concept_labels = batch["concept_labels"].to(self.device)

            optimizer.zero_grad()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                _, _, concept_logits = self.model.concept_model(images)
                loss = self.concept_bce(concept_logits, concept_labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    # ── Stage 2: Local Concept Vector Generation ──────────────────────────────

    def run_stage2(self, train_loader: DataLoader) -> torch.Tensor:
        """
        Generate local concept vectors {v_k^i} for all training images.
        No training — pure inference pass using the Stage 1 concept model.

        Returns:
            all_vectors: (N, K, D) local concept vectors for all N training images
        """
        print("\n" + "=" * 60)
        print("STAGE 2: Local Concept Vector Generation")
        print("=" * 60)

        self.model.eval()
        all_vectors = []

        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(self.device)
                v = self.model.concept_model.generate_concept_vectors(images)  # (B, K, C)
                # Project to proto_dim
                B, K, C = v.shape
                v_flat = v.view(B * K, C)
                v_proj = self.model.projector(v_flat).view(B, K, -1)
                all_vectors.append(v_proj.cpu())

        all_vectors = torch.cat(all_vectors, dim=0)                            # (N, K, D)
        print(f"  Generated {all_vectors.shape[0]} concept vector sets.")

        # Initialize prototypes from these vectors
        self.model.prototype_learner.initialize_from_vectors(all_vectors)
        print("  Prototypes initialized from concept vectors.")

        return all_vectors

    # ── Stage 3: Prototype + Projector Learning ───────────────────────────────

    def run_stage3(self, train_loader: DataLoader):
        """
        Train projector P and prototypes {p_km} with:
          L = L_con-m  +  λ_align * L_align   (if use_vlm=True)
        """
        print("\n" + "=" * 60)
        print("STAGE 3: Prototype Learning")
        print("=" * 60)

        cfg = self.config["training"]
        params = self.model.get_stage_parameters(stage=3)
        optimizer = AdamW(params, lr=cfg["stage3_lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["stage3_epochs"])

        for epoch in range(cfg["stage3_epochs"]):
            self.model.train()
            epoch_loss = self._train_stage3_epoch(train_loader, optimizer)
            scheduler.step()
            self.history["stage3_train_loss"].append(epoch_loss)
            print(f"  Epoch {epoch+1:3d}/{cfg['stage3_epochs']}  loss={epoch_loss:.4f}")

        self._save_checkpoint("stage3")
        print("Stage 3 complete.\n")

    def _train_stage3_epoch(self, loader: DataLoader, optimizer) -> float:
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device)
            concept_labels = batch["concept_labels"].to(self.device)

            optimizer.zero_grad()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                # Get projected concept vectors
                with torch.no_grad():
                    v_raw = self.model.concept_model.generate_concept_vectors(images)

                B, K, C = v_raw.shape
                v_flat = v_raw.view(B * K, C)
                v_proj = self.model.projector(v_flat).view(B, K, -1)

                con_loss = self.model.prototype_learner.contrastive_loss(v_proj, concept_labels)

                align_loss = None
                if self.model.use_vlm and hasattr(self.model, "vlm_aligner"):
                    protos = self.model.prototype_learner.normalized_prototypes
                    align_loss = self.model.vlm_aligner.compute_alignment_loss(protos)

                loss = self.proto_loss(con_loss, align_loss)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    # ── Stage 4: Task Head Training ───────────────────────────────────────────

    def run_stage4(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        build_graph_from: DataLoader = None,
    ):
        """
        Train GNN (or linear) task head on similarity scores.

        Args:
            train_loader:    training data
            val_loader:      optional validation data for monitoring
            build_graph_from: if provided and use_gnn=True, build the concept
                             co-occurrence graph from this loader's labels
                             (usually the training loader)
        """
        print("\n" + "=" * 60)
        print("STAGE 4: Task Head Training")
        print("=" * 60)

        # Build and register concept graph for GNN
        if self.model.use_gnn:
            graph_loader = build_graph_from or train_loader
            self._build_and_register_graph(graph_loader)

        cfg = self.config["training"]
        params = self.model.get_stage_parameters(stage=4)
        optimizer = AdamW(params, lr=cfg["stage4_lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["stage4_epochs"])

        best_val_f1 = 0.0
        for epoch in range(cfg["stage4_epochs"]):
            self.model.train()
            epoch_loss = self._train_stage4_epoch(train_loader, optimizer)
            scheduler.step()
            self.history["stage4_train_loss"].append(epoch_loss)

            val_info = ""
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_f1 = self._eval_task_head(val_loader)
                self.history["stage4_val_f1"].append(val_f1)
                val_info = f"  val_f1={val_f1:.4f}"
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    self._save_checkpoint("stage4_best")

            print(f"  Epoch {epoch+1:3d}/{cfg['stage4_epochs']}  "
                  f"loss={epoch_loss:.4f}{val_info}")

        self._save_checkpoint("stage4")
        print(f"Stage 4 complete. Best val F1: {best_val_f1:.4f}\n")

    def _train_stage4_epoch(self, loader: DataLoader, optimizer) -> float:
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device)
            class_labels = batch["class_label"].to(self.device)

            optimizer.zero_grad()
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                with torch.no_grad():
                    f, _, _ = self.model.concept_model(images)
                    f_prime = self.model.projector.project_feature_map(f)
                    sim_scores = self.model.prototype_learner.get_similarity_scores(f_prime)

                if self.model.use_gnn:
                    logits = self.model.task_head(sim_scores)
                else:
                    B = sim_scores.shape[0]
                    logits = self.model.task_head(sim_scores.view(B, -1))

                loss = self.cls_loss(logits, class_labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _build_and_register_graph(self, loader: DataLoader):
        """Build concept co-occurrence graph from all training labels."""
        print("  Building concept co-occurrence graph...")
        all_labels = []
        for batch in loader:
            all_labels.append(batch["concept_labels"])
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        tau = self.config.get("gnn", {}).get("graph_threshold", 0.1)
        edge_index, edge_weight = build_cooccurrence_graph(all_labels, threshold=tau)
        edge_weight = normalize_edge_weights(edge_index, edge_weight, all_labels.shape[1])

        self.model.set_concept_graph(edge_index, edge_weight)
        print(f"  Graph built: {all_labels.shape[1]} nodes, {edge_index.shape[1]} edges "
              f"(threshold τ={tau})")

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Full evaluation on the test set.

        Returns:
            dict with 'macro_f1' and optionally 'pointing_game' metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        self.model.eval()
        all_preds, all_targets = [], []
        all_sim_maps, all_bboxes = [], []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                class_labels = batch["class_label"]

                out = self.model(images, return_maps=True)
                all_preds.append(out["logits"].cpu())
                all_targets.append(class_labels)

                # Collect similarity maps for Pointing Game (max over M prototypes)
                if "sim_maps" in out:
                    maps = out["sim_maps"].cpu()                # (B, K, M, H, W)
                    maps_max = maps.amax(dim=2)                 # (B, K, H, W)
                    all_sim_maps.append(maps_max)
                    all_bboxes.extend(batch["bbox"])

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        f1 = macro_f1(all_preds, all_targets)
        print(f"  Macro F1: {f1:.4f}")

        results = {"macro_f1": f1}

        if all_sim_maps:
            sim_maps_tensor = torch.cat(all_sim_maps)           # (N, K, H, W)
            pg = pointing_game(sim_maps_tensor, all_bboxes)
            results["pointing_game"] = pg["hit_rate"]
            print(f"  Pointing Game: {pg['hit_rate']:.4f} ({pg['num_evaluated']} evaluated)")

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _eval_concept_model(self, loader: DataLoader) -> float:
        """Quick concept-level F1 for Stage 1 monitoring."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                _, _, logits = self.model.concept_model(images)
                all_preds.append(torch.sigmoid(logits).cpu())
                all_labels.append(batch["concept_labels"])
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        # Use argmax of concept scores as a proxy class
        return macro_f1(preds, labels.argmax(dim=-1))

    def _eval_task_head(self, loader: DataLoader) -> float:
        """Macro F1 for Stage 4 monitoring."""
        self.model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                out = self.model(images)
                all_preds.append(out["logits"].cpu())
                all_targets.append(batch["class_label"])
        return macro_f1(torch.cat(all_preds), torch.cat(all_targets))

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"csr_{tag}.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, tag: str):
        path = os.path.join(self.checkpoint_dir, f"csr_{tag}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        print(f"  Loaded checkpoint: {path}")
