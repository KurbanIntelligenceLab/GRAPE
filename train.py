"""
CSR++ Training Entry Point

Usage:
  # Full training on TBX11K with all improvements:
  python train.py --dataset tbx11k --data_dir /data/TBX11K

  # Ablation: GNN only, no VLM, no uncertainty:
  python train.py --dataset tbx11k --data_dir /data/TBX11K --no_vlm --no_uncertainty

  # Dry run with synthetic data (no real dataset needed):
  python train.py --dataset tbx11k --synthetic --epochs 2
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from src.models.csr_baseline import CSRModel
from src.training.trainer import CSRTrainer
from src.data.datasets import get_dataset, get_dataloader, SyntheticDataset, CONCEPTS, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSR++ model")

    # Dataset
    # tbx11k and nih are the paper datasets; vindrcxr/isic are unused scaffolding.
    parser.add_argument("--dataset", type=str, default="tbx11k",
                        choices=["tbx11k", "nih"])
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Root directory of the dataset")
    parser.add_argument("--image_size", type=int, default=224)

    # Synthetic mode (for testing without real data)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random data for testing the pipeline")
    parser.add_argument("--synthetic_n", type=int, default=128,
                        help="Number of synthetic samples")

    # Model
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--num_prototypes", type=int, default=100)
    parser.add_argument("--proto_dim", type=int, default=256)

    # Improvement toggles (ablation)
    parser.add_argument("--no_gnn", action="store_true", help="Disable [A] GNN task head")
    parser.add_argument("--no_uncertainty", action="store_true", help="Disable [B] uncertainty head")
    parser.add_argument("--no_vlm", action="store_true", help="Disable [C] VLM alignment")

    # Training
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override all stage epoch counts")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Config
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    # Stages to run (useful for resuming)
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated list of stages to run, e.g. '3,4' to skip Stage 1")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Load checkpoint before training (e.g. 'stage1')")

    return parser.parse_args()


def set_seed(seed: int):
    """Seed all RNGs (Python, NumPy, PyTorch CPU+CUDA) and request deterministic cuDNN.

    Some CUDA kernels remain nondeterministic, so runs are not guaranteed bit-for-bit
    identical, but variation stays within the std reported in the paper.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def build_model(args, config: dict, num_concepts: int, num_classes: int) -> CSRModel:
    return CSRModel(
        num_concepts=num_concepts,
        num_prototypes=args.num_prototypes,
        num_classes=num_classes,
        backbone=args.backbone,
        proto_dim=args.proto_dim,
        use_gnn=not args.no_gnn,
        use_uncertainty=not args.no_uncertainty,
        use_vlm=not args.no_vlm,
        gnn_hidden_dim=config.get("gnn", {}).get("hidden_dim", 64),
        gnn_num_heads=config.get("gnn", {}).get("num_heads", 4),
        vlm_text_dim=config.get("vlm", {}).get("text_dim", 768),
        vlm_lambda=config.get("vlm", {}).get("lambda_align", 0.1),
        vlm_model_name=config.get("vlm", {}).get("model_name", "microsoft/BiomedVLP-BioViL-T"),
    )


def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)
    print(f"Seed: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    config = load_config(args.config)

    # Override epochs if specified
    if args.epochs is not None:
        for key in ["stage1_epochs", "stage3_epochs", "stage4_epochs"]:
            config.setdefault("training", {})[key] = args.epochs

    # Override batch size
    config.setdefault("training", {})["batch_size"] = args.batch_size

    num_concepts = len(CONCEPTS[args.dataset])
    num_classes = len(CLASS_NAMES[args.dataset])

    print(f"\nDataset:      {args.dataset}  ({num_concepts} concepts, {num_classes} classes)")
    print(f"Improvements: GNN={'OFF' if args.no_gnn else 'ON'}  "
          f"Uncertainty={'OFF' if args.no_uncertainty else 'ON'}  "
          f"VLM={'OFF' if args.no_vlm else 'ON'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        print(f"\nUsing SYNTHETIC data ({args.synthetic_n} samples)")
        train_ds = SyntheticDataset(args.dataset, args.synthetic_n, args.image_size, "train")
        val_ds   = SyntheticDataset(args.dataset, args.synthetic_n // 4, args.image_size, "val")
        test_ds  = SyntheticDataset(args.dataset, args.synthetic_n // 4, args.image_size, "test")
    else:
        train_ds = get_dataset(args.dataset, args.data_dir, "train", args.image_size)
        val_ds   = get_dataset(args.dataset, args.data_dir, "val",   args.image_size)
        test_ds  = get_dataset(args.dataset, args.data_dir, "test",  args.image_size)

    train_loader = get_dataloader(train_ds, args.batch_size, args.num_workers)
    val_loader   = get_dataloader(val_ds,   args.batch_size, args.num_workers)
    test_loader  = get_dataloader(test_ds,  args.batch_size, args.num_workers)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── VLM concept encoding (Improvement C) ─────────────────────────────────
    if not args.no_vlm:
        from src.models.vlm_alignment import CONCEPT_DESCRIPTIONS
        concept_descriptions = CONCEPT_DESCRIPTIONS.get(args.dataset, {})
        if not concept_descriptions:
            print(f"Warning: no concept descriptions found for {args.dataset}, disabling VLM.")
            args.no_vlm = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args, config, num_concepts, num_classes)

    # Encode concept text embeddings before training starts
    if not args.no_vlm and hasattr(model, "vlm_aligner"):
        from src.models.vlm_alignment import CONCEPT_DESCRIPTIONS
        print("\nEncoding concept descriptions with VLM...")
        model.vlm_aligner.encode_concepts(CONCEPT_DESCRIPTIONS[args.dataset])

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = CSRTrainer(model, config, device, args.checkpoint_dir)

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    stages = [int(s) for s in args.stages.split(",")]
    print(f"\nRunning stages: {stages}\n")

    if 1 in stages:
        trainer.run_stage1(train_loader, val_loader)

    if 2 in stages:
        trainer.run_stage2(train_loader)

    if 3 in stages:
        trainer.run_stage3(train_loader)

    if 4 in stages:
        trainer.run_stage4(train_loader, val_loader, build_graph_from=train_loader)

    # ── Final evaluation ──────────────────────────────────────────────────────
    results = trainer.evaluate(test_loader)
    print("\nFinal results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
