"""
N9 -- Robustness of the static co-occurrence graph under distribution shift.

Protocol: resample the NIH training set to
shift concept prevalence and co-occurrence structure, rebuild the
co-occurrence graph from that resampled set, retrain Stage 4 (GAT task head)
only from the frozen Stage 1-3 checkpoint, and compare macro-F1 and graph
structure (edges, self-loop vs. cross-concept weight) against the original
graph and Stage-4 result.

Resampling: oversample images containing any of a chosen "shifted" concept
subset (simulating an institution/case-mix with a different disease
prevalence than the original training distribution), by sampling with
replacement so that the shifted concepts' prevalence roughly doubles while
keeping the total training-set size fixed. This changes both marginal
prevalence and pairwise co-occurrence counts, since concepts that co-occur
with the shifted set become relatively more frequent too.

Usage:
  python3 scripts/n9_prevalence_shift.py \
      --stage1_ckpt checkpoints/nih_seed42/csr_stage1.pt \
      --stage3_ckpt checkpoints/nih_seed42/csr_stage3.pt \
      --train_data_dir data/nih_official --test_data_dir data/nih_official \
      --results_path results/n9_prevalence_shift_nih.json
"""

import sys, os, json, random, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1

# Concepts to oversample -- chosen to plausibly co-vary in a shifted case mix
# (e.g. an infectious/inflammatory-disease-heavy institution): Infiltration,
# Consolidation, Pneumonia, Effusion. Indices per CONCEPTS["nih"].
SHIFTED_CONCEPTS = [8, 2, 12, 4]  # Infiltration, Consolidation, Pneumonia, Effusion
OVERSAMPLE_FACTOR = 3.0  # shifted-concept images sampled ~3x their base rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage3_ckpt", required=True)
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--graph_tau", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--stage4_epochs", type=int, default=20)
    p.add_argument("--stage4_lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS["nih"])
    NUM_CLASSES = len(CLASS_NAMES["nih"])
    concept_names = CONCEPTS["nih"]

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"K={NUM_CONCEPTS}  C={NUM_CLASSES}  shifted concepts: "
          f"{[concept_names[i] for i in SHIFTED_CONCEPTS]}  device={DEVICE}")

    stage3_ckpt = torch.load(args.stage3_ckpt, map_location=DEVICE, weights_only=False)
    cfg = stage3_ckpt.get("config", {})
    proto_dim = cfg.get("model", {}).get("proto_dim", 256)
    num_proto = cfg.get("model", {}).get("num_prototypes", 100)
    backbone = cfg.get("model", {}).get("backbone", "resnet50")

    print("Loading datasets...")
    train_ds = CSRDataset(args.train_data_dir, "nih", split="train", image_size=args.image_size)
    test_ds = CSRDataset(args.test_data_dir, "nih", split="test", image_size=args.image_size)
    print(f"  Train: {len(train_ds)}  Test: {len(test_ds)}")

    all_labels = torch.stack([s["concept_labels"] for s in train_ds.samples])

    def build_resampled_indices(labels, shifted_concepts, factor, seed):
        """Sample-with-replacement indices, oversampling images with any shifted concept."""
        rng = np.random.RandomState(seed)
        n = labels.shape[0]
        has_shifted = (labels[:, shifted_concepts].sum(dim=1) > 0).numpy()
        base_weight = 1.0
        weights = np.where(has_shifted, base_weight * factor, base_weight)
        weights = weights / weights.sum()
        idx = rng.choice(n, size=n, replace=True, p=weights)
        return idx

    def report_prevalence(labels, tag):
        prev = labels.float().mean(dim=0).numpy()
        print(f"  {tag} prevalence: " +
              ", ".join(f"{concept_names[i]}={prev[i]:.3f}" for i in SHIFTED_CONCEPTS))
        return prev

    print("\n=== Original training-set prevalence ===")
    report_prevalence(all_labels, "original")

    resampled_idx = build_resampled_indices(all_labels, SHIFTED_CONCEPTS, OVERSAMPLE_FACTOR, args.seed)
    resampled_labels = all_labels[resampled_idx]
    print("\n=== Resampled (shifted) training-set prevalence ===")
    report_prevalence(resampled_labels, "shifted")

    # ── Build both co-occurrence graphs ──────────────────────────────────────
    ei_orig, ew_orig = build_cooccurrence_graph(all_labels.to(DEVICE), threshold=args.graph_tau)
    ew_orig_norm = normalize_edge_weights(ei_orig, ew_orig, NUM_CONCEPTS)

    ei_shift, ew_shift = build_cooccurrence_graph(resampled_labels.to(DEVICE), threshold=args.graph_tau)
    ew_shift_norm = normalize_edge_weights(ei_shift, ew_shift, NUM_CONCEPTS)

    print(f"\nOriginal graph: {ei_orig.shape[1]} edges")
    print(f"Shifted graph:  {ei_shift.shape[1]} edges")

    def edge_set(ei):
        return set(zip(ei[0].tolist(), ei[1].tolist()))

    orig_edges = edge_set(ei_orig)
    shift_edges = edge_set(ei_shift)
    added = shift_edges - orig_edges
    removed = orig_edges - shift_edges
    print(f"Edges added under shift: {len(added)}  ({[(concept_names[a], concept_names[b]) for a, b in added]})")
    print(f"Edges removed under shift: {len(removed)}  ({[(concept_names[a], concept_names[b]) for a, b in removed]})")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    from torch.utils.data import Subset
    resampled_train_ds = Subset(train_ds, resampled_idx.tolist())

    def make_model():
        m = CSRModel(
            num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
            backbone=backbone, proto_dim=proto_dim,
            use_gnn=True, use_uncertainty=True, use_vlm=True,
            pretrained_backbone=False,
        ).to(DEVICE)
        m.load_state_dict(stage3_ckpt["model_state"], strict=False)
        return m

    @torch.no_grad()
    def eval_f1(model, loader):
        model.eval()
        all_logits, all_labels_list = [], []
        for batch in loader:
            imgs = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            out = model(imgs)
            all_logits.append(out["logits"].cpu())
            all_labels_list.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels_list)
        return macro_f1(logits, labels, num_classes=NUM_CLASSES)

    def train_stage4(model, loader, ei, ew):
        model.set_concept_graph(ei, ew)
        for name, p in model.named_parameters():
            p.requires_grad = "task_head" in name
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.stage4_lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.stage4_epochs)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

        for ep in range(args.stage4_epochs):
            model.train()
            for batch in loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["class_label"].to(DEVICE)
                opt.zero_grad()
                out = model(imgs)
                loss = criterion(out["logits"], labels)
                loss.backward()
                opt.step()
            scheduler.step()
        model.eval()
        return model

    test_loader = get_dataloader(test_ds, args.batch_size, args.num_workers, shuffle=False)

    # ── Original graph, Stage 4 retrain on ORIGINAL training set ─────────────
    print("\n=== Stage 4 on original training set / original graph ===")
    orig_loader = get_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    model_orig = make_model()
    model_orig = train_stage4(model_orig, orig_loader, ei_orig, ew_orig_norm)
    f1_orig = eval_f1(model_orig, test_loader)
    print(f"Original: test F1={f1_orig:.4f}")

    # ── Shifted graph, Stage 4 retrain on RESAMPLED training set ─────────────
    print("\n=== Stage 4 on resampled training set / shifted graph ===")
    shift_loader = get_dataloader(resampled_train_ds, args.batch_size, args.num_workers, shuffle=True)
    model_shift = make_model()
    model_shift = train_stage4(model_shift, shift_loader, ei_shift, ew_shift_norm)
    f1_shift_on_shift_test = eval_f1(model_shift, test_loader)
    print(f"Shifted (eval on original test dist): test F1={f1_shift_on_shift_test:.4f}")

    # ── Cross check: shifted-graph model evaluated normally; original-graph
    #    model would perform under a shifted graph if deployed without
    #    retraining (the actual "robustness" question) ────────────────────────
    print("\n=== Original-trained Stage 4 head evaluated WITH the shifted graph swapped in ===")
    model_orig.set_concept_graph(ei_shift, ew_shift_norm)
    f1_orig_with_shifted_graph = eval_f1(model_orig, test_loader)
    print(f"Original Stage-4 weights + shifted graph: test F1={f1_orig_with_shifted_graph:.4f}")

    results = {
        "shifted_concepts": [concept_names[i] for i in SHIFTED_CONCEPTS],
        "oversample_factor": OVERSAMPLE_FACTOR,
        "graph_tau": args.graph_tau,
        "original_graph_edges": ei_orig.shape[1],
        "shifted_graph_edges": ei_shift.shape[1],
        "edges_added_under_shift": [[concept_names[a], concept_names[b]] for a, b in added],
        "edges_removed_under_shift": [[concept_names[a], concept_names[b]] for a, b in removed],
        "f1_original_graph_original_data": f1_orig,
        "f1_shifted_graph_shifted_data": f1_shift_on_shift_test,
        "f1_original_weights_shifted_graph": f1_orig_with_shifted_graph,
        "delta_retrained_on_shift": f1_shift_on_shift_test - f1_orig,
        "delta_graph_swap_no_retrain": f1_orig_with_shifted_graph - f1_orig,
    }

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== N9 Summary ===")
    print(f"Original graph/data:              F1={f1_orig:.4f}")
    print(f"Retrained on shifted graph/data:  F1={f1_shift_on_shift_test:.4f}  "
          f"(Δ={results['delta_retrained_on_shift']:+.4f})")
    print(f"Original weights + shifted graph: F1={f1_orig_with_shifted_graph:.4f}  "
          f"(Δ={results['delta_graph_swap_no_retrain']:+.4f})")
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
