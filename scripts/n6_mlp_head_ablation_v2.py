"""
N6 v2 — GAT vs. Parameter-Matched MLP Task Head Ablation (dataset-agnostic)

Is the GAT gain from graph structure, or from
added nonlinearity over a flat linear head? Generalizes the original TBX11K-only
n6_mlp_head_ablation.py to run on either TBX11K (K=3) or NIH (K=14), loading a
Stage 1-3 checkpoint produced by train.py and re-training only Stage 4 for each
of: linear (CSR baseline), parameter-matched MLP, and GAT (GRAPE).

Usage:
  python3 scripts/n6_mlp_head_ablation_v2.py --dataset tbx11k --data_dir data/tbx11k \
      --stage3_ckpt checkpoints/tbx11k_seed42/csr_stage3.pt \
      --gat_ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --out_dir checkpoints/tbx11k_n6 --results_path results/n6_tbx11k_seed42.json --seed 42

  python3 scripts/n6_mlp_head_ablation_v2.py --dataset nih --data_dir data/nih_official \
      --stage3_ckpt checkpoints/nih_seed42/csr_stage3.pt \
      --gat_ckpt checkpoints/nih_seed42/csr_stage4_best.pt \
      --out_dir checkpoints/nih_n6 --results_path results/n6_nih_seed42.json --seed 42
"""

import sys, os, json, argparse
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, pointing_game


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tbx11k", "nih"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--stage3_ckpt", required=True)
    p.add_argument("--gat_ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--graph_tau", type=float, default=None,
                   help="Co-occurrence threshold; defaults to 0.10 (tbx11k) / 0.05 (nih)")
    p.add_argument("--pg_concept", type=int, default=0,
                   help="Concept index to evaluate Pointing Game on")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS[args.dataset])
    NUM_CLASSES  = len(CLASS_NAMES[args.dataset])
    GRAPH_TAU    = args.graph_tau if args.graph_tau is not None else (0.10 if args.dataset == "tbx11k" else 0.05)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"Dataset: {args.dataset}  K={NUM_CONCEPTS}  C={NUM_CLASSES}  tau={GRAPH_TAU}  device={DEVICE}")

    # ── Load Stage 3 checkpoint + embedded config ────────────────────────────
    stage3_ckpt = torch.load(args.stage3_ckpt, map_location=DEVICE, weights_only=False)
    cfg = stage3_ckpt.get("config", {})
    proto_dim   = cfg.get("model", {}).get("proto_dim", 256)
    num_proto   = cfg.get("model", {}).get("num_prototypes", 100)
    backbone    = cfg.get("model", {}).get("backbone", "resnet50")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = CSRDataset(args.data_dir, args.dataset, split="train", image_size=args.image_size)
    val_ds   = CSRDataset(args.data_dir, args.dataset, split="val",   image_size=args.image_size)
    test_ds  = CSRDataset(args.data_dir, args.dataset, split="test",  image_size=args.image_size)
    try:
        bbox_ds = CSRDataset(args.data_dir, args.dataset, split="bbox_eval", image_size=args.image_size)
    except Exception as e:
        print(f"  No bbox_eval split available ({e}); Pointing Game will be skipped.")
        bbox_ds = None

    train_loader = get_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader   = get_dataloader(val_ds,   args.batch_size, args.num_workers, shuffle=False)
    test_loader  = get_dataloader(test_ds,  args.batch_size, args.num_workers, shuffle=False)

    print("Building co-occurrence graph...")
    all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
    ei, ew  = build_cooccurrence_graph(all_labels, threshold=GRAPH_TAU)
    ew_norm = normalize_edge_weights(ei, ew, NUM_CONCEPTS)

    # ── Parameter-matched MLP task head ──────────────────────────────────────
    class MLPTaskHead(nn.Module):
        """2-layer MLP over flattened K*M prototype scores, sized to roughly
        match the GAT head's parameter count for the given K."""
        def __init__(self, num_concepts, num_prototypes, num_classes, hidden1=128, hidden2=64):
            super().__init__()
            in_dim = num_concepts * num_prototypes
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden1),
                nn.LayerNorm(hidden1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_classes),
            )

        def forward(self, sim_scores, edge_index=None, edge_weight=None):
            return self.net(sim_scores.flatten(1))

        def count_params(self):
            return sum(p.numel() for p in self.parameters())

    def load_stage3_base():
        m = CSRModel(
            num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
            backbone=backbone, proto_dim=proto_dim,
            use_gnn=False, use_uncertainty=False, use_vlm=False,
            pretrained_backbone=False,
        ).to(DEVICE)
        m.load_state_dict(stage3_ckpt["model_state"], strict=False)
        return m

    @torch.no_grad()
    def evaluate(model, loader):
        model.eval()
        all_logits, all_labels_list = [], []
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            out    = model(imgs)
            all_logits.append(out["logits"].cpu())
            all_labels_list.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels_list)
        return macro_f1(logits, labels, num_classes=NUM_CLASSES)

    @torch.no_grad()
    def eval_pg(model, bbox_ds_ref, concept_idx):
        if bbox_ds_ref is None:
            return 0.0
        model.eval()
        hits, total = 0, 0
        for i in range(len(bbox_ds_ref)):
            item = bbox_ds_ref[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if concept_idx not in bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            f, _, _ = model.concept_model(img)
            fp = model.projector.project_feature_map(f)
            sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()
            raw = sm[concept_idx].amax(0).numpy()
            H, W = raw.shape
            x1, y1, x2, y2 = bboxes[concept_idx]
            bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
            by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))
            idx = int(np.argmax(raw))
            mh, mw = divmod(idx, W)
            hits  += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
            total += 1
        return hits / total if total > 0 else 0.0

    def train_stage4(model, head_name):
        for name, p in model.named_parameters():
            p.requires_grad = "task_head" in name
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_val_f1, best_state = 0.0, None

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n── {head_name} Stage 4 training ── trainable params: {trainable:,}")

        for ep in range(args.epochs):
            model.train()
            for batch in train_loader:
                imgs   = batch["image"].to(DEVICE)
                labels = batch["class_label"].to(DEVICE)
                opt.zero_grad()
                out  = model(imgs)
                loss = criterion(out["logits"], labels)
                loss.backward()
                opt.step()
            val_f1 = evaluate(model, val_loader)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                print(f"   Ep {ep+1:02d}/{args.epochs}  val_F1={val_f1:.4f}  best={best_val_f1:.4f}")

        model.load_state_dict(best_state)
        return model

    # ── MLP head ──────────────────────────────────────────────────────────────
    print("\n=== MLP Task Head ===")
    model_mlp = load_stage3_base()
    model_mlp.task_head = MLPTaskHead(NUM_CONCEPTS, num_proto, NUM_CLASSES).to(DEVICE)
    print(f"MLP params: {model_mlp.task_head.count_params():,}")
    model_mlp = train_stage4(model_mlp, "MLP")
    test_f1_mlp = evaluate(model_mlp, test_loader)
    pg_mlp      = eval_pg(model_mlp, bbox_ds, args.pg_concept)
    print(f"MLP  → test F1={test_f1_mlp:.4f}  PG={pg_mlp:.4f}")
    torch.save({"model_state": model_mlp.state_dict()},
               os.path.join(args.out_dir, "mlp_stage4_best.pt"))

    # ── Linear head (CSR baseline) ────────────────────────────────────────────
    print("\n=== Linear Task Head (CSR) ===")
    model_lin = load_stage3_base()
    lin_params = sum(p.numel() for p in model_lin.task_head.parameters())
    print(f"Linear params: {lin_params:,}")
    model_lin = train_stage4(model_lin, "Linear")
    test_f1_lin = evaluate(model_lin, test_loader)
    pg_lin      = eval_pg(model_lin, bbox_ds, args.pg_concept)
    print(f"Linear → test F1={test_f1_lin:.4f}  PG={pg_lin:.4f}")

    # ── GAT head (load existing trained GRAPE checkpoint) ───────────────────
    print("\n=== GAT Task Head (GRAPE) ===")
    model_gat = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
        backbone=backbone, proto_dim=proto_dim,
        use_gnn=True, use_uncertainty=False, use_vlm=False,
        pretrained_backbone=False,
    ).to(DEVICE)
    gat_ckpt = torch.load(args.gat_ckpt, map_location=DEVICE, weights_only=False)
    model_gat.load_state_dict(gat_ckpt["model_state"], strict=False)
    model_gat.set_concept_graph(ei, ew_norm)
    gat_params = sum(p.numel() for p in model_gat.task_head.parameters())
    print(f"GAT params: {gat_params:,}")
    test_f1_gat = evaluate(model_gat, test_loader)
    pg_gat      = eval_pg(model_gat, bbox_ds, args.pg_concept)
    print(f"GAT  → test F1={test_f1_gat:.4f}  PG={pg_gat:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "num_concepts": NUM_CONCEPTS,
        "num_classes": NUM_CLASSES,
        "Linear (CSR)":  {"f1": float(test_f1_lin), "pg": float(pg_lin), "params": lin_params},
        "MLP (matched)": {"f1": float(test_f1_mlp), "pg": float(pg_mlp), "params": model_mlp.task_head.count_params()},
        "GAT (GRAPE)":   {"f1": float(test_f1_gat), "pg": float(pg_gat), "params": gat_params},
    }

    print("\n=== N6 Summary ===")
    print(f"{'Head':<18} {'F1':>8} {'PG':>8} {'Params':>10}")
    print("-" * 48)
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:<18} {r['f1']:>8.4f} {r['pg']:>8.4f} {r['params']:>10,}")

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
