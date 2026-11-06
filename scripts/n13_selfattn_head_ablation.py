"""
N13 -- Self-Attention Task Head Ablation (Table 29)

Would a Transformer-style self-attention head over concept nodes outperform the
GAT, which requires a pre-defined adjacency matrix?

Extends n6_mlp_head_ablation_v2.py's protocol exactly: same frozen Stage 1-3
checkpoint, same Stage-4-only retraining, same parameter-matching discipline
(~47K params, matching the GAT head's ~43-47K). Adds a fourth head
(self-attention) alongside Linear / MLP / GAT already in Table 27.

Usage:
  python3 scripts/n13_selfattn_head_ablation.py --dataset tbx11k --data_dir data/tbx11k \
      --stage3_ckpt checkpoints/tbx11k_seed42/csr_stage3.pt \
      --gat_ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --out_dir checkpoints/tbx11k_n13 --results_path results/n13_selfattn_tbx11k_seed42.json --seed 42
"""

import sys, os, json, argparse, math
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1


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
    p.add_argument("--graph_tau", type=float, default=None)
    p.add_argument("--pg_concept", type=int, default=0)
    return p.parse_args()


class SelfAttentionTaskHead(nn.Module):
    """
    Single-layer multi-head self-attention over concept nodes, no predefined
    adjacency -- each concept attends freely to every other concept, letting
    the data determine which relationships matter rather than fixing them
    via a co-occurrence graph. Parameter-matched to the GAT head (~43-47K).

    sim_scores (B, K, M) -> per-concept node embedding (B, K, d_model)
      -> multi-head self-attention (B, K, d_model)
      -> flatten + linear readout -> (B, num_classes)
    """

    def __init__(self, num_concepts, num_prototypes, num_classes,
                 d_model=96, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model

        self.embed = nn.Linear(num_prototypes, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_concepts * d_model, num_classes),
        )

    def forward(self, sim_scores, edge_index=None, edge_weight=None):
        # sim_scores may arrive flattened (B, K*M) when use_gnn=False
        # (CSRModel.forward flattens before calling non-GNN heads) or as
        # (B, K, M) -- handle both, reshape to (B, K, M) for attention.
        if sim_scores.dim() == 2:
            B = sim_scores.shape[0]
            sim_scores = sim_scores.view(B, self.num_concepts, -1)
        h = self.embed(sim_scores)              # (B, K, d_model)
        attn_out, _ = self.attn(h, h, h)         # (B, K, d_model)
        h = self.norm(h + attn_out)              # residual + norm
        return self.readout(h.flatten(1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS[args.dataset])
    NUM_CLASSES = len(CLASS_NAMES[args.dataset])
    GRAPH_TAU = args.graph_tau if args.graph_tau is not None else (0.10 if args.dataset == "tbx11k" else 0.05)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"Dataset: {args.dataset}  K={NUM_CONCEPTS}  C={NUM_CLASSES}  tau={GRAPH_TAU}  device={DEVICE}")

    stage3_ckpt = torch.load(args.stage3_ckpt, map_location=DEVICE, weights_only=False)
    cfg = stage3_ckpt.get("config", {})
    proto_dim = cfg.get("model", {}).get("proto_dim", 256)
    num_proto = cfg.get("model", {}).get("num_prototypes", 100)
    backbone = cfg.get("model", {}).get("backbone", "resnet50")

    print("Loading datasets...")
    train_ds = CSRDataset(args.data_dir, args.dataset, split="train", image_size=args.image_size)
    val_ds = CSRDataset(args.data_dir, args.dataset, split="val", image_size=args.image_size)
    test_ds = CSRDataset(args.data_dir, args.dataset, split="test", image_size=args.image_size)
    try:
        bbox_ds = CSRDataset(args.data_dir, args.dataset, split="bbox_eval", image_size=args.image_size)
    except Exception as e:
        print(f"  No bbox_eval split available ({e}); Pointing Game will be skipped.")
        bbox_ds = None

    train_loader = get_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = get_dataloader(val_ds, args.batch_size, args.num_workers, shuffle=False)
    test_loader = get_dataloader(test_ds, args.batch_size, args.num_workers, shuffle=False)

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
            imgs = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            out = model(imgs)
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
            hits += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
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
        print(f"\n-- {head_name} Stage 4 training -- trainable params: {trainable:,}")

        for ep in range(args.epochs):
            model.train()
            for batch in train_loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["class_label"].to(DEVICE)
                opt.zero_grad()
                out = model(imgs)
                loss = criterion(out["logits"], labels)
                loss.backward()
                opt.step()
            val_f1 = evaluate(model, val_loader)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                print(f"   Ep {ep+1:02d}/{args.epochs}  val_F1={val_f1:.4f}  best={best_val_f1:.4f}")

        model.load_state_dict(best_state)
        return model

    print("\n=== Self-Attention Task Head ===")
    model_sa = load_stage3_base()
    model_sa.task_head = SelfAttentionTaskHead(NUM_CONCEPTS, num_proto, NUM_CLASSES).to(DEVICE)
    sa_params = model_sa.task_head.count_params()
    print(f"Self-attention params: {sa_params:,}")
    model_sa = train_stage4(model_sa, "SelfAttention")
    test_f1_sa = evaluate(model_sa, test_loader)
    pg_sa = eval_pg(model_sa, bbox_ds, args.pg_concept)
    print(f"SelfAttn -> test F1={test_f1_sa:.4f}  PG={pg_sa:.4f}")
    torch.save({"model_state": model_sa.state_dict()},
               os.path.join(args.out_dir, "selfattn_stage4_best.pt"))

    print("\n=== GAT Task Head (for reference, from existing checkpoint) ===")
    model_gat = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
        backbone=backbone, proto_dim=proto_dim,
        use_gnn=True, use_uncertainty=False, use_vlm=False,
        pretrained_backbone=False,
    ).to(DEVICE)
    gat_ckpt = torch.load(args.gat_ckpt, map_location=DEVICE, weights_only=False)
    model_gat.load_state_dict(gat_ckpt["model_state"], strict=False)
    all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
    ei, ew = build_cooccurrence_graph(all_labels, threshold=GRAPH_TAU)
    ew_norm = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
    model_gat.set_concept_graph(ei, ew_norm)
    gat_params = sum(p.numel() for p in model_gat.task_head.parameters())
    test_f1_gat = evaluate(model_gat, test_loader)
    pg_gat = eval_pg(model_gat, bbox_ds, args.pg_concept)
    print(f"GAT (ref) -> test F1={test_f1_gat:.4f}  PG={pg_gat:.4f}  params={gat_params:,}")

    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "num_concepts": NUM_CONCEPTS,
        "num_classes": NUM_CLASSES,
        "SelfAttention (matched)": {"f1": float(test_f1_sa), "pg": float(pg_sa), "params": sa_params},
        "GAT (GRAPE, reference)": {"f1": float(test_f1_gat), "pg": float(pg_gat), "params": gat_params},
    }

    print("\n=== N13 Summary ===")
    print(f"{'Head':<24} {'F1':>8} {'PG':>8} {'Params':>10}")
    print("-" * 54)
    for name, r in results.items():
        if isinstance(r, dict):
            print(f"{name:<24} {r['f1']:>8.4f} {r['pg']:>8.4f} {r['params']:>10,}")

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
