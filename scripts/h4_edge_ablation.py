"""
H4 — GAT Edge-Importance Ablation (TBX11K, K=3)

For each of the 4 edges in the TBX11K co-occurrence graph, retrain Stage 4
with that edge removed from the graph.  Demonstrates which edges carry
task-meaningful information for the GAT head.

Results → results/h4_edge_ablation.json
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1
from src.training.trainer import CSRTrainer as Trainer

DATA_DIR  = "data/tbx11k"
CKPT_BASE = "checkpoints/tbx11k_no_C_novlm"
OUT_DIR   = "checkpoints/h4_edge_ablation"
RESULTS   = "results/h4_edge_ablation.json"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

NUM_CONCEPTS = 3
NUM_CLASSES  = 3
IMAGE_SIZE   = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("configs/tbx11k_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# ── Build base graph ──────────────────────────────────────────────────────────
print("Building base co-occurrence graph...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_ldr = get_dataloader(train_ds, 128, 4, shuffle=True)
val_ds  = CSRDataset(DATA_DIR, "tbx11k", split="val",  image_size=IMAGE_SIZE)
val_ldr = get_dataloader(val_ds,  128, 4, shuffle=False)
test_ds = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
test_ldr = get_dataloader(test_ds, 128, 4, shuffle=False)

probe_ldr = get_dataloader(train_ds, 128, 4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in probe_ldr]).to(DEVICE)
base_ei, base_ew = build_cooccurrence_graph(all_labels, threshold=0.10)
base_ew_norm = normalize_edge_weights(base_ei, base_ew, NUM_CONCEPTS)

print(f"Base graph: {base_ei.shape[1]} edges")
edge_names = []
for i in range(base_ei.shape[1]):
    src, dst = base_ei[0, i].item(), base_ei[1, i].item()
    edge_names.append(f"({src}→{dst})")
    print(f"  Edge {i}: {src}→{dst}  weight={base_ew_norm[i]:.4f}")


def load_stage123_weights():
    """Return model with Stages 1-3 frozen, task head freshly initialized."""
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt_path = os.path.join(CKPT_BASE, "csr_stage4_best.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # Load full state (includes old task head weights), then reset task head
    m.load_state_dict(ckpt["model_state"], strict=False)
    # Re-init task head so we train from scratch
    for layer in m.task_head.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return m


def train_with_graph(edge_index, edge_weight, run_tag):
    """Retrain Stage 4 only with the given graph; return test macro-F1."""
    m = load_stage123_weights()
    ew_norm = normalize_edge_weights(edge_index, edge_weight, NUM_CONCEPTS)
    target_ei = edge_index.to(DEVICE)
    target_ew = ew_norm.to(DEVICE)
    m.set_concept_graph(target_ei, target_ew)

    cfg_copy = copy.deepcopy(CONFIG)
    out_dir = os.path.join(OUT_DIR, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    trainer = Trainer(m, cfg_copy, DEVICE, checkpoint_dir=out_dir)
    # Monkey-patch graph builder so run_stage4 uses our modified graph
    # (not the full graph rebuilt from training labels)
    trainer._build_and_register_graph = lambda loader: m.set_concept_graph(
        target_ei, target_ew
    )

    # Freeze everything except task head
    for name, p in m.named_parameters():
        if "task_head" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainer.run_stage4(train_ldr, val_ldr)
    trainer.load_checkpoint("stage4_best")

    # Evaluate
    m.eval()
    preds, gt = [], []
    with torch.no_grad():
        for batch in test_ldr:
            out = m(batch["image"].to(DEVICE))
            logit = out["logits"] if isinstance(out, dict) else out
            preds.append(logit.argmax(-1).cpu())
            gt.append(batch["class_label"])
    f1 = float(macro_f1(torch.cat(preds), torch.cat(gt), num_classes=NUM_CLASSES))
    return f1


# ── Full graph (baseline) ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("H4 Edge Ablation")
print("=" * 60)
print("\nTraining full graph (all 4 edges)...")
f1_full = train_with_graph(base_ei.clone(), base_ew_norm.clone(), "full")
print(f"Full graph F1: {f1_full:.4f}")

results = {"full_graph_f1": f1_full, "edges": {}}

# ── One-edge-removed ablations ───────────────────────────────────────────────
for edge_idx in range(base_ei.shape[1]):
    keep = [j for j in range(base_ei.shape[1]) if j != edge_idx]
    ablated_ei = base_ei[:, keep]
    ablated_ew = base_ew[keep]
    tag = f"remove_edge{edge_idx}"
    src, dst = base_ei[0, edge_idx].item(), base_ei[1, edge_idx].item()
    name = f"{src}→{dst}"
    print(f"\nRemoving edge {edge_idx} ({name}), retraining Stage 4...")
    f1 = train_with_graph(ablated_ei, ablated_ew, tag)
    delta = f1 - f1_full
    print(f"  F1: {f1:.4f}  ΔF1: {delta:+.4f}")
    results["edges"][f"edge{edge_idx}"] = {
        "src": src, "dst": dst, "name": name,
        "f1": f1, "delta_f1": delta,
    }

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("H4 Summary")
print("=" * 60)
print(f"Full graph F1: {f1_full:.4f}")
for k, v in results["edges"].items():
    print(f"  Remove {v['name']}: F1={v['f1']:.4f}  ΔF1={v['delta_f1']:+.4f}")

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nH4 done. Results → {RESULTS}")
