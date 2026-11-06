"""
N2 — Per-class F1 and Confusion Matrix on TBX11K

Computes per-class F1 and confusion matrix for GRAPE (best: GNN+Uncert, no VLM)
and CSR baseline across all 5 seeds {42, 0, 1, 2, 3}.

Reports:
  - Mean ± std per-class F1 for each model
  - Single-seed (seed 42) confusion matrix
  - Saves results/n2_perclass_f1.json
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR   = "data/tbx11k"
CKPT_BASE  = "checkpoints"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
SEEDS = [42, 0, 1, 2, 3]
CLASS_NAMES = ["healthy", "sick_but_non_tb", "active_tb"]

GRAPE_CKPTS = {
    42: "tbx11k_no_C_novlm",
    0:  "tbx11k_grape_seed0",
    1:  "tbx11k_grape_seed1",
    2:  "tbx11k_grape_seed2",
    3:  "tbx11k_grape_seed3",
}
CSR_CKPTS = {
    42: "tbx11k_baseline_csr",
    0:  "tbx11k_csr_seed0",
    1:  "tbx11k_csr_seed1",
    2:  "tbx11k_csr_seed2",
    3:  "tbx11k_csr_seed3",
}


def load_model(ckpt_name, use_gnn):
    path = os.path.join(CKPT_BASE, ckpt_name, "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_gnn,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    if use_gnn:
        train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
        loader   = get_dataloader(train_ds, 128, 4, shuffle=False)
        labels   = torch.cat([b["concept_labels"] for b in loader]).to(DEVICE)
        ei, ew   = build_cooccurrence_graph(labels, threshold=0.10)
        ew_norm  = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
        m.set_concept_graph(ei, ew_norm)
    return m


def evaluate(model):
    test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    loader   = get_dataloader(test_ds, 128, 4, shuffle=False)
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            out   = model(batch["image"].to(DEVICE))
            logit = out["logits"] if isinstance(out, dict) else out
            preds.append(logit.argmax(-1).cpu())
            gts.append(batch["class_label"])
    preds = torch.cat(preds).numpy()
    gts   = torch.cat(gts).numpy()
    return preds, gts


def per_class_f1(preds, gts, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        f1s.append(float(f1))
    return f1s


def confusion_matrix(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    return cm.tolist()


results = {"grape": {}, "csr": {}}
cm_seed42 = {"grape": None, "csr": None}

for variant, ckpt_map, use_gnn in [("csr", CSR_CKPTS, False), ("grape", GRAPE_CKPTS, True)]:
    print(f"\n=== {variant.upper()} ===")
    for seed in SEEDS:
        ckpt_name = ckpt_map[seed]
        path = os.path.join(CKPT_BASE, ckpt_name, "csr_stage4_best.pt")
        if not os.path.exists(path):
            print(f"  seed {seed}: missing checkpoint — skip")
            continue
        print(f"  seed {seed} ({ckpt_name}) ...", end=" ", flush=True)
        model = load_model(ckpt_name, use_gnn)
        preds, gts = evaluate(model)
        f1s = per_class_f1(preds, gts, NUM_CLASSES)
        results[variant][seed] = {"per_class_f1": f1s}
        if seed == 42:
            cm_seed42[variant] = confusion_matrix(preds, gts, NUM_CLASSES)
        print(f"F1={np.mean(f1s):.4f}  per-class={[f'{v:.3f}' for v in f1s]}")
        del model

# Aggregate across seeds
print("\n" + "="*60)
print("N2 Summary — Per-class F1 (mean ± std over 5 seeds)")
print("="*60)

out = {"class_names": CLASS_NAMES, "confusion_matrix_seed42": cm_seed42}
for variant in ["csr", "grape"]:
    all_f1s = [results[variant][s]["per_class_f1"] for s in SEEDS if s in results[variant]]
    if not all_f1s:
        continue
    arr = np.array(all_f1s)   # (n_seeds, n_classes)
    means = arr.mean(0)
    stds  = arr.std(0)
    macro_mean = means.mean()
    print(f"\n{variant.upper()}:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {means[i]:.4f} ± {stds[i]:.4f}")
    print(f"  {'macro':20s}: {macro_mean:.4f}")
    out[variant] = {
        "per_class_f1_mean": means.tolist(),
        "per_class_f1_std":  stds.tolist(),
        "per_seed": {str(s): results[variant][s] for s in SEEDS if s in results[variant]},
    }

print("\nConfusion matrix seed 42 (rows=GT, cols=Pred):")
for variant in ["grape", "csr"]:
    if cm_seed42[variant]:
        print(f"\n{variant.upper()}:")
        print(f"  {'':20s} " + "  ".join(f"{n:>16}" for n in CLASS_NAMES))
        for i, row in enumerate(cm_seed42[variant]):
            print(f"  {CLASS_NAMES[i]:20s} " + "  ".join(f"{v:>16}" for v in row))

os.makedirs("results", exist_ok=True)
with open("results/n2_perclass_f1.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nN2 done. Results → results/n2_perclass_f1.json")
