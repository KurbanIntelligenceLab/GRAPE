"""
H2 — TBX11K Staged-Curriculum Training, 5 Seeds

The paper currently reports multi-seed results for seeds {42, 0, 1} (3 seeds).
This script extends to 5 seeds {42, 0, 1, 2, 3} for tighter confidence intervals.

For each seed:
  - GRAPE (GNN + Uncertainty, no VLM): best TBX11K configuration
  - CSR baseline

Evaluates macro-F1 and Pointing Game.
Reuses existing checkpoints for seeds already trained.

Results → results/h2_tbx11k_5seeds.json
Compute:  ~4 GPU-h (2 new seeds × 2 variants × ~1h each)
Updates:  Table 4 (tab:tbx11k) — replace 3-seed with 5-seed mean ± std.
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

DATA_DIR  = "data/tbx11k"
CKPT_BASE = "checkpoints"
LOG_BASE  = "logs"
RESULTS   = "results/h2_tbx11k_5seeds.json"

os.makedirs(LOG_BASE, exist_ok=True)
os.makedirs("results", exist_ok=True)

SEEDS        = [42, 0, 1, 2, 3]
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
IMAGE_SIZE   = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map variant names to existing checkpoint directories (seeds 42/0/1 already trained)
EXISTING_CKPT = {
    ("grape", 42): "tbx11k_no_C_novlm",     # best GRAPE config from ablations
    ("grape",  0): "tbx11k_grape_seed0",
    ("grape",  1): "tbx11k_grape_seed1",
    ("csr",   42): "tbx11k_baseline_csr",
    ("csr",    0): "tbx11k_csr_seed0",
    ("csr",    1): "tbx11k_csr_seed1",
}


def ckpt_dir(variant, seed):
    if (variant, seed) in EXISTING_CKPT:
        return os.path.join(CKPT_BASE, EXISTING_CKPT[(variant, seed)])
    return os.path.join(CKPT_BASE, f"tbx11k_{variant}_seed{seed}")


def needs_training(variant, seed):
    path = os.path.join(ckpt_dir(variant, seed), "csr_stage4_best.pt")
    return not os.path.exists(path)


def train_run(variant, seed):
    out_dir  = ckpt_dir(variant, seed)
    log_path = os.path.join(LOG_BASE, f"tbx11k_{variant}_seed{seed}.log")
    os.makedirs(out_dir, exist_ok=True)

    flags = [
        "--dataset", "tbx11k",
        "--data_dir", DATA_DIR,
        "--config", "configs/tbx11k_config.yaml",
        "--checkpoint_dir", out_dir,
        "--seed", str(seed),
        "--batch_size", "128",
        "--num_workers", "8",
    ]
    if variant == "csr":
        flags += ["--no_gnn", "--no_vlm", "--no_uncertainty"]
    else:
        # GRAPE: GNN + Uncertainty, no VLM (best TBX11K config)
        flags += ["--no_vlm"]

    print(f"\n  Training tbx11k_{variant}_seed{seed} ...")
    with open(log_path, "w") as flog:
        result = subprocess.run(
            [sys.executable, "train.py"] + flags,
            stdout=flog, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        print(f"  WARNING: tbx11k_{variant}_seed{seed} exited {result.returncode}")
        return False
    print(f"  Done: tbx11k_{variant}_seed{seed}")
    return True


# Train missing runs
for variant in ["csr", "grape"]:
    for seed in SEEDS:
        if needs_training(variant, seed):
            train_run(variant, seed)
        else:
            print(f"  Checkpoint exists: tbx11k_{variant}_seed{seed} — skipping.")


# ── Evaluation ────────────────────────────────────────────────────────────────
from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, pointing_game


def load_model(variant, seed):
    path = os.path.join(ckpt_dir(variant, seed), "csr_stage4_best.pt")
    use_gnn = variant == "grape"
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


def eval_model(model):
    # Macro-F1 on test set
    test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    test_ldr = get_dataloader(test_ds, 128, 4, shuffle=False)
    preds, gtlabels = [], []
    with torch.no_grad():
        for batch in test_ldr:
            out   = model(batch["image"].to(DEVICE))
            logit = out["logits"] if isinstance(out, dict) else out
            preds.append(logit.argmax(-1).cpu())
            gtlabels.append(batch["class_label"])
    f1 = float(macro_f1(torch.cat(preds), torch.cat(gtlabels), num_classes=NUM_CLASSES))

    # Pointing Game on bbox_eval split
    bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
    hits = total = 0
    with torch.no_grad():
        for i in range(len(bbox_ds)):
            item   = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if not bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            f, _, _ = model.concept_model(img)
            fp  = model.projector.project_feature_map(f)
            sm  = model.prototype_learner.get_similarity_maps(fp)[0].cpu()  # (K,M,h,w)
            # Per-image PG: HIT if any concept's max-sim patch lands in its box
            image_hit = False
            for c_idx, bbox in bboxes.items():
                raw = sm[c_idx].mean(0).numpy()   # mean over M → (h, w)
                H, W = raw.shape
                x1, y1, x2, y2 = bbox
                bx1 = max(0, min(int(x1*W/IMAGE_SIZE), W-1))
                bx2 = max(bx1+1, min(int(x2*W/IMAGE_SIZE), W))
                by1 = max(0, min(int(y1*H/IMAGE_SIZE), H-1))
                by2 = max(by1+1, min(int(y2*H/IMAGE_SIZE), H))
                idx = int(np.argmax(raw))
                mh, mw = divmod(idx, W)
                if (bx1 <= mw < bx2) and (by1 <= mh < by2):
                    image_hit = True
            hits  += int(image_hit)
            total += 1
    pg = hits / total if total > 0 else 0.0
    return f1, pg


print("\n" + "="*60)
print("H2 TBX11K Evaluation (5 seeds)")
print("="*60)

results = {"grape": {}, "csr": {}}
for variant in ["csr", "grape"]:
    for seed in SEEDS:
        path = os.path.join(ckpt_dir(variant, seed), "csr_stage4_best.pt")
        if not os.path.exists(path):
            print(f"  {variant} seed {seed}: checkpoint missing — skipping")
            results[variant][seed] = {"f1": None, "pg": None}
            continue
        print(f"  Evaluating {variant} seed {seed} ...")
        m       = load_model(variant, seed)
        f1, pg  = eval_model(m)
        results[variant][seed] = {"f1": f1, "pg": pg}
        print(f"    F1={f1:.4f}  PG={pg:.4f}")
        del m


# ── Statistics ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("H2 Summary (5-seed TBX11K)")
print("="*60)

grape_f1s = [results["grape"][s]["f1"] for s in SEEDS if results["grape"][s].get("f1")]
csr_f1s   = [results["csr"][s]["f1"]   for s in SEEDS if results["csr"][s].get("f1")]
grape_pgs = [results["grape"][s]["pg"] for s in SEEDS if results["grape"][s].get("pg") is not None]
csr_pgs   = [results["csr"][s]["pg"]   for s in SEEDS if results["csr"][s].get("pg") is not None]

def summarize(vals, label):
    if not vals: return
    m, s = np.mean(vals), np.std(vals)
    print(f"  {label}: {m:.4f} ± {s:.4f}  n={len(vals)}")

summarize(grape_f1s, "GRAPE F1")
summarize(csr_f1s,   "CSR   F1")
summarize(grape_pgs, "GRAPE PG")
summarize(csr_pgs,   "CSR   PG")

n = min(len(grape_f1s), len(csr_f1s))
t_stat, p_val = None, None
if n >= 2:
    t_stat, p_val = stats.ttest_rel(grape_f1s[:n], csr_f1s[:n])
    t_stat, p_val = float(t_stat), float(p_val)
    print(f"  Paired t-test F1: t({n-1})={t_stat:.3f}  p={p_val:.4f}")

out = {
    "results_per_seed": {v: {str(s): r for s, r in d.items()}
                         for v, d in results.items()},
    "grape_f1_mean": float(np.mean(grape_f1s)) if grape_f1s else None,
    "grape_f1_std":  float(np.std(grape_f1s))  if grape_f1s else None,
    "csr_f1_mean":   float(np.mean(csr_f1s))   if csr_f1s   else None,
    "csr_f1_std":    float(np.std(csr_f1s))    if csr_f1s   else None,
    "grape_pg_mean": float(np.mean(grape_pgs)) if grape_pgs else None,
    "grape_pg_std":  float(np.std(grape_pgs))  if grape_pgs else None,
    "t_stat": t_stat, "p_val": p_val,
    "seeds": SEEDS,
}
with open(RESULTS, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nH2 done. Results → {RESULTS}")
print("Update Table 4 (tab:tbx11k) — replace 3-seed with 5-seed mean ± std.")
