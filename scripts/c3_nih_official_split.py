"""
C3 — NIH ChestX-ray14 with Official Patient-Level Split, 5 Seeds

Uses a fixed split because the HuggingFace streaming
approach returns different 20k subsets across sessions (different shards).

This script:
  1. Downloads NIH with patient-level train/val/test split (no patient leak).
  2. Trains GRAPE (full) and CSR baseline for 5 seeds {42, 0, 1, 2, 3}.
  3. Evaluates macro-F1 and PG for each run.
  4. Computes paired t-test on per-seed F1 differences.
  5. Saves JSON results for paper update.

Results → results/c3_nih_official.json
Compute:  ~80 GPU-h on A100 (10 runs × ~8h each)
"""

import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

DATA_DIR  = "data/nih_official"
CKPT_BASE = "checkpoints"
LOG_BASE  = "logs"
RESULTS   = "results/c3_nih_official.json"
HF_CACHE  = "hf_cache"

os.makedirs(LOG_BASE, exist_ok=True)
os.makedirs("results", exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE

SEEDS        = [42, 0, 1, 2, 3]
NUM_CONCEPTS = 14
NUM_CLASSES  = 2


# ── Step 1: Download NIH official split if missing ────────────────────────────
if not os.path.exists(os.path.join(DATA_DIR, "labels.csv")):
    print("=== Downloading NIH ChestX-ray14 (patient-level split) ===")
    result = subprocess.run(
        [sys.executable, "scripts/download_nih_official.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: download_nih_official.py failed")
        sys.exit(1)
    print("NIH official data ready.\n")
else:
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))
    vc = df["split"].value_counts().to_dict()
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else "unknown"
    print(f"NIH official data found: {len(df)} images, {n_patients} patients, {vc}\n")


# ── Step 2: Training helpers ──────────────────────────────────────────────────
def needs_training(variant, seed):
    ckpt = os.path.join(CKPT_BASE, f"nih_official_{variant}_seed{seed}",
                        "csr_stage4_best.pt")
    return not os.path.exists(ckpt)


def train_run(variant, seed):
    ckpt_dir = os.path.join(CKPT_BASE, f"nih_official_{variant}_seed{seed}")
    log_path = os.path.join(LOG_BASE, f"nih_official_{variant}_seed{seed}.log")
    os.makedirs(ckpt_dir, exist_ok=True)

    flags = [
        "--dataset", "nih",
        "--data_dir", DATA_DIR,
        "--config", "configs/nih_config.yaml",
        "--checkpoint_dir", ckpt_dir,
        "--seed", str(seed),
        "--batch_size", "128",
        "--num_workers", "8",
    ]
    if variant == "csr":
        flags += ["--no_gnn", "--no_vlm", "--no_uncertainty"]

    print(f"\n  Training nih_official_{variant}_seed{seed} ...")
    with open(log_path, "w") as flog:
        result = subprocess.run(
            [sys.executable, "train.py"] + flags,
            stdout=flog, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        print(f"  WARNING: nih_official_{variant}_seed{seed} exited {result.returncode}")
        return False
    print(f"  Done: nih_official_{variant}_seed{seed}")
    return True


# Train all missing runs
for variant in ["csr", "grape"]:
    for seed in SEEDS:
        if needs_training(variant, seed):
            train_run(variant, seed)
        else:
            print(f"  Checkpoint exists: nih_official_{variant}_seed{seed} — skipping.")


# ── Step 3: Evaluation ────────────────────────────────────────────────────────
from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224


def load_model(variant, seed):
    ckpt_path = os.path.join(CKPT_BASE, f"nih_official_{variant}_seed{seed}",
                             "csr_stage4_best.pt")
    use_gnn = variant == "grape"

    model = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_gnn,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    if use_gnn:
        train_ds = CSRDataset(DATA_DIR, "nih", split="train", image_size=IMAGE_SIZE)
        loader   = get_dataloader(train_ds, 128, 4, shuffle=False)
        labels   = torch.cat([b["concept_labels"] for b in loader]).to(DEVICE)
        ei, ew   = build_cooccurrence_graph(labels, threshold=0.05)
        ew_norm  = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
        model.set_concept_graph(ei, ew_norm)
    return model


def evaluate(model):
    test_ds  = CSRDataset(DATA_DIR, "nih", split="test", image_size=IMAGE_SIZE)
    test_ldr = get_dataloader(test_ds, 128, 4, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_ldr:
            imgs  = batch["image"].to(DEVICE)
            out   = model(imgs)
            logit = out["logits"] if isinstance(out, dict) else out
            preds.append(logit.argmax(-1).cpu())
            labels.append(batch["class_label"])
    f1 = macro_f1(torch.cat(preds), torch.cat(labels), num_classes=NUM_CLASSES)
    return float(f1)


print("\n" + "="*60)
print("C3 Evaluation (patient-level split, 5 seeds)")
print("="*60)

results = {"grape": {}, "csr": {}}
for variant in ["csr", "grape"]:
    for seed in SEEDS:
        ckpt = os.path.join(CKPT_BASE, f"nih_official_{variant}_seed{seed}",
                            "csr_stage4_best.pt")
        if not os.path.exists(ckpt):
            print(f"  {variant} seed {seed}: checkpoint missing — skipping")
            results[variant][seed] = {"f1": None}
            continue
        print(f"  Evaluating {variant} seed {seed} ...")
        model  = load_model(variant, seed)
        f1     = evaluate(model)
        results[variant][seed] = {"f1": f1}
        print(f"    F1={f1:.4f}")
        del model


# ── Step 4: Statistics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("C3 Statistical Summary")
print("="*60)

grape_f1s = [results["grape"][s]["f1"] for s in SEEDS if results["grape"][s].get("f1")]
csr_f1s   = [results["csr"][s]["f1"]   for s in SEEDS if results["csr"][s].get("f1")]

if grape_f1s and csr_f1s:
    grape_mean = float(np.mean(grape_f1s))
    grape_std  = float(np.std(grape_f1s))
    csr_mean   = float(np.mean(csr_f1s))
    csr_std    = float(np.std(csr_f1s))
    delta      = float(np.mean([g - c for g, c in zip(grape_f1s, csr_f1s)]))

    print(f"\nCSR   F1: {csr_mean:.4f} ± {csr_std:.4f}  seeds={csr_f1s}")
    print(f"GRAPE F1: {grape_mean:.4f} ± {grape_std:.4f}  seeds={grape_f1s}")
    print(f"Mean Δ:   {delta:+.4f}")

    t_stat, p_val = None, None
    n_complete = min(len(grape_f1s), len(csr_f1s))
    if n_complete >= 2:
        t_stat, p_val = stats.ttest_rel(
            grape_f1s[:n_complete], csr_f1s[:n_complete]
        )
        t_stat, p_val = float(t_stat), float(p_val)
        df = n_complete - 1
        print(f"Paired t-test: t({df})={t_stat:.3f}  p={p_val:.4f}")
        print("  → " + ("Significant (p<0.05)" if p_val < 0.05 else "NOT significant"))

    out = {
        "results_per_seed": {v: {str(s): r for s, r in d.items()}
                             for v, d in results.items()},
        "grape_f1_mean": grape_mean, "grape_f1_std": grape_std,
        "csr_f1_mean":   csr_mean,   "csr_f1_std":   csr_std,
        "delta_f1_mean": delta,
        "t_stat": t_stat, "p_val": p_val,
        "seeds": SEEDS,
        "split_type": "patient_level",
        "data_dir": DATA_DIR,
    }
else:
    out = {"error": "Insufficient completed runs", "results_per_seed": results}

with open(RESULTS, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nC3 done. Results → {RESULTS}")
print("Update Table 5 (tab:nih), Table 6 (tab:compare).")
