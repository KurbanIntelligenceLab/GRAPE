"""
C7 — Module B Safety Evaluation (Simulated Miscorrection Experiment)

For each image in TBX11K bbox_eval with a concept bounding box:
  - At rate p, randomly reassign the box to a WRONG concept
  - Run Module B safety check (Eq. 10, η = 0.05)
  - Report TP rate (misdraw caught) and FP rate (correct draw falsely warned)

Produces results for Table 6 (new §4.6).
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.training.interaction import TestTimeInteraction, BoundingBox

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
ETA          = 0.05       # safety threshold from paper
MISC_RATES   = [0.0, 0.10, 0.25, 0.50, 1.0]
N_TRIALS     = 5          # repeat each p with different random seeds, average
FEAT_H = FEAT_W = 7       # 7×7 feature map from ResNet-50

os.makedirs("results", exist_ok=True)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── Load GRAPE model ──────────────────────────────────────────────────────────
def load_model():
    path = os.path.join(CKPT_BASE, "tbx11k_no_C_novlm", "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


def get_sim_maps(model, image_t):
    """Return (K, M, 7, 7) similarity maps for one image."""
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)  # (1, K, M, h, w)
    return sm[0].cpu()  # (K, M, h, w)


def run_safety_check(sim_maps_7x7, bbox_224, intended_concept, eta=ETA):
    """
    Matches paper Eq. 10: mean of mean-prototype similarity inside bbox.
    bbox_224: (x1, y1, x2, y2) in 224-space.
    Returns: (warned, dominant_concept)
    """
    K, M, H, W = sim_maps_7x7.shape
    x1, y1, x2, y2 = bbox_224

    # Scale bbox to 7×7 feature space
    bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
    bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
    by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
    by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))

    # Mean over M prototypes at each patch → (K, h, w)
    mu_k = sim_maps_7x7.mean(dim=1)  # (K, H, W)

    # Mean over bbox region → (K,)
    region = mu_k[:, by1:by2, bx1:bx2]   # (K, h_reg, w_reg)
    s_bar = region.mean(dim=(-2, -1))     # (K,)

    dominant = int(s_bar.argmax())
    dom_score = float(s_bar[dominant])
    int_score = float(s_bar[intended_concept])

    warned = (dominant != intended_concept) and (dom_score - int_score > eta)
    return warned, dominant


# ── Main experiment ───────────────────────────────────────────────────────────
print("Loading model...")
model = load_model()

# Load co-occurrence graph (needed for GNN)
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_loader = get_dataloader(train_ds, batch_size=128, num_workers=4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
ei, ew = build_cooccurrence_graph(all_labels, threshold=0.1)
ew = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
model.set_concept_graph(ei, ew)

print("Loading bbox_eval split...")
bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

# Collect all (image_idx, concept_idx, bbox) triples
samples = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    bboxes = {int(k): v for k, v in item["bbox"].items()}
    for concept_idx, bbox in bboxes.items():
        samples.append((i, concept_idx, bbox))

print(f"  Total (image, concept, bbox) triples: {len(samples)}")

# Pre-compute sim maps for all unique images (avoid redundant forward passes)
unique_imgs = list(set(s[0] for s in samples))
print(f"  Pre-computing similarity maps for {len(unique_imgs)} images...")
sim_map_cache = {}
for idx, img_i in enumerate(unique_imgs):
    if idx % 20 == 0:
        print(f"    {idx}/{len(unique_imgs)}")
    item = bbox_ds[img_i]
    sim_map_cache[img_i] = get_sim_maps(model, item["image"])

print("Running safety check experiment...")
all_concepts = list(range(NUM_CONCEPTS))

results = {}
for p in MISC_RATES:
    tp_rates_over_trials = []
    fp_rates_over_trials = []

    for trial in range(N_TRIALS):
        rng = random.Random(RANDOM_SEED + trial)
        tp_warned = 0
        tp_total = 0
        fp_warned = 0
        fp_total = 0

        for img_i, concept_idx, bbox in samples:
            sim_maps = sim_map_cache[img_i]

            # Decide if this box is misdrawn
            is_misdraw = rng.random() < p

            if is_misdraw:
                # Randomly pick a different concept as the "claimed" one
                wrong_concepts = [c for c in all_concepts if c != concept_idx]
                claimed_concept = rng.choice(wrong_concepts)
                warned, _ = run_safety_check(sim_maps, bbox, claimed_concept)
                tp_total += 1
                if warned:
                    tp_warned += 1
            else:
                # Correct draw — use the true concept
                warned, _ = run_safety_check(sim_maps, bbox, concept_idx)
                fp_total += 1
                if warned:
                    fp_warned += 1

        if tp_total > 0:
            tp_rates_over_trials.append(tp_warned / tp_total)
        if fp_total > 0:
            fp_rates_over_trials.append(fp_warned / fp_total)

    results[p] = {
        "tp_rate_mean": float(np.mean(tp_rates_over_trials)) if tp_rates_over_trials else None,
        "tp_rate_std":  float(np.std(tp_rates_over_trials))  if tp_rates_over_trials else None,
        "fp_rate_mean": float(np.mean(fp_rates_over_trials)) if fp_rates_over_trials else None,
        "fp_rate_std":  float(np.std(fp_rates_over_trials))  if fp_rates_over_trials else None,
        "n_samples": len(samples),
    }

# ── Print results ─────────────────────────────────────────────────────────────
print("\n=== C7 Module B Safety Check Results (TBX11K bbox_eval) ===")
print(f"η = {ETA}, {N_TRIALS} trials, {len(samples)} (image, concept, bbox) pairs")
print(f"{'p':>6}  {'TP rate':>10}  {'FP rate':>10}")
print("-" * 32)
for p in MISC_RATES:
    r = results[p]
    tp_str = f"{r['tp_rate_mean']:.3f}±{r['tp_rate_std']:.3f}" if r["tp_rate_mean"] is not None else "  —"
    fp_str = f"{r['fp_rate_mean']:.3f}±{r['fp_rate_std']:.3f}" if r["fp_rate_mean"] is not None else "  —"
    print(f"{p:>6.2f}  {tp_str:>10}  {fp_str:>10}")

# Save JSON for LaTeX table generation
out_path = "results/c7_safety_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
