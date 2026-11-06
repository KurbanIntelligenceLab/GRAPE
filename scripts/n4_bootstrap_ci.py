"""
N4 — Bootstrap CIs on Module B TP/FP rates

Replaces the T=5 trial approach with 1000-sample bootstrap over the 207
(image, concept, bbox) triples for each miscorrection rate p.

For each bootstrap iteration:
  - Resample 207 triples with replacement
  - For each sampled triple, apply misdraw with probability p (fixed RNG per iter)
  - Compute TP rate (fraction of misdrawn boxes warned) and FP rate
  - Repeat 1000 times; report 2.5/97.5 percentiles as 95% CI

Saves results/n4_bootstrap_ci.json
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
ETA          = 0.05
MISC_RATES   = [0.0, 0.10, 0.25, 0.50, 1.0]
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
os.makedirs("results", exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────
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
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
    return sm[0].cpu()


def run_safety_check(sim_maps, bbox_224, intended_concept, eta=ETA):
    K, M, H, W = sim_maps.shape
    x1, y1, x2, y2 = bbox_224
    bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
    bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
    by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
    by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
    region   = sim_maps[:, :, by1:by2, bx1:bx2]
    s_bar    = region.mean(dim=(-2, -1)).mean(dim=1)   # (K,)
    dominant = int(s_bar.argmax())
    dom_score = float(s_bar[dominant])
    int_score = float(s_bar[intended_concept])
    return (dominant != intended_concept) and (dom_score - int_score > eta)


# ── Load data and build sim cache ─────────────────────────────────────────────
print("Loading model...")
model = load_model()

train_ds  = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
train_ldr = get_dataloader(train_ds, 128, 4, shuffle=False)
all_labels = torch.cat([b["concept_labels"] for b in train_ldr]).to(DEVICE)
ei, ew = build_cooccurrence_graph(all_labels, threshold=0.1)
ew     = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
model.set_concept_graph(ei, ew)

bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
samples = []
for i in range(len(bbox_ds)):
    item = bbox_ds[i]
    for concept_idx, bbox in {int(k): v for k, v in item["bbox"].items()}.items():
        samples.append((i, concept_idx, bbox))

print(f"  Total triples: {len(samples)}")
unique_imgs = list(set(s[0] for s in samples))
print(f"  Pre-computing sim maps for {len(unique_imgs)} images...")
sim_cache = {}
for idx, img_i in enumerate(unique_imgs):
    if idx % 20 == 0:
        print(f"    {idx}/{len(unique_imgs)}", flush=True)
    sim_cache[img_i] = get_sim_maps(model, bbox_ds[img_i]["image"])

all_concepts = list(range(NUM_CONCEPTS))
N = len(samples)
samples_arr = np.array(samples, dtype=object)

# Pre-compute warned flags for all triples with correct and wrong labels
# correct_warned[i] = True if triple i fires a warning when the correct concept is used
# wrong_warned[i][c] = True if triple i fires a warning when wrong concept c is used
print("\nPre-computing safety check for all triples...")
correct_warned = np.zeros(N, dtype=bool)
wrong_warned   = {}   # (i, wrong_concept) → bool

for i, (img_i, concept_idx, bbox) in enumerate(samples):
    correct_warned[i] = run_safety_check(sim_cache[img_i], bbox, concept_idx)
    wrong_concepts = [c for c in all_concepts if c != concept_idx]
    for wc in wrong_concepts:
        wrong_warned[(i, wc)] = run_safety_check(sim_cache[img_i], bbox, wc)

print("  Done.")

# ── Bootstrap ─────────────────────────────────────────────────────────────────
print(f"\n=== N4: Bootstrap (B={N_BOOTSTRAP}) ===")
rng = np.random.default_rng(RANDOM_SEED)

out = {"misc_rates": MISC_RATES, "n_bootstrap": N_BOOTSTRAP, "n_samples": N, "eta": ETA}

for p in MISC_RATES:
    tp_rates, fp_rates = [], []
    for b in range(N_BOOTSTRAP):
        # Resample triples with replacement
        idx = rng.integers(0, N, size=N)
        # Assign misdraw per triple (vectorised)
        misdraw_mask = rng.random(N) < p
        tp_w = tp_t = fp_w = fp_t = 0
        for j, orig_idx in enumerate(idx):
            img_i, concept_idx, bbox = samples[orig_idx]
            if misdraw_mask[j]:
                wrong_concepts = [c for c in all_concepts if c != concept_idx]
                wc = wrong_concepts[rng.integers(0, len(wrong_concepts))]
                warned = wrong_warned[(orig_idx, wc)]
                tp_t += 1
                if warned: tp_w += 1
            else:
                warned = correct_warned[orig_idx]
                fp_t += 1
                if warned: fp_w += 1
        if tp_t > 0: tp_rates.append(tp_w / tp_t)
        if fp_t > 0: fp_rates.append(fp_w / fp_t)

    def ci(vals):
        if not vals: return None, None, None, None
        a = np.array(vals)
        return float(a.mean()), float(a.std()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    tp_mean, tp_std, tp_lo, tp_hi = ci(tp_rates)
    fp_mean, fp_std, fp_lo, fp_hi = ci(fp_rates)
    tp_str = f"{tp_mean:.3f} [{tp_lo:.3f},{tp_hi:.3f}]" if tp_mean is not None else "N/A"
    fp_str = f"{fp_mean:.3f} [{fp_lo:.3f},{fp_hi:.3f}]" if fp_mean is not None else "N/A"
    print(f"  p={p:.2f}  TP={tp_str}  FP={fp_str}")
    out[str(p)] = {
        "tp_mean": tp_mean, "tp_std": tp_std, "tp_ci95_lo": tp_lo, "tp_ci95_hi": tp_hi,
        "fp_mean": fp_mean, "fp_std": fp_std, "fp_ci95_lo": fp_lo, "fp_ci95_hi": fp_hi,
    }

with open("results/n4_bootstrap_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nN4 done. Results → results/n4_bootstrap_ci.json")
