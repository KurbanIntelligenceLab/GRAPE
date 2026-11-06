"""
N7 — Module B Safety Check on NIH ChestX-ray14 (K=14, denser multi-label)

How does the safety check perform on denser multi-label settings (e.g., NIH
with overlapping findings)?

Same simulated miscorrection protocol as C7 (TBX11K), but on NIH bbox split
with K=14 concepts and potentially overlapping findings. We additionally
report TP/FP broken down by concept to surface any concept-size effects.

Results → results/n7_nih_safety_eval.json
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR     = "data/nih_bbox"
DATASET_NAME = "nih"
CKPT_PATH    = "checkpoints/nih_official_grape_seed0/csr_stage4_best.pt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 14
NUM_CLASSES  = 2
NUM_PROTO    = 100
ETA          = 0.05
MISC_RATES   = [0.0, 0.10, 0.25, 0.50, 1.0]
N_TRIALS     = 5
FEAT_H = FEAT_W = 7

CONCEPT_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

os.makedirs("results", exist_ok=True)
random.seed(42); np.random.seed(42); torch.manual_seed(42)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=NUM_PROTO, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


def get_sim_maps(model, image_t):
    with torch.no_grad():
        f, _, _ = model.concept_model(image_t.unsqueeze(0).to(DEVICE))
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)  # (1, K, M, H, W)
    return sm[0].cpu()


def run_safety_check(sim_maps, bbox_224, intended_concept, eta=ETA):
    K, M, H, W = sim_maps.shape
    x1, y1, x2, y2 = bbox_224
    bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
    bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
    by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
    by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
    mu_k   = sim_maps.mean(dim=1)               # (K, H, W)
    region = mu_k[:, by1:by2, bx1:bx2]         # (K, h_r, w_r)
    s_bar  = region.mean(dim=(-2, -1))          # (K,)
    dominant  = int(s_bar.argmax())
    dom_score = float(s_bar[dominant])
    int_score = float(s_bar[intended_concept])
    warned = (dominant != intended_concept) and (dom_score - int_score > eta)
    return warned, dominant


# ── Load model and data ───────────────────────────────────────────────────────
print("Loading NIH model...")
model = load_model()

print("Building NIH co-occurrence graph...")
train_ds = CSRDataset(DATA_DIR, DATASET_NAME, split="bbox_eval", image_size=IMAGE_SIZE)

# Build graph: collect concept_labels one-by-one (bbox dict prevents batching)
print("  Collecting labels for graph construction...")
label_list = []
for i in range(len(train_ds)):
    label_list.append(train_ds[i]["concept_labels"])
all_labels = torch.stack(label_list).to(DEVICE)
ei, ew     = build_cooccurrence_graph(all_labels, threshold=0.10)
ew_norm    = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
model.set_concept_graph(ei, ew_norm)

print("Loading NIH bbox_eval split...")
bbox_ds = CSRDataset(DATA_DIR, DATASET_NAME, split="bbox_eval", image_size=IMAGE_SIZE)

# Collect all (image_idx, concept_idx, bbox) triples
samples = []
for i in range(len(bbox_ds)):
    item   = bbox_ds[i]
    bboxes = {int(k): v for k, v in item["bbox"].items()}
    for concept_idx, bbox in bboxes.items():
        if concept_idx < NUM_CONCEPTS:
            samples.append((i, concept_idx, bbox))

print(f"  Triples: {len(samples)}  across {len(set(s[0] for s in samples))} images")

# ── Pre-compute similarity maps ───────────────────────────────────────────────
unique_imgs = list(set(s[0] for s in samples))
print(f"Pre-computing similarity maps for {len(unique_imgs)} images...")
sim_map_cache = {}
for idx, img_i in enumerate(unique_imgs):
    if idx % 50 == 0:
        print(f"  {idx}/{len(unique_imgs)}", flush=True)
    sim_map_cache[img_i] = get_sim_maps(model, bbox_ds[img_i]["image"])

# ── Main experiment ───────────────────────────────────────────────────────────
all_concepts    = list(range(NUM_CONCEPTS))
results         = {}
saved_concept_tp = None   # per-concept rates saved from the p=0.5 iteration
saved_concept_fp = None

for p in MISC_RATES:
    tp_trials, fp_trials = [], []
    concept_tp = {c: [] for c in all_concepts}
    concept_fp = {c: [] for c in all_concepts}

    for trial in range(N_TRIALS):
        rng = random.Random(42 + trial)
        tp_w, tp_t = 0, 0
        fp_w, fp_t = 0, 0
        c_tp_w = {c: 0 for c in all_concepts}
        c_tp_t = {c: 0 for c in all_concepts}
        c_fp_w = {c: 0 for c in all_concepts}
        c_fp_t = {c: 0 for c in all_concepts}

        for img_i, concept_idx, bbox in samples:
            sm = sim_map_cache[img_i]
            is_misdraw = rng.random() < p

            if is_misdraw:
                wrong = [c for c in all_concepts if c != concept_idx]
                claimed = rng.choice(wrong)
                warned, _ = run_safety_check(sm, bbox, claimed)
                tp_t += 1; c_tp_t[concept_idx] += 1
                if warned:
                    tp_w += 1; c_tp_w[concept_idx] += 1
            else:
                warned, _ = run_safety_check(sm, bbox, concept_idx)
                fp_t += 1; c_fp_t[concept_idx] += 1
                if warned:
                    fp_w += 1; c_fp_w[concept_idx] += 1

        if tp_t > 0: tp_trials.append(tp_w / tp_t)
        if fp_t > 0: fp_trials.append(fp_w / fp_t)

        if abs(p - 0.5) < 0.01:
            for c in all_concepts:
                if c_tp_t[c] > 0:
                    concept_tp[c].append(c_tp_w[c] / c_tp_t[c])
                if c_fp_t[c] > 0:
                    concept_fp[c].append(c_fp_w[c] / c_fp_t[c])

    if abs(p - 0.5) < 0.01:
        saved_concept_tp = {c: list(v) for c, v in concept_tp.items()}
        saved_concept_fp = {c: list(v) for c, v in concept_fp.items()}

    results[p] = {
        "tp_rate_mean": float(np.mean(tp_trials)) if tp_trials else None,
        "tp_rate_std":  float(np.std(tp_trials))  if tp_trials else None,
        "fp_rate_mean": float(np.mean(fp_trials)) if fp_trials else None,
        "fp_rate_std":  float(np.std(fp_trials))  if fp_trials else None,
        "n_samples": len(samples),
    }

# Attach per-concept breakdown at p=0.5
per_concept = {}
for c in all_concepts:
    ct = saved_concept_tp.get(c, []) if saved_concept_tp else []
    cf = saved_concept_fp.get(c, []) if saved_concept_fp else []
    per_concept[CONCEPT_NAMES[c]] = {
        "tp_mean": float(np.mean(ct)) if ct else None,
        "fp_mean": float(np.mean(cf)) if cf else None,
        "n_bbox":  sum(1 for s in samples if s[1] == c),
    }
results["per_concept_at_p0.5"] = per_concept

# ── Print results ─────────────────────────────────────────────────────────────
print("\n=== N7 Module B Safety Check — NIH (K=14) ===")
print(f"η={ETA}, {N_TRIALS} trials, {len(samples)} triples")
print(f"{'p':>6}  {'TP rate':>14}  {'FP rate':>14}")
print("-" * 42)
for p in MISC_RATES:
    r = results[p]
    tp = f"{r['tp_rate_mean']:.3f}±{r['tp_rate_std']:.3f}" if r["tp_rate_mean"] else "   —"
    fp = f"{r['fp_rate_mean']:.3f}±{r['fp_rate_std']:.3f}" if r["fp_rate_mean"] else "   —"
    print(f"{p:>6.2f}  {tp:>14}  {fp:>14}")

print("\nPer-concept at p=0.5:")
print(f"{'Concept':<22} {'TP':>8} {'FP':>8} {'N bbox':>8}")
for name, v in per_concept.items():
    tp = f"{v['tp_mean']:.3f}" if v["tp_mean"] is not None else "  —"
    fp = f"{v['fp_mean']:.3f}" if v["fp_mean"] is not None else "  —"
    print(f"  {name:<20} {tp:>8} {fp:>8} {v['n_bbox']:>8}")

with open("results/n7_nih_safety_eval.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved → results/n7_nih_safety_eval.json")
