"""
H5 — Insertion / Deletion AUC Faithfulness Metric

Petsiuk et al. (RISE, 2018): faithfulness of saliency maps measured by
how much class probability drops when the most-salient pixels are removed
(Deletion AUC — lower is better) and rises when they are revealed from
a blurred baseline (Insertion AUC — higher is better).

We apply this to the prototype similarity maps:
  saliency(i,j) = max_{k,m} sim_map_{k,m}(i,j)  (highest prototype response)

Then:
  Deletion: replace top-p% of pixels with channel-mean (gray) → forward pass
            → measure softmax prob of GT class at each step p ∈ {5,10,...,100}%
  Insertion: reveal top-p% of pixels from blurred baseline → forward pass
             → measure softmax prob of GT class at each step p ∈ {5,10,...,100}%

AUC = trapezoid integral of the curve, normalised to [0,1].

Models compared:
  1. CSR baseline (no GNN, no VLM, no Uncert)
  2. GRAPE best   (GNN + Uncert, no VLM)    — same backbone, best PG config

Results → results/h5_faithfulness.json
Use to add a row to Table 6 (safety eval) or Table 7 (new faithfulness table).
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR  = "data/tbx11k"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
RESULTS_PATH = "results/h5_faithfulness.json"

os.makedirs("results", exist_ok=True)

# Pixel fraction steps for Deletion/Insertion curves
STEPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
N_IMAGES = 200   # use the bbox_eval split (200 images)

# ── Blurring kernel for Insertion baseline ─────────────────────────────────────
BLUR = transforms.GaussianBlur(kernel_size=51, sigma=20.0)


def load_model(ckpt_path, use_gnn, use_uncertainty):
    model = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=100, num_classes=NUM_CLASSES,
        proto_dim=256, use_gnn=use_gnn, use_uncertainty=use_uncertainty,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
    loader   = get_dataloader(train_ds, 128, 4, shuffle=False)
    labels   = torch.cat([b["concept_labels"] for b in loader]).to(DEVICE)
    ei, ew   = build_cooccurrence_graph(labels, threshold=0.10)
    ew_norm  = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
    if use_gnn:
        model.set_concept_graph(ei, ew_norm)
    return model


def get_saliency_map(model, img_t):
    """
    Compute per-pixel saliency = max over all (k, m) of the similarity map.
    Returns: (H_feat, W_feat) saliency map (upsample to IMAGE_SIZE × IMAGE_SIZE).
    """
    img = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)  # (1, K, M, h, w)
    # sm: (1, K, M, h, w) — already returned from get_similarity_maps
    # Actually returns list or (1, K, M, h, w)?  check implementation:
    # get_similarity_maps returns (B, K, M, H, W) based on src inspection
    sal = sm[0].amax(dim=(0, 1))     # (h, w) — max over K and M
    sal_up = F.interpolate(
        sal.unsqueeze(0).unsqueeze(0),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False,
    )[0, 0]   # (IMAGE_SIZE, IMAGE_SIZE)
    return sal_up.cpu()


def predict_prob(model, img_t, class_idx):
    """Softmax probability for class_idx given a (3, H, W) image tensor."""
    img = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(img)
        logits = out["logits"] if isinstance(out, dict) else out
        prob = F.softmax(logits, dim=-1)[0, class_idx].item()
    return prob


def deletion_curve(model, img_t, sal, gt_class):
    """
    Deletion: replace top-p% most salient pixels with channel mean (gray).
    Returns list of class probs at each step.
    """
    img_np   = img_t.numpy()                     # (3, H, W)
    sal_np   = sal.numpy().flatten()              # (H*W,)
    sorted_idx = np.argsort(sal_np)[::-1]        # descending saliency
    n_pixels   = sal_np.size
    channel_mean = img_np.mean(axis=(1, 2), keepdims=True)   # (3, 1, 1)

    probs = []
    for step in STEPS:
        n_remove = int(step * n_pixels)
        mask = np.ones(n_pixels, dtype=bool)
        mask[sorted_idx[:n_remove]] = False
        mask_2d = mask.reshape(IMAGE_SIZE, IMAGE_SIZE)

        img_mod = img_np.copy()
        for c in range(3):
            img_mod[c][~mask_2d] = float(channel_mean[c])

        prob = predict_prob(model, torch.tensor(img_mod, dtype=torch.float32), gt_class)
        probs.append(prob)
    return probs


def insertion_curve(model, img_t, sal, gt_class):
    """
    Insertion: start from blurred image, reveal top-p% most salient pixels.
    Returns list of class probs at each step.
    """
    img_np   = img_t.numpy()                     # (3, H, W)
    blurred  = BLUR(img_t).numpy()               # (3, H, W)
    sal_np   = sal.numpy().flatten()
    sorted_idx = np.argsort(sal_np)[::-1]
    n_pixels   = sal_np.size

    probs = []
    for step in STEPS:
        n_reveal = int(step * n_pixels)
        mask = np.zeros(n_pixels, dtype=bool)
        mask[sorted_idx[:n_reveal]] = True
        mask_2d = mask.reshape(IMAGE_SIZE, IMAGE_SIZE)

        img_mod = blurred.copy()
        for c in range(3):
            img_mod[c][mask_2d] = img_np[c][mask_2d]

        prob = predict_prob(model, torch.tensor(img_mod, dtype=torch.float32), gt_class)
        probs.append(prob)
    return probs


def auc_trapz(probs):
    """Trapezoidal AUC, normalised to [0,1] by dividing by number of intervals."""
    return float(np.trapz(probs, x=STEPS) / (STEPS[-1] - STEPS[0]))


def eval_model(model, name):
    """Compute mean Deletion AUC and Insertion AUC over N_IMAGES images."""
    test_ds = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    del_aucs, ins_aucs = [], []

    for i in range(min(N_IMAGES, len(test_ds))):
        item    = test_ds[i]
        img_t   = item["image"]       # (3, H, W) normalised tensor
        gt_cls  = int(item["class_label"])

        sal = get_saliency_map(model, img_t)

        del_probs = deletion_curve(model, img_t, sal, gt_cls)
        ins_probs = insertion_curve(model, img_t, sal, gt_cls)

        del_aucs.append(auc_trapz(del_probs))
        ins_aucs.append(auc_trapz(ins_probs))

        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i+1}/{N_IMAGES}  "
                  f"Del={np.mean(del_aucs):.4f}  Ins={np.mean(ins_aucs):.4f}")

    return {
        "deletion_auc_mean":  float(np.mean(del_aucs)),
        "deletion_auc_std":   float(np.std(del_aucs)),
        "insertion_auc_mean": float(np.mean(ins_aucs)),
        "insertion_auc_std":  float(np.std(ins_aucs)),
        "n_images": len(del_aucs),
    }


# ── Run ────────────────────────────────────────────────────────────────────────
MODELS = {
    "CSR_baseline": {
        "ckpt": "checkpoints/tbx11k_baseline_CSR/csr_stage4_best.pt",
        "use_gnn": False, "use_uncertainty": False,
    },
    "GRAPE_best": {
        "ckpt": "checkpoints/tbx11k_no_C_novlm/csr_stage4_best.pt",
        "use_gnn": True, "use_uncertainty": True,
    },
}

all_results = {}
for model_name, cfg in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    if not os.path.exists(cfg["ckpt"]):
        print(f"  Checkpoint not found: {cfg['ckpt']}")
        all_results[model_name] = {"error": "checkpoint not found"}
        continue
    model = load_model(cfg["ckpt"], cfg["use_gnn"], cfg["use_uncertainty"])
    res   = eval_model(model, model_name)
    all_results[model_name] = res
    del model

    print(f"\n  {model_name}:")
    print(f"    Deletion  AUC: {res['deletion_auc_mean']:.4f} ± {res['deletion_auc_std']:.4f}  (lower = more faithful)")
    print(f"    Insertion AUC: {res['insertion_auc_mean']:.4f} ± {res['insertion_auc_std']:.4f}  (higher = more faithful)")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("H5 Summary — Insertion/Deletion AUC Faithfulness")
print(f"{'='*60}")
for name, res in all_results.items():
    if "error" in res:
        print(f"  {name}: {res['error']}")
    else:
        print(f"  {name}:  Del={res['deletion_auc_mean']:.4f}±{res['deletion_auc_std']:.4f}"
              f"  Ins={res['insertion_auc_mean']:.4f}±{res['insertion_auc_std']:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nH5 done. Results → {RESULTS_PATH}")
print("Add Deletion/Insertion AUC rows to the faithfulness table.")
