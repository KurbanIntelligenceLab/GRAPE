"""
E1 — Module C Zero-Shot Concept Addition (NIH K=14 partial LOO)

For each of the 4–6 most common NIH concepts with bbox annotations
(Infiltration=8, Effusion=4, Atelectasis=0, Consolidation=2):
  1. Load the fully trained NIH GRAPE model (nih_official_grape_seed0).
  2. Reinitialise that concept's M=100 prototypes from BioViL-T text anchor (Eq. 9).
  3. Fine-tune with n ∈ {0, 1, 5, 20} labelled images, 3 seeds each.
  4. Measure Pointing Game on nih_bbox bbox_eval split.

Prerequisites:
  - data/nih_bbox/  (download via scripts/download_nih_bbox.py)
  - checkpoints/nih_official_grape_seed0/csr_stage4_best.pt

Results → results/e1_nih_zeroshot_loo.json
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, CONCEPTS

# ── Config ────────────────────────────────────────────────────────────────────
NIH_TRAIN_DIR  = "data/nih_official"   # train split (patient-level)
NIH_BBOX_DIR   = "data/nih_bbox"        # bbox images + bboxes.csv
CKPT_PATH      = "checkpoints/nih_official_grape_seed0/csr_stage4_best.pt"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE     = 224
NUM_CONCEPTS   = 14
NUM_CLASSES    = 2
M_PROTOS       = 100
PROTO_DIM      = 256
SIGMA          = 0.1
FINETUNE_EPOCHS = 10
FINETUNE_LR     = 1e-4
FEW_SHOT_COUNTS = [0, 1, 5, 20]
N_SEEDS         = 3

# Concepts with reliable bbox annotations in NIH BBox_List_2017.csv.
# Ordered by annotation frequency (most common first).
EVAL_CONCEPTS = [8, 4, 0, 2]   # Infiltration, Effusion, Atelectasis, Consolidation

os.makedirs("results", exist_ok=True)
torch.manual_seed(42)
random.seed(42)

NIH_CONCEPT_NAMES = CONCEPTS["nih"]

# BioViL-T text descriptions for each NIH concept (matching Appendix B style)
NIH_CONCEPT_DESCRIPTIONS = {
    0:  "Atelectasis: partial or complete lung collapse with opacity, volume loss, "
        "and displacement of fissures or mediastinum on chest X-ray.",
    1:  "Cardiomegaly: enlarged cardiac silhouette exceeding 50% of the thoracic "
        "width on a PA chest X-ray.",
    2:  "Consolidation: lobar or segmental opacity with air bronchograms indicating "
        "airspace filling on chest X-ray.",
    3:  "Edema: bilateral perihilar or interstitial haziness with vascular "
        "redistribution and Kerley B lines on chest X-ray.",
    4:  "Pleural effusion: blunting of costophrenic angle, homogeneous opacity "
        "layering dependently on chest X-ray.",
    5:  "Emphysema: bilateral hyperinflation, flattened diaphragms, increased "
        "retrosternal space, and decreased vascular markings on chest X-ray.",
    6:  "Fibrosis: bilateral reticular or reticulonodular opacities with volume "
        "loss and honeycombing on chest X-ray.",
    7:  "Hiatal hernia: gas or soft-tissue density behind the cardiac silhouette "
        "on chest X-ray.",
    8:  "Infiltration: unilateral or bilateral airspace opacity without lobar "
        "consolidation, suggesting inflammatory or infectious process on chest X-ray.",
    9:  "Pulmonary mass: solitary rounded opacity greater than 3 cm with smooth "
        "or irregular margins on chest X-ray.",
    10: "Pulmonary nodule: small rounded opacity less than 3 cm with well-defined "
        "margins on chest X-ray.",
    11: "Pleural thickening: irregular pleural surface or blunted costophrenic "
        "angle without free fluid on chest X-ray.",
    12: "Pneumonia: lobar or patchy consolidation with or without air bronchograms "
        "and parapneumonic effusion on chest X-ray.",
    13: "Pneumothorax: absence of lung markings peripheral to a pleural line "
        "indicating collapsed lung on chest X-ray.",
}


def get_sim_maps(model, image_t):
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
    return sm[0].cpu()


def evaluate_pg_for_concept(model, bbox_ds, held_out_k):
    hits, total = 0, 0
    model.eval()
    for i in range(len(bbox_ds)):
        item = bbox_ds[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if held_out_k not in bboxes:
            continue
        sm = get_sim_maps(model, item["image"])
        raw = sm[held_out_k].amax(0).numpy()
        H, W = raw.shape
        x1, y1, x2, y2 = bboxes[held_out_k]
        bx1 = max(0, min(int(x1 * W / IMAGE_SIZE), W - 1))
        bx2 = max(bx1 + 1, min(int(x2 * W / IMAGE_SIZE), W))
        by1 = max(0, min(int(y1 * H / IMAGE_SIZE), H - 1))
        by2 = max(by1 + 1, min(int(y2 * H / IMAGE_SIZE), H))
        idx = np.argmax(raw)
        mh, mw = divmod(idx, W)
        hit = (bx1 <= mw < bx2) and (by1 <= mh < by2)
        hits  += int(hit)
        total += 1
    return hits / total if total > 0 else 0.0, total


def get_text_anchor(model, concept_description):
    if not hasattr(model, "vlm_aligner") or model.vlm_aligner is None:
        vec = torch.randn(PROTO_DIM)
        return F.normalize(vec, dim=0)
    with torch.no_grad():
        t_k = model.vlm_aligner.encode_text([concept_description])
        t_hat = model.vlm_aligner.project(t_k)[0]
        return F.normalize(t_hat, dim=0)


def init_zeroshot_prototypes(model, held_out_k, concept_description, sigma=SIGMA):
    t_hat = get_text_anchor(model, concept_description)
    protos = model.prototype_learner.prototypes.data
    new_protos = torch.zeros(M_PROTOS, PROTO_DIM)
    for m in range(M_PROTOS):
        eps_m = F.normalize(torch.randn(PROTO_DIM), dim=0)
        p = t_hat + sigma * eps_m
        new_protos[m] = F.normalize(p, dim=0)
    protos[held_out_k] = new_protos.to(DEVICE)
    return model


def finetune_concept_prototypes(model, train_ds, held_out_k, n_examples, seed=42):
    if n_examples == 0:
        return model

    rng = random.Random(seed)
    train_items = [
        train_ds[i] for i in range(len(train_ds))
        if train_ds[i]["concept_labels"][held_out_k] == 1
    ]
    rng.shuffle(train_items)
    finetune_items = train_items[:n_examples]

    if not finetune_items:
        return model

    for param in model.parameters():
        param.requires_grad = False
    model.prototype_learner.prototypes.requires_grad_(True)

    opt = torch.optim.Adam([model.prototype_learner.prototypes], lr=FINETUNE_LR)

    model.train()
    for _ in range(FINETUNE_EPOCHS):
        for item in finetune_items:
            img = item["image"].unsqueeze(0).to(DEVICE)
            opt.zero_grad()
            with torch.no_grad():
                f, _, _ = model.concept_model(img)
                fp = model.projector.project_feature_map(f)

            protos_k = model.prototype_learner.prototypes[held_out_k]
            protos_k_norm = F.normalize(protos_k, dim=-1)
            fp_flat = fp[0].permute(1, 2, 0).reshape(-1, PROTO_DIM)
            fp_norm = F.normalize(fp_flat, dim=-1)
            sim = fp_norm @ protos_k_norm.T
            max_sim = sim.max(dim=0).values
            score = max_sim.mean()
            loss = -score
            loss.backward()
            opt.step()

            with torch.no_grad():
                model.prototype_learner.prototypes[held_out_k] = F.normalize(
                    model.prototype_learner.prototypes[held_out_k], dim=-1
                )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_nih_grape_base(seed=0):
    """Load NIH GRAPE model for given seed (0, 1, 2, 3, or 42)."""
    ckpt_path = f"checkpoints/nih_official_grape_seed{seed}/csr_stage4_best.pt"
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=M_PROTOS, num_classes=NUM_CLASSES,
        proto_dim=PROTO_DIM, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


# ── Check prerequisites ───────────────────────────────────────────────────────
if not os.path.exists(NIH_BBOX_DIR):
    print(f"ERROR: NIH bbox data not found at {NIH_BBOX_DIR}")
    print("Run: python scripts/download_nih_bbox.py")
    sys.exit(1)

if not os.path.exists(CKPT_PATH):
    print(f"ERROR: Checkpoint not found: {CKPT_PATH}")
    sys.exit(1)

# ── Datasets ──────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = CSRDataset(NIH_TRAIN_DIR, "nih", split="train", image_size=IMAGE_SIZE)
bbox_ds  = CSRDataset(NIH_BBOX_DIR,  "nih", split="bbox_eval", image_size=IMAGE_SIZE)
print(f"  Train: {len(train_ds)} images, BBox eval: {len(bbox_ds)} images")

# ── Leave-one-concept-out loop ────────────────────────────────────────────────
results = {}

for held_out_k in EVAL_CONCEPTS:
    cname = NIH_CONCEPT_NAMES[held_out_k]
    cdesc = NIH_CONCEPT_DESCRIPTIONS[held_out_k]
    print(f"\n=== Held-out concept: {cname} (idx={held_out_k}) ===")

    # Check bbox coverage
    model_tmp = load_nih_grape_base(seed=0)
    _, n_pairs = evaluate_pg_for_concept(model_tmp, bbox_ds, held_out_k)
    del model_tmp

    if n_pairs == 0:
        print(f"  No bbox_eval pairs for concept {held_out_k}. Skipping.")
        results[held_out_k] = {"concept": cname, "n_pairs": 0, "pg": {}}
        continue

    # Fully supervised upper bound (seed 0)
    model_full = load_nih_grape_base(seed=0)
    pg_full, _ = evaluate_pg_for_concept(model_full, bbox_ds, held_out_k)
    del model_full
    print(f"  Fully supervised PG = {pg_full:.4f}  ({n_pairs} pairs)")

    concept_results = {"concept": cname, "n_pairs": n_pairs,
                       "pg_full_supervised": pg_full, "pg": {}}

    for n_ex in FEW_SHOT_COUNTS:
        seed_pgs = []
        n_seeds_to_use = 1 if n_ex == 0 else N_SEEDS
        for seed in range(n_seeds_to_use):
            model = load_nih_grape_base(seed=0)
            model = init_zeroshot_prototypes(model, held_out_k, cdesc)
            model = finetune_concept_prototypes(model, train_ds, held_out_k, n_ex,
                                                seed=seed)
            pg, _ = evaluate_pg_for_concept(model, bbox_ds, held_out_k)
            seed_pgs.append(pg)
            del model

        mean_pg = float(np.mean(seed_pgs))
        std_pg  = float(np.std(seed_pgs)) if len(seed_pgs) > 1 else 0.0
        concept_results["pg"][n_ex] = {"mean": mean_pg, "std": std_pg,
                                        "all_seeds": seed_pgs}
        print(f"  n_ex={n_ex:>2}: PG = {mean_pg:.4f} ± {std_pg:.4f}")

    results[held_out_k] = concept_results

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== E1 NIH Partial LOO Results ===")
eval_names = [NIH_CONCEPT_NAMES[k] for k in EVAL_CONCEPTS
              if k in results and results[k]["n_pairs"] > 0]
if eval_names:
    print(f"{'n_shots':>8}  " + "  ".join(f"{n:>16}" for n in eval_names))
    for n_ex in FEW_SHOT_COUNTS:
        vals = []
        for k in EVAL_CONCEPTS:
            if k not in results or results[k]["n_pairs"] == 0:
                continue
            if n_ex in results[k]["pg"]:
                r = results[k]["pg"][n_ex]
                vals.append(f"{r['mean']:.3f}±{r['std']:.3f}")
            else:
                vals.append("N/A")
        print(f"{str(n_ex):>8}  " + "  ".join(f"{v:>16}" for v in vals))

with open("results/e1_nih_zeroshot_loo.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

print("\nE1 done. Results in results/e1_nih_zeroshot_loo.json")
