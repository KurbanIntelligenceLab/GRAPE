"""
C8 — Module C Zero-Shot Concept Addition (Leave-One-Concept-Out)

For each concept k ∈ {0,1,2} in TBX11K:
  1. Train Stage 1+2 on remaining K−1=2 concepts (or reuse full backbone).
  2. Initialize held-out concept's prototypes from text anchor only (Eq. 13, σ=0.1).
  3. Evaluate at 0 / 1 / 5 / 20 labelled examples (fine-tune only the held-out
     concept's M=100 prototype vectors, all other parameters frozen).
  4. Report per-concept Pointing Game (PG) at each label count.

Since TBX11K has K=3 concepts (a smaller scale than NIH K=14), this experiment
serves as a proof-of-concept. Full NIH 14-concept leave-one-out results require
additional compute (~24 h on 2× A100).

Results → Table 7 (§4.7).
"""

import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.training.trainer import CSRTrainer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
M_PROTOS     = 100
PROTO_DIM    = 256
SIGMA        = 0.1          # perturbation scale (Eq. 13)
FINETUNE_EPOCHS = 10        # fine-tune epochs per few-shot level
FINETUNE_LR     = 1e-4
FEW_SHOT_COUNTS = [0, 1, 5, 20]   # 0 = text-only (zero-shot)
N_SEEDS         = 3                # repeat few-shot with different random seeds

os.makedirs("results", exist_ok=True)
torch.manual_seed(42)
random.seed(42)

# Concept text descriptions (same as Appendix B in paper)
CONCEPT_DESCRIPTIONS = [
    "Active pulmonary tuberculosis with cavitary lesions, nodular opacities, "
    "and upper-lobe infiltrates, often with signs of consolidation on chest X-ray.",

    "Healed or inactive tuberculosis with calcified granulomas, fibrous scarring, "
    "and volume loss in the upper lobes on chest X-ray.",

    "Pulmonary tuberculosis presenting as patchy or confluent consolidation, "
    "tree-in-bud opacities, or miliary nodules on chest X-ray.",
]
CONCEPT_NAMES = CONCEPTS["tbx11k"]


def get_sim_maps(model, image_t):
    """Return (K, M, h, w) sim maps for one image."""
    img = image_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
    return sm[0].cpu()


def evaluate_pg_for_concept(model, bbox_ds, held_out_k):
    """Evaluate Pointing Game for the held-out concept only."""
    hits, total = 0, 0
    model.eval()
    for i in range(len(bbox_ds)):
        item = bbox_ds[i]
        bboxes = {int(k): v for k, v in item["bbox"].items()}
        if held_out_k not in bboxes:
            continue
        sm = get_sim_maps(model, item["image"])
        raw = sm[held_out_k].amax(0).numpy()   # (h, w)
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
    """
    Get the projected text anchor t̂_k for a concept description.
    Returns unit-norm vector in R^D.
    """
    if not hasattr(model, "vlm_aligner") or model.vlm_aligner is None:
        # No VLM: return random unit vector as fallback
        vec = torch.randn(PROTO_DIM)
        return F.normalize(vec, dim=0)
    with torch.no_grad():
        t_k = model.vlm_aligner.encode_text([concept_description])   # (1, 768)
        t_hat = model.vlm_aligner.project(t_k)[0]                    # (D,)
        return F.normalize(t_hat, dim=0)


def init_zeroshot_prototypes(model, held_out_k, concept_description, sigma=SIGMA):
    """
    Initialize held-out concept's prototypes from text anchor + noise (Eq. 13).
    p^(0)_{k'm} = (t̂_{k'} + σ ε_m) / ‖...‖
    """
    t_hat = get_text_anchor(model, concept_description)   # (D,) on CPU
    protos = model.prototype_learner.prototypes.data       # (K, M, D)

    new_protos = torch.zeros(M_PROTOS, PROTO_DIM)
    for m in range(M_PROTOS):
        eps_m = F.normalize(torch.randn(PROTO_DIM), dim=0)
        p = t_hat + sigma * eps_m
        new_protos[m] = F.normalize(p, dim=0)

    protos[held_out_k] = new_protos.to(DEVICE)
    return model


def finetune_concept_prototypes(model, train_ds, held_out_k, n_examples, seed=42):
    """
    Fine-tune ONLY the held-out concept's prototypes using n_examples labelled images.
    Returns the fine-tuned model (modified in-place copy).
    """
    if n_examples == 0:
        return model   # pure zero-shot, no fine-tuning

    rng = random.Random(seed)

    # Collect images that have the held-out concept annotated (bbox_eval split has boxes)
    # Use train split images that have concept label = 1 for held_out_k
    train_items = [
        train_ds[i] for i in range(len(train_ds))
        if train_ds[i]["concept_labels"][held_out_k] == 1
    ]
    rng.shuffle(train_items)
    finetune_items = train_items[:n_examples]

    if not finetune_items:
        return model

    # Only the held-out concept's prototypes are trainable
    for name, param in model.named_parameters():
        param.requires_grad = False
    # prototype shape: (K, M, D) — enable grad only for [held_out_k]
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

            # Compute similarity map for held-out concept's prototypes
            protos_k = model.prototype_learner.prototypes[held_out_k]  # (M, D)
            protos_k_norm = F.normalize(protos_k, dim=-1)

            fp_flat = fp[0].permute(1, 2, 0).reshape(-1, PROTO_DIM)   # (h*w, D)
            fp_norm = F.normalize(fp_flat, dim=-1)

            sim = fp_norm @ protos_k_norm.T                             # (h*w, M)
            max_sim = sim.max(dim=0).values                            # (M,) best patch per proto
            score = max_sim.mean()                                      # pull prototypes toward image

            loss = -score   # maximise similarity = minimise negative
            loss.backward()
            opt.step()

            # Re-normalize prototypes after update
            with torch.no_grad():
                model.prototype_learner.prototypes[held_out_k] = F.normalize(
                    model.prototype_learner.prototypes[held_out_k], dim=-1
                )

    model.eval()
    # Freeze everything again
    for param in model.parameters():
        param.requires_grad = False
    return model


# ── Load base model ───────────────────────────────────────────────────────────
def load_grape_base():
    path = os.path.join(CKPT_BASE, "tbx11k_no_C_novlm", "csr_stage4_best.pt")
    m = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=M_PROTOS, num_classes=NUM_CLASSES,
        proto_dim=PROTO_DIM, use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


# ── Datasets ──────────────────────────────────────────────────────────────────
print("Loading datasets...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)
bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

# ── Leave-one-concept-out loop ────────────────────────────────────────────────
results = {}

for held_out_k in range(NUM_CONCEPTS):
    cname = CONCEPT_NAMES[held_out_k]
    cdesc = CONCEPT_DESCRIPTIONS[held_out_k]
    print(f"\n=== Held-out concept: {cname} (idx={held_out_k}) ===")

    # Check if this concept has any bbox_eval samples
    _, n_pairs = evaluate_pg_for_concept(load_grape_base(), bbox_ds, held_out_k)
    if n_pairs == 0:
        print(f"  No bbox_eval pairs for concept {held_out_k}. Skipping.")
        results[held_out_k] = {"concept": cname, "n_pairs": 0, "pg": {}}
        continue

    # Also compute fully supervised PG (upper bound = original GRAPE)
    model_full = load_grape_base()
    pg_full, _ = evaluate_pg_for_concept(model_full, bbox_ds, held_out_k)
    print(f"  Fully supervised PG (upper bound) = {pg_full:.4f}  ({n_pairs} pairs)")

    concept_results = {"concept": cname, "n_pairs": n_pairs,
                       "pg_full_supervised": pg_full, "pg": {}}

    for n_ex in FEW_SHOT_COUNTS:
        seed_pgs = []
        n_seeds_to_use = 1 if n_ex == 0 else N_SEEDS   # zero-shot is deterministic
        for seed in range(n_seeds_to_use):
            model = load_grape_base()

            # Step 1: Zero-shot init from text anchor
            model = init_zeroshot_prototypes(model, held_out_k, cdesc)

            # Step 2: Fine-tune with n_ex labelled examples
            model = finetune_concept_prototypes(model, train_ds, held_out_k, n_ex, seed=seed)

            pg, _ = evaluate_pg_for_concept(model, bbox_ds, held_out_k)
            seed_pgs.append(pg)
            del model

        mean_pg = float(np.mean(seed_pgs))
        std_pg  = float(np.std(seed_pgs)) if len(seed_pgs) > 1 else 0.0
        concept_results["pg"][n_ex] = {"mean": mean_pg, "std": std_pg}
        print(f"  n_ex={n_ex:>2}: PG = {mean_pg:.4f} ± {std_pg:.4f}")

    results[held_out_k] = concept_results

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== C8 Zero-Shot Results (TBX11K leave-one-concept-out) ===")
print(f"{'Labelled examples':>18}  " +
      "  ".join(f"{c:>12}" for c in CONCEPT_NAMES))
header_done = False
for n_ex in FEW_SHOT_COUNTS:
    vals = []
    for k in range(NUM_CONCEPTS):
        if k in results and results[k]["n_pairs"] > 0 and n_ex in results[k]["pg"]:
            r = results[k]["pg"][n_ex]
            vals.append(f"{r['mean']:.3f}±{r['std']:.3f}")
        else:
            vals.append("N/A")
    print(f"{str(n_ex):>18}  " + "  ".join(f"{v:>12}" for v in vals))

with open("results/c8_zeroshot_results.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

print("\nC8 done. Results in results/c8_zeroshot_results.json")
