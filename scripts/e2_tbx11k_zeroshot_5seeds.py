"""
E2 — Module C Zero-Shot (TBX11K) re-run with 5 seeds

Identical to c8_zeroshot_eval.py but uses N_SEEDS=5 instead of 3,
to determine whether the non-monotonic 1-shot > 5-shot trend
(0.079 > 0.063) is seed noise or a systematic effect.

Results → results/e2_zeroshot_5seeds.json
"""

import sys, os, json, random, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = "data/tbx11k"
CKPT_BASE    = "checkpoints"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE   = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
M_PROTOS     = 100
PROTO_DIM    = 256
SIGMA        = 0.1
FINETUNE_EPOCHS = 10
FINETUNE_LR     = 1e-4
FEW_SHOT_COUNTS = [0, 1, 5, 20]
N_SEEDS         = 5   # increased from 3

os.makedirs("results", exist_ok=True)
torch.manual_seed(42)
random.seed(42)

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

    for name, param in model.named_parameters():
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


print("Loading datasets...")
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)
bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)

results = {}

for held_out_k in range(NUM_CONCEPTS):
    cname = CONCEPT_NAMES[held_out_k]
    cdesc = CONCEPT_DESCRIPTIONS[held_out_k]
    print(f"\n=== Held-out concept: {cname} (idx={held_out_k}) ===")

    _, n_pairs = evaluate_pg_for_concept(load_grape_base(), bbox_ds, held_out_k)
    if n_pairs == 0:
        print(f"  No bbox_eval pairs. Skipping.")
        results[held_out_k] = {"concept": cname, "n_pairs": 0, "pg": {}}
        continue

    model_full = load_grape_base()
    pg_full, _ = evaluate_pg_for_concept(model_full, bbox_ds, held_out_k)
    print(f"  Fully supervised PG = {pg_full:.4f}  ({n_pairs} pairs)")

    concept_results = {"concept": cname, "n_pairs": n_pairs,
                       "pg_full_supervised": pg_full, "pg": {}}

    for n_ex in FEW_SHOT_COUNTS:
        seed_pgs = []
        n_seeds_to_use = 1 if n_ex == 0 else N_SEEDS
        for seed in range(n_seeds_to_use):
            model = load_grape_base()
            model = init_zeroshot_prototypes(model, held_out_k, cdesc)
            model = finetune_concept_prototypes(model, train_ds, held_out_k, n_ex, seed=seed)
            pg, _ = evaluate_pg_for_concept(model, bbox_ds, held_out_k)
            seed_pgs.append(pg)
            del model

        mean_pg = float(np.mean(seed_pgs))
        std_pg  = float(np.std(seed_pgs)) if len(seed_pgs) > 1 else 0.0
        concept_results["pg"][n_ex] = {"mean": mean_pg, "std": std_pg,
                                        "all_seeds": seed_pgs}
        print(f"  n_ex={n_ex:>2}: PG = {mean_pg:.4f} ± {std_pg:.4f}  seeds={seed_pgs}")

    results[held_out_k] = concept_results

print("\n=== E2 Zero-Shot Results (TBX11K, 5 seeds) ===")
print(f"{'n_shots':>8}  " + "  ".join(f"{c:>22}" for c in CONCEPT_NAMES))
for n_ex in FEW_SHOT_COUNTS:
    vals = []
    for k in range(NUM_CONCEPTS):
        if k in results and results[k]["n_pairs"] > 0 and n_ex in results[k]["pg"]:
            r = results[k]["pg"][n_ex]
            vals.append(f"{r['mean']:.3f}±{r['std']:.3f}")
        else:
            vals.append("N/A")
    print(f"{str(n_ex):>8}  " + "  ".join(f"{v:>22}" for v in vals))

# Monotonicity check for active_tb
if 0 in results and results[0]["n_pairs"] > 0:
    pg0 = results[0]["pg"]
    print(f"\nMonotonicity check (active_tb): "
          f"0→{pg0.get(0,{}).get('mean',0):.3f}, "
          f"1→{pg0.get(1,{}).get('mean',0):.3f}, "
          f"5→{pg0.get(5,{}).get('mean',0):.3f}, "
          f"20→{pg0.get(20,{}).get('mean',0):.3f}")
    is_mono = (pg0.get(1,{}).get('mean',0) >= pg0.get(0,{}).get('mean',0) and
               pg0.get(5,{}).get('mean',0) >= pg0.get(1,{}).get('mean',0) and
               pg0.get(20,{}).get('mean',0) >= pg0.get(5,{}).get('mean',0))
    print(f"Monotonic: {is_mono}")

with open("results/e2_zeroshot_5seeds.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

print("\nE2 done. Results in results/e2_zeroshot_5seeds.json")
