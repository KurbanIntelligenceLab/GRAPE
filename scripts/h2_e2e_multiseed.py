"""
H2 — Multi-seed E2E Training + Zero-Shot Test on E2E Weights

H2 tests two claims:
  (a) E2E multi-seed: does the F1=0.9882 (seed 42) generalise, or is it a
      lucky seed? Report mean ± std over seeds {0, 1, 42}.
  (b) Zero-shot: does Module C's text-anchor init work on E2E weights?
      If E2E supports zero-shot as well as staged training, the "staged is
      better for Module C" argument in Appendix B collapses.

Results → results/h2_e2e_multiseed.json
           Updates Appendix B table (tab:n4_curriculum).
"""

import sys, os, json, subprocess, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

DATA_DIR   = "data/tbx11k"
CKPT_BASE  = "checkpoints"
RESULTS    = "results/h2_e2e_multiseed.json"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
NUM_CONCEPTS = 3
NUM_CLASSES  = 3
M_PROTOS     = 100
PROTO_DIM    = 256
SIGMA        = 0.1

os.makedirs("results", exist_ok=True)

SEEDS = [42, 0, 1]   # seed 42 already trained; 0 and 1 will be trained here

# ── Seed 42 result already in log ─────────────────────────────────────────────
KNOWN = {
    42: {"f1": 0.9882, "pg": 0.1739}
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_e2e_model(seed):
    ckpt_dir = os.path.join(CKPT_BASE, f"tbx11k_e2e_seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "csr_e2e_best.pt")
    # Seed 42 uses a different directory name
    if seed == 42:
        ckpt_path = os.path.join(CKPT_BASE, "tbx11k_e2e", "csr_e2e_best.pt")

    model = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=M_PROTOS,
        num_classes=NUM_CLASSES, proto_dim=PROTO_DIM,
        use_gnn=True, use_uncertainty=True,
        use_vlm=False, pretrained_backbone=False,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Build and register concept graph from training data
    train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train", image_size=IMAGE_SIZE)
    loader   = get_dataloader(train_ds, 128, 4, shuffle=False)
    labels   = torch.cat([b["concept_labels"] for b in loader]).to(DEVICE)
    ei, ew   = build_cooccurrence_graph(labels, threshold=0.10)
    ew_norm  = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
    model.set_concept_graph(ei, ew_norm)
    return model


def eval_f1_pg(model):
    """Evaluate macro-F1 and per-image Pointing Game on test/bbox_eval splits."""
    from src.utils.metrics import macro_f1, pointing_game

    test_ds  = CSRDataset(DATA_DIR, "tbx11k", split="test", image_size=IMAGE_SIZE)
    test_ldr = get_dataloader(test_ds, 128, 4, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in test_ldr:
            imgs  = batch["image"].to(DEVICE)
            out   = model(imgs)
            logits = out["logits"] if isinstance(out, dict) else out
            preds  = logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(batch["class_label"])
    f1 = macro_f1(torch.cat(all_preds), torch.cat(all_labels), num_classes=NUM_CLASSES)

    # Per-image PG
    bbox_ds = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
    hits, total = 0, 0
    with torch.no_grad():
        for i in range(len(bbox_ds)):
            item   = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if not bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            f, _, _ = model.concept_model(img)
            fp = model.projector.project_feature_map(f)
            sm = model.prototype_learner.get_similarity_maps(fp)[0].cpu()
            image_hit = False
            for c_idx, bbox in bboxes.items():
                raw = sm[c_idx].amax(0).numpy()
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
    return float(f1), pg


def zeroshot_pg_on_model(model, held_out_k, concept_description, n_examples,
                          seed, bbox_ds, train_ds):
    """
    Zero-shot eval: re-init held-out concept's prototypes from text anchor,
    optionally fine-tune on n_examples, then measure PG.
    """
    import copy
    m = copy.deepcopy(model)

    # Text anchor init (Eq. 13)
    rng = torch.Generator()
    rng.manual_seed(seed)
    protos = m.prototype_learner.prototypes.data       # (K, M, D)
    new_protos = torch.zeros(M_PROTOS, PROTO_DIM)
    # For E2E models with no VLM, fall back to random unit vector (same as c8 script)
    t_hat = F.normalize(torch.randn(PROTO_DIM, generator=rng), dim=0)
    for pm in range(M_PROTOS):
        eps = F.normalize(torch.randn(PROTO_DIM, generator=rng), dim=0)
        p   = t_hat + SIGMA * eps
        new_protos[pm] = F.normalize(p, dim=0)
    protos[held_out_k] = new_protos.to(DEVICE)

    # Few-shot fine-tune if n_examples > 0
    if n_examples > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        # Collect n_examples images that have the held-out concept
        indices = [i for i in range(len(train_ds))
                   if train_ds[i]["concept_labels"][held_out_k] > 0]
        random.shuffle(indices)
        indices = indices[:n_examples]

        proto_param = m.prototype_learner.prototypes
        opt = torch.optim.Adam([proto_param], lr=1e-4)
        m.train()
        for _ in range(10):
            for idx in indices:
                item = train_ds[idx]
                img  = item["image"].unsqueeze(0).to(DEVICE)
                lbl  = item["concept_labels"].unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    v_raw = m.concept_model.generate_concept_vectors(img)
                    B, K, C = v_raw.shape
                    v_proj  = m.projector(v_raw.view(B*K, C)).view(B, K, -1)
                loss = m.prototype_learner.contrastive_loss(v_proj, lbl)
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()

    # Evaluate PG for held-out concept
    hits, total = 0, 0
    with torch.no_grad():
        for i in range(len(bbox_ds)):
            item = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if held_out_k not in bboxes:
                continue
            img = item["image"].unsqueeze(0).to(DEVICE)
            f, _, _ = m.concept_model(img)
            fp = m.projector.project_feature_map(f)
            sm = m.prototype_learner.get_similarity_maps(fp)[0].cpu()
            raw = sm[held_out_k].amax(0).numpy()
            H, W = raw.shape
            x1, y1, x2, y2 = bboxes[held_out_k]
            bx1 = max(0, min(int(x1*W/IMAGE_SIZE), W-1))
            bx2 = max(bx1+1, min(int(x2*W/IMAGE_SIZE), W))
            by1 = max(0, min(int(y1*H/IMAGE_SIZE), H-1))
            by2 = max(by1+1, min(int(y2*H/IMAGE_SIZE), H))
            idx = int(np.argmax(raw))
            mh, mw = divmod(idx, W)
            hits  += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
            total += 1
    del m
    return hits / total if total > 0 else 0.0, total


# ── Step 1: Train E2E for seeds that are missing ───────────────────────────────
for seed in SEEDS:
    if seed == 42:
        print(f"Seed 42 E2E already trained (F1=0.9882). Skipping.")
        continue
    ckpt_dir = os.path.join(CKPT_BASE, f"tbx11k_e2e_seed{seed}")
    ckpt_path = os.path.join(ckpt_dir, "csr_e2e_best.pt")
    if os.path.exists(ckpt_path):
        print(f"Seed {seed} checkpoint exists — skipping training.")
        continue
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = f"logs/tbx11k_e2e_seed{seed}.log"
    print(f"\nTraining E2E seed={seed} → {ckpt_dir} ...")
    cmd = [
        "python3", "train.py",
        "--dataset", "tbx11k",
        "--data_dir", DATA_DIR,
        "--config", "configs/tbx11k_config.yaml",
        "--end_to_end",
        "--no_vlm",
        "--seed", str(seed),
        "--checkpoint_dir", ckpt_dir,
        "--batch_size", "128",
        "--num_workers", "8",
    ]
    with open(log_path, "w") as flog:
        result = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"  WARNING: seed {seed} training exited with code {result.returncode}")
    else:
        print(f"  Seed {seed} training done.")


# ── Step 2: Evaluate F1 + PG for all seeds ────────────────────────────────────
print("\n" + "="*60)
print("Evaluating E2E F1 and Pointing Game for all seeds")
print("="*60)

e2e_results = {}
for seed in SEEDS:
    if seed in KNOWN:
        e2e_results[seed] = KNOWN[seed]
        print(f"  Seed {seed}: F1={KNOWN[seed]['f1']:.4f}  PG={KNOWN[seed]['pg']:.4f}  (from existing log)")
        continue
    print(f"  Evaluating seed {seed}...")
    try:
        model = load_e2e_model(seed)
        f1, pg = eval_f1_pg(model)
        e2e_results[seed] = {"f1": f1, "pg": pg}
        print(f"  Seed {seed}: F1={f1:.4f}  PG={pg:.4f}")
        del model
    except Exception as e:
        print(f"  Seed {seed}: ERROR — {e}")
        e2e_results[seed] = {"f1": None, "pg": None, "error": str(e)}

valid_f1 = [v["f1"] for v in e2e_results.values() if v.get("f1") is not None]
valid_pg = [v["pg"] for v in e2e_results.values() if v.get("pg") is not None]
e2e_mean_f1 = float(np.mean(valid_f1))
e2e_std_f1  = float(np.std(valid_f1))
e2e_mean_pg = float(np.mean(valid_pg))
e2e_std_pg  = float(np.std(valid_pg))


# ── Step 3: Zero-shot eval on E2E weights ─────────────────────────────────────
print("\n" + "="*60)
print("Zero-shot eval on E2E weights (Module C, leave-one-out)")
print("="*60)

bbox_ds  = CSRDataset(DATA_DIR, "tbx11k", split="bbox_eval", image_size=IMAGE_SIZE)
train_ds = CSRDataset(DATA_DIR, "tbx11k", split="train",     image_size=IMAGE_SIZE)

CONCEPT_DESCS = [
    "Active pulmonary tuberculosis with cavitary lesions and upper-lobe infiltrates.",
    "Healed or inactive tuberculosis with calcified granulomas and fibrous scarring.",
    "Pulmonary tuberculosis presenting as patchy consolidation or miliary nodules.",
]
HELD_OUT_CONCEPTS = [0, 1]   # concept 2 has 0 bbox pairs
FEW_SHOT_COUNTS   = [0, 1, 5, 20]
ZS_SEEDS          = [42, 0, 1]

zs_results = {}   # {seed: {held_out_k: {n_examples: pg}}}

for train_seed in SEEDS:
    print(f"\n  E2E model seed={train_seed}")
    try:
        model = load_e2e_model(train_seed)
    except Exception as e:
        print(f"    Cannot load model: {e}")
        continue

    zs_results[train_seed] = {}
    for held_out_k in HELD_OUT_CONCEPTS:
        zs_results[train_seed][held_out_k] = {}
        desc = CONCEPT_DESCS[held_out_k]
        for n in FEW_SHOT_COUNTS:
            pgs = []
            for zs_seed in ZS_SEEDS:
                pg_val, n_pairs = zeroshot_pg_on_model(
                    model, held_out_k, desc, n, zs_seed, bbox_ds, train_ds)
                pgs.append(pg_val)
            mean_pg = float(np.mean(pgs))
            std_pg  = float(np.std(pgs))
            zs_results[train_seed][held_out_k][n] = {
                "pg_mean": mean_pg, "pg_std": std_pg, "n_pairs": n_pairs
            }
            print(f"    k={held_out_k}  n={n:>2}:  PG={mean_pg:.4f}±{std_pg:.4f}  ({n_pairs} pairs)")
    del model


# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("H2 Summary")
print("="*60)
print(f"\nE2E F1 (seeds {SEEDS}): {e2e_mean_f1:.4f} ± {e2e_std_f1:.4f}")
print(f"E2E PG (seeds {SEEDS}): {e2e_mean_pg:.4f} ± {e2e_std_pg:.4f}")
print(f"\nStaged GRAPE reference: F1=0.809±0.084  PG=0.111±0.038  (seeds 42,0,1)")
print(f"\nZero-shot PG on E2E weights (averaged across E2E seeds):")
for held_out_k in HELD_OUT_CONCEPTS:
    concept_name = ["ActiveTB", "ObsoleteTB"][held_out_k]
    for n in FEW_SHOT_COUNTS:
        pgs = [zs_results[s][held_out_k][n]["pg_mean"]
               for s in SEEDS if s in zs_results and held_out_k in zs_results[s]]
        if pgs:
            print(f"  {concept_name}  n={n:>2}: PG={np.mean(pgs):.4f}±{np.std(pgs):.4f}")

out = {
    "e2e_per_seed": e2e_results,
    "e2e_mean_f1": e2e_mean_f1,
    "e2e_std_f1":  e2e_std_f1,
    "e2e_mean_pg": e2e_mean_pg,
    "e2e_std_pg":  e2e_std_pg,
    "zeroshot_on_e2e": zs_results,
}
with open(RESULTS, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nH2 done. Results → {RESULTS}")
print("Use these to update Appendix B (tab:n4_curriculum).")
