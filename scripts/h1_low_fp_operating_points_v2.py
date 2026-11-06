"""
H1 v3 -- Module B low-FP operating points, with a confusion-weighted
reassignment option (open_items.md #14, second half).

Identical to h1_low_fp_operating_points.py except for one addition: the
simulated misdrawn claim can be drawn either uniformly over the K-1 wrong
concepts (the paper's existing protocol, --reassign_mode uniform, and the
default so old commands reproduce old results unchanged) or weighted by the
training co-occurrence matrix A (--reassign_mode confusion), so a clinician
"error" is more likely to name a concept that actually co-occurs with the
true one, a more realistic and more adversarial reassignment than uniform
random, per the manuscript's own \\RF{} note that a uniform reassignment
"is not how clinicians err."

Usage (reproduces the original uniform-reassignment results exactly):
  python3 scripts/h1_low_fp_operating_points_v2.py --dataset tbx11k --data_dir data/tbx11k \
      --ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --results_path results/h1_low_fp_tbx11k_v2_uniform.json \
      --reassign_mode uniform

New confusion-weighted variant:
  python3 scripts/h1_low_fp_operating_points_v2.py --dataset tbx11k --data_dir data/tbx11k \
      --ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --results_path results/h1_low_fp_tbx11k_confusion.json \
      --reassign_mode confusion
"""

import sys, os, json, random, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tbx11k", "nih"], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--train_data_dir", default=None,
                   help="Data dir for building the co-occurrence graph from training labels "
                        "(defaults to --data_dir; use e.g. data/nih_official for NIH since "
                        "the bbox_eval split alone is too small to build a graph from).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--graph_tau", type=float, default=None)
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp_targets", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--reassign_mode", choices=["uniform", "confusion"], default="uniform",
                    help="uniform: draw the misdrawn claim uniformly over the K-1 wrong "
                         "concepts (original protocol). confusion: draw it weighted by "
                         "the training co-occurrence matrix A (row for the true concept, "
                         "self-entry excluded, renormalized).")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS[args.dataset])
    NUM_CLASSES  = len(CLASS_NAMES[args.dataset])
    GRAPH_TAU    = args.graph_tau if args.graph_tau is not None else (0.10 if args.dataset == "tbx11k" else 0.05)
    MISC_RATES   = [0.0, 0.10, 0.25, 0.50, 1.0]
    THETA_V_SWEEP = [None, 0.0000025, 0.000005, 0.0000075, 0.00001, 0.0000125,
                     0.000015, 0.0000175, 0.00002, 0.0000225, 0.000025, 0.00003,
                     0.000035, 0.00004, 0.000045, 0.00005, 0.00010, 0.00015, 0.00020,
                     0.00025, 0.00030, 0.00035, 0.00040, 0.00045, 0.00050, 0.00060,
                     0.00080, 0.00100]

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"Dataset: {args.dataset}  K={NUM_CONCEPTS}  eta={args.eta}  tau={GRAPH_TAU}  "
          f"reassign_mode={args.reassign_mode}  device={DEVICE}")

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    proto_dim = cfg.get("model", {}).get("proto_dim", 256)
    num_proto = cfg.get("model", {}).get("num_prototypes", 100)
    backbone  = cfg.get("model", {}).get("backbone", "resnet50")

    model = CSRModel(
        num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
        backbone=backbone, proto_dim=proto_dim,
        use_gnn=True, use_uncertainty=True, use_vlm=False,
        pretrained_backbone=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    train_dir = args.train_data_dir or args.data_dir
    train_ds  = CSRDataset(train_dir, args.dataset, split="train", image_size=args.image_size)
    train_ldr = get_dataloader(train_ds, 128, 4, shuffle=False)
    all_labels = torch.cat([b["concept_labels"] for b in train_ldr]).to(DEVICE)
    ei, ew = build_cooccurrence_graph(all_labels, threshold=GRAPH_TAU)
    ew = normalize_edge_weights(ei, ew, NUM_CONCEPTS)
    model.set_concept_graph(ei, ew)

    # Full (unthresholded) co-occurrence matrix for confusion-weighted reassignment,
    # independent of the GRAPH_TAU-thresholded graph the model itself reads.
    labels_np = all_labels.float().cpu().numpy()
    co = labels_np.T @ labels_np
    A_full = co / np.maximum(labels_np.sum(0)[None, :], 1e-9)
    np.fill_diagonal(A_full, 0.0)  # exclude self before renormalizing rows below

    def confusion_weights(true_concept):
        row = A_full[true_concept].copy()
        if row.sum() <= 0:
            # no co-occurrence signal for this concept -> fall back to uniform
            row = np.ones(NUM_CONCEPTS)
            row[true_concept] = 0.0
        return row / row.sum()

    @torch.no_grad()
    def get_sim_maps(image_t):
        img = image_t.unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
        return sm[0].cpu()

    def run_safety_check(sim_maps, bbox_224, intended_concept, eta, theta_v=None):
        K, M, H, W = sim_maps.shape
        x1, y1, x2, y2 = bbox_224
        bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
        bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
        by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
        by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))

        region = sim_maps[:, :, by1:by2, bx1:bx2]
        proto_means = region.mean(dim=(-2, -1))
        s_bar = proto_means.mean(dim=1)
        dominant = int(s_bar.argmax())
        dom_score = float(s_bar[dominant])
        int_score = float(s_bar[intended_concept])
        mismatch = (dominant != intended_concept) and (dom_score - int_score > eta)

        if mismatch and theta_v is not None:
            v_intended = float(proto_means[intended_concept].var().item())
            if v_intended >= theta_v:
                mismatch = False
        return mismatch

    print("Loading bbox_eval split...")
    bbox_ds = CSRDataset(args.data_dir, args.dataset, split="bbox_eval", image_size=args.image_size)
    samples = []
    for i in range(len(bbox_ds)):
        item = bbox_ds[i]
        for concept_idx, bbox in {int(k): v for k, v in item["bbox"].items()}.items():
            samples.append((i, concept_idx, bbox))
    print(f"  Total (image, concept, bbox) triples: {len(samples)}")

    unique_imgs = sorted(set(s[0] for s in samples))
    print(f"  Pre-computing sim maps for {len(unique_imgs)} images...")
    sim_cache = {}
    for idx, img_i in enumerate(unique_imgs):
        if idx % 50 == 0:
            print(f"    {idx}/{len(unique_imgs)}", flush=True)
        sim_cache[img_i] = get_sim_maps(bbox_ds[img_i]["image"])

    all_concepts = list(range(NUM_CONCEPTS))
    confusion_cache = {c: confusion_weights(c) for c in all_concepts} if args.reassign_mode == "confusion" else None

    def draw_claimed(rng, np_rng, concept_idx):
        if args.reassign_mode == "uniform":
            wrong = [c for c in all_concepts if c != concept_idx]
            return rng.choice(wrong)
        else:
            return int(np_rng.choice(all_concepts, p=confusion_cache[concept_idx]))

    def run_experiment(theta_v):
        results_p = {}
        for p in MISC_RATES:
            tp_trials, fp_trials = [], []
            for trial in range(args.n_trials):
                rng = random.Random(args.seed + trial)
                np_rng = np.random.RandomState(args.seed + trial)
                tp_w = tp_t = fp_w = fp_t = 0
                for img_i, concept_idx, bbox in samples:
                    is_misdraw = rng.random() < p
                    if is_misdraw:
                        claimed = draw_claimed(rng, np_rng, concept_idx)
                        warned = run_safety_check(sim_cache[img_i], bbox, claimed,
                                                   args.eta, theta_v=theta_v)
                        tp_t += 1
                        if warned: tp_w += 1
                    else:
                        warned = run_safety_check(sim_cache[img_i], bbox, concept_idx,
                                                   args.eta, theta_v=theta_v)
                        fp_t += 1
                        if warned: fp_w += 1
                if tp_t > 0: tp_trials.append(tp_w / tp_t)
                if fp_t > 0: fp_trials.append(fp_w / fp_t)
            results_p[p] = {
                "tp_rate_mean": float(np.mean(tp_trials)) if tp_trials else None,
                "tp_rate_std":  float(np.std(tp_trials))  if tp_trials else None,
                "fp_rate_mean": float(np.mean(fp_trials)) if fp_trials else None,
                "fp_rate_std":  float(np.std(fp_trials))  if fp_trials else None,
            }
        return results_p

    print(f"\n=== H1 v3: Low-FP Operating Points ({args.dataset}, reassign_mode={args.reassign_mode}) ===")
    print(f"{'theta_V':>10}  {'FP@p=0':>8}  {'TP@p=1':>8}")
    print("-" * 32)

    all_results = {}
    for theta_v in THETA_V_SWEEP:
        label = "None (orig)" if theta_v is None else f"{theta_v:.5f}"
        res = run_experiment(theta_v)
        fp0 = res[0.0]["fp_rate_mean"] or 0.0
        tp1 = res[1.0]["tp_rate_mean"] or 0.0
        print(f"{label:>10}  {fp0:.3f}     {tp1:.3f}")
        all_results[str(theta_v)] = res

    operating_points = {}
    for target in args.fp_targets:
        candidates = [(tv, res) for tv, res in all_results.items() if tv != "None"]
        below = [(tv, res) for tv, res in candidates if (res[0.0]["fp_rate_mean"] or 1.0) <= target]
        if below:
            best = max(below, key=lambda kv: kv[1][0.0]["fp_rate_mean"] or 0.0)
            achieved = True
        else:
            best = min(candidates, key=lambda kv: abs((kv[1][0.0]["fp_rate_mean"] or 1.0) - target))
            achieved = False
        tv, res = best
        operating_points[str(target)] = {
            "theta_v": tv,
            "fp_rate": res[0.0]["fp_rate_mean"],
            "tp_rate": res[1.0]["tp_rate_mean"],
            "target_achieved": achieved,
        }
        print(f"\nFP target <= {target:.0%}: theta_V={tv}  FP={res[0.0]['fp_rate_mean']:.3f}  "
              f"TP={res[1.0]['tp_rate_mean']:.3f}  (target achieved: {achieved})")

    out = {
        "dataset": args.dataset,
        "reassign_mode": args.reassign_mode,
        "sweep": all_results,
        "operating_points": operating_points,
        "n_samples": len(samples),
        "eta": args.eta,
        "n_trials": args.n_trials,
        "misc_rates": MISC_RATES,
    }
    with open(args.results_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
