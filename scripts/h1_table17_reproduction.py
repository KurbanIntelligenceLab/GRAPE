"""
Module B gated operating-point sweep (re-run behind the low-FP rows of Table 18
and the re-run note in Table 17).

Sweeps theta_V, reports TP and FP at every miscorrection rate p, and reports the
minimum-FP theta_V subject to TP >= 0.85 on this checkpoint.
h1_low_fp_operating_points_v2.py targets fixed FP budgets instead.

Usage:
  python3 scripts/h1_table17_reproduction.py --dataset tbx11k --data_dir data/tbx11k \
      --ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --results_path results/h1_table17_reproduction_tbx11k.json
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
    # Finer grid around the 0.0006-0.0025 region where TP crosses back toward
    # 0.85 (previous coarse sweep: theta_V=0.0006 -> TP=0.445, theta_V=0.001 -> TP=0.722);
    # extended above 0.001 in case the TP>=0.85 crossover sits higher than the
    # original coarse grid covered.
    THETA_V_SWEEP = [None, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0011,
                     0.0012, 0.0013, 0.0014, 0.0015, 0.0017, 0.0020, 0.0025,
                     0.0030, 0.0040, 0.0050]

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"Dataset: {args.dataset}  K={NUM_CONCEPTS}  eta={args.eta}  tau={GRAPH_TAU}  device={DEVICE}")

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

    def run_experiment(theta_v):
        results_p = {}
        for p in MISC_RATES:
            tp_trials, fp_trials = [], []
            for trial in range(args.n_trials):
                rng = random.Random(args.seed + trial)
                tp_w = tp_t = fp_w = fp_t = 0
                for img_i, concept_idx, bbox in samples:
                    is_misdraw = rng.random() < p
                    if is_misdraw:
                        wrong = [c for c in all_concepts if c != concept_idx]
                        claimed = rng.choice(wrong)
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

    print(f"\n=== H1 v2: Low-FP Operating Points ({args.dataset}) ===")
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

    # For each FP target, find the theta_V whose FP@p=0 is closest to (and at or
    # below) the target, and report its TP@p=1. If none is at-or-below, report the
    # closest overall and flag it.
    operating_points = {}
    for target in args.fp_targets:
        candidates = [(tv, res) for tv, res in all_results.items() if tv != "None"]
        below = [(tv, res) for tv, res in candidates if (res[0.0]["fp_rate_mean"] or 1.0) <= target]
        if below:
            best = max(below, key=lambda kv: kv[1][0.0]["fp_rate_mean"] or 0.0)  # closest to target from below
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

    # Reproduce the paper's original stated selection rule: minimize FP subject
    # to TP >= 0.85. Report whichever theta_V (including "None"/ungated) clears
    # the constraint with lowest FP -- do not assume theta_V=2e-4 is the answer.
    tp85_candidates = [(tv, res) for tv, res in all_results.items()
                        if (res[1.0]["tp_rate_mean"] or 0.0) >= 0.85]
    if tp85_candidates:
        best_tp85 = min(tp85_candidates, key=lambda kv: kv[1][0.0]["fp_rate_mean"] or 1.0)
        tv85, res85 = best_tp85
        tp85_result = {
            "theta_v": tv85,
            "fp_rate": res85[0.0]["fp_rate_mean"],
            "tp_rate": res85[1.0]["tp_rate_mean"],
            "constraint_satisfied": True,
        }
        print(f"\nMinimize-FP-subject-to-TP>=0.85: theta_V={tv85}  "
              f"FP={res85[0.0]['fp_rate_mean']:.4f}  TP={res85[1.0]['tp_rate_mean']:.4f}")
    else:
        best_overall = max(all_results.items(), key=lambda kv: kv[1][1.0]["tp_rate_mean"] or 0.0)
        tv85, res85 = best_overall
        tp85_result = {
            "theta_v": tv85,
            "fp_rate": res85[0.0]["fp_rate_mean"],
            "tp_rate": res85[1.0]["tp_rate_mean"],
            "constraint_satisfied": False,
        }
        print(f"\nNo theta_V in the swept grid reaches TP>=0.85. "
              f"Closest: theta_V={tv85}  FP={res85[0.0]['fp_rate_mean']:.4f}  "
              f"TP={res85[1.0]['tp_rate_mean']:.4f}")

    out = {
        "dataset": args.dataset,
        "sweep": all_results,
        "operating_points": operating_points,
        "tp_geq_085_selection": tp85_result,
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
