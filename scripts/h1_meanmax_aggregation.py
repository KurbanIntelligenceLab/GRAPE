"""
h1_meanmax_aggregation.py -- mean-vs-max box-region aggregation ablation
for the NIH safety curve (open_items.md #11, second half).

Eq. sbar (the manuscript's Sec. 3, "Module B") defines s_bar_k(B) as a
spatial MEAN over the box's patches, then a mean over the M prototypes.
The manuscript itself notes the classifier's own readout instead takes a
per-prototype spatial MAX before the GAT reads it, so this script asks
whether the safety check's numbers are an artefact of using the mean
aggregation rather than the classifier's own max aggregation.

Reports, for both aggregation rules and at p=0/p=1 (false alarms / detection,
uniform reassignment, ungated dominance rule -- no theta_V sweep, this is a
single-point comparison, not the operating-curve sweep), the resulting rate,
matching the "0.895/0.891 detection, 0.217/0.213 false alarms" comparison
referenced in the manuscript's now-removed \\RF{VERIFY} tag.

Usage:
  python3 scripts/h1_meanmax_aggregation.py --dataset nih --data_dir data/nih_bbox \
      --train_data_dir data/nih_official \
      --ckpt checkpoints/nih_seed42/csr_stage4_best.pt \
      --out results/h1_meanmax_nih.json
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
    p.add_argument("--train_data_dir", default=None)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--graph_tau", type=float, default=None)
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
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

    def run_safety_check(sim_maps, bbox_224, intended_concept, eta, agg):
        K, M, H, W = sim_maps.shape
        x1, y1, x2, y2 = bbox_224
        bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
        bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
        by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
        by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))

        region = sim_maps[:, :, by1:by2, bx1:bx2]
        if agg == "mean":
            proto_stat = region.mean(dim=(-2, -1))
        else:  # max, matching the classifier's own per-prototype spatial max readout
            proto_stat = region.amax(dim=(-2, -1))
        s_bar = proto_stat.mean(dim=1)
        dominant = int(s_bar.argmax())
        dom_score = float(s_bar[dominant])
        int_score = float(s_bar[intended_concept])
        mismatch = (dominant != intended_concept) and (dom_score - int_score > eta)
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

    def run_ungated(agg):
        tp_trials, fp_trials = [], []
        for trial in range(args.n_trials):
            rng = random.Random(args.seed + trial)
            tp_w = tp_t = fp_w = fp_t = 0
            for img_i, concept_idx, bbox in samples:
                wrong = [c for c in all_concepts if c != concept_idx]
                claimed = rng.choice(wrong)
                warned = run_safety_check(sim_cache[img_i], bbox, claimed, args.eta, agg)
                tp_t += 1
                if warned: tp_w += 1

                warned0 = run_safety_check(sim_cache[img_i], bbox, concept_idx, args.eta, agg)
                fp_t += 1
                if warned0: fp_w += 1
            tp_trials.append(tp_w / tp_t)
            fp_trials.append(fp_w / fp_t)
        return {
            "detection_mean": float(np.mean(tp_trials)),
            "detection_std": float(np.std(tp_trials)),
            "false_alarm_mean": float(np.mean(fp_trials)),
            "false_alarm_std": float(np.std(fp_trials)),
        }

    result = {
        "dataset": args.dataset,
        "eta": args.eta,
        "n_trials": args.n_trials,
        "mean_aggregation": run_ungated("mean"),
        "max_aggregation": run_ungated("max"),
    }
    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
