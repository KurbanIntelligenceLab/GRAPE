"""
E3 — Module C downstream effects: interference, calibration, and graph-adjacency
shift after zero-shot concept addition.

Q3: After adding a new concept via Module C, what happens to downstream
    diagnostic classification, calibration, and competition with existing
    concepts?
Q4: Does Module C support reliable diagnosis with the newly added concept,
    or only localization of prototype similarity maps?

Protocol: reinitialize Effusion's M=100 prototypes from its BioViL-T text
anchor (Eq. 9), fine-tune with n in {0,1,5,20} labeled images (concept-local,
matching e1_nih_zeroshot_loo.py), all other parameters frozen. Measure, before
vs. after, for each n:
  1. Macro-F1 on the *class* head (GRAPE's binary no-finding/finding task) on
     the held-out test split -- this is the only classification head GRAPE
     exposes; it is not per-concept, but it is what the GAT task head
     actually produces and is therefore the correct "interference" measure
     for whether the graph-structured classifier's output degrades when one
     concept's prototype geometry changes.
  2. Expected Calibration Error (ECE) of that class prediction.
  3. Localization (Pointing Game) for Effusion itself, at each n (reproduces
     e1's per-concept result as a sanity check).
  4. Dominant-concept-assignment shift inside annotated boxes for concepts
     graph-adjacent to Effusion in the co-occurrence graph (per Appendix G,
     Effusion-Atelectasis is a documented strong edge): for each adjacent
     concept's own bbox-annotated images, check whether the model's dominant
     concept inside the box flips between before/after Effusion is modified.

Usage:
  python3 scripts/e3_moduleC_downstream_effects.py \
      --ckpt checkpoints/nih_seed42/csr_stage4_best.pt \
      --train_data_dir data/nih_official --test_data_dir data/nih_official \
      --bbox_data_dir data/nih_bbox --results_path results/e3_moduleC_nih.json
"""

import sys, os, json, random, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights
from src.utils.metrics import macro_f1, expected_calibration_error, per_concept_f1

HELD_OUT_CONCEPT = 4  # Effusion
GRAPH_ADJACENT_CONCEPTS = {0: "Atelectasis"}  # per Appendix G: Effusion-Atelectasis strong edge

NIH_CONCEPT_DESCRIPTIONS = {
    0:  "Atelectasis: partial or complete lung collapse with opacity, volume loss, "
        "and displacement of fissures or mediastinum on chest X-ray.",
    4:  "Pleural effusion: blunting of costophrenic angle, homogeneous opacity "
        "layering dependently on chest X-ray.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train_data_dir", required=True)
    p.add_argument("--test_data_dir", required=True)
    p.add_argument("--bbox_data_dir", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--graph_tau", type=float, default=0.05)
    p.add_argument("--few_shot_counts", type=int, nargs="+", default=[0, 1, 5, 20])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--sigma_init", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS["nih"])
    NUM_CLASSES = len(CLASS_NAMES["nih"])
    concept_names = CONCEPTS["nih"]

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)

    print(f"K={NUM_CONCEPTS}  C={NUM_CLASSES}  held-out concept: "
          f"{concept_names[HELD_OUT_CONCEPT]} (idx={HELD_OUT_CONCEPT})  device={DEVICE}")

    base_ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = base_ckpt.get("config", {})
    proto_dim = cfg.get("model", {}).get("proto_dim", 256)
    num_proto = cfg.get("model", {}).get("num_prototypes", 100)
    backbone = cfg.get("model", {}).get("backbone", "resnet50")

    print("Loading datasets...")
    train_ds = CSRDataset(args.train_data_dir, "nih", split="train", image_size=args.image_size)
    test_ds  = CSRDataset(args.test_data_dir, "nih", split="test", image_size=args.image_size)
    bbox_ds  = CSRDataset(args.bbox_data_dir, "nih", split="bbox_eval", image_size=args.image_size)

    train_loader = get_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=False)
    all_labels = torch.cat([b["concept_labels"] for b in train_loader]).to(DEVICE)
    ei, ew = build_cooccurrence_graph(all_labels, threshold=args.graph_tau)
    ew_norm = normalize_edge_weights(ei, ew, NUM_CONCEPTS)

    def load_base_model():
        m = CSRModel(
            num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=NUM_CLASSES,
            backbone=backbone, proto_dim=proto_dim,
            use_gnn=True, use_uncertainty=True, use_vlm=True,
            pretrained_backbone=False,
        ).to(DEVICE)
        m.load_state_dict(base_ckpt["model_state"], strict=False)
        m.set_concept_graph(ei, ew_norm)
        m.eval()
        return m

    def get_text_anchor(model, concept_description):
        if not hasattr(model, "vlm_aligner") or model.vlm_aligner is None:
            vec = torch.randn(proto_dim)
            return F.normalize(vec, dim=0)
        with torch.no_grad():
            t_hat = model.vlm_aligner.zero_shot_prototype(concept_description)
            return F.normalize(t_hat, dim=0).cpu()

    def init_zeroshot_prototypes(model, held_out_k, concept_description):
        t_hat = get_text_anchor(model, concept_description)
        protos = model.prototype_learner.prototypes.data
        new_protos = torch.zeros(num_proto, proto_dim)
        for m_i in range(num_proto):
            eps_m = F.normalize(torch.randn(proto_dim), dim=0)
            p = t_hat + args.sigma_init * eps_m
            new_protos[m_i] = F.normalize(p, dim=0)
        protos[held_out_k] = new_protos.to(DEVICE)
        return model

    def finetune_concept_prototypes(model, held_out_k, n_examples, seed=42):
        if n_examples == 0:
            return model
        rng = random.Random(seed)
        idxs = [i for i in range(len(train_ds)) if train_ds[i]["concept_labels"][held_out_k] == 1]
        rng.shuffle(idxs)
        finetune_idxs = idxs[:n_examples]
        if not finetune_idxs:
            return model

        for param in model.parameters():
            param.requires_grad = False
        model.prototype_learner.prototypes.requires_grad_(True)
        opt = torch.optim.Adam([model.prototype_learner.prototypes], lr=args.finetune_lr)

        model.train()
        for _ in range(args.finetune_epochs):
            for i in finetune_idxs:
                img = train_ds[i]["image"].unsqueeze(0).to(DEVICE)
                opt.zero_grad()
                with torch.no_grad():
                    f, _, _ = model.concept_model(img)
                    fp = model.projector.project_feature_map(f)
                protos_k = model.prototype_learner.prototypes[held_out_k]
                protos_k_norm = F.normalize(protos_k, dim=-1)
                fp_flat = fp[0].permute(1, 2, 0).reshape(-1, proto_dim)
                fp_norm = F.normalize(fp_flat, dim=-1)
                sim = fp_norm @ protos_k_norm.T
                max_sim = sim.max(dim=0).values
                loss = -max_sim.mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    model.prototype_learner.prototypes[held_out_k] = F.normalize(
                        model.prototype_learner.prototypes[held_out_k], dim=-1)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @torch.no_grad()
    def eval_class_head(model):
        """Macro-F1 and ECE on the class head, over the held-out test split."""
        all_logits, all_labels_list = [], []
        test_loader = get_dataloader(test_ds, args.batch_size, args.num_workers, shuffle=False)
        for batch in test_loader:
            imgs = batch["image"].to(DEVICE)
            labels = batch["class_label"].to(DEVICE)
            out = model(imgs)
            all_logits.append(out["logits"].cpu())
            all_labels_list.append(labels.cpu())
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels_list)
        f1 = macro_f1(logits, labels, num_classes=NUM_CLASSES)

        probs = torch.softmax(logits, dim=1)
        confidences, preds = probs.max(dim=1)
        accuracies = (preds == labels).float()
        ece = expected_calibration_error(confidences.numpy(), accuracies.numpy())
        return f1, ece

    @torch.no_grad()
    def eval_other_concepts_f1(model, held_out_k):
        """Per-concept F1 on the 13 pre-existing (non-held-out) concepts, derived
        from prototype max-similarity scores thresholded at 0. This measures
        interference on the other concepts, not the aggregate binary class head, which is a different quantity."""
        all_scores, all_labels_list = [], []
        test_loader = get_dataloader(test_ds, args.batch_size, args.num_workers, shuffle=False)
        for batch in test_loader:
            imgs = batch["image"].to(DEVICE)
            labels = batch["concept_labels"]
            out = model(imgs)
            all_scores.append(out["sim_scores"].amax(dim=-1).cpu())  # (B, K) max over M
            all_labels_list.append(labels)
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels_list)
        # Per-concept score threshold: use each concept's own median as the
        # binary decision boundary (scores are cosine similarities in [-1,1]
        # with no shared calibrated threshold across concepts).
        thresholds = scores.median(dim=0).values
        preds = (scores >= thresholds.unsqueeze(0)).float()
        f1_per_concept = per_concept_f1(preds, labels, threshold=0.5)
        other_concepts = [k for k in range(NUM_CONCEPTS) if k != held_out_k]
        macro_over_others = float(np.mean([f1_per_concept[k] for k in other_concepts]))
        return macro_over_others, {concept_names[k]: f1_per_concept[k] for k in other_concepts}

    @torch.no_grad()
    def get_sim_maps(model, image_t):
        img = image_t.unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
        return sm[0].cpu()

    def dominant_concept_in_box(sim_maps, bbox_224):
        K, M, H, W = sim_maps.shape
        x1, y1, x2, y2 = bbox_224
        bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
        bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
        by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
        by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))
        region = sim_maps[:, :, by1:by2, bx1:bx2]
        s_bar = region.mean(dim=(1, 2, 3))
        return int(s_bar.argmax())

    def eval_pg_for_concept(model, held_out_k):
        hits, total = 0, 0
        for i in range(len(bbox_ds)):
            item = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if held_out_k not in bboxes:
                continue
            sm = get_sim_maps(model, item["image"])
            raw = sm[held_out_k].amax(0).numpy()
            H, W = raw.shape
            x1, y1, x2, y2 = bboxes[held_out_k]
            bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
            by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))
            idx = int(np.argmax(raw))
            mh, mw = divmod(idx, W)
            hits += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
            total += 1
        return (hits / total if total > 0 else 0.0), total

    def eval_adjacent_dominant_shift(model_before, model_after, adjacent_k):
        """For images with a bbox for `adjacent_k`, does the dominant concept
        inside that box change between before/after modifying Effusion?"""
        flips, total = 0, 0
        for i in range(len(bbox_ds)):
            item = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if adjacent_k not in bboxes:
                continue
            sm_before = get_sim_maps(model_before, item["image"])
            sm_after  = get_sim_maps(model_after, item["image"])
            dom_before = dominant_concept_in_box(sm_before, bboxes[adjacent_k])
            dom_after  = dominant_concept_in_box(sm_after, bboxes[adjacent_k])
            flips += int(dom_before != dom_after)
            total += 1
        return (flips / total if total > 0 else 0.0), total

    # ── Baseline (before any Module C modification) ─────────────────────────
    print("\n=== Baseline (before Module C) ===")
    model_baseline = load_base_model()
    f1_before, ece_before = eval_class_head(model_baseline)
    other_f1_before, other_f1_before_per_concept = eval_other_concepts_f1(model_baseline, HELD_OUT_CONCEPT)
    pg_before, pg_before_n = eval_pg_for_concept(model_baseline, HELD_OUT_CONCEPT)
    print(f"  Class head: F1={f1_before:.4f}  ECE={ece_before:.4f}")
    print(f"  Other-13-concepts macro-F1 (prototype scores): {other_f1_before:.4f}")
    print(f"  Effusion PG (fully supervised): {pg_before:.4f} ({pg_before_n} pairs)")

    results = {
        "held_out_concept": concept_names[HELD_OUT_CONCEPT],
        "baseline": {"class_f1": f1_before, "class_ece": ece_before,
                     "other13_concepts_f1": other_f1_before,
                     "other13_concepts_f1_per_concept": other_f1_before_per_concept,
                     "effusion_pg_full_supervised": pg_before, "n_pairs": pg_before_n},
        "per_n_shot": {},
    }

    # ── For each n-shot count ─────────────────────────────────────────────────
    for n_ex in args.few_shot_counts:
        print(f"\n=== n_shots={n_ex} ===")
        n_seeds_to_use = 1 if n_ex == 0 else args.n_seeds
        seed_results = []
        for seed in range(n_seeds_to_use):
            model = load_base_model()
            model = init_zeroshot_prototypes(model, HELD_OUT_CONCEPT,
                                              NIH_CONCEPT_DESCRIPTIONS[HELD_OUT_CONCEPT])
            model = finetune_concept_prototypes(model, HELD_OUT_CONCEPT, n_ex, seed=seed)

            f1_after, ece_after = eval_class_head(model)
            other_f1_after, other_f1_after_per_concept = eval_other_concepts_f1(model, HELD_OUT_CONCEPT)
            pg_after, pg_after_n = eval_pg_for_concept(model, HELD_OUT_CONCEPT)

            adjacency_shifts = {}
            for adj_k, adj_name in GRAPH_ADJACENT_CONCEPTS.items():
                shift_rate, shift_n = eval_adjacent_dominant_shift(model_baseline, model, adj_k)
                adjacency_shifts[adj_name] = {"flip_rate": shift_rate, "n_pairs": shift_n}

            seed_results.append({
                "seed": seed,
                "class_f1": f1_after,
                "class_ece": ece_after,
                "other13_concepts_f1": other_f1_after,
                "other13_concepts_f1_per_concept": other_f1_after_per_concept,
                "effusion_pg": pg_after,
                "adjacency_shifts": adjacency_shifts,
            })
            print(f"  seed={seed}: class F1={f1_after:.4f} (Δ={f1_after-f1_before:+.4f})  "
                  f"other13 F1={other_f1_after:.4f} (Δ={other_f1_after-other_f1_before:+.4f})  "
                  f"ECE={ece_after:.4f} (Δ={ece_after-ece_before:+.4f})  "
                  f"Effusion PG={pg_after:.4f}")
            for adj_name, v in adjacency_shifts.items():
                print(f"    {adj_name} dominant-concept flip rate: {v['flip_rate']:.4f} "
                      f"({v['n_pairs']} pairs)")
            del model

        f1s = [r["class_f1"] for r in seed_results]
        other_f1s = [r["other13_concepts_f1"] for r in seed_results]
        eces = [r["class_ece"] for r in seed_results]
        pgs = [r["effusion_pg"] for r in seed_results]
        results["per_n_shot"][str(n_ex)] = {
            "class_f1_mean": float(np.mean(f1s)), "class_f1_std": float(np.std(f1s)),
            "class_f1_delta_from_baseline": float(np.mean(f1s) - f1_before),
            "other13_concepts_f1_mean": float(np.mean(other_f1s)),
            "other13_concepts_f1_std": float(np.std(other_f1s)),
            "other13_concepts_f1_delta_from_baseline": float(np.mean(other_f1s) - other_f1_before),
            "class_ece_mean": float(np.mean(eces)), "class_ece_std": float(np.std(eces)),
            "class_ece_delta_from_baseline": float(np.mean(eces) - ece_before),
            "effusion_pg_mean": float(np.mean(pgs)), "effusion_pg_std": float(np.std(pgs)),
            "seed_results": seed_results,
        }

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.results_path}")

    print("\n=== E3 Summary ===")
    print(f"Baseline class F1={f1_before:.4f}  ECE={ece_before:.4f}  "
          f"Effusion PG={pg_before:.4f}")
    print(f"{'n_shots':>8} {'class_F1':>10} {'ΔF1':>8} {'ECE':>8} {'ΔECE':>8} {'Effusion_PG':>12}")
    for n_ex in args.few_shot_counts:
        r = results["per_n_shot"][str(n_ex)]
        print(f"{n_ex:>8} {r['class_f1_mean']:>10.4f} {r['class_f1_delta_from_baseline']:>+8.4f} "
              f"{r['class_ece_mean']:>8.4f} {r['class_ece_delta_from_baseline']:>+8.4f} "
              f"{r['effusion_pg_mean']:>12.4f}")


if __name__ == "__main__":
    main()
