"""
m5_moduleC_confound_controls.py -- two confound controls on top of e3_moduleC_downstream_effects.py's existing Effusion
zero-shot-addition experiment.

(i) Within-run comparison: for the SAME Effusion-addition runs (n in
    {1,5,20}, 3 seeds), report the dominant-concept flip rate for every one
    of the other 12 non-Atelectasis concepts, not just Atelectasis. If the
    non-neighbour flip rate is close to Atelectasis's, the effect is not
    specific to co-occurrence adjacency.
(ii) Negative control across runs: repeat the identical zero-shot-addition
    protocol, but for the concept with the WEAKEST co-occurrence edge to
    Atelectasis (rather than Effusion, Atelectasis's strongest neighbour),
    and check whether Atelectasis's own flip rate stays near the n=0
    baseline. If it also jumps to a third of boxes, the co-occurrence
    account is wrong.

Reuses the same model, dataset, and protocol code as
e3_moduleC_downstream_effects.py (same CSRModel, same LOO zero-shot-init +
fine-tune procedure, same dominant-concept-in-box logic) so the numbers are
directly comparable to Table tab:modc_interference.

Usage:
  python3 scripts/m5_moduleC_confound_controls.py \
      --ckpt checkpoints/nih_seed42/csr_stage4_best.pt \
      --train_data_dir data/nih_official --test_data_dir data/nih_official \
      --bbox_data_dir data/nih_cxr14 --results_path results/m5_moduleC_controls.json
"""

import sys, os, json, random, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, get_dataloader, CONCEPTS, CLASS_NAMES
from src.utils.graph_builder import build_cooccurrence_graph, normalize_edge_weights

HELD_OUT_CONCEPT = 4   # Effusion (positive: Atelectasis's strongest neighbour)
ATELECTASIS = 0

NIH_CONCEPT_DESCRIPTIONS = {
    0:  "Atelectasis: partial or complete lung collapse with opacity, volume loss, "
        "and displacement of fissures or mediastinum on chest X-ray.",
    4:  "Pleural effusion: blunting of costophrenic angle, homogeneous opacity "
        "layering dependently on chest X-ray.",
    6:  "Fibrosis: chronic reticular or reticulonodular opacity from scarring, "
        "with volume loss and architectural distortion on chest X-ray.",
    7:  "Hernia: displacement of abdominal or mediastinal contents through a "
        "diaphragmatic defect, visible as an anomalous opacity or gas shadow "
        "on chest X-ray.",
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

    print(f"K={NUM_CONCEPTS}  C={NUM_CLASSES}  device={DEVICE}")

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

    # Co-occurrence matrix (for picking the negative-control concept).
    labels_np = all_labels.float().cpu().numpy()
    co = labels_np.T @ labels_np
    A = co / np.maximum(labels_np.sum(0)[None, :], 1e-9)
    np.fill_diagonal(A, 0.0)
    edge_to_atelectasis = A[:, ATELECTASIS]
    print("\nCo-occurrence edge weight to Atelectasis, by concept:")
    order = np.argsort(edge_to_atelectasis)
    for k in order:
        print(f"    {concept_names[k]:<20} {edge_to_atelectasis[k]:.4f}"
              + ("  <- Effusion (positive: strongest edge)" if k == HELD_OUT_CONCEPT else ""))
    weakest_candidates = [k for k in order if k not in (ATELECTASIS, HELD_OUT_CONCEPT)]
    negative_control_concept = int(weakest_candidates[0])
    print(f"\nNegative control concept (weakest edge to Atelectasis, excluding "
          f"Atelectasis itself and Effusion): {concept_names[negative_control_concept]} "
          f"(idx={negative_control_concept}, edge={edge_to_atelectasis[negative_control_concept]:.4f})")

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

    def flip_rate_for_concept(model_before, model_after, target_k):
        flips, total = 0, 0
        for i in range(len(bbox_ds)):
            item = bbox_ds[i]
            bboxes = {int(k): v for k, v in item["bbox"].items()}
            if target_k not in bboxes:
                continue
            sm_before = get_sim_maps(model_before, item["image"])
            sm_after  = get_sim_maps(model_after, item["image"])
            dom_before = dominant_concept_in_box(sm_before, bboxes[target_k])
            dom_after  = dominant_concept_in_box(sm_after, bboxes[target_k])
            flips += int(dom_before != dom_after)
            total += 1
        return (flips / total if total > 0 else None), total

    def run_addition_experiment(held_out_k, description, track_concepts):
        """Full LOO zero-shot addition + fine-tune protocol for `held_out_k`,
        tracking dominant-concept flip rate for every concept in
        `track_concepts` (list of concept indices) at each n-shot count."""
        model_baseline = load_base_model()
        per_n = {}
        for n_ex in args.few_shot_counts:
            n_seeds_to_use = 1 if n_ex == 0 else args.n_seeds
            seed_flip_rates = {k: [] for k in track_concepts}
            for seed in range(n_seeds_to_use):
                model = load_base_model()
                model = init_zeroshot_prototypes(model, held_out_k, description)
                model = finetune_concept_prototypes(model, held_out_k, n_ex, seed=seed)
                for k in track_concepts:
                    if k == held_out_k:
                        continue
                    rate, n_pairs = flip_rate_for_concept(model_baseline, model, k)
                    if rate is not None:
                        seed_flip_rates[k].append(rate)
                del model
            per_n[str(n_ex)] = {
                concept_names[k]: {
                    "flip_rate_mean": float(np.mean(v)) if v else None,
                    "flip_rate_std": float(np.std(v)) if v else None,
                    "n_seeds": len(v),
                }
                for k, v in seed_flip_rates.items()
            }
            summary = "  ".join(f"{concept_names[k]}={per_n[str(n_ex)][concept_names[k]]['flip_rate_mean']:.3f}"
                                 for k in track_concepts if k != held_out_k
                                 and per_n[str(n_ex)][concept_names[k]]['flip_rate_mean'] is not None)
            print(f"  n={n_ex}: {summary}")
        return per_n

    results = {}

    # ── (i) Within-run: flip rate for every non-Atelectasis concept, same
    #        Effusion-addition runs as the manuscript's Table tab:modc_interference.
    print("\n=== (i) Within-run: Effusion addition, flip rate for ALL other 13 concepts ===")
    all_others = [k for k in range(NUM_CONCEPTS) if k != HELD_OUT_CONCEPT]
    effusion_all_flips = run_addition_experiment(
        HELD_OUT_CONCEPT, NIH_CONCEPT_DESCRIPTIONS[HELD_OUT_CONCEPT], all_others)
    results["effusion_addition_all_concept_flip_rates"] = effusion_all_flips

    # ── (ii) Negative control: add the weakest-edge-to-Atelectasis concept
    #         instead, and track Atelectasis's own flip rate.
    print(f"\n=== (ii) Negative control: add {concept_names[negative_control_concept]} "
          f"(weak edge to Atelectasis), track Atelectasis flip rate ===")
    neg_control_flips = run_addition_experiment(
        negative_control_concept, NIH_CONCEPT_DESCRIPTIONS[negative_control_concept],
        [ATELECTASIS])
    results["negative_control"] = {
        "concept_added": concept_names[negative_control_concept],
        "concept_added_idx": negative_control_concept,
        "edge_weight_to_atelectasis": float(edge_to_atelectasis[negative_control_concept]),
        "atelectasis_flip_rates": neg_control_flips,
    }
    results["edge_weights_to_atelectasis"] = {
        concept_names[k]: float(edge_to_atelectasis[k]) for k in range(NUM_CONCEPTS)
    }

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
