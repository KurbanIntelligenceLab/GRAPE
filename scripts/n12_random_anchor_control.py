"""
N12 -- Random-Anchor Control for Module C (Table 21)

Without a control, there is no evidence the BioViL-T text anchor contributes
anything to zero-shot concept addition: the same leave-one-out protocol with a
random initialization might work just as well.

Adapts e2_tbx11k_zeroshot_5seeds.py's leave-one-concept-out protocol
(itself adapted from c8_zeroshot_eval.py) and adds the random-anchor control: for each held-out
concept, run BOTH text-anchored initialization (the real Module C
mechanism) and random-unit-vector initialization (the control), matched on
seed and n-shot count, and compare.

Usage:
  python3 scripts/n12_random_anchor_control.py --dataset tbx11k --data_dir data/tbx11k \
      --ckpt checkpoints/tbx11k_seed42/csr_stage4_best.pt \
      --results_path results/n12_random_anchor_tbx11k.json
"""

import sys, os, json, random, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.csr_baseline import CSRModel
from src.data.datasets import CSRDataset, CONCEPTS
from src.models.vlm_alignment import CONCEPT_DESCRIPTIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tbx11k"], default="tbx11k")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--n_seeds", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CONCEPTS = len(CONCEPTS[args.dataset])
    CONCEPT_NAMES = CONCEPTS[args.dataset]
    DESCRIPTIONS = CONCEPT_DESCRIPTIONS[args.dataset]
    FEW_SHOT_COUNTS = [0, 1, 5, 20]

    torch.manual_seed(42)
    random.seed(42)

    print(f"Dataset: {args.dataset}  K={NUM_CONCEPTS}  device={DEVICE}")

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    proto_dim = cfg.get("model", {}).get("proto_dim", 256)
    num_proto = cfg.get("model", {}).get("num_prototypes", 100)
    backbone = cfg.get("model", {}).get("backbone", "resnet50")

    print("Loading datasets...")
    train_ds = CSRDataset(args.data_dir, args.dataset, split="train", image_size=args.image_size)
    bbox_ds = CSRDataset(args.data_dir, args.dataset, split="bbox_eval", image_size=args.image_size)

    def load_base_model():
        m = CSRModel(
            num_concepts=NUM_CONCEPTS, num_prototypes=num_proto, num_classes=len(set(
                s["class_label"] for s in train_ds.samples)),
            backbone=backbone, proto_dim=proto_dim,
            use_gnn=True, use_uncertainty=True, use_vlm=True,
            pretrained_backbone=False,
        ).to(DEVICE)
        m.load_state_dict(ckpt["model_state"], strict=False)
        m.eval()
        return m

    @torch.no_grad()
    def get_sim_maps(model, image_t):
        img = image_t.unsqueeze(0).to(DEVICE)
        f, _, _ = model.concept_model(img)
        fp = model.projector.project_feature_map(f)
        sm = model.prototype_learner.get_similarity_maps(fp)
        return sm[0].cpu()

    def evaluate_pg_for_concept(model, held_out_k):
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
            bx1 = max(0, min(int(x1 * W / args.image_size), W - 1))
            bx2 = max(bx1 + 1, min(int(x2 * W / args.image_size), W))
            by1 = max(0, min(int(y1 * H / args.image_size), H - 1))
            by2 = max(by1 + 1, min(int(y2 * H / args.image_size), H))
            idx = np.argmax(raw)
            mh, mw = divmod(idx, W)
            hits += int((bx1 <= mw < bx2) and (by1 <= mh < by2))
            total += 1
        return (hits / total if total > 0 else 0.0), total

    def get_text_anchor(model, concept_description):
        with torch.no_grad():
            t_hat = model.vlm_aligner.zero_shot_prototype(concept_description)
        return F.normalize(t_hat, dim=0)

    def get_random_anchor(proto_dim_):
        vec = torch.randn(proto_dim_)
        return F.normalize(vec, dim=0)

    def init_prototypes(model, held_out_k, anchor_vec, seed, sigma):
        g = torch.Generator().manual_seed(seed)
        protos = model.prototype_learner.prototypes.data
        new_protos = torch.zeros(num_proto, proto_dim)
        for m in range(num_proto):
            eps_m = F.normalize(torch.randn(proto_dim, generator=g), dim=0)
            p = anchor_vec.cpu() + sigma * eps_m
            new_protos[m] = F.normalize(p, dim=0)
        protos[held_out_k] = new_protos.to(DEVICE)
        return model

    def finetune_concept_prototypes(model, held_out_k, n_examples, seed):
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

        for p in model.parameters():
            p.requires_grad = False
        model.prototype_learner.prototypes.requires_grad_(True)
        opt = torch.optim.Adam([model.prototype_learner.prototypes], lr=args.finetune_lr)

        model.train()
        for _ in range(args.finetune_epochs):
            for item in finetune_items:
                img = item["image"].unsqueeze(0).to(DEVICE)
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
        for p in model.parameters():
            p.requires_grad = False
        return model

    results = {}
    for held_out_k in range(NUM_CONCEPTS):
        cname = CONCEPT_NAMES[held_out_k]
        cdesc = DESCRIPTIONS.get(cname, "")
        print(f"\n=== Held-out concept: {cname} (idx={held_out_k}) ===")

        model_probe = load_base_model()
        _, n_pairs = evaluate_pg_for_concept(model_probe, held_out_k)
        del model_probe
        if n_pairs == 0:
            print("  No bbox_eval pairs. Skipping.")
            results[cname] = {"n_pairs": 0}
            continue

        text_anchor_model = load_base_model()
        text_anchor = get_text_anchor(text_anchor_model, cdesc)
        del text_anchor_model

        concept_results = {"n_pairs": n_pairs, "text_anchor": {}, "random_anchor": {}}

        for anchor_name, anchor_vec in [("text_anchor", text_anchor),
                                         ("random_anchor", get_random_anchor(proto_dim))]:
            for n_ex in FEW_SHOT_COUNTS:
                seed_pgs = []
                n_seeds_to_use = 1 if n_ex == 0 else args.n_seeds
                for seed in range(n_seeds_to_use):
                    model = load_base_model()
                    model = init_prototypes(model, held_out_k, anchor_vec, seed, args.sigma)
                    model = finetune_concept_prototypes(model, held_out_k, n_ex, seed)
                    pg, _ = evaluate_pg_for_concept(model, held_out_k)
                    seed_pgs.append(pg)
                    del model
                mean_pg = float(np.mean(seed_pgs))
                std_pg = float(np.std(seed_pgs)) if len(seed_pgs) > 1 else 0.0
                concept_results[anchor_name][n_ex] = {
                    "mean": mean_pg, "std": std_pg, "all_seeds": seed_pgs}
                print(f"  [{anchor_name}] n_ex={n_ex:>2}: PG = {mean_pg:.4f} +/- {std_pg:.4f}  seeds={seed_pgs}")

        results[cname] = concept_results

    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.results_path}")


if __name__ == "__main__":
    main()
