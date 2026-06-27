# GRAPE: Graph-Augmented Prototype Explanations for Interactive Medical Image Diagnosis

Reference implementation for the paper **"GRAPE: Graph-Augmented Prototype Explanations
for Interactive Medical Image Diagnosis"** (Khanbayov & Kurban).

GRAPE extends the Concept-based Similarity Reasoning (CSR) prototype classifier with three
modules that share a single four-stage training pipeline:

- **Module A — Graph Attention Task Head.** Replaces the flat linear readout with a 2-layer
  GAT over a concept co-occurrence graph built from training labels.
- **Module B — Concept-Mismatch Safety Check.** Uses per-patch prototype-disagreement
  variance to warn when a clinician-drawn box is dominated by a concept other than the
  claimed one, before feedback is applied. Inference-only.
- **Module C — Open-Vocabulary Prototype Anchoring.** Anchors prototypes to frozen BioViL-T
  text embeddings so a new concept can be added from a text description without retraining.

The classifier head operates on scalar prototype-similarity scores, so the spatial
similarity maps (and therefore the explanations) are never reshaped by the task gradient.

---

## Installation

```bash
git clone <repo-url> grape
cd grape
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e .                                    # installs deps from pyproject.toml
```

This installs the `src` package so scripts run from any working directory. To install only
the pinned dependencies without the package, use `pip install -r requirements.txt`.

The stack reported in the paper is **PyTorch 2.4.1 + CUDA 12.1, BF16 AMP, on an
NVIDIA A100 80GB**. Other CUDA versions work after adjusting the `torch`/`torchvision`
build (see https://pytorch.org/get-started/previous-versions/).

### Smoke test (no data required)

```bash
python train.py --dataset tbx11k --synthetic --epochs 2
```

---

## Data

Two public datasets are used. Neither is redistributed here; download each from its source
and prepare it into the layout below.

| Dataset | Source | Used for |
|---|---|---|
| **TBX11K** | Liu et al., *Revisiting Computer-Aided Tuberculosis Diagnosis*, TPAMI 2023 (research license) | Primary 3-class TB benchmark; Pointing Game |
| **NIH ChestX-ray14** | Wang et al., CVPR 2017 (NIH public-access release) | 14-finding classification; Pointing Game |

Expected directory layout (per dataset root):

```
<dataset_root>/
  images/        # .jpg / .jpeg / .png
  labels.csv     # columns: image_id, split, class_label, concept_0, concept_1, ...
  bboxes.csv     # columns: image_id, concept_idx, x1, y1, x2, y2   (TBX11K / bbox eval only)
```

Preparation / download helpers (all paths are passed by flag or environment variable — see
each script's `--help` or header):

```bash
python scripts/prepare_tbx11k.py     --src <raw_tbx11k> --dest data/tbx11k
python scripts/download_nih.py       --data-root data/nih_cxr14
python scripts/download_nih_bbox.py  --data-root data/nih_bbox --bbox-csv <BBox_List_2017.csv>
```

---

## Training

A full run executes Stages 1–4 (concept supervision → concept-vector generation →
prototype learning with VLM alignment → graph-structured classification):

```bash
# TBX11K (K=3), all modules
python train.py --dataset tbx11k --data_dir data/tbx11k --config configs/tbx11k_config.yaml

# NIH ChestX-ray14 (K=14), all modules
python train.py --dataset nih --data_dir data/nih_cxr14 --config configs/nih_config.yaml
```

Module ablations via flags (each disables one module):

```bash
python train.py --dataset tbx11k --data_dir data/tbx11k --config configs/tbx11k_config.yaml \
    --no_gnn          # linear task head (no Module A)
    --no_uncertainty  # no Module B
    --no_vlm          # no Module C
```

Reproducible multi-seed runs use the `--seed` flag; the paper reports mean ± std over
seeds `{0, 1, 42}`. Seeding covers Python, NumPy, and PyTorch (CPU + CUDA); see
[Reproducibility](#reproducibility).

The ablation runners reproduce the paper's tables:

```bash
DATA_DIR=data/tbx11k CKPT_BASE=checkpoints bash scripts/run_tbx11k_ablations.sh
DATA_DIR=data/nih_cxr14 CKPT_BASE=checkpoints bash scripts/run_ablations.sh
```

---

## Evaluation

```bash
# TBX11K Pointing Game + macro-F1 over trained ablation checkpoints
DATA_DIR=data/tbx11k CKPT_BASE=checkpoints python scripts/eval_tbx11k_bbox.py

# NIH bbox Pointing Game
DATA_DIR=data/nih_bbox CKPT_BASE=checkpoints python scripts/eval_bbox.py

# GNN-vs-linear inference latency benchmark (Table: inference speed)
python scripts/benchmark_gnn_speed.py
```

---

## Configuration

Hyperparameters live in [`configs/`](configs/). `base_config.yaml` holds shared defaults;
`tbx11k_config.yaml` and `nih_config.yaml` override per-dataset values to match the paper
(Appendix B, Tables 7–8): learning rates and weight decay per stage, graph threshold
`τ = 0.10`, batch size 128, label smoothing 0.05, `M = 100`, `D = 256`,
`λ = 10`, `γ = 5`, `δ = 0.1`, `λ_align = 0.1`.

A command-line `--config` selects the file; `--batch_size`, `--epochs`, and `--seed`
override config values for quick experiments.

---

## Reproducibility

- **Seeding.** `train.py` seeds Python's `random`, NumPy, and PyTorch (CPU and CUDA) from
  `--seed`, and enables deterministic cuDNN. Some CUDA kernels remain nondeterministic;
  expect small run-to-run variation, consistent with the std reported in the paper.
- **Checkpoints.** Trained weights are **not** included in this repository. Re-train with
  the commands above (each full run is well under the A100's memory budget at batch size 128).
- **Hardware.** Results were produced on a single NVIDIA A100 80GB. Smaller GPUs work by
  lowering `--batch_size`.

---

## Repository structure

```
src/
  models/      backbone + CAM head, projector, prototype learner,
               GAT task head (A), uncertainty head (B), VLM aligner (C)
  data/        dataset loaders + a SyntheticDataset for pipeline smoke tests
  training/    four-stage trainer, losses, interaction/safety-check logic
  utils/       metrics (macro-F1, Pointing Game), co-occurrence graph builder
configs/       base + per-dataset hyperparameters
scripts/       data prep, ablation runners, evaluation, figures, benchmarks
train.py       entry point
```

---

## Citation

```bibtex
@article{khanbayov2026grape,
  title   = {GRAPE: Graph-Augmented Prototype Explanations for Interactive Medical Image Diagnosis},
  author  = {Khanbayov, Rasul and Kurban, Hasan},
  year    = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The TBX11K and NIH ChestX-ray14 datasets
are governed by their own licenses and are not redistributed here.
