# GRAPE: Graph-Augmented Prototype Explanations for Interactive Medical Image Diagnosis

[![arXiv](https://img.shields.io/badge/arXiv-2606.30901-b31b1b.svg)](https://arxiv.org/abs/2606.30901)

Reference implementation for the NeurIPS 2026 paper **"GRAPE: Graph-Augmented Prototype
Explanations for Interactive Medical Image Diagnosis"** (Khanbayov, Serpedin & Kurban).

GRAPE extends the Concept-based Similarity Reasoning (CSR) prototype classifier with three
modules that share a single four-stage training pipeline:

- **Module A — Graph Attention Task Head.** Replaces the flat linear readout with a 2-layer
  GAT-style head built over a concept co-occurrence graph from training labels. In the
  evaluated implementation, the graph does not affect the head's outputs at inference (see
  [Implementation notes](#implementation-notes)).
- **Module B — Concept-Mismatch Safety Check.** Compares mean prototype similarity per
  concept inside a clinician-drawn box and warns when a concept other than the claimed one
  dominates, before feedback is applied. Prototype variance is only an optional second-stage
  gate. Inference-only.
- **Module C — Open-Vocabulary Prototype Anchoring.** Anchors prototypes to frozen BioViL-T
  text embeddings so a new concept can be added from a text description without retraining.
  Evaluated for localization only; with one example, Pointing Game stays near or below chance.

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

Preparation / download helpers (all paths are passed by environment variable — see each
script's header):

```bash
TBX11K_SRC=<raw_tbx11k> DATA_DIR=data/tbx11k python scripts/prepare_tbx11k.py
DATA_DIR=data/nih_cxr14 python scripts/download_nih.py
DATA_DIR=data/nih_bbox BBOX_CSV=<BBox_List_2017.csv> python scripts/download_nih_bbox.py
```

---

## Training

A full run executes Stages 1–4 (concept supervision → concept-vector generation →
prototype learning with VLM alignment → task-head classification):

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
seeds `{0, 1, 42}` (Table 1) and `{0, 1, 2, 3, 42}` (Table 3). Seeding covers Python, NumPy, and PyTorch (CPU + CUDA); see
[Reproducibility](#reproducibility).

The ablation runners reproduce the module ablation tables:

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
`τ = 0.10` on TBX11K and `τ = 0.05` on NIH, batch size 128, label smoothing 0.05, `M = 100`, `D = 256`,
`λ = 10`, `γ = 5`, `δ = 0.1`, `λ_align = 0.1`.

A command-line `--config` selects the file; `--batch_size`, `--epochs`, and `--seed`
override config values for quick experiments.

---

## Reproducing the paper

Numbering follows the NeurIPS 2026 camera-ready PDF. All scripts read their paths from the
environment or flags (`DATA_DIR`, `CKPT_BASE`, or relative `data/`, `checkpoints/`, `results/`).

| Paper result | Script |
|---|---|
| Tables 1, 2: module ablations | `scripts/run_tbx11k_ablations.sh`, `scripts/run_ablations.sh` |
| Table 1: CBM and PIPNet baselines | `scripts/c3_baselines.py` |
| Table 1: ResNet-50 (backbone only) | No script in this repository (ResNet-50 + global average pooling + linear head, seed 42) |
| Table 3: TBX11K, 5 seeds | `scripts/h2_tbx11k_5seeds.py` |
| Table 3: NIH, 5 seeds | `scripts/download_nih_official.py`, `scripts/c3_nih_official_split.py` |
| Tables 5, 6: threshold sensitivity | `scripts/n3_sensitivity.py` |
| Tables 9, 10: end-to-end training | `scripts/h2_e2e_multiseed.py` (`train.py --end_to_end`) |
| Table 11: graph head comparison | `scripts/h3_graph_baselines.py` (`train.py --task_head_type`) |
| Table 12: latency | `scripts/benchmark_gnn_speed.py` |
| Table 13: edge importance | `scripts/h4_edge_ablation.py` |
| Tables 15, 16: per-class F1, confusion matrices | `scripts/n2_perclass_f1.py` |
| Table 17: ungated safety check | `scripts/c7_safety_eval.py`, `scripts/n4_bootstrap_ci.py` |
| Table 18: gated operating points | `scripts/h1_uk_gated_safety.py`, `scripts/h1_table17_reproduction.py` |
| Table 19: Pointing Game aggregation | `scripts/c4_pg_reconcile.py` |
| Table 20: TBX11K zero-shot | `scripts/e2_tbx11k_zeroshot_5seeds.py`, `scripts/c8_zeroshot_eval.py` |
| Table 21: random-anchor control | `scripts/n12_random_anchor_control.py` |
| Table 22: NIH zero-shot | `scripts/e1_nih_zeroshot_loo.py` |
| Appendix M.2: interference from adding a concept | `scripts/e3_moduleC_downstream_effects.py`, `scripts/m5_moduleC_confound_controls.py` |
| Table 23: Insertion/Deletion AUC | `scripts/h5_faithfulness.py` |
| Table 24: MC-Dropout comparison | `scripts/n5_mc_dropout_eval.py` |
| Table 25: mean vs. max aggregation | `scripts/h1_meanmax_aggregation.py` |
| Appendix Q, Table 26: NIH safety check | `scripts/n7_nih_safety_eval.py`, `scripts/h1_low_fp_operating_points_v2.py` |
| Table 27: TBX11K head ablation | `scripts/n6_mlp_head_ablation.py`, `scripts/n6_mlp_head_ablation_v2.py` |
| Table 28: NIH head ablation | `scripts/run_nih_n6_multiseed.sh` |
| Table 29: self-attention head | `scripts/n13_selfattn_head_ablation.py` |
| Table 30: distribution shift | `scripts/n9_prevalence_shift.py` |
| Table 31: in-domain backbone | `train.py --backbone xrv_densenet121` (requires `torchxrayvision`) |
| Figures 2, 3 | `scripts/plot_pg_dots_18.py`, `scripts/plot_simmaps_comparison.py` |
| Figure 4 | `scripts/n8_proto_dispersion.py` (`scripts/n8_proto_dispersion_v2.py` is a later re-run that adds the M-ablation checkpoints and writes a separate figure) |

[`results/`](results/) holds the raw outputs of the later ablations (Tables 21, 27 re-run, 28, 29,
30, 31, the Table 17 re-run, Table 18's low-FP rows, Table 26, and the Appendix M.2 interference
experiment).

**Expect different exact values on a re-run.** The main tables came from an earlier run on an
A100, and the files in `results/` come from later runs on a P100, so the two differ in exact
values. Table 18's `θ_V = 2×10⁻⁴` operating point is specific to its checkpoint, and re-running
the sweep on a new checkpoint can select a different `θ_V`.

---

## Reproducibility

- **Seeding.** `train.py` seeds Python's `random`, NumPy, and PyTorch (CPU and CUDA) from
  `--seed`, and enables deterministic cuDNN. Some CUDA kernels remain nondeterministic;
  expect small run-to-run variation, consistent with the std reported in the paper.
- **Checkpoints.** Trained weights are **not** included in this repository. Re-train with
  the commands above (each full run is well under the A100's memory budget at batch size 128).
- **Hardware.** The main results were produced on a single NVIDIA A100 80GB with BF16 AMP,
  and the later ablations on a single Tesla P100 16GB with FP16 AMP (selected automatically
  from the GPU). Smaller GPUs work by lowering `--batch_size`; the NIH runs on the P100 used 32.
- **TBX11K splits.** TBX11K does not release test labels, so `scripts/prepare_tbx11k.py` uses
  the 1,800-image validation set as the test split, and the Stage 4 checkpoint is selected on
  it. `bbox_eval` is a subset of the same set.

### Implementation notes

The paper's numbers come from the code as it stands, including these behaviors:

- **Task head.** `src/models/gnn_task_head.py` aggregates each node's own transformed features
  (`Wh_dst`), and attention weights sum to one, so with fixed weights the head's outputs do not
  depend on the graph. The graph affects training only through attention dropout: on NIH,
  self-loops alone give F1 0.546, or 0.605 with that dropout off, against 0.604 as shipped.
  Using neighbor features (`Wh_src`) leaves TBX11K F1 at 0.904 and lowers NIH F1 from 0.604 to
  0.547 (5 seeds).
- **Zero-shot anchors.** The scripts for Tables 20 and 22 load the model without its text aligner,
  so they use a random anchor. `scripts/n12_random_anchor_control.py` (Table 21) compares the
  BioViL-T text anchor with a random one.
- **NIH boxes.** `scripts/download_nih_bbox.py` places each patient's boxes on that patient's first
  streamed image, so 37% of image–concept pairs lie on an image labeled with that finding. NIH
  Pointing Game and Module B values are approximate.

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
results/       raw outputs of the later ablations
train.py       entry point
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{khanbayov2026grapegraphaugmentedprototypeexplanations,
      title={GRAPE: Graph-Augmented Prototype Explanations for Interactive Medical Image Diagnosis}, 
      author={Rasul Khanbayov and Erchin Serpedin and Hasan Kurban},
      year={2026},
      eprint={2606.30901},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.30901}, 
}
```

## License

Code is released under the [MIT License](LICENSE). The TBX11K and NIH ChestX-ray14 datasets
are governed by their own licenses and are not redistributed here.
