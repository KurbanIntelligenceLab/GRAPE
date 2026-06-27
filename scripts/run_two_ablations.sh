#!/bin/bash
# Run only GRAPE (no_C_novlm) and baseline_CSR — needed for the qualitative figure.
# Paths can be overridden via environment variables, e.g.:
#   DATA_DIR=data/tbx11k CKPT_BASE=checkpoints bash scripts/run_two_ablations.sh

DATA_DIR="${DATA_DIR:-data/tbx11k}"
CKPT_BASE="${CKPT_BASE:-checkpoints}"
LOG_BASE="${LOG_BASE:-logs}"
mkdir -p "$LOG_BASE"

echo "=== GRAPE (no_C_novlm): GNN + Uncertainty ==="
python3 train.py \
    --dataset tbx11k --data_dir "$DATA_DIR" \
    --config configs/tbx11k_config.yaml \
    --checkpoint_dir "$CKPT_BASE/tbx11k_no_C_novlm" \
    --batch_size 128 --num_workers 8 \
    --no_vlm \
    2>&1 | tee "$LOG_BASE/tbx11k_no_C_novlm.log"

echo "=== baseline_CSR: no GNN, no VLM, no Uncertainty ==="
python3 train.py \
    --dataset tbx11k --data_dir "$DATA_DIR" \
    --config configs/tbx11k_config.yaml \
    --checkpoint_dir "$CKPT_BASE/tbx11k_baseline_CSR" \
    --batch_size 128 --num_workers 8 \
    --no_gnn --no_vlm --no_uncertainty \
    2>&1 | tee "$LOG_BASE/tbx11k_baseline_CSR.log"

echo "Both done."
