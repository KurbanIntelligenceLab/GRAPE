#!/bin/bash
# TBX11K ablation runs — 4 variants to compare improvements A/B/C
# 3-class task (healthy / sick-non-TB / active-TB): GNN advantage most visible here
#
# Each run: full 4-stage training → checkpoint in ${CKPT_BASE}/tbx11k_{variant}/
#
# Paths can be overridden via environment variables, e.g.:
#   DATA_DIR=data/tbx11k CKPT_BASE=checkpoints bash scripts/run_tbx11k_ablations.sh

set -e
DATA_DIR="${DATA_DIR:-data/tbx11k}"
CKPT_BASE="${CKPT_BASE:-checkpoints}"
LOG_BASE="${LOG_BASE:-logs}"
mkdir -p "${LOG_BASE}"

run_ablation() {
    local NAME=$1
    shift
    local FLAGS="$@"
    local CKPT_DIR="${CKPT_BASE}/tbx11k_${NAME}"
    mkdir -p "${CKPT_DIR}"
    echo "============================================================"
    echo "Running: ${NAME}  [${FLAGS}]"
    echo "Checkpoint: ${CKPT_DIR}"
    echo "============================================================"
    python3 train.py \
        --dataset tbx11k \
        --data_dir "${DATA_DIR}" \
        --config configs/tbx11k_config.yaml \
        --checkpoint_dir "${CKPT_DIR}" \
        --batch_size 128 \
        --num_workers 8 \
        ${FLAGS} \
        2>&1 | tee "${LOG_BASE}/tbx11k_${NAME}.log"
    echo "Done: ${NAME}"
}

# 1. Full model: GNN + Uncertainty + VLM  [Improvements A+B+C]
run_ablation "full_ABC"

# 2. No GNN — linear task head (baseline for Improvement A)
run_ablation "no_A_linear"  --no_gnn

# 3. No VLM alignment (baseline for Improvement C)
run_ablation "no_C_novlm"   --no_vlm

# 4. Baseline CSR — no improvements
run_ablation "baseline_CSR" --no_gnn --no_vlm --no_uncertainty

echo ""
echo "All TBX11K ablations done."
echo "Checkpoints: ${CKPT_BASE}/tbx11k_*/"
echo "Logs:        ${LOG_BASE}/tbx11k_*.log"
