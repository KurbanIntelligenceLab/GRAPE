#!/bin/bash
# Sequential driver: for each seed, train GRAPE (Stage 1-4, GAT) on NIH official split,
# then run the N6 (Linear/MLP/GAT) Stage-4 ablation reusing that seed's Stage 3 checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 2 3 42)
DATA_DIR=data/nih_official

for SEED in "${SEEDS[@]}"; do
  CKPT_DIR="checkpoints/nih_seed${SEED}"
  TRAIN_LOG="data/nih_train_seed${SEED}.log"
  N6_LOG="data/n6_nih_seed${SEED}.log"
  N6_OUT="checkpoints/nih_n6_seed${SEED}"
  N6_RESULTS="results/n6_nih_seed${SEED}.json"

  echo "=== [$(date)] Training GRAPE full on NIH, seed ${SEED} ==="
  mkdir -p "$CKPT_DIR"
  python3 train.py --dataset nih --data_dir "$DATA_DIR" --config configs/nih_config.yaml \
    --batch_size 32 --seed "$SEED" --checkpoint_dir "$CKPT_DIR" \
    > "$TRAIN_LOG" 2>&1
  TRAIN_STATUS=$?
  if [ $TRAIN_STATUS -ne 0 ]; then
    echo "=== [$(date)] Training FAILED for seed ${SEED} (exit $TRAIN_STATUS), skipping N6 ==="
    continue
  fi
  echo "=== [$(date)] Training complete for seed ${SEED} ==="

  echo "=== [$(date)] Running N6 MLP-vs-GAT ablation on NIH, seed ${SEED} ==="
  mkdir -p "$N6_OUT" results
  python3 scripts/n6_mlp_head_ablation_v2.py --dataset nih --data_dir "$DATA_DIR" \
    --stage3_ckpt "${CKPT_DIR}/csr_stage3.pt" \
    --gat_ckpt "${CKPT_DIR}/csr_stage4_best.pt" \
    --out_dir "$N6_OUT" --results_path "$N6_RESULTS" \
    --seed "$SEED" --batch_size 32 \
    > "$N6_LOG" 2>&1
  N6_STATUS=$?
  if [ $N6_STATUS -ne 0 ]; then
    echo "=== [$(date)] N6 ablation FAILED for seed ${SEED} (exit $N6_STATUS) ==="
  else
    echo "=== [$(date)] N6 ablation complete for seed ${SEED} -> ${N6_RESULTS} ==="
  fi
done

echo "=== [$(date)] All seeds done ==="
