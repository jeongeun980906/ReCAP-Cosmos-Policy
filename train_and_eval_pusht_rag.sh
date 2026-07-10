#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# SPDX-License-Identifier: Apache-2.0
# Part of ReCAP, built on NVIDIA Cosmos Policy (Apache-2.0). See NOTICE.md.

# train_and_eval_pusht_rag.sh — train the residual RAG PushT policy, then eval 9 configs.
#
#   experiment : cosmos_predict2p5_2b_480p_pusht_ret_top100_residual
#   eval config: cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only
#
# Env:
#   BASE_DATASETS_DIR  dataset root (has PushT-Cosmos-Policy/)  default: ./data
#   TRAIN_GPUS         CUDA_VISIBLE_DEVICES for training        default: 0,1,2,3,4,5,6,7
#   IMAGINAIRE_OUTPUT_ROOT  checkpoint output root              default: ./output
set -euo pipefail

export BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-./data}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-./output}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
TRAIN_NPROC=$(echo "$TRAIN_GPUS" | awk -F, '{print NF}')
MASTER_PORT="${MASTER_PORT:-12342}"

EXP=cosmos_predict2p5_2b_480p_pusht_ret_top100_residual
CKPT_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/$EXP/checkpoints"

# 1. Train ------------------------------------------------------------------
echo "=== TRAIN: $EXP ($TRAIN_NPROC GPUs) ==="
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
  uv run --extra cu128 --group pusht \
    torchrun --nproc_per_node=$TRAIN_NPROC --master_port=$MASTER_PORT \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment="$EXP"

# 2. Convert DCP -> .pt (read final iter from latest_checkpoint.txt) ---------
LATEST=$(cat "$CKPT_DIR/latest_checkpoint.txt")     # e.g. iter_000007000
ITER=$((10#${LATEST#iter_}))
echo "=== CONVERT: $EXP iter=$ITER ==="
uv run --extra cu128 --group pusht -m convert_dcp_to_pt --tag "$EXP" --iter "$ITER"

# 3. Eval -------------------------------------------------------------------
export CKPT="$CKPT_DIR/model_$(printf '%09d' "$ITER").pt"
echo "=== EVAL: $CKPT ==="
./eval_pusht_rag.sh
