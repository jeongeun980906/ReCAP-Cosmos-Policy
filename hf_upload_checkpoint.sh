#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# SPDX-License-Identifier: Apache-2.0
# Part of ReCAP, built on NVIDIA Cosmos Policy (Apache-2.0). See NOTICE.md.

# hf_upload_checkpoint.sh — upload the trained PushT-RAG residual checkpoint to HF.
#
# Usage:
#   HF_USER=your-hf-org \         # or set HF_REPO directly
#   CKPT=/mnt/ddn/tmp/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual/checkpoints/model_000007000.pt \
#   DATA=/mnt/ddn/dataset/PushT-Cosmos-Policy/success_only \
#   ./hf_upload_checkpoint.sh
#
# Requires: `pip install -U huggingface_hub` and `hf auth login`.
set -euo pipefail

# Use the project's .venv (has huggingface_hub + `hf` CLI) if present.
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$HERE/.venv/bin/hf" ]; then export PATH="$HERE/.venv/bin:$PATH"; fi
command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found. Run: uv sync --extra cu128 --group pusht  (or: pip install -U huggingface_hub)"; exit 1; }

HF_REPO="${HF_REPO:-${HF_USER:?set HF_USER=<your-hf-org> or HF_REPO=<org>/ReCAP-Cosmos2.5-pusht}/ReCAP-Cosmos2.5-pusht}"
CKPT="${CKPT:-/mnt/ddn/tmp/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual/checkpoints/model_000007000.pt}"
DATA="${DATA:-/mnt/ddn/dataset/PushT-Cosmos-Policy/success_only}"
STAGE="$(mktemp -d)"

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }

cp "$CKPT" "$STAGE/model_000007000.pt"
for f in dataset_statistics.json delta_dataset_statistics.json t5_embeddings.pkl; do
  [ -f "$DATA/$f" ] && cp "$DATA/$f" "$STAGE/"
done

echo "Creating (private) model repo if needed: $HF_REPO"
python -c "from huggingface_hub import create_repo; create_repo('$HF_REPO', repo_type='model', private=True, exist_ok=True)"
echo "Uploading to https://huggingface.co/$HF_REPO"
hf upload "$HF_REPO" "$STAGE" . --repo-type model
echo "Done. (staging: $STAGE)"
