#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# SPDX-License-Identifier: Apache-2.0
# Part of ReCAP, built on NVIDIA Cosmos Policy (Apache-2.0). See NOTICE.md.

# hf_upload_dataset.sh — upload ONLY the PushT-RAG files actually used by
# training + evaluation of cosmos_predict2p5_2b_480p_pusht_ret_top100_residual.
#
# The source `success_only/` dir contains ~140 unused rotation/variant pools
# (2.5 GB); this script stages only the 57 used pools + stats (~655 MB).
#
# Usage:
#   HF_USER=your-hf-org \         # or set HF_REPO directly
#   SRC=/mnt/ddn/dataset/PushT-Cosmos-Policy \
#   EVAL_ONLY=0 \                 # 1 = skip training-only files
#   ./hf_upload_dataset.sh
#
# Requires: `pip install -U huggingface_hub` and `hf auth login`.
set -euo pipefail

# Use the project's .venv (has huggingface_hub + `hf` CLI) if present.
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -x "$HERE/.venv/bin/hf" ]; then export PATH="$HERE/.venv/bin:$PATH"; fi
command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found. Run: uv sync --extra cu128 --group pusht  (or: pip install -U huggingface_hub)"; exit 1; }

HF_REPO="${HF_REPO:-${HF_USER:?set HF_USER=<your-hf-org> or HF_REPO=<org>/ReCAP-Cosmos2.5-pusht}/ReCAP-Cosmos2.5-pusht}"
SRC="${SRC:-/mnt/ddn/dataset/PushT-Cosmos-Policy}"
EVAL_ONLY="${EVAL_ONLY:-0}"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/success_only"

copy_dir() { [ -d "$SRC/success_only/$1" ] && cp -r "$SRC/success_only/$1" "$STAGE/success_only/" || echo "  WARN missing dir: $1"; }
copy_file(){ [ -f "$SRC/$1" ] && cp "$SRC/$1" "$STAGE/$2" || echo "  WARN missing file: $1"; }

# ── EVAL live-retrieval pools (one per visual config × 5 shards) ────────────
EVAL_POOLS=(base goal_flipped rot0 rot15 rot-15 rot30 rot-30 rot60 rot-60)
for p in "${EVAL_POOLS[@]}"; do for i in 0 1 2 3 4; do copy_dir "${p}_${i}"; done; done

# ── EVAL stats / embeddings (required) ─────────────────────────────────────
for f in t5_embeddings.pkl dataset_statistics.json dataset_statistics_post_norm.json \
         delta_dataset_statistics.json; do
  copy_file "success_only/$f" "success_only/"
done

# ── TRAINING-only data (query splits, allowlists, retrieval npz) ───────────
if [ "$EVAL_ONLY" != "1" ]; then
  # extra retrieval-source shards used at train time (base_5 / goal_flipped_5)
  copy_dir base_5; copy_dir goal_flipped_5
  # query splits referenced by the two retrieval npz / allowlists
  for d in tri_default_predict2_1 \
           tri_default_predict2p5_distilled_0 tri_default_predict2p5_distilled_1 \
           tri_default_predict2p5_no_pred_0   tri_default_predict2p5_no_pred_1 \
           tri_goal_0 tri_goal_1 tri_goal_2 tri_goal_3 tri_goal_4; do
    copy_dir "$d"
  done
  copy_file "success_only/episode_action_error_ranking_tri_default_p.json" "success_only/"
  copy_file "success_only/episode_action_error_ranking_tri_goal.json"      "success_only/"
  # precomputed retrieval indices (repo root, NOT success_only/)
  copy_file "retrieval_results_state_action_tri_default_p_base.npz"     ""
  copy_file "retrieval_results_state_action_tri_goal_goal_flipped.npz"  ""
fi

echo ""
echo "Staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
echo "Creating (private) dataset repo if needed: $HF_REPO"
python -c "from huggingface_hub import create_repo; create_repo('$HF_REPO', repo_type='dataset', private=True, exist_ok=True)"
echo "Uploading to https://huggingface.co/datasets/$HF_REPO"
hf upload "$HF_REPO" "$STAGE" . --repo-type dataset
echo "Done."
