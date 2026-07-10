# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------
# Modifications Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# This file is derived from NVIDIA Cosmos Policy
# (https://github.com/NVlabs/cosmos-policy) and was modified for the ReCAP
# project (https://github.com/jeongeun980906/ReCAP-Cosmos-Policy).
# Modifications are released under the Apache License, Version 2.0. See NOTICE.md.
# -----------------------------------------------------------------------------

"""
Shared utilities for precomputing Reason1 text embeddings (for T2V / Cosmos Predict2.5 backbone).

The Reason1 encoder is a fine-tuned Qwen2.5-VL-7B model whose hidden states are used as
text conditioning instead of T5 embeddings. This produces (1, seq_len, hidden_dim) embeddings.
"""

import os
import pickle
from typing import Dict, Iterable

import torch
from tqdm import tqdm

from cosmos_policy._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig


def _resolve_ckpt_path(ckpt_path: str) -> str:
    """
    Resolve a checkpoint path to a local path.

    - For `hf://org/repo` (no file path) → uses snapshot_download to get the full repo dir.
    - For `hf://org/repo/path/to/file` → delegates to get_checkpoint_path (single file download).
    - For local or S3 paths → returned as-is (TextEncoder handles them internally).
    """
    if ckpt_path.startswith("hf://"):
        parts = ckpt_path[5:].split("/")  # strip "hf://"
        if len(parts) == 2:
            # Repo-level path: download the entire model folder
            from huggingface_hub import snapshot_download
            repo_id = "/".join(parts)
            print(f"Downloading Reason1 model from HuggingFace: {repo_id} ...")
            return snapshot_download(repo_id=repo_id)
        # File-level path: TextEncoder will call get_checkpoint_path internally
    return ckpt_path


def build_text_encoder(ckpt_path: str) -> TextEncoder:
    """
    Build and load a Reason1 TextEncoder.

    Args:
        ckpt_path: Path to the Reason1 model checkpoint.
                   Accepts:
                   - HuggingFace repo:  ``hf://nvidia/Cosmos-Reason1-7B``
                   - HuggingFace file:  ``hf://nvidia/Cosmos-Reason1-7B/model.pt``
                   - Local directory or file path
                   - S3 path

    Returns:
        Loaded TextEncoder ready for inference.
    """
    resolved = _resolve_ckpt_path(ckpt_path)
    config = TextEncoderConfig(
        ckpt_path=resolved,
        embedding_concat_strategy="full_concat",
    )
    return TextEncoder(config, device="cuda")


def generate_reason1_embeddings(
    unique_commands: Iterable[str],
    ckpt_path: str = "hf://nvidia/Cosmos-Reason1-7B",
) -> Dict[str, torch.Tensor]:
    """
    Generate Reason1 text embeddings for a collection of commands.

    Args:
        unique_commands: Iterable of unique command/instruction strings.
        ckpt_path: Path to the Reason1 model checkpoint.

    Returns:
        Dict mapping each command to its embedding tensor (bfloat16, CPU).
        Embedding shape: (1, seq_len, hidden_dim) with full_concat strategy.
    """
    text_encoder = build_text_encoder(ckpt_path)
    embeddings = {}
    print("Computing Reason1 text embeddings...")
    for command in tqdm(unique_commands):
        with torch.no_grad():
            emb = text_encoder.compute_text_embeddings_online({"text": [command]}, "text")
        embeddings[command] = emb.to(dtype=torch.bfloat16).cpu()
    return embeddings


def save_reason1_embeddings(embeddings: Dict[str, torch.Tensor], data_dir: str) -> str:
    """
    Save Reason1 embeddings to reason1_embeddings.pkl in data_dir.

    Args:
        embeddings: Dict of {command: tensor} to save.
        data_dir: Directory where the pickle file will be written.

    Returns:
        Path to the saved file.
    """
    save_path = os.path.join(data_dir, "reason1_embeddings.pkl")
    print(f"Saving Reason1 embeddings to {save_path} ...")
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings.")
    return save_path
