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
Precomputes Reason1 text embeddings for LIBERO task descriptions and saves them to disk.
Required for training with the T2V (Cosmos Predict2.5) backbone.

Usage:
    uv run -m cosmos_policy.datasets.save_libero_reason1_embeddings --data_dir <DATA_DIR>

Example:
    uv run -m cosmos_policy.datasets.save_libero_reason1_embeddings \\
        --data_dir /data/LIBERO-Cosmos-Policy/success_only

The script saves reason1_embeddings.pkl in the specified data_dir.
"""

import argparse

from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.reason1_embedding_utils import generate_reason1_embeddings, save_reason1_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Reason1 text embeddings for LIBERO task descriptions")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing LIBERO dataset (same dir as t5_embeddings.pkl)",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="hf://nvidia/Cosmos-Reason1-7B",
        help="Path to the Reason1 model checkpoint (HuggingFace hub or local path)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading LIBERO dataset to collect unique task descriptions...")
    dataset = LIBERODataset(data_dir=args.data_dir)

    print(f"Found {len(dataset.unique_commands)} unique task descriptions")
    embeddings = generate_reason1_embeddings(dataset.unique_commands, ckpt_path=args.ckpt_path)
    save_reason1_embeddings(embeddings, args.data_dir)


if __name__ == "__main__":
    main()
