# SPDX-FileCopyrightText: Copyright (c) 2026 Jeongeun Park et al. (ReCAP). All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of ReCAP ("Retrieve, Don't Retrain: Extending Vision Language
# Action Models to New Tasks at Test Time"), built on NVIDIA Cosmos Policy
# (https://github.com/NVlabs/cosmos-policy). See NOTICE.md for attribution.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.pusht_dataset2 import PushTDataset2
from cosmos_policy.datasets.pusht_dataset_ret import PushTRetDataset
from cosmos_policy.models.policy_video2world_model_rectified_flow import CosmosPolicyVideo2WorldModelRectifiedFlow
from cosmos_policy.models.policy_video2world_model_pusht_ret import CosmosPolicyPushTRetModelRectifiedFlow

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")

# ── Lightweight DiT configs for PushT (train from scratch, no pretrained ckpt) ──
#
# Architecture S (~300M): model_channels=1024, num_blocks=16, num_heads=8
#   Per block: ~14 × 1024² ≈ 14.7M → 16 blocks = 235M + embeddings ≈ ~300M
#
# Inherits all dataset / tokenizer / scheduler settings from the 2B configs;
# only overrides the net size, checkpoint (empty → random init), max_iter, and lr.

_NET_300M = dict(
    model_channels=1024,
    num_blocks=16,
    num_heads=8,
    use_crossattn_projection=False,
    crossattn_emb_channels=1024,
)

_CKPT_SCRATCH = dict(
    load_path="",
    load_training_state=False,
    strict_resume=False,
    save_iter=1000,
    load_ema_to_reg=False,
)

# ── 300M, state_t=7 (future prediction ON) ──────────────────────────────────
cosmos_v1_300m_pusht = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        checkpoint=_CKPT_SCRATCH,
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(net=_NET_300M),
        ),
        trainer=dict(max_iter=10_000),
        optimizer=dict(lr=3e-4),
        job=dict(group="cosmos_v1_light", name="cosmos_v1_300m_pusht"),
    )
)

cosmos_v1_300m_pusht__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_v1_300m_pusht", "_self_"],
        job=dict(group="cosmos_v1_light_inference", name="cosmos_v1_300m_pusht__inference_only"),
    )
)

# ── 300M, state_t=4 (future prediction OFF) ─────────────────────────────────
cosmos_v1_300m_pusht_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        checkpoint=_CKPT_SCRATCH,
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(net=_NET_300M),
        ),
        trainer=dict(max_iter=10_000),
        optimizer=dict(lr=3e-4),
        job=dict(group="cosmos_v1_light", name="cosmos_v1_300m_pusht_no_pred"),
    )
)

cosmos_v1_300m_pusht_no_pred__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_v1_300m_pusht_no_pred", "_self_"],
        job=dict(group="cosmos_v1_light_inference", name="cosmos_v1_300m_pusht_no_pred__inference_only"),
    )
)

# ── 300M no_pred — data-scaling ablation datasets (no future pred, limited episodes) ─
def _pusht_no_pred_dataset(max_num_episodes):
    return L(PushTDataset2)(
        data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
        t5_text_embeddings_path=os.path.join(
            BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
        ),
        chunk_size=8,
        use_image_aug=True,
        use_wrist_images=False,
        use_third_person_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,
        use_stronger_image_aug=True,
        rollout_data_dir="",
        demonstration_sampling_prob=1.0,
        success_rollout_sampling_prob=1.0,
        return_value_function_returns=False,
        predict_future_states=False,
        gamma=0.99,
        max_num_episodes=max_num_episodes,
    )


def _dataloader(dataset):
    return L(DataLoader)(
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        dataset=dataset,
        sampler=L(DistributedSampler)(
            dataset=dataset,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=True,
            seed=0,
        ),
        batch_size=25,
        drop_last=True,
    )


for _n in [50, 100, 150, 200]:
    _ds = _pusht_no_pred_dataset(_n)
    globals()[f"cosmos_v1_300m_pusht_no_pred_{_n}"] = LazyDict(
        dict(
            defaults=["/experiment/cosmos_v1_300m_pusht_no_pred", "_self_"],
            dataloader_train=_dataloader(_ds),
            job=dict(group="cosmos_v1_light", name=f"cosmos_v1_300m_pusht_no_pred_{_n}"),
            upload_reproducible_setup=False,
        )
    )

# ── 300M Retrieval configs ─────────────────────────────────────────────────────

def _pusht_ret_dataset(max_num_episodes=None, predict_future_states=True):
    kw = dict(
        data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
        retrieval_npz_path=[
            os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10.npz"),
            os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_goal_goal_flipped.npz"),
        ],
        task_split=["tri_default", "tri_goal"],
        retrieval_source_splits=["base", "goal_flipped"],
        t5_text_embeddings_path=os.path.join(
            BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
        ),
        chunk_size=8,
        use_image_aug=True,
        use_wrist_images=False,
        use_third_person_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,
        use_stronger_image_aug=True,
        return_value_function_returns=predict_future_states,
        predict_future_states=predict_future_states,
        gamma=0.99,
        retrieval_dropout_prob=0.3,
    )
    if max_num_episodes is not None:
        kw["max_num_episodes"] = max_num_episodes
    return L(PushTRetDataset)(**kw)


_RET_MODEL_CONFIG = dict(net=_NET_300M)

# ── 300M ret, future prediction ON (state_t=10) ──────────────────────────────
cosmos_v1_300m_pusht_ret = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_200", "_self_"],
        checkpoint=_CKPT_SCRATCH,
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=_RET_MODEL_CONFIG,
        ),
        trainer=dict(max_iter=10_000),
        optimizer=dict(lr=3e-4),
        job=dict(group="cosmos_v1_light", name="cosmos_v1_300m_pusht_ret"),
    )
)

cosmos_v1_300m_pusht_ret__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_v1_300m_pusht_ret", "_self_"],
        job=dict(group="cosmos_v1_light_inference", name="cosmos_v1_300m_pusht_ret__inference_only"),
    )
)

# ── 300M ret, future prediction OFF (state_t=7) ──────────────────────────────
cosmos_v1_300m_pusht_ret_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_200_no_pred", "_self_"],
        checkpoint=_CKPT_SCRATCH,
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=_RET_MODEL_CONFIG,
        ),
        trainer=dict(max_iter=10_000),
        optimizer=dict(lr=3e-4),
        job=dict(group="cosmos_v1_light", name="cosmos_v1_300m_pusht_ret_no_pred"),
    )
)

cosmos_v1_300m_pusht_ret_no_pred__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_v1_300m_pusht_ret_no_pred", "_self_"],
        job=dict(group="cosmos_v1_light_inference", name="cosmos_v1_300m_pusht_ret_no_pred__inference_only"),
    )
)

# ── 300M ret no_pred — data-scaling ablation (limited episodes) ──────────────
for _n in [50, 100, 150, 200]:
    _ds_ret = _pusht_ret_dataset(_n, predict_future_states=False)
    globals()[f"cosmos_v1_300m_pusht_ret_no_pred_{_n}"] = LazyDict(
        dict(
            defaults=["/experiment/cosmos_v1_300m_pusht_ret_no_pred", "_self_"],
            dataloader_train=_dataloader(_ds_ret),
            job=dict(group="cosmos_v1_light", name=f"cosmos_v1_300m_pusht_ret_no_pred_{_n}"),
            upload_reproducible_setup=False,
        )
    )

ALL_LIGHT_PUSHT_CONFIGS = [
    cosmos_v1_300m_pusht,
    cosmos_v1_300m_pusht__inference_only,
    cosmos_v1_300m_pusht_no_pred,
    cosmos_v1_300m_pusht_no_pred__inference_only,
    # data-scaling ablation
    cosmos_v1_300m_pusht_no_pred_50,
    cosmos_v1_300m_pusht_no_pred_100,
    cosmos_v1_300m_pusht_no_pred_150,
    cosmos_v1_300m_pusht_no_pred_200,
    # retrieval configs
    cosmos_v1_300m_pusht_ret,
    cosmos_v1_300m_pusht_ret__inference_only,
    cosmos_v1_300m_pusht_ret_no_pred,
    cosmos_v1_300m_pusht_ret_no_pred__inference_only,
    cosmos_v1_300m_pusht_ret_no_pred_50,
    cosmos_v1_300m_pusht_ret_no_pred_100,
    cosmos_v1_300m_pusht_ret_no_pred_150,
    cosmos_v1_300m_pusht_ret_no_pred_200,
]
