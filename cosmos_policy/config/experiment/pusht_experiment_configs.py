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

from cosmos_policy.config.experiment.light_pusht_experiment_configs import ALL_LIGHT_PUSHT_CONFIGS
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.pusht_dataset import PushTDataset
from cosmos_policy.datasets.pusht_dataset2 import PushTDataset2
from cosmos_policy.datasets.pusht_dataset_ret import PushTRetDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.models.policy_video2world_model_rectified_flow import CosmosPolicyVideo2WorldModelRectifiedFlow
from cosmos_policy.models.policy_video2world_model_pusht_ret import (
    CosmosPolicyPushTRetModel,
    CosmosPolicyPushTRetModelRectifiedFlow,
)
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")

# Per-task ranking JSONs written by dataset/compute_episode_action_error.py.
# Order must match the dataset's `task_split` order so that each task gets its
# own top-K allowlist (N episodes per task).
_EPISODE_ACTION_ERROR_RANKING_PATH = [
    os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only",
        "episode_action_error_ranking_tri_default_p.json",
    ),
    os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only",
        "episode_action_error_ranking_tri_goal.json",
    ),
]

# *** PushT Datasets ***
# All visual configs: base (206 demos), tri_color (27), tri_default (27) = 260 total
pusht_dataset = L(PushTDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    task_split=["rot0"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
)

# Without future state prediction
pusht_dataset_no_pred = L(PushTDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    task_split=["rot0"],
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
)

# Data-scaling ablation datasets (random N episodes)
pusht_dataset_50 = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    max_num_episodes=50,
)

pusht_dataset_100 = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    max_num_episodes=100,
)

pusht_dataset_150 = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    max_num_episodes=150,
)

pusht_dataset_50_no_pred = L(PushTDataset2)(
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
    max_num_episodes=50,
)
pusht_dataset_200 = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    # max_num_episodes=200,
)


# ── Mixed baseline: tri_default(top100) + tri_goal(top100) + full base/goal_flipped/rot* ──
# Triangle side == existing pusht_dataset_top100_no_pred (tri_default + tri_goal,
# each filtered to top-100 by action-error ranking). On top of that we add the
# full circle-agent pool: all base_*, goal_flipped_*, and rot*_* suite folders.
# Shares the same output schema + dataset_statistics.json as the existing
# _no_pred baselines so run_pusht_eval.py is fully compatible.
pusht_dataset_tri100_mixed_no_pred = L(PushTDataset2)(
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
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,   # top-100 per task (tri_default, tri_goal)
    extra_task_splits=[
        ("base", -1),
        ("goal_flipped", -1),
        ("rot0", -1),
        ("rot15", -1),
        ("rot-15", -1),
        ("rot30", -1),
        ("rot-30", -1),
        ("rot60", -1),
        ("rot-60", -1),
        ("rot90", -1),
        ("rot-90", -1),
        ("rot105", -1),
        ("rot120", -1),
        ("rot135", -1),
        ("rot150", -1),
        ("rot165", -1),
        ("rot180", -1),
        ("rot-105", -1),
        ("rot-120", -1),
        ("rot-135", -1),
        ("rot-150", -1),
        ("rot-165", -1),
    ],
)

# Pred variant of pusht_dataset_tri100_mixed_no_pred — same data mix, but with
# future-state prediction + value-function returns enabled (matches the pred
# pipeline used by pusht_dataset_50/100/150/200).
pusht_dataset_tri100_mixed = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
    extra_task_splits=[
        ("base", -1),
        ("goal_flipped", -1),
        ("rot0", -1),
        ("rot15", -1),
        ("rot-15", -1),
        ("rot30", -1),
        ("rot-30", -1),
        ("rot60", -1),
        ("rot-60", -1),
        ("rot-90", -1),
        ("rot90", -1),
        ("rot105", -1),
        ("rot120", -1),
        ("rot135", -1),
        ("rot150", -1),
        ("rot165", -1),
        ("rot180", -1),
        ("rot-105", -1),
        ("rot-120", -1),
        ("rot-135", -1),
        ("rot-150", -1),
        ("rot-165", -1),
    ],
)

pusht_dataset_tri100_mixed2 = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
    extra_task_splits=[
        ("base", -1),
        ("goal_flipped", -1),
    ],
)

pusht_dataset_tri100_mixed_no_pred2 = L(PushTDataset2)(
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
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,   # top-100 per task (tri_default, tri_goal)
    extra_task_splits=[
        ("base", 30),
        ("goal_flipped", 30),
        ("rot0", 20),
        ("rot15", 20),
        ("rot-15", 20),
        ("rot30", 20),
        ("rot-30", 20),
        ("rot60", 20),
        ("rot-60", 20),
    ],
)

# ── Pretrain split for 2-stage training ────────────────────────────────────
# Pretrain ONLY on circle-agent data (base/goal_flipped/rot*), then fine-tune
# on triangle data (pusht_100_no_pred). Uses PushTDataset (not 2) because
# PushTDataset2 hardcodes tri_default+tri_goal loading, which we want to
# exclude at pretrain time. task_split is substring-matched against filepaths.
# Matches the circle pool used by pusht_dataset_tri100_mixed_no_pred:
#   base_*, goal_flipped_*, rot0_*, rot{±15,±30,±60}_*
pusht_dataset_pretrain_bgr_no_pred = L(PushTDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    task_split=[
        "base",
        "goal_flipped",
        "rot0",
        "rot15",
        "rot-15",
        "rot30",
        "rot-30",
        "rot60",
        "rot-60",
        "rot90",
        "rot-90",
        "rot105",
        "rot120",
        "rot135",
        "rot150",
        "rot165",
        "rot180",
        "rot-105",
        "rot-120",
        "rot-135",
        "rot-150",
        "rot-165",
    ],
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
)

# Pred variant of pusht_dataset_pretrain_bgr_no_pred — same task split,
# future-state prediction + value-function returns enabled.
pusht_dataset_pretrain_bgr = L(PushTDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    task_split=[
        "base",
        "goal_flipped",
        "rot0",
        "rot15",
        "rot-15",
        "rot30",
        "rot-30",
        "rot60",
        "rot-60",
        "rot90",
        "rot-90",
        "rot105",
        "rot120",
        "rot135",
        "rot150",
        "rot165",
        "rot180",
        "rot-105",
        "rot-120",
        "rot-135",
        "rot-150",
        "rot-165",
    ],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
)

pusht_dataset_pretrain_bgr2 = L(PushTDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    task_split=[
        "base",
        "goal_flipped",
    ],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
)

pusht_dataset_100_no_pred = L(PushTDataset2)(
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
    max_num_episodes=100,
)

pusht_dataset_150_no_pred = L(PushTDataset2)(
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
    max_num_episodes=150,
)

pusht_dataset_200_no_pred = L(PushTDataset2)(
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
    # max_num_episodes=200,
)


# *** WAN2.1 / EDM backbone ***
cosmos_predict2_2b_480p_pusht = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                # state_t=7: blank, proprio, primary, action, future_proprio, future_primary, value
                state_t=7,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                tokenizer=dict(chunk_duration=25),
            ),
        ),
        optimizer=dict(lr=2e-4),
        scheduler=dict(
            cycle_lengths=[10000, 100000000000000],
            f_start=[1e-6, 0.06],
            warm_up_steps=[500, 0],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2_2b_480p_pusht"),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2_2b_480p_pusht_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                # state_t=4: blank, proprio, primary, action
                state_t=4,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                tokenizer=dict(chunk_duration=13),
            ),
        ),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[10000, 100000000000000],
            f_start=[1e-6, 0.06],
            warm_up_steps=[500, 0],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2_2b_480p_pusht_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_pusht__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_pusht", "_self_"],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4))
        ),
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2_2b_480p_pusht__inference_only"),
    )
)

cosmos_predict2_2b_480p_pusht_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_pusht_no_pred", "_self_"],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4))
        ),
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2_2b_480p_pusht_no_pred_inference_only"),
    )
)


# *** Predict2.5 / Rectified Flow backbone ***
cosmos_predict2p5_2b_480p_pusht = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_libero", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                # state_t=7: blank, proprio, primary, action, future_proprio, future_primary, value
                state_t=7,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0.0, 4: 1.0},
                tokenizer=dict(chunk_duration=25),
                text_encoder_class="T5",
                action_dim=2,
                proprio_dim=2,
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
            ),
        ),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            f_start=[1e-6, 0.06],
            warm_up_steps=[500, 0],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht__inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_libero", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                # state_t=4: blank, proprio, primary, action
                state_t=4,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0.0, 4: 1.0},
                tokenizer=dict(chunk_duration=13),
                text_encoder_class="T5",
                action_dim=2,
                proprio_dim=2,
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
            ),
        ),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[10000, 100000000000000],
            f_start=[1e-6, 0.06],
            warm_up_steps=[500, 0],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_no_pred_inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_no_pred_distilled = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        checkpoint=dict(
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt"
            ),
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_no_pred_distilled"),
    )
)

# *** Predict2.5 — DMD2 distilled backbone ***
cosmos_predict2p5_2b_480p_pusht_distilled = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        checkpoint=dict(
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Predict2.5-2B/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt"
            ),
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_distilled"),
    )
)

cosmos_predict2p5_2b_480p_pusht_distilled__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_distilled", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_distilled__inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_50 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_50,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_50,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_50"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_100 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_100,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_100,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_100"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_150 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_150,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_150,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_150"),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_200 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_200,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_200,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_200"),
        upload_reproducible_setup=False,
    )
)



cosmos_predict2p5_2b_480p_pusht_50_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_50_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_50_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_50_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_100_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_100_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_100_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_100_no_pred"),
        upload_reproducible_setup=False,
    )
)

# Mixed baseline: tri_default(100) + full base/goal_flipped/rot* splits
cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_tri100_mixed_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_tri100_mixed_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred_inference_only",
        ),
    )
)


# Pred variant of cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred —
# same mix, future-state prediction enabled (inherits pusht base config).
cosmos_predict2p5_2b_480p_pusht_tri100_mixed = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=7_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_tri100_mixed,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_tri100_mixed,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_tri100_mixed_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_tri100_mixed",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed_inference_only",
        ),
    )
)


# Pred variant of pusht_dataset_tri100_mixed2 (tri100 + base/goal_flipped only).
cosmos_predict2p5_2b_480p_pusht_tri100_mixed2 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=7_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_tri100_mixed2,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_tri100_mixed2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed2",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_tri100_mixed2_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_tri100_mixed2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed2_inference_only",
        ),
    )
)


# Mixed baseline v2: tri_default(100) + tri_goal(100) + capped circle pool
# (base×15, goal_flipped×15, rot{0,±15,±30,±60}×10 each)
cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_tri100_mixed_no_pred2,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_tri100_mixed_no_pred2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2_inference_only",
        ),
    )
)


# ── 2-stage: pretrain on circle pool (base/goal_flipped/rot*), then finetune ──
# Stage 1: pretrain ~4K steps on pusht_dataset_pretrain_bgr_no_pred.
# Stage 2: finetune ~3K steps on pusht_dataset_top100_no_pred (tri_default+tri_goal
# filtered to top-100 each by action-error ranking), resuming from the stage-1
# checkpoint (model weights only).
#
# Note: the trainer writes DCP checkpoints under
#   $IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/<name>/checkpoints/
# and convert_dcp_to_pt.py produces model_<iter:09d>.pt. Run that between
# stages so stage 2's load_path resolves.
cosmos_predict2p5_2b_480p_pusht_pretrain_bgr_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=4_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_pretrain_bgr_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_pretrain_bgr_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_pretrain_bgr_no_pred",
        ),
        upload_reproducible_setup=False,
    )
)


_PRETRAIN_BGR_CKPT_PATH = os.path.join(
    os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output"),
    "cosmos_policy",
    "cosmos_v2_finetune",
    "cosmos_predict2p5_2b_480p_pusht_pretrain_bgr_no_pred",
    "checkpoints",
    "model_000004000.pt",
)

cosmos_predict2p5_2b_480p_pusht_100_no_pred_ft_bgr = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100_no_pred", "_self_"],
        trainer=dict(max_iter=3_000),
        checkpoint=dict(
            load_path=_PRETRAIN_BGR_CKPT_PATH,
            load_training_state=False,
            strict_resume=False,
            load_ema_to_reg=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_100_no_pred_ft_bgr",
        ),
        upload_reproducible_setup=False,
    )
)


# Pred variants of the 2-stage pretrain→finetune pipeline.
cosmos_predict2p5_2b_480p_pusht_pretrain_bgr = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=4_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_pretrain_bgr,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_pretrain_bgr,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_pretrain_bgr",
        ),
        upload_reproducible_setup=False,
    )
)


_PRETRAIN_BGR_PRED_CKPT_PATH = os.path.join(
    os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output"),
    "cosmos_policy",
    "cosmos_v2_finetune",
    "cosmos_predict2p5_2b_480p_pusht_pretrain_bgr",
    "checkpoints",
    "model_000004000.pt",
)

cosmos_predict2p5_2b_480p_pusht_100_ft_bgr = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100", "_self_"],
        trainer=dict(max_iter=2_000),
        checkpoint=dict(
            load_path=_PRETRAIN_BGR_PRED_CKPT_PATH,
            load_training_state=False,
            strict_resume=False,
            load_ema_to_reg=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_100_ft_bgr",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_100_ft_bgr_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_100_ft_bgr",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_100_ft_bgr_inference_only",
        ),
    )
)


# Pred 2-stage variant using pusht_dataset_pretrain_bgr2 (base + goal_flipped only).
cosmos_predict2p5_2b_480p_pusht_pretrain_bgr2 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=4_000),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_pretrain_bgr2,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_pretrain_bgr2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_pretrain_bgr2",
        ),
        upload_reproducible_setup=False,
    )
)


_PRETRAIN_BGR2_PRED_CKPT_PATH = os.path.join(
    os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire4-output"),
    "cosmos_policy",
    "cosmos_v2_finetune",
    "cosmos_predict2p5_2b_480p_pusht_pretrain_bgr2",
    "checkpoints",
    "model_000004000.pt",
)

cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100", "_self_"],
        trainer=dict(max_iter=2_000),
        checkpoint=dict(
            load_path=_PRETRAIN_BGR2_PRED_CKPT_PATH,
            load_training_state=False,
            strict_resume=False,
            load_ema_to_reg=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2",
        ),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2_inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2_inference_only",
        ),
    )
)


cosmos_predict2p5_2b_480p_pusht_150_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_150_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_150_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_150_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_200_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_200_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_200_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_200_no_pred"),
        upload_reproducible_setup=False,
    )
)

# ── PushT Retrieval (retrieved future frames + actions as conditioning) ─────────
# Retrieved video: 8 frames, 1× each → 2 latents
# Retrieved action: 1 latent (blank image, action values injected into latent)
# state_t = 7 (original) + 2 (ret_video) + 1 (ret_action) = 10
# chunk_duration = 25 (original) + 8 (ret video) + 4 (ret action) = 37
pusht_ret_dataset_200 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=True,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
)


pusht_ret_dataset_50 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
    max_num_episodes=50,
)

pusht_ret_dataset_150 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
    max_num_episodes=150,
)

pusht_ret_dataset_100 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_top_k_choice = 1,
    retrieval_dropout_prob=0.0,
    max_num_episodes=100,
)

### No pred version


pusht_ret_dataset_200_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
)


pusht_ret_dataset_50_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
    max_num_episodes=50,
)

pusht_ret_dataset_150_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_dropout_prob=0.1,
    max_num_episodes=150,
)

pusht_ret_dataset_100_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice = 1,
    retrieval_dropout_prob=0.0,
    max_num_episodes=100,
)

# ── Top-K ablation: per-episode future-action L1 error ranking ──────────────
# Ranking file generated by dataset/compute_episode_action_error.py using
# top-1 retrieval + raw action units.  Two downstream datasets consume it:
#   1. pusht_dataset_top{50,100}_no_pred   → no-retrieval baseline (PushTDataset2)
#   2. pusht_ret_dataset_top{50,100}_no_pred_residual → residual learning
# Training uses retrieval_top_k_choice=1 so the trained matcher == the ranker.

# WAN2.1 / EDM backbone + retrieval
cosmos_predict2_2b_480p_pusht_ret = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_pusht", "_self_"],
        model=L(CosmosPolicyPushTRetModel)(
            config=dict(
                # Layout: blank(0), cur_frame(1), cur_state(2), ret_action(3),
                #         ret_frame(4-5), ret_state(6), pred_action(7),
                #         pred_frame(8), pred_state(9)
                state_t=10,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=37),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,

            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_200,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_200,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2_2b_480p_pusht_ret"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2_2b_480p_pusht_ret__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_pusht_ret", "_self_"],
        model=L(CosmosPolicyPushTRetModel)(
            config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4))
        ),
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2_2b_480p_pusht_ret__inference_only"),
    )
)

# Predict2.5 / Rectified Flow backbone + retrieval
cosmos_predict2p5_2b_480p_pusht_ret_200 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), cur_frame(1), cur_state(2), ret_action(3),
                #         ret_frame(4-5), ret_state(6), pred_action(7),
                #         pred_frame(8), pred_state(9)
                state_t=10,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=37),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_200,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_200,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_200"),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_ret_200_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=8_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                # Latent projection: replace tiling with learned MLP encoder/decoder
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,

            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_200_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_200_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_200_no_pred"),
        upload_reproducible_setup=False,
    )
)



# Predict2.5 / Rectified Flow backbone + retrieval
cosmos_predict2p5_2b_480p_pusht_ret_150 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=8_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), cur_frame(1), cur_state(2), ret_action(3),
                #         ret_frame(4-5), ret_state(6), pred_action(7),
                #         pred_frame(8), pred_state(9)
                state_t=10,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=37),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_150,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_150,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_150"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_150_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=8_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_150_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_150_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_150_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_100 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), cur_frame(1), cur_state(2), ret_action(3),
                #         ret_frame(4-5), ret_state(6), pred_action(7),
                #         pred_frame(8), pred_state(9)
                state_t=10,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=37),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100"),
        upload_reproducible_setup=False,
    )
)



cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_50 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_50_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_50_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_50"),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_ret_50_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=5_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={0: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_50_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_50_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_50_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret__inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret__inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_no_pred_inference_only"),
    )
)

# ── Residual learning: predict delta = action - retrieved_action ──────────────
# Instead of predicting absolute actions, the model learns to predict the
# residual (correction) relative to the retrieved action.  At inference the
# predicted delta is added back to the retrieved action.
# Inspired by rag_toy/run_noise01.py residual FM approach.

pusht_ret_dataset_100_no_pred_residual = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    max_num_episodes=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100_no_pred_residual,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100_no_pred_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_inference_only"),
    )
)

# ── PRED variant: residual learning on top of pusht_ret_100 (predicts future
#    state + value alongside actions). Mirrors *_no_pred_residual but defaults
#    from the predict-mode config so state_t / chunk_duration / dataset signal
#    set match. Inference variant included so eval scripts can target it.

pusht_ret_dataset_100_residual = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    max_num_episodes=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_100_residual = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100_residual,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100_residual"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_100_residual_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_residual", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_100_residual_inference_only"),
    )
)

# ── inference_only variants for the absolute baselines (plain 100-episode,
#    predict version). Eval scripts target these.

cosmos_predict2p5_2b_480p_pusht_100_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_100", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_100_inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_100_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_100_inference_only"),
    )
)

# ── Residual learning with state_action5 retrieval ────────────────────────────
# Same as _residual but uses retrieval_results_state_action5_* npz files.

pusht_ret_dataset_100_no_pred_residual_sa5 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action5_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action5_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    max_num_episodes=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual_sa5 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100_no_pred_residual_sa5,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100_no_pred_residual_sa5,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual_sa5"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_sa5_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual_sa5", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_sa5_inference_only"),
    )
)

# ── Top-K ablation datasets + experiments ─────────────────────────────────────
# 4 training configs total: {baseline, residual} × {top50, top100}.
# All consume `_EPISODE_ACTION_ERROR_RANKING_PATH` (smallest future-action L1
# first) and use top-1 retrieval at train time.

# 2-A. Baseline (no retrieval): PushTDataset2 + allowlist
pusht_dataset_top50_no_pred = L(PushTDataset2)(
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
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=50,
)

pusht_dataset_top100_no_pred = L(PushTDataset2)(
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
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

cosmos_predict2p5_2b_480p_pusht_top50_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_top50_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_top50_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_top50_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_top100_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_top100_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_top100_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_top100_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_top50_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top50_no_pred", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_top50_no_pred_inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_top100_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100_no_pred", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_top100_no_pred_inference_only"),
    )
)

# 2-B. Residual learning: PushTRetDataset + use_residual_actions + allowlist (top-1)
pusht_ret_dataset_top50_no_pred_residual = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=50,
)

pusht_ret_dataset_top50_no_pred_residual_k3 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=50,
)

pusht_ret_dataset_top100_no_pred_residual = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)


pusht_ret_dataset_top100_no_pred_residual_k3 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.05,
    use_residual_actions=True,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top50_no_pred_residual,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top50_no_pred_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual"),
        upload_reproducible_setup=False,
    )
)


cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_k3 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top50_no_pred_residual_k3,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top50_no_pred_residual_k3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_k3"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_no_pred_residual,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_no_pred_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_k3 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_no_pred_residual_k3,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_no_pred_residual_k3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_k3"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_inference_only"),
    )
)

# ── Absolute (non-residual) control, NO aux ──────────────────────────────────
# Clone of top100_no_pred_residual (k=1) with the only change being
# use_residual_actions=False. Serves as the comparison baseline for absolute_aux30:
# same RAG data, same architecture, just no 30% aux branch.
pusht_ret_dataset_top100_no_pred_absolute = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=False,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_no_pred_absolute,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_no_pred_absolute,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_inference_only"),
    )
)

# ── Absolute (non-residual) + auxiliary cotrain-RAG (30%) ─────────────────────
# Clone of top100_no_pred_residual (k=1) but with use_residual_actions=False and
# a 30% aux branch that samples from extra_task_splits (base, goal_flipped, rot*)
# and conditions the model on that sample's own future. 70% path keeps the
# existing tri → base/goal_flipped NPZ retrieval unchanged.
pusht_ret_dataset_top100_no_pred_absolute_aux30 = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=False,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
    extra_task_splits=[
        ("base", -1),
        ("goal_flipped", -1),
        ("rot0", -1),
        ("rot15", -1),
        ("rot-15", -1),
        ("rot30", -1),
        ("rot-30", -1),
        ("rot60", -1),
        ("rot-60", -1),
        ("rot90", -1),
        ("rot-90", -1),
        ("rot105", -1),
        ("rot120", -1),
        ("rot135", -1),
        ("rot150", -1),
        ("rot165", -1),
        ("rot180", -1),
        ("rot-105", -1),
        ("rot-120", -1),
        ("rot-135", -1),
        ("rot-150", -1),
        ("rot-165", -1),
    ],
    aux_sampling_prob=0.3,
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_no_pred_absolute_aux30,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_no_pred_absolute_aux30,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30_inference_only"),
    )
)

# ── Top-K with future-state prediction ───────────────────────────────────────
# State-predicting counterparts of:
#   pusht_top100_no_pred                 → pusht_top100
#   pusht_ret_top100_no_pred_residual    → pusht_ret_top100_residual
#   pusht_ret_top100_no_pred_absolute    → pusht_ret_top100_absolute
# Same top-100 allowlist and dataset wiring, but with
# predict_future_states=True and value-function returns enabled.

pusht_dataset_top100_pred = L(PushTDataset2)(
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

pusht_ret_dataset_top100_residual = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

pusht_ret_dataset_top100_absolute = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=True,
    predict_future_states=True,
    gamma=0.99,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=False,
    episode_allowlist_path=_EPISODE_ACTION_ERROR_RANKING_PATH,
    episode_allowlist_top_k=100,
)

cosmos_predict2p5_2b_480p_pusht_top100 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_dataset_top100_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_dataset_top100_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_top100"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_top100_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_top100_inference_only"),
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_residual = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_residual,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_residual"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only"),
    )
)

# ── Initialized from Self-Forcing Stage 3 backbone ───────────────────────────
# These two configs reuse the same dataset / loss / state_t setup as their
# non-_sfstage3 counterparts, but swap the initial checkpoint to the Cosmos
# Self-Forcing Stage 3 student (causal-distilled video DiT). Attention is left
# bidirectional in cosmos_policy; the backbone is fine-tuned during policy
# training to absorb the train-time distribution shift.
#
# Source: scripts/train_cosmos_sf.sh produces sf_v2w_student.pt via the bridge
# step at the end of Stage 3. The raw bridge output uses ``transformer.*`` key
# prefix (fastgen layout); cosmos_policy expects ``net.*``. Use the remapped
# .pt produced by:
#   python scripts/remap_stage3_for_policy.py \
#       --input  /mnt/ddn/tmp/cosmos_sf_stage3/sf_v2w_student.pt \
#       --output /mnt/ddn/tmp/cosmos_sf_stage3/sf_v2w_student_policy.pt
# Architecture-wise the two networks are identical (572/572 params match base
# post-trained Cosmos-Predict2.5-2B after the prefix swap). The Stage 3 chunk-
# causal mask is dropped here - cosmos_policy uses bidirectional attention.
_SF_STAGE3_BACKBONE_PT = os.environ.get(
    "SF_STAGE3_BACKBONE_PT", "/mnt/ddn/tmp/cosmos_sf_stage3/sf_v2w_student_policy.pt"
)

cosmos_predict2p5_2b_480p_pusht_top100_sfstage3 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100", "_self_"],
        checkpoint=dict(
            load_path=_SF_STAGE3_BACKBONE_PT,
            load_training_state=False,
            strict_resume=False,
            load_ema_to_reg=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_top100_sfstage3",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_top100_sfstage3_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_top100_sfstage3", "_self_"],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_top100_sfstage3_inference_only",
        ),
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual", "_self_"],
        checkpoint=dict(
            load_path=_SF_STAGE3_BACKBONE_PT,
            load_training_state=False,
            strict_resume=False,
            load_ema_to_reg=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3",
        ),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3", "_self_"],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3_inference_only",
        ),
    )
)


cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_top100_absolute,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_top100_absolute,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute_inference_only"),
    )
)

# ── Image dropout ablation: retrieval_dropout_prob=0 ───────────────────────────
# Same as no_pred but with dropout completely disabled at training time.

pusht_ret_dataset_100_no_pred_imgdrop = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_default_p_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state_action_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.0,
    max_num_episodes=100,
)

cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_imgdrop = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred", "_self_"],
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_dataset_100_no_pred_imgdrop,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_dataset_100_no_pred_imgdrop,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_imgdrop"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_no_pred_imgdrop_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_imgdrop", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_no_pred_imgdrop_inference_only"),
    )
)

# ── Retrieval with absolute state only (state10, no future action features) ────
# Uses retrieval_results_state10_tri_default_base.npz / state10_tri_goal_goal_flipped.npz

pusht_ret_s10_dataset_50_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_default_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.1,
    max_num_episodes=50,
)

pusht_ret_s10_dataset_100_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_default_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.1,
    max_num_episodes=100,
)

pusht_ret_s10_dataset_150_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_default_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.1,
    max_num_episodes=150,
)

pusht_ret_s10_dataset_200_no_pred = L(PushTRetDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "success_only"),
    retrieval_npz_path=[
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_default_base.npz"),
        os.path.join(BASE_DATASETS_DIR, "PushT-Cosmos-Policy", "retrieval_results_state10_tri_goal_goal_flipped.npz"),
    ],
    task_split=["tri_default_p", "tri_goal"],
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
    return_value_function_returns=False,
    predict_future_states=False,
    gamma=0.99,
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.1,
)

# Predict2.5 / RF + state10 retrieval experiments
cosmos_predict2p5_2b_480p_pusht_ret_s10_50_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                # Layout: blank(0), ret_frame(1-2), ret_state(3), ret_action(4),
                #         cur_frame(5), cur_state(6), pred_action(7)
                state_t=8,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_s10_dataset_50_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_s10_dataset_50_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_s10_50_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_s10_100_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                state_t=8,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_s10_dataset_100_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_s10_dataset_100_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_s10_100_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_s10_150_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                state_t=8,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_s10_dataset_150_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_s10_dataset_150_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_s10_150_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_s10_200_no_pred = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_no_pred", "_self_"],
        trainer=dict(max_iter=7_000),
        model=L(CosmosPolicyPushTRetModelRectifiedFlow)(
            config=dict(
                state_t=8,
                min_num_conditional_frames=7,
                max_num_conditional_frames=7,
                conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                tokenizer=dict(chunk_duration=29),
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                use_action_projection=False,
                use_proprio_projection=False,
                action_dim=2,
                proprio_dim=2,
                projection_hidden_dim=256,
                action_loss_multiplier=16,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=pusht_ret_s10_dataset_200_no_pred,
            sampler=L(DistributedSampler)(
                dataset=pusht_ret_s10_dataset_200_no_pred,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_pusht_ret_s10_200_no_pred"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_480p_pusht_ret_s10_no_pred_inference_only = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_s10_100_no_pred", "_self_"],
        job=dict(group="cosmos_v2_inference", name="cosmos_predict2p5_2b_480p_pusht_ret_s10_no_pred_inference_only"),
    )
)

# All PushT experiment configs — imported by cosmos_policy_experiment_configs.py
ALL_PUSHT_CONFIGS = [
    # WAN2.1 / EDM backbone
    cosmos_predict2_2b_480p_pusht,
    cosmos_predict2_2b_480p_pusht__inference_only,
    cosmos_predict2_2b_480p_pusht_no_pred,
    cosmos_predict2_2b_480p_pusht_no_pred_inference_only,
    # Predict2.5 / Rectified Flow backbone
    cosmos_predict2p5_2b_480p_pusht,
    cosmos_predict2p5_2b_480p_pusht__inference_only,
    cosmos_predict2p5_2b_480p_pusht_no_pred,
    cosmos_predict2p5_2b_480p_pusht_no_pred_inference_only,
    # Predict2.5 — data scaling ablation
    cosmos_predict2p5_2b_480p_pusht_50,
    cosmos_predict2p5_2b_480p_pusht_100,
    cosmos_predict2p5_2b_480p_pusht_150,
    cosmos_predict2p5_2b_480p_pusht_200,
    cosmos_predict2p5_2b_480p_pusht_50_no_pred,
    cosmos_predict2p5_2b_480p_pusht_100_no_pred,
    cosmos_predict2p5_2b_480p_pusht_150_no_pred,
    cosmos_predict2p5_2b_480p_pusht_200_no_pred,
    # Predict2.5 — mixed baseline (tri100 + full base/goal_flipped/rot*)
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed_inference_only,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed2,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed2_inference_only,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred_inference_only,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2,
    cosmos_predict2p5_2b_480p_pusht_tri100_mixed_no_pred2_inference_only,
    # Predict2.5 — 2-stage pretrain (base/goal_flipped/rot*) + finetune (pusht_100)
    cosmos_predict2p5_2b_480p_pusht_pretrain_bgr,
    cosmos_predict2p5_2b_480p_pusht_100_ft_bgr,
    cosmos_predict2p5_2b_480p_pusht_100_ft_bgr_inference_only,
    cosmos_predict2p5_2b_480p_pusht_pretrain_bgr2,
    cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2,
    cosmos_predict2p5_2b_480p_pusht_100_ft_bgr2_inference_only,
    cosmos_predict2p5_2b_480p_pusht_pretrain_bgr_no_pred,
    cosmos_predict2p5_2b_480p_pusht_100_no_pred_ft_bgr,
    # Retrieval conditioning (WAN2.1 / EDM)
    cosmos_predict2_2b_480p_pusht_ret,
    cosmos_predict2_2b_480p_pusht_ret__inference_only,
    # Retrieval conditioning (Predict2.5 / RF)
    cosmos_predict2p5_2b_480p_pusht_ret_50,
    cosmos_predict2p5_2b_480p_pusht_ret_100,
    cosmos_predict2p5_2b_480p_pusht_ret_150,
    cosmos_predict2p5_2b_480p_pusht_ret_200,
    cosmos_predict2p5_2b_480p_pusht_ret_50_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_150_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_200_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret__inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_no_pred_inference_only,
    # Residual learning (predict delta = action - retrieved_action)
    cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual,
    cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_residual_sa5,
    cosmos_predict2p5_2b_480p_pusht_ret_no_pred_residual_sa5_inference_only,
    # Plain-100 (predict version) — baseline, retrieval-absolute, retrieval-residual.
    cosmos_predict2p5_2b_480p_pusht_100_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_100_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_100_residual,
    cosmos_predict2p5_2b_480p_pusht_ret_100_residual_inference_only,
    # Top-K episode ablation (lowest per-episode future-action L1 error)
    # Baseline (no retrieval)
    cosmos_predict2p5_2b_480p_pusht_top50_no_pred,
    cosmos_predict2p5_2b_480p_pusht_top100_no_pred,
    cosmos_predict2p5_2b_480p_pusht_top50_no_pred_inference_only,
    cosmos_predict2p5_2b_480p_pusht_top100_no_pred_inference_only,
    # Residual (retrieval + delta prediction)
    cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual,
    cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_k3,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_k3,
    cosmos_predict2p5_2b_480p_pusht_ret_top50_no_pred_residual_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_residual_inference_only,
    # Absolute (non-residual) — control baseline for aux30 comparison
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_inference_only,
    # Absolute (non-residual) + auxiliary cotrain-RAG (30%) from extra_task_splits
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_no_pred_absolute_aux30_inference_only,
    # Top-K with future-state prediction (counterparts of the three _no_pred configs)
    cosmos_predict2p5_2b_480p_pusht_top100,
    cosmos_predict2p5_2b_480p_pusht_top100_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_residual,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only,
    # Initialized from Self-Forcing Stage 3 backbone (causal-distilled video DiT)
    cosmos_predict2p5_2b_480p_pusht_top100_sfstage3,
    cosmos_predict2p5_2b_480p_pusht_top100_sfstage3_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_sfstage3_inference_only,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute,
    cosmos_predict2p5_2b_480p_pusht_ret_top100_absolute_inference_only,
    # Image dropout ablation (retrieval_dropout_prob=0)
    cosmos_predict2p5_2b_480p_pusht_ret_100_no_pred_imgdrop,
    cosmos_predict2p5_2b_480p_pusht_ret_no_pred_imgdrop_inference_only,
    # Retrieval with absolute state only (state10, no future action features)
    cosmos_predict2p5_2b_480p_pusht_ret_s10_50_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_s10_100_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_s10_150_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_s10_200_no_pred,
    cosmos_predict2p5_2b_480p_pusht_ret_s10_no_pred_inference_only,
    # Lightweight DiT (train from scratch)
    *ALL_LIGHT_PUSHT_CONFIGS,
]
