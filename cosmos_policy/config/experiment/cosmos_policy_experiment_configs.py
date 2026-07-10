# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# Modifications Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# This file is derived from NVIDIA Cosmos Policy
# (https://github.com/NVlabs/cosmos-policy) and was modified for the ReCAP
# project (https://github.com/jeongeun980906/ReCAP-Cosmos-Policy).
# Modifications are released under the Apache License, Version 2.0. See NOTICE.md.
# -----------------------------------------------------------------------------

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.models.policy_video2world_model_rectified_flow import CosmosPolicyVideo2WorldModelRectifiedFlow
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.config.experiment.pusht_experiment_configs import ALL_PUSHT_CONFIGS  # noqa: F401

cs = ConfigStore.instance()
val_sampling_size_override = dict(
    video_length=121,
    video_height=704,
    video_width=1280,
)
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


# *** Main checkpoint ***
libero_all_4_suites_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
# Dataset with precomputed Reason1 embeddings (for rectified flow / T2V models - faster training)
libero_all_4_suites_dataset_reason1 = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)

cosmos_predict2_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=30,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version
cosmos_predict2_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_libero__inference_only",
        ),
    )
)


# *** Cosmos Predict2.5 (T2V / Rectified Flow backbone) ***
cosmos_predict2p5_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
            {"override /data_train": "mock"},
            {"override /model": "policy_rectified_flow_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=2e-4,
            weight_decay=0.1,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            # LR decay for ~43K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[42800, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                conditional_frame_timestep=0,  # Near-clean timestep for conditional frames (0=clean, 1=noise)
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 1.0},
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    vae_pth = "hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                shift=5,
                action_loss_multiplier=16,
                use_dynamic_shift=False,
                train_time_distribution="uniform", #"logitnormal",#,
                train_time_weight="uniform",
                resolution="224",
                text_encoder_class="T5",
                text_encoder_config=None,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"),
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_libero__inference_only",
        ),
    )
)


# *** Main checkpoint ***
robocasa_50_demos_per_task_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
# Dataset with precomputed Reason1 embeddings (for rectified flow / T2V models)
robocasa_50_demos_stove_only_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    # Use precomputed Reason1 embeddings instead of T5 (stored under same key name for compatibility)
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    # rollout_data_dir= None,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=10,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                tokenizer=dict(
                    chunk_duration=37,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
                mask_value_prediction_loss_for_policy_prediction = True,  # Mask value prediction loss for the first 5 frames (policy prediction frames); only compute value prediction loss for future frames
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_per_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_per_task_dataset,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_stove_only_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_stove_only_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only__inference",
        ),
    )
)


robocasa_50_demos_kitchen_stove_only = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    rollout_data_dir= None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50_000,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=41,
                ),
                # mask_value_prediction_loss_for_policy_prediction = True,  # Mask value prediction loss
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_kitchen_stove_only,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_kitchen_stove_only,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove",
        ),
    )
)


robocasa_50_demos_kitchen_stove_only_no_pred = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    rollout_data_dir= None,
    predict_future_states=False,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=20_000,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=6,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action) — no future states, no value
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=21,  # 1 blank + 20 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action)
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_kitchen_stove_only_no_pred,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_kitchen_stove_only_no_pred,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred",
        ),
    )
)


# Train-Seen splits baseline (Cosmos Predict 2.5, with future-state prediction)
# Splits (see success_only/Splits.md):
#   OpenDoor (Single or Double) -> OpenSingleDoor, OpenDoubleDoor
#   CountertoX                  -> PnPCounterToCab, PnPCounterToMicrowave,
#                                  PnPCounterToSink, PnPCounterToStove
#   Closedrawer                 -> CloseDrawer
#   TurnOffStove                -> TurnOffStove
ROBOCASA_TRAIN_SEEN_TASKS = [
    "OpenSingleDoor",
    "OpenDoubleDoor",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "CloseDrawer",
    "TurnOffStove",
]

robocasa_train_seen_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    # Predict 2.5 uses Reason1 text embeddings (stored under the same key for compatibility)
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split=ROBOCASA_TRAIN_SEEN_TASKS,
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_train_seen = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50_000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,  # 1 blank + proprio + wrist + primary + secondary + action + future proprio + future wrist + future primary + future secondary + value
                min_num_conditional_frames=5,  # 1 blank + 4 conditioning (proprio, wrist, primary, secondary)
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 each for 10 categories)
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_train_seen_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_train_seen_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_train_seen",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_train_seen__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_train_seen",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_train_seen__inference",
        ),
    )
)


# OpenDoor-only baseline (Single + Double door, Cosmos Predict 2.5, with future-state prediction)
ROBOCASA_OPEN_DOOR_TASKS = [
    "OpenSingleDoor",
    "OpenDoubleDoor",
]

robocasa_open_door_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    task_split=ROBOCASA_OPEN_DOOR_TASKS,
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_open_door = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=10_000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_open_door_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_open_door_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_open_door",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_open_door__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door__inference",
        ),
    )
)


# *** Main checkpoint ***
aloha_cosmos_policy_dataset_185_demos = L(ALOHADataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=True,  # Include demos as success rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, left wrist, right wrist, primary, action, future proprio, future left wrist, future right wrist, future primary, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 left wrist image, 4 right wrist image, 4 primary image, 4 action, 4 future proprio, 4 future left wrist, 4 future right wrist, 4 future primary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_cosmos_policy_dataset_185_demos,
            sampler=L(DistributedSampler)(
                dataset=aloha_cosmos_policy_dataset_185_demos,
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
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ),
    )
)


# ALOHA planning model
# Dataset: 648 rollouts from evaluations with Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
# NOTE: This rollouts dataset is not released; you will need to replace `rollout_data_dir` below with your own rollouts dataset
aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset = L(
    ALOHADataset
)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
    demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
    rollout_data_dir=os.path.join(BASE_DATASETS_DIR, "PATH/TO/YOUR/ROLLOUTS/DATASET"),  # JPEG images
    use_jpeg_for_rollouts=True,  # JPEG images
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        checkpoint=dict(
            # Resume from 50K checkpoint of base Cosmos Policy run
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
            ),
        ),
        scheduler=dict(
            # LR decay for 15K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
            sampler=L(DistributedSampler)(
                dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_current_state_action_for_value_prediction=True,  # Use input masking to mask out irrelevant inputs (current state and action) during value prediction
            ),
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only",
        ),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        # LIBERO (T2V / Rectified Flow)
        cosmos_predict2p5_2b_480p_libero,  # *** Main checkpoint (T2V backbone) ***
        cosmos_predict2p5_2b_480p_libero__inference_only,
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove,  # *** Main checkpoint with smaller dataset (50 demos from a single task) for ablating data scaling ***
        cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred,
        # RoboCasa (T2V / Rectified Flow)
        # cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint (T2V backbone) ***
        # cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task__inference,
        cosmos_predict2p5_2b_480p_robocasa_train_seen,  # Train-Seen splits baseline (with future-state prediction)
        cosmos_predict2p5_2b_480p_robocasa_train_seen__inference,
        cosmos_predict2p5_2b_480p_robocasa_open_door,  # OpenDoor-only baseline (Single + Double)
        cosmos_predict2p5_2b_480p_robocasa_open_door__inference,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
        *ALL_PUSHT_CONFIGS,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
