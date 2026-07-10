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

"""
run_pusht_eval.py

Evaluates a trained Cosmos Policy on the PushT simulation task.

Usage example:
    uv run --extra cu128 --group pusht \
        -m cosmos_policy.experiments.robot.pusht.run_pusht_eval \
        --config cosmos_predict2_2b_480p_pusht__inference_only \
        --ckpt_path /path/to/checkpoint \
        --config_file cosmos_policy/config/config.py \
        --t5_text_embeddings_path /mnt/ssd/sangdoo/vla_je/dataset/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl \
        --dataset_stats_path /mnt/ssd/sangdoo/vla_je/dataset/PushT-Cosmos-Policy/success_only/dataset_stats.json \
        --visual_config base \
        --num_trials 50 \
        --chunk_size 8 \
        --num_open_loop_steps 8 \
        --local_log_dir cosmos_policy/experiments/robot/pusht/logs/ \
        --seed 42
"""

import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Add the pusht experiment directory to sys.path so local gym_pusht is importable
sys.path.insert(0, os.path.dirname(__file__))

import draccus
import imageio
import numpy as np

  # noqa: F401 — registers "gym_pusht/PushT-v0"
import gymnasium as gym

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robot_utils import (
    log_message,
    save_rollout_video_with_future_image_predictions,
    setup_logging,
)
from cosmos_policy.utils.utils import set_seed_everywhere

# PushT latent sequence (state_t=7, no wrist):
# [blank(0), proprio(1), primary(2), action(3), future_proprio(4), future_primary(5), value(6)]
CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 2
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = -1, -1

TASK_DESCRIPTION = "push t shaped block to the location"
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.85  # must match env.success_threshold

VISUAL_CONFIGS = {
    "base": {},
    "goal_flipped": {"goal_flipped": True},
    "color": {
        "block_color": "salmon",
    },
    "tri_color": {
        "agent_shape": "triangle",
        "block_color": "salmon",
    },
    "tri_default": {
        "agent_shape": "triangle",
    },
    "tri_goal_flipped": {
        "agent_shape": "triangle",
        "goal_flipped": True,
    },
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "pusht"                                        # Evaluation suite name

    ##########################################################################
    # Cosmos Policy model parameters
    ##########################################################################
    model_family: str = "cosmos"
    config: str = ""                                            # Inference config name
    ckpt_path: str = ""                                         # Pretrained checkpoint path
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = False                               # PushT has no wrist camera
    num_wrist_images: int = 0
    use_proprio: bool = True
    flip_images: bool = False                                   # PushT images don't need flipping
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    shift: float = 5.0                                          # Shift for rectified flow scheduler (predict2.5); ignored for EDM models
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 8
    num_open_loop_steps: int = 8

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 42
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 15

    ##########################################################################
    # PushT-specific parameters
    ##########################################################################
    visual_config: str = "base"                                 # Visual config: base | tri_color | tri_default | ...
    goal_angle: Optional[float] = None                            # Custom goal angle in radians (overrides goal_flipped)
    num_trials: int = 50                                        # Number of evaluation episodes
    env_img_res: int = 128                                      # Rendered image size (pixels, square)
    predict_future_states: bool = False                          # Whether to predict / visualize future states (set False for no_pred models)
    predict_values: bool = False                                 # Whether to predict values (set False for no_pred models)

    ##########################################################################
    # Logging
    ##########################################################################
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "YOUR_ENTITY"
    wandb_project: str = "YOUR_PROJECT"
    # fmt: on


def validate_config(cfg: PolicyEvalConfig) -> None:
    assert cfg.visual_config in VISUAL_CONFIGS, (
        f"Unknown visual_config '{cfg.visual_config}'. Choose from: {list(VISUAL_CONFIGS.keys())}"
    )
    assert not cfg.use_wrist_image, "PushT has no wrist camera. Set use_wrist_image=False."
    assert cfg.num_third_person_images == 1, "PushT uses exactly 1 primary (third-person) image."
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and cfg.dataset_stats_path == "":
        raise ValueError("Must provide dataset_stats_path when unnormalize_actions or normalize_proprio is True.")


def create_pusht_env(cfg: PolicyEvalConfig):
    """Create a PushT gymnasium environment with the selected visual config."""
    visual_opts = VISUAL_CONFIGS[cfg.visual_config]
    if cfg.goal_angle is not None:
        visual_opts["goal_angle"] = cfg.goal_angle
    import gym_pusht
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=cfg.env_img_res,
        observation_height=cfg.env_img_res,
    )
    env.success_threshold = SUCCESS_THRESHOLD  # ensure env uses the same success threshold as our evaluation metric
    return env, visual_opts


def prepare_observation(obs: dict) -> dict:
    """Extract primary image and 2-D proprio (agent position) from env obs."""
    return {
        "primary_image": obs["pixels"],            # (H, W, 3) uint8
        "proprio": obs["agent_pos"].astype(np.float32),  # [ax, ay] in [0, 512]
    }


def save_video(frames: list, filepath: str, fps: int = 10) -> None:
    """Save a list of (H, W, 3) uint8 frames as an MP4 video."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with imageio.get_writer(filepath, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    visual_opts: dict,
    model,
    planning_model,
    dataset_stats: dict,
    episode_idx: int,
    log_file=None,
) -> tuple:
    """Run a single evaluation episode. Returns (success, coverage, replay_frames, future_image_predictions_list)."""
    obs, _ = env.reset(seed=cfg.seed + episode_idx, options=visual_opts)

    action_queue = deque()
    replay_frames = []
    future_image_predictions_list = []
    success = False
    coverage = 0.0

    try:
        for t in range(MAX_STEPS):
            observation = prepare_observation(obs)
            replay_frames.append(observation["primary_image"].copy())

            # Query policy when the action queue is empty
            if len(action_queue) == 0:
                action_return_dict = get_action(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    TASK_DESCRIPTION,
                    seed=cfg.seed + episode_idx,
                    randomize_seed=cfg.randomize_seed,
                    num_denoising_steps_action=cfg.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=cfg.predict_future_states,
                )
                actions = action_return_dict["actions"]  # list of np arrays, each (action_dim,)
                for a in actions[: cfg.num_open_loop_steps]:
                    action_queue.append(a)

                future_preds = action_return_dict.get("future_image_predictions")
                if future_preds is not None:
                    future_image_predictions_list.append(future_preds)

            action = action_queue.popleft()
            # Clip action to valid range
            action = np.clip(action, 0.0, 512.0)

            obs, _reward, terminated, _truncated, info = env.step(action)
            coverage = info.get("coverage", 0.0)
            log_message(
                f"  t={t:3d}  coverage={coverage:.3f}  action={np.round(action, 1)}",
                log_file,
            )

            if terminated or info.get("is_success", False) or coverage >= SUCCESS_THRESHOLD:
                success = True
                break

    except Exception as e:
        log_message(f"Episode error: {e}\n{traceback.format_exc()}", log_file)

    return success, coverage, replay_frames, future_image_predictions_list


@draccus.wrap()
def eval_pusht(cfg: PolicyEvalConfig) -> None:
    """Main evaluation entry point for PushT."""
    validate_config(cfg)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    set_seed_everywhere(cfg.seed)

    # Initialize T5 embedding cache
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load dataset stats for action un-normalization / proprio normalization
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # Load model
    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"Train/test chunk size mismatch! "
        f"Train: {cosmos_config.dataloader_train.dataset.chunk_size}, Test: {cfg.chunk_size}"
    )
    planning_model = None
    if cfg.planning_model_ckpt_path:
        planning_model, _ = get_planning_model(cfg)

    # Setup logging
    log_file, _, run_id = setup_logging(
        cfg=cfg,
        task_identifier=f"pusht_{cfg.visual_config}",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)
    log_message(f"Visual config: {cfg.visual_config} → {VISUAL_CONFIGS[cfg.visual_config]}", log_file)

    # Create environment
    env, visual_opts = create_pusht_env(cfg)

    # Video output directory
    video_dir = os.path.join(cfg.local_log_dir, run_id, "videos")

    total_successes = 0
    total_coverage  = 0.0
    for episode_idx in range(cfg.num_trials):
        log_message(f"\n--- Episode {episode_idx + 1}/{cfg.num_trials} ---", log_file)

        t0 = time.time()
        success, coverage, replay_frames, future_image_predictions_list = run_episode(
            cfg,
            env,
            visual_opts,
            model,
            planning_model,
            dataset_stats,
            episode_idx,
            log_file,
        )
        elapsed = time.time() - t0

        if success:
            total_successes += 1
        total_coverage += coverage
        success_rate     = total_successes / (episode_idx + 1)
        average_coverage = total_coverage  / (episode_idx + 1)
        log_message(
            f"Episode {episode_idx + 1}: {'SUCCESS' if success else 'FAIL'} "
            f"({elapsed:.1f}s)  "
            f"Running success rate: {success_rate * 100:.1f}% ({total_successes}/{episode_idx + 1})  "
            f"Running avg coverage: {average_coverage * 100:.1f}%",
            log_file,
        )

        ep_suffix = f"ep{episode_idx + 1:03d}--{'success' if success else 'fail'}"

        # Save replay video
        video_path = os.path.join(video_dir, f"{ep_suffix}.mp4")
        save_video(replay_frames, video_path)

        # Save video with future image predictions (if available)
        future_primary_preds = [x["future_image"] for x in future_image_predictions_list if x.get("future_image") is not None]
        save_rollout_video_with_future_image_predictions(
            rollout_images=replay_frames,
            idx=episode_idx + 1,
            success=success,
            task_description=TASK_DESCRIPTION,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            output_dir=video_dir,
            future_primary_image_predictions=future_primary_preds,
            log_file=log_file,
        )

    final_success_rate = total_successes / cfg.num_trials
    final_coverage     = total_coverage  / cfg.num_trials
    log_message(
        f"\n=== FINAL RESULTS ===\n"
        f"Visual config   : {cfg.visual_config}\n"
        f"Trials          : {cfg.num_trials}\n"
        f"Successes       : {total_successes}\n"
        f"Success rate    : {final_success_rate * 100:.1f}%\n"
        f"Average coverage: {final_coverage * 100:.1f}%",
        log_file,
    )

    env.close()


if __name__ == "__main__":
    eval_pusht()
