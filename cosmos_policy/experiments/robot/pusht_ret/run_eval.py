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
cosmos_policy/experiments/robot/pusht_ret/run_eval.py

Evaluates a retrieval-augmented Cosmos Policy on PushT.
Uses gt pose (block_pos, agent_pos, block_angle) for 10-dim state retrieval.
No perception models (SAM3 / CoTracker) needed.

Usage:
    uv run --extra cu128 --group pusht \\
        -m cosmos_policy.experiments.robot.pusht_ret.run_eval \\
        --config cosmos_predict2p5_2b_480p_pusht_ret__inference_only \\
        --ckpt_path /path/to/checkpoint \\
        --config_file cosmos_policy/config/config.py \\
        --t5_text_embeddings_path .../t5_embeddings.pkl \\
        --dataset_stats_path .../dataset_stats.json \\
        --retrieval_data_dir .../success_only \\
        --visual_config tri_default \\
        --num_trials 50 \\
        --chunk_size 8 \\
        --num_open_loop_steps 8 \\
        --local_log_dir .../logs/ \\
        --seed 42
"""

import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Register local gym_pusht
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pusht"))

import draccus
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym  # noqa: F401

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.experiments.robot.robot_utils import (
    log_message,
    save_rollout_video_with_future_image_predictions,
    setup_logging,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from .retrieval import PushTRetrieval, resolve_retrieval_split
from .retrieval_consistent import PushTConsistentRetrieval
from .retrieval_cumulative import PushTCumulativeRetrieval

CURR_STATE_START_LATENT_IDX, CURR_STATE_END_LATENT_IDX = 1, 2
FUTURE_STATE_START_LATENT_IDX, FUTURE_STATE_END_LATENT_IDX = -1, -1

TASK_DESCRIPTION = "push t shaped block to the location"
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.85

VISUAL_CONFIGS = {
    "base": {},
    "goal_flipped":{"goal_flipped": True},
    "color": {
        'block_color': 'salmon',
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
    "tri_rot60": {
        "agent_shape": "triangle",
        "goal_angle": 60,
    },
    "tri_rot0": {
        "agent_shape": "triangle",
        "goal_angle": 0,
    },
    "tri_rot-60": {
        "agent_shape": "triangle",
        "goal_angle": -60,
    },
    "tri_rot30": {
        "agent_shape": "triangle",
        "goal_angle": 30,
    },
    "tri_rot-30": {
        "agent_shape": "triangle",
        "goal_angle": -30,
    },
    "tri_rot15": {
        "agent_shape": "triangle",
        "goal_angle": 15,
    },
    "tri_rot-15": {
        "agent_shape": "triangle",
        "goal_angle": -15,
    }

}

# Which retrieval pool split to use for each visual config.
# Each entry is a list of exact pool-directory names so substring matching in
# get_hdf5_files doesn't accidentally include *_flipcolor_* pools.
RETRIEVAL_SPLIT_MAP = {
    "color":            [f"color_{i}"        for i in range(5)],
    "tri_color":        [f"color_{i}"        for i in range(5)],
    "tri_rot60":        [f"rot60_{i}"        for i in range(5)],
    "tri_rot0":         [f"rot0_{i}"         for i in range(5)],
    "tri_rot-60":       [f"rot-60_{i}"       for i in range(5)],
    "tri_rot30":        [f"rot30_{i}"        for i in range(5)],
    "tri_rot-30":       [f"rot-30_{i}"       for i in range(5)],
    "tri_rot15":        [f"rot15_{i}"        for i in range(5)],
    "tri_rot-15":       [f"rot-15_{i}"       for i in range(5)],
    "tri_default":      [f"base_{i}"         for i in range(5)],
    "tri_goal_flipped": [f"goal_flipped_{i}" for i in range(5)],
}


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "pusht_ret"

    ##########################################################################
    # Cosmos Policy model
    ##########################################################################
    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = False
    num_wrist_images: int = 0
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    shift: float = 5.0
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 8
    num_open_loop_steps: int = 8

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 0
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 15

    ##########################################################################
    # PushT environment
    ##########################################################################
    visual_config: str = "tri_default"
    goal_angle: Optional[float] = None  # if set, overrides visual_config's goal setting (in degrees)
    num_trials: int = 50
    env_img_res: int = 128
    predict_future_states: bool = False
    predict_values: bool = False

    ##########################################################################
    # Retrieval
    ##########################################################################
    retrieval_data_dir: str = os.path.join(
        os.environ.get("BASE_DATASETS_DIR", "."), "PushT-Cosmos-Policy", "success_only"
    )
    block_rel: bool = False  # True: agent pos를 block 좌표계 기준 상대 좌표로 변환
    ret_context_multiplier: int = 1  # 1 = original window, 2 = 2x window [-l/2, 3l/2)
    ret_image_subsample: int = 1  # 1 = all frames, 2 = ::2 subsampling on ret images
    # Pool resolution. Priority: retrieval_pool_split > goal_angle K-NN > visual_config map.
    retrieval_pool_split: Optional[str] = None       # explicit comma-separated dir names, e.g. "rot30_0,rot30_1,rot60_0"
    retrieval_pool_k: int = 2                        # K nearest rot pools when goal_angle is set
    retrieval_pool_pattern: str = "plain"            # "plain" | "flipcolor" | "both"
    retrieval_strategy: str = "standard"  # "standard", "consistent", or "cumulative"
    ret_top_n: int = 3               # consistent/cumulative: number of diverse tracks
    ret_dist_threshold: float = 0.3  # consistent: re-retrieve when dist² > this
    ret_gamma: float = 0.95          # cumulative: EMA decay
    ret_switch_cost: float = 0.05    # cumulative: penalty for fresh start
    use_ret_image: bool = True       # False → zero out retrieved frames AND state (mask=0, like image+state dropout)
    use_ret_state: bool = True       # False → zero out retrieved state only (proprio not injected)
    use_residual_actions: bool = False  # True → model predicts delta; add retrieved action back at inference
    delta_stats_path: str = ""         # path to delta_dataset_statistics.json (required when use_residual_actions)

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
        f"Unknown visual_config '{cfg.visual_config}'. "
        f"Choose from: {list(VISUAL_CONFIGS.keys())}"
    )
    assert not cfg.use_wrist_image, "PushT has no wrist camera."
    assert cfg.num_third_person_images == 1
    if (cfg.unnormalize_actions or cfg.normalize_proprio) and not cfg.dataset_stats_path:
        raise ValueError("dataset_stats_path required.")
    if not cfg.retrieval_data_dir:
        raise ValueError("retrieval_data_dir is required.")


def create_pusht_env(cfg: PolicyEvalConfig):
    visual_opts = VISUAL_CONFIGS[cfg.visual_config]
    if cfg.goal_angle is not None:
        visual_opts["goal_angle"] = cfg.goal_angle
    import gym_pusht  # noqa: F401
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=cfg.env_img_res,
        observation_height=cfg.env_img_res,
    )
    env.success_threshold = SUCCESS_THRESHOLD
    return env, visual_opts


def prepare_observation(obs: dict) -> dict:
    return {
        "primary_image": obs["pixels"],
        "proprio": obs["agent_pos"].astype(np.float32),
    }


def save_video(frames: list, filepath: str, fps: int = 10) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with imageio.get_writer(filepath, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_combined_video(
    replay_frames: list,
    chunk_future_preds: list,
    chunk_ret_frames: list,
    num_open_loop_steps: int,
    filepath: str,
    fps: int = 10,
) -> None:
    """Save 3-panel video: real | generated | retrieved (side by side)."""
    if not replay_frames:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    H, W, C = replay_frames[0].shape
    text_height = 28
    labels = ["real", "generated", "retrieved"]

    def resize_to(img):
        if img.shape[:2] == (H, W):
            return img
        return np.array(Image.fromarray(img).resize((W, H), Image.LANCZOS))

    # Build label bar once
    bar_arr = np.ones((text_height, W * 3, C), dtype=np.uint8) * 40
    bar_pil = Image.fromarray(bar_arr)
    draw = ImageDraw.Draw(bar_pil)
    try:
        font = ImageFont.truetype("DejaVuSans", 13)
    except IOError:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x = i * W + W // 2
        tw = draw.textlength(label, font=font)
        draw.text((x - tw // 2, 7), label, font=font, fill=(255, 255, 255))
    label_bar = np.array(bar_pil)

    blank = np.zeros((H, W, C), dtype=np.uint8)

    with imageio.get_writer(filepath, fps=fps) as writer:
        for t, real in enumerate(replay_frames):
            real = resize_to(real)
            chunk_idx = t // num_open_loop_steps
            frame_in_chunk = t % num_open_loop_steps

            # Generated: future image predicted at this chunk
            gen = blank
            if chunk_idx < len(chunk_future_preds):
                preds = chunk_future_preds[chunk_idx]
                if preds is not None and preds.get("future_image") is not None:
                    gen = resize_to(preds["future_image"])

            # Retrieved: frame within the retrieved demo clip
            ret = blank
            if chunk_idx < len(chunk_ret_frames) and chunk_ret_frames[chunk_idx] is not None:
                rf = chunk_ret_frames[chunk_idx]
                ret = resize_to(rf[min(frame_in_chunk, len(rf) - 1)])

            combined = np.concatenate([real, gen, ret], axis=1)
            writer.append_data(np.vstack([label_bar, combined]))


def run_episode(
    cfg: PolicyEvalConfig,
    env,
    visual_opts: dict,
    model,
    planning_model,
    dataset_stats: dict,
    episode_idx: int,
    retrieval: PushTRetrieval,
    log_file=None,
    delta_stats: dict = None,
) -> tuple:
    """Run one episode with gt-pose retrieval. Returns (success, coverage, replay_frames, ...)."""
    obs, _ = env.reset(seed=cfg.seed + episode_idx, options=visual_opts)

    action_queue = deque()
    replay_frames = []
    future_image_predictions_list = []
    chunk_future_preds = []
    chunk_ret_frames = []
    success = False
    coverage = 0.0

    # Position history buffers for velocity computation
    block_pos_history = deque(maxlen=retrieval.WINDOW_SIZE)
    agent_pos_history = deque(maxlen=retrieval.WINDOW_SIZE)

    try:
        for t in range(MAX_STEPS):
            observation = prepare_observation(obs)
            frame = observation["primary_image"]
            replay_frames.append(frame.copy())

            # Collect gt pose every timestep for velocity history
            unwrapped   = env.unwrapped
            block_pos   = np.array(unwrapped.block.position, dtype=np.float32)
            agent_pos   = observation["proprio"].copy()
            block_pos_history.append(block_pos)
            agent_pos_history.append(agent_pos)

            if len(action_queue) == 0:
                block_angle = float(unwrapped.block.angle)

                bph = np.array(block_pos_history) if len(block_pos_history) >= 2 else None
                aph = np.array(agent_pos_history) if len(agent_pos_history) >= 2 else None

                ret_frames, ret_actions, ret_proprio = retrieval.get_retrieved_data(
                    agent_pos=agent_pos,
                    block_pos=block_pos,
                    block_angle=block_angle,
                    block_pos_history=bph,
                    agent_pos_history=aph,
                )
                observation["retrieved_frames"]  = ret_frames
                observation["retrieved_actions"] = ret_actions
                observation["retrieved_proprio"] = ret_proprio
                chunk_ret_frames.append(ret_frames)

                # For residual mode: disable unnormalization in get_action;
                # we unnormalize the delta with delta_stats ourselves.
                if cfg.use_residual_actions:
                    _orig_unnorm = cfg.unnormalize_actions
                    cfg.unnormalize_actions = False

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

                if cfg.use_residual_actions:
                    cfg.unnormalize_actions = _orig_unnorm
                    # Model output is delta normalized with delta_stats → unnorm to raw delta
                    d_min = delta_stats["delta_actions_min"]
                    d_max = delta_stats["delta_actions_max"]
                    norm_delta = np.stack(action_return_dict["actions"])  # (chunk, action_dim)
                    raw_delta = 0.5 * (norm_delta + 1.0) * (d_max - d_min) + d_min
                    # Add raw retrieved actions → raw final action
                    raw_actions = raw_delta + ret_actions[:len(raw_delta)]
                    action_return_dict["actions"] = [raw_actions[i] for i in range(len(raw_actions))]

                for a in action_return_dict["actions"][: cfg.num_open_loop_steps]:
                    action_queue.append(a)

                future_preds = action_return_dict.get("future_image_predictions")
                chunk_future_preds.append(future_preds)
                if future_preds is not None:
                    future_image_predictions_list.append(future_preds)

            action = np.clip(action_queue.popleft(), 0.0, 512.0)
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

    return (success, coverage, replay_frames, future_image_predictions_list,
            chunk_future_preds, chunk_ret_frames)


@draccus.wrap()
def eval_pusht_ret(cfg: PolicyEvalConfig) -> None:
    validate_config(cfg)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # Load delta statistics for residual mode
    delta_stats = None
    if cfg.use_residual_actions:
        assert cfg.delta_stats_path, "delta_stats_path required when use_residual_actions=True"
        delta_stats = load_dataset_stats(cfg.delta_stats_path)

    model, cosmos_config = get_model(cfg)
    assert cfg.chunk_size == cosmos_config.dataloader_train.dataset.chunk_size, (
        f"Chunk-size mismatch: train={cosmos_config.dataloader_train.dataset.chunk_size} "
        f"vs eval={cfg.chunk_size}"
    )
    planning_model = None
    if cfg.planning_model_ckpt_path:
        planning_model, _ = get_planning_model(cfg)

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

    env, visual_opts = create_pusht_env(cfg)

    retrieval_split = resolve_retrieval_split(
        data_dir=cfg.retrieval_data_dir,
        visual_config=cfg.visual_config,
        goal_angle=cfg.goal_angle,
        explicit_split=cfg.retrieval_pool_split,
        k=cfg.retrieval_pool_k,
        fallback_map=RETRIEVAL_SPLIT_MAP,
        pool_pattern=cfg.retrieval_pool_pattern,
    )
    log_message(f"Retrieval: strategy='{cfg.retrieval_strategy}', split='{retrieval_split}'", log_file)
    if cfg.retrieval_strategy == "consistent":
        retrieval = PushTConsistentRetrieval(
            data_dir=cfg.retrieval_data_dir,
            chunk_size=cfg.chunk_size,
            split=retrieval_split,
            block_rel=cfg.block_rel,
            ret_context_multiplier=cfg.ret_context_multiplier,
            ret_image_subsample=cfg.ret_image_subsample,
            top_n=cfg.ret_top_n,
            dist_threshold=cfg.ret_dist_threshold,
        )
    elif cfg.retrieval_strategy == "cumulative":
        retrieval = PushTCumulativeRetrieval(
            data_dir=cfg.retrieval_data_dir,
            chunk_size=cfg.chunk_size,
            split=retrieval_split,
            block_rel=cfg.block_rel,
            ret_context_multiplier=cfg.ret_context_multiplier,
            ret_image_subsample=cfg.ret_image_subsample,
            top_n=cfg.ret_top_n,
            gamma=cfg.ret_gamma,
            switch_cost=cfg.ret_switch_cost,
        )
    else:
        retrieval = PushTRetrieval(
            data_dir=cfg.retrieval_data_dir,
            chunk_size=cfg.chunk_size,
            split=retrieval_split,
            block_rel=cfg.block_rel,
            ret_context_multiplier=cfg.ret_context_multiplier,
            ret_image_subsample=cfg.ret_image_subsample,
        )

    video_dir       = os.path.join(cfg.local_log_dir, run_id, "videos")
    total_successes = 0
    total_coverage  = 0.0

    for episode_idx in range(cfg.num_trials):
        if hasattr(retrieval, "reset"):
            retrieval.reset()
        log_message(f"\n--- Episode {episode_idx + 1}/{cfg.num_trials} ---", log_file)
        t0 = time.time()
        (success, coverage, replay_frames, future_preds_list,
         chunk_future_preds, chunk_ret_frames) = run_episode(
            cfg, env, visual_opts, model, planning_model,
            dataset_stats, episode_idx, retrieval, log_file,
            delta_stats=delta_stats,
        )
        elapsed = time.time() - t0

        if success:
            total_successes += 1
        total_coverage  += coverage
        success_rate     = total_successes / (episode_idx + 1)
        average_coverage = total_coverage  / (episode_idx + 1)
        ret_stats = ""
        if hasattr(retrieval, "_retrieval_count"):
            total = retrieval._retrieval_count + retrieval._follow_count
            if total > 0:
                ret_stats = (f"  [ret: {retrieval._retrieval_count} retrieve, "
                             f"{retrieval._follow_count} follow]")
        log_message(
            f"Episode {episode_idx + 1}: {'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s)  "
            f"Success rate: {success_rate * 100:.1f}% ({total_successes}/{episode_idx + 1})  "
            f"Avg coverage: {average_coverage * 100:.1f}%{ret_stats}",
            log_file,
        )

        ep_suffix = f"ep{episode_idx + 1:03d}--{'success' if success else 'fail'}"
        save_video(replay_frames, os.path.join(video_dir, f"{ep_suffix}.mp4"))
        save_combined_video(
            replay_frames=replay_frames,
            chunk_future_preds=chunk_future_preds,
            chunk_ret_frames=chunk_ret_frames,
            num_open_loop_steps=cfg.num_open_loop_steps,
            filepath=os.path.join(video_dir, f"{ep_suffix}--combined.mp4"),
            fps=10,
        )

        future_imgs = [x["future_image"] for x in future_preds_list
                       if x.get("future_image") is not None]
        save_rollout_video_with_future_image_predictions(
            rollout_images=replay_frames,
            idx=episode_idx + 1,
            success=success,
            task_description=TASK_DESCRIPTION,
            chunk_size=cfg.chunk_size,
            num_open_loop_steps=cfg.num_open_loop_steps,
            output_dir=video_dir,
            future_primary_image_predictions=future_imgs,
            log_file=log_file,
        )

    final_sr  = total_successes / cfg.num_trials
    final_cov = total_coverage  / cfg.num_trials
    log_message(
        f"\n=== FINAL RESULTS ===\n"
        f"Visual config   : {cfg.visual_config}\n"
        f"Trials          : {cfg.num_trials}\n"
        f"Successes       : {total_successes}\n"
        f"Success rate    : {final_sr  * 100:.1f}%\n"
        f"Average coverage: {final_cov * 100:.1f}%",
        log_file,
    )
    env.close()


if __name__ == "__main__":
    eval_pusht_ret()
