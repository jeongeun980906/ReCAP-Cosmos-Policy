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
collect_success_demos.py

Run one or more Cosmos policies on PushT and save successful demos to HDF5 files.
uv run --extra cu128 --group pusht \
      cosmos_policy/experiments/robot/pusht/collect_success_demos.py

Output structure:
  OUTPUT_DIR/
    {visual_config}_{policy_name}_0/data.hdf5   (demos 0 .. DEMOS_PER_FILE-1)
    {visual_config}_{policy_name}_1/data.hdf5   (next batch, if any)
    ...

Each HDF5 file layout:
  data/
    demo_0/
      obs/
        images   shape=(T,)  dtype=vlen uint8  (JPEG-encoded frames)
        states   shape=(T, 2) float32  [agent x, agent y]
      actions    shape=(T, 2) float32  [target_x, target_y]
    demo_1/ ...
  attrs: num_demos

Usage:
  Edit the POLICIES list below, then run:
    python cosmos_policy/experiments/robot/pusht/collect_success_demos.py
"""

import io
import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import List

# Local gym_pusht must be on path
sys.path.insert(0, os.path.dirname(__file__))

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers "gym_pusht/PushT-v0")
import h5py
import numpy as np
from PIL import Image

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.utils.utils import set_seed_everywhere

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_DESCRIPTION = "push t shaped block to the location"
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.862
JPEG_QUALITY = 85

VISUAL_CONFIGS = {
    "base": {},
    "tri_color": {
        "background_color": (100, 100, 100),
        "agent_shape": "triangle",
        "block_color": "Gold",
    },
    "tri_default": {"agent_shape": "triangle"},
}

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = "/mnt/ssd/sangdoo/vla_je/dataset/PushT-Cosmos-Policy/success_only"

# ── Policy specification ───────────────────────────────────────────────────────

@dataclass
class PolicySpec:
    """Describes one policy to evaluate and collect demos from."""

    name: str                    # short id used in output folder names
    config: str                  # experiment config name (inference config)
    ckpt_path: str               # path to checkpoint directory
    t5_text_embeddings_path: str
    dataset_stats_path: str

    config_file: str = "cosmos_policy/config/config.py"
    chunk_size: int = 8
    num_open_loop_steps: int = 8
    num_denoising_steps: int = 5
    shift: float = 5.0           # rectified-flow shift (predict2.5); ignored for EDM

    visual_configs : List[str] = field(default_factory=lambda: ["base"])
    num_trials: int = 150        # episodes to run per (policy × visual_config)
    demos_per_file: int = 50     # demos per HDF5 file

    predict_future_states: bool = True #False
    predict_values: bool = True # False

    seed: int = 5000
    env_img_res: int = 128


# ── Configure policies here ────────────────────────────────────────────────────
POLICIES: List[PolicySpec] = [
    PolicySpec(
        name="predict2",
        config="cosmos_predict2_2b_480p_pusht__inference_only",
        ckpt_path="/tmp/imaginaire4-output/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_pusht/checkpoints/model_000003000.pt",
        t5_text_embeddings_path=(
            "/mnt/ssd/sangdoo/vla_je/dataset/PushT-Cosmos-Policy/success_only/t5_embeddings.pkl"
        ),
        dataset_stats_path=(
            "/mnt/ssd/sangdoo/vla_je/dataset/PushT-Cosmos-Policy/success_only/dataset_statistics.json"
        ),
        visual_configs=["tri_default"],
        demos_per_file=40,
        num_denoising_steps=5,
    ),
    # Add more policies here:
    # PolicySpec(
    #     name="predict2_edm",
    #     config="cosmos_predict2_2b_480p_pusht__inference_only",
    #     ckpt_path="/path/to/predict2/checkpoint",
    #     ...
    # ),
]
# ──────────────────────────────────────────────────────────────────────────────


# ── Thin cfg wrapper (accepted by get_action / get_model) ────────────────────

class _Cfg:
    """Minimal cfg duck-type accepted by cosmos_utils helpers."""

    # Fixed PushT settings
    suite = "pusht"
    use_third_person_image = True
    num_third_person_images = 1
    use_wrist_image = False
    num_wrist_images = 0
    use_proprio = True
    flip_images = False
    use_variance_scale = False
    use_jpeg_compression = True
    ar_future_prediction = False
    ar_value_prediction = False
    ar_qvalue_prediction = False
    unnormalize_actions = True
    normalize_proprio = True
    trained_with_image_aug = True
    deterministic = True
    randomize_seed = False
    num_queries_best_of_n = 1
    use_parallel_inference = False

    def __init__(self, spec: PolicySpec) -> None:
        self.config = spec.config
        self.ckpt_path = spec.ckpt_path
        self.config_file = spec.config_file
        self.t5_text_embeddings_path = spec.t5_text_embeddings_path
        self.dataset_stats_path = spec.dataset_stats_path
        self.chunk_size = spec.chunk_size
        self.num_open_loop_steps = spec.num_open_loop_steps
        self.num_denoising_steps_action = spec.num_denoising_steps
        self.shift = spec.shift
        self.predict_future_states = spec.predict_future_states
        self.predict_values = spec.predict_values
        self.seed = spec.seed


# ── HDF5 helpers ──────────────────────────────────────────────────────────────

def _encode_jpeg(img_hwc: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img_hwc).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def save_demos_to_hdf5(filepath: str, demos: list) -> None:
    """Save (images, states, actions) tuples to an HDF5 file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    with h5py.File(filepath, "w") as f:
        data_grp = f.create_group("data")
        for i, (images, states, actions) in enumerate(demos):
            grp = data_grp.create_group(f"demo_{i}")
            obs_grp = grp.create_group("obs")

            img_dset = obs_grp.create_dataset(
                "images", shape=(len(images),), dtype=vlen_uint8
            )
            for j, img in enumerate(images):
                img_dset[j] = _encode_jpeg(img)

            obs_grp.create_dataset(
                "states", data=np.array(states, dtype=np.float32)
            )
            grp.create_dataset(
                "actions", data=np.array(actions, dtype=np.float32)
            )
        f.attrs["num_demos"] = len(demos)
    print(f"  Saved {len(demos)} demos → {filepath}")


# ── Environment helpers ────────────────────────────────────────────────────────

def make_env(img_res: int):
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=img_res,
        observation_height=img_res,
    )
    env.unwrapped.success_threshold = SUCCESS_THRESHOLD
    return env


def get_full_state(obs: dict, raw_env) -> np.ndarray:
    """2-D state: [agent_x, agent_y]."""
    agent_pos = obs["agent_pos"]                        # (2,)
    block_pos = np.array(raw_env.block.position)        # (2,)
    block_angle = float(raw_env.block.angle)
    return np.array(
        [agent_pos[0], agent_pos[1], block_pos[0], block_pos[1], block_angle],
        dtype=np.float32,
    )


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    cfg_obj: _Cfg,
    model,
    dataset_stats: dict,
    env,
    raw_env,
    visual_opts: dict,
    episode_idx: int,
) -> tuple:
    """
    Run one episode.

    Returns:
        (success, images, states, actions, max_coverage)
        images:  list of (H, W, 3) uint8
        states:  list of (5,) float32
        actions: list of (2,) float32
    """
    obs, _ = env.reset(seed=cfg_obj.seed + episode_idx, options=visual_opts)

    action_queue: deque = deque()
    images, states, actions = [], [], []
    max_coverage = 0.0
    success = False

    try:
        for _ in range(MAX_STEPS):
            # Record observation before acting
            images.append(obs["pixels"].copy())
            states.append(get_full_state(obs, raw_env))

            # Query policy when queue is empty
            if not action_queue:
                observation = {
                    "primary_image": obs["pixels"],
                    "proprio": obs["agent_pos"].astype(np.float32),
                }
                action_return = get_action(
                    cfg_obj,
                    model,
                    dataset_stats,
                    observation,
                    TASK_DESCRIPTION,
                    seed=cfg_obj.seed + episode_idx,
                    randomize_seed=False,
                    num_denoising_steps_action=cfg_obj.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=cfg_obj.predict_future_states,
                )
                for a in action_return["actions"][: cfg_obj.num_open_loop_steps]:
                    action_queue.append(
                        np.clip(a, 0.0, 512.0).astype(np.float32)
                    )

            action = action_queue.popleft()
            actions.append(action)

            obs, _, terminated, __, info = env.step(action)
            cov = info.get("coverage", 0.0)
            print(f"coverage={cov:.3f}  ")
            if cov > max_coverage:
                max_coverage = cov

            if terminated or info.get("is_success", False) or cov >= SUCCESS_THRESHOLD:
                success = True
                break

    except Exception:
        print(f"    Episode error:\n{traceback.format_exc()}")

    return success, images, states, actions, max_coverage


# ── Per-policy collection ──────────────────────────────────────────────────────

def collect_for_policy(spec: PolicySpec) -> None:
    print(f"\n{'=' * 60}")
    print(f"Policy : {spec.name}")
    print(f"Config : {spec.config}")
    print(f"Ckpt   : {spec.ckpt_path}")
    print(f"{'=' * 60}")

    set_seed_everywhere(spec.seed)
    os.environ["DETERMINISTIC"] = "True"

    cfg_obj = _Cfg(spec)
    init_t5_text_embeddings_cache(spec.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(spec.dataset_stats_path)
    model, _ = get_model(cfg_obj)

    env = make_env(spec.env_img_res)
    raw_env = env.unwrapped

    for vc_name in spec.visual_configs:
        if vc_name not in VISUAL_CONFIGS:
            raise ValueError(
                f"Unknown visual_config '{vc_name}'. "
                f"Available: {list(VISUAL_CONFIGS.keys())}"
            )
        visual_opts = VISUAL_CONFIGS[vc_name]
        folder_prefix = f"{vc_name}_{spec.name}"
        print(f"\n  Visual config : {vc_name}  →  folder prefix: {folder_prefix}")

        success_demos: list = []
        file_idx = 0
        total_success = 0

        for ep_idx in range(spec.num_trials):
            t0 = time.time()
            success, images, states, actions, max_cov = run_episode(
                cfg_obj, model, dataset_stats, env, raw_env, visual_opts, ep_idx
            )
            elapsed = time.time() - t0

            print(
                f"    ep {ep_idx:3d}: coverage={max_cov:.3f}  "
                f"{'SUCCESS' if success else 'fail   '}  "
                f"({elapsed:.1f}s)  collected={total_success}"
            )

            if success:
                total_success += 1
                success_demos.append((images, states, actions))

                if len(success_demos) == spec.demos_per_file:
                    path = os.path.join(OUTPUT_DIR, f"{folder_prefix}_{file_idx}", "data.hdf5")
                    save_demos_to_hdf5(path, success_demos)
                    success_demos = []
                    file_idx += 1

        # Flush remaining partial batch
        if success_demos:
            path = os.path.join(OUTPUT_DIR, f"{folder_prefix}_{file_idx}", "data.hdf5")
            save_demos_to_hdf5(path, success_demos)
            file_idx += 1

        print(
            f"  [{spec.name} / {vc_name}] Done.  "
            f"Successes: {total_success}/{spec.num_trials}  Files written: {file_idx}"
        )

    env.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for spec in POLICIES:
        collect_for_policy(spec)
    print(f"\nAll done. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
