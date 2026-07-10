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
LIBERO simulation benchmark task suites dataloader.

Run this command to print a few samples from the LIBERO dataset:
    python -m cosmos_policy.datasets.libero_dataset
"""

import json
import os
import pickle
import random
import re
from collections import defaultdict

import h5py
import imageio
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    build_rollout_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    decode_single_jpeg_frame,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    rescale_episode_data,
)
from cosmos_policy.utils.utils import duplicate_array

# Set floating point precision to 3 decimal places and disable line wrapping
np.set_printoptions(precision=3, linewidth=np.inf)


class PushTDataset2(Dataset):
    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = False,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        rollout_data_dir: str = "",
        task_split: list = ['base'],
        demonstration_sampling_prob: float = 0.5,
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = True,
        predict_future_states: bool = True,
        task_type: str = 'baseline',
        gamma: float = 0.99,
        max_num_episodes: int = -1,
        # ── Top-K episode filtering (by precomputed per-episode action error) ──
        # Either a single path (applied to both tri_default and tri_goal splits)
        # or a list of two paths mapped to [tri_default, tri_goal]. Top-K is
        # applied per-allowlist so each task contributes at most K episodes.
        episode_allowlist_path: "str | list[str]" = "",
        episode_allowlist_top_k: int = -1,
        # ── Mixed-baseline extension (opt-in; default preserves old behavior) ──
        # extra_task_splits=[(prefix, max_eps), ...] loads additional suite
        # folders matching '^{prefix}_\\d+$' (strict match: 'tri_default' will
        # NOT pick up 'tri_default_predict2_*'). max_eps<=0 means "all".
        extra_task_splits: "list[tuple[str, int]] | None" = None,
    ):
        """
        Initialize LIBERO dataset for training.

        Args:
            data_dir (str): Path to directory containing LIBERO task suite HDF5 files
            chunk_size (int): Action chunk size
            final_image_size (int): Target size for resized images (square), defaults to 224
            t5_text_embeddings_path (str): Path to precomputed T5 text embeddings dictionary (key: instruction, val: embedding)
            num_images_per_sample (int): Number of images to return per sample
            normalize_images (bool): Whether to normalize the images and return as torch.float32
            normalize_actions (bool): Whether to normalize the actions
            normalize_proprio (bool): Whether to normalize the proprioceptive state
            use_image_aug (bool): Whether to apply image augmentations
            use_stronger_image_aug (bool): Whether to apply stronger image augmentations
            use_wrist_images (bool): If True, loads wrist-mounted camera images
            use_third_person_images (bool): If True, loads third-person images
            use_proprio (bool): If True, adds proprio to image observations
            num_duplicates_per_image (int): Number of times to duplicate each image (so that each type of image fills 1 latent frame when encoded with the tokenizer)
            rollout_data_dir (str): Path to directory containing rollout data (if provided, will load rollout data in addition to base dataset)
            demonstration_sampling_prob (float): Probability of sampling from demonstration data instead of rollout data
            success_rollout_sampling_prob (float): Probability of sampling from success rollout data instead of failure rollout data
            treat_success_rollouts_as_demos (bool): If True, copy successful rollout episodes into demonstration dataset (self.data)
            return_value_function_returns (bool): If True, returns value function returns for rollout episodes
            gamma (float): Discount factor for value function returns
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.rollout_data_dir = rollout_data_dir
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.success_rollout_sampling_prob = success_rollout_sampling_prob
        self.treat_success_rollouts_as_demos = treat_success_rollouts_as_demos
        self.return_value_function_returns = return_value_function_returns
        self.predict_future_states = predict_future_states
        self.gamma = gamma

        assert self.use_wrist_images or self.use_third_person_images, (
            "Must use at least one of wrist images or third-person images!"
        )

        # Build per-split (suite, demo_key) allowlists. Splits are hardcoded
        # below as ['tri_default', 'tri_goal']; the allowlist path param may
        # be either a single path (applied to both) or a list paired 1-to-1.
        _hardcoded_splits = ["tri_default", "tri_goal"]
        if isinstance(episode_allowlist_path, str):
            _allowlist_paths = (
                [episode_allowlist_path] * len(_hardcoded_splits)
                if episode_allowlist_path
                else []
            )
        else:
            _allowlist_paths = list(episode_allowlist_path)
        if _allowlist_paths:
            assert len(_allowlist_paths) == len(_hardcoded_splits), (
                f"episode_allowlist_path length ({len(_allowlist_paths)}) must "
                f"match hardcoded splits {_hardcoded_splits}"
            )

        split_to_allowed: dict[str, "set | None"] = {s: None for s in _hardcoded_splits}
        for split, path in zip(_hardcoded_splits, _allowlist_paths):
            if not path:
                continue
            with open(path, "r") as f:
                ranking = json.load(f)
            ranked = ranking["episodes"]
            if episode_allowlist_top_k > 0:
                ranked = ranked[:episode_allowlist_top_k]
            split_to_allowed[split] = {(r["suite"], r["demo_key"]) for r in ranked}
            print(
                f"[PushTDataset2] split={split}: keeping {len(split_to_allowed[split])} episodes "
                f"(top-{episode_allowlist_top_k if episode_allowlist_top_k > 0 else 'ALL'} "
                f"from {path})"
            )
        self._split_to_allowed = split_to_allowed

        # Get all HDF5 files in data directory
        hdf5_files = get_hdf5_files(data_dir, task_split=['tri_default'])
        # In debug mode, only load the first demo
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]
        self.data = {}
        self.rollout_episode_metadata = {}  # For lazy loading: episode_idx -> metadata dict
        self.num_episodes = 0
        self.num_steps = 0
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.unique_commands = set()

        # Global step mapping from task suite name to list[global_step_idx]
        # Populated later in `_build_step_index_mapping()`
        self._suite_to_step_indices = {}
        self.max_num_episodes = max_num_episodes
        
        # Collect all (file, demo_key) pairs first (tri_default split)
        tri_default_allowed = split_to_allowed.get("tri_default")
        all_episode_refs = []
        for file in hdf5_files:
            with h5py.File(file, "r") as f:
                demo_keys_list = list(f["data"].keys())
                sorted_demo_keys = sorted(demo_keys_list, key=lambda x: int(x.split("_")[1]))
                suite = os.path.relpath(file, data_dir).split(os.sep)[0]
                for demo_key in sorted_demo_keys:
                    if tri_default_allowed is not None and (suite, demo_key) not in tri_default_allowed:
                        continue
                    all_episode_refs.append((file, demo_key))

        # Cap tri_default episodes to N (applied per-split)
        if max_num_episodes > 0 and max_num_episodes < len(all_episode_refs):
            all_episode_refs = all_episode_refs[:max_num_episodes]

        # tri_goal split
        tri_goal_allowed = split_to_allowed.get("tri_goal")
        hdf5_files = get_hdf5_files(data_dir, task_split=['tri_goal'])
        all_episode_refs2 = []
        for file in hdf5_files:
            with h5py.File(file, "r") as f:
                demo_keys_list = list(f["data"].keys())
                sorted_demo_keys = sorted(demo_keys_list, key=lambda x: int(x.split("_")[1]))
                suite = os.path.relpath(file, data_dir).split(os.sep)[0]
                for demo_key in sorted_demo_keys:
                    if tri_goal_allowed is not None and (suite, demo_key) not in tri_goal_allowed:
                        continue
                    all_episode_refs2.append((file, demo_key))
        if max_num_episodes > 0 and max_num_episodes < len(all_episode_refs2):
            all_episode_refs2 = all_episode_refs2[:max_num_episodes]
        all_episode_refs.extend(all_episode_refs2)

        # Extra splits (strict ^{prefix}_\d+$ suite match so that, e.g.,
        # 'tri_default' does not pull in 'tri_default_predict2_0').
        # Each tuple: (prefix, per_split_max_eps); per_split_max_eps<=0 means all.
        if extra_task_splits:
            extra_counts: dict[str, int] = {}
            for prefix, max_eps in extra_task_splits:
                suite_pattern = re.compile(rf"^{re.escape(prefix)}_\d+$")
                try:
                    suite_names = sorted(os.listdir(data_dir))
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Data directory not found: {data_dir}") from e
                matched_suites = [
                    s for s in suite_names
                    if suite_pattern.match(s) and os.path.isdir(os.path.join(data_dir, s))
                ]
                extra_refs: list = []
                for suite in matched_suites:
                    suite_dir = os.path.join(data_dir, suite)
                    for root, _dirs, names in os.walk(suite_dir, followlinks=True):
                        for name in sorted(names):
                            if not name.lower().endswith((".h5", ".hdf5", ".he5")):
                                continue
                            fp = os.path.join(root, name)
                            with h5py.File(fp, "r") as f:
                                demo_keys_list = list(f["data"].keys())
                            sorted_demo_keys = sorted(
                                demo_keys_list, key=lambda x: int(x.split("_")[1])
                            )
                            for dk in sorted_demo_keys:
                                extra_refs.append((fp, dk))
                if max_eps is not None and max_eps > 0 and len(extra_refs) > max_eps:
                    extra_refs = extra_refs[:max_eps]
                all_episode_refs.extend(extra_refs)
                extra_counts[prefix] = len(extra_refs)
                print(
                    f"[PushTDataset2] extra split '{prefix}': {len(extra_refs)} episodes "
                    f"from {len(matched_suites)} suite folder(s)"
                )
            self._extra_split_counts = extra_counts
        # Group by file to minimize HDF5 open/close overhead
        file_to_keys = defaultdict(list)
        for file, demo_key in all_episode_refs:
            file_to_keys[file].append(demo_key)

        for file, demo_keys in tqdm(file_to_keys.items()):
            with h5py.File(file, "r") as f:
                for demo_key in tqdm(demo_keys):
                    # Determine whether the dataset stores raw RGB frames or JPEG bytes
                    obs_group = f[f"data/{demo_key}/obs"]
                    # Agent-view (third-person) images
                    if "images" in obs_group:
                        images = decode_jpeg_bytes_dataset(obs_group["images"])
                    else:
                        raise KeyError("Neither 'images' nor 'images_jpeg' found in HDF5 file.")
                    # Actions
                    actions = f[f"data/{demo_key}/actions"][:].astype(
                        np.float32
                    )  # (episode_len, action_dim=7), float32
                    # Proprio states
                    proprio = f[f"data/{demo_key}/obs/states"][:][:,:2].astype(
                        np.float32
                    )  # (episode_len, proprio_dim=2), float32
                    # language instruction]
                    command = 'push t shaped block to the location'
                    self.unique_commands.add(command)
                    num_steps = len(images)
                    # Add value function returns if applicable
                    if self.return_value_function_returns:
                        returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                    # Add entry to dataset dict
                    self.data[self.num_episodes] = dict(
                        images=images,
                        proprio=proprio,
                        actions=actions,
                        command=command,
                        num_steps=num_steps,
                        suite=os.path.relpath(file, self.data_dir).split(os.sep)[
                            0
                        ],  # Task suite folder name (e.g. libero_spatial_no_noops_rerendered)
                        returns=returns.copy() if self.return_value_function_returns else None,
                    )
                    # Update number of episodes
                    self.num_episodes += 1
                    # Update number of steps
                    self.num_steps += num_steps
        if task_type == 'cotrain':
            hdf5_files_ref = get_hdf5_files(data_dir, task_split = ['base', 'color', 'goal_flip'])
            # add all demos in the other splits as well
            for file in hdf5_files_ref:
                with h5py.File(file, "r") as f:
                    demo_keys_list = list(f["data"].keys())
                    sorted_demo_keys = sorted(demo_keys_list, key=lambda x: int(x.split("_")[1]))
                    for demo_key in sorted_demo_keys:
                        obs_group = f[f"data/{demo_key}/obs"]
                        # Agent-view (third-person) images
                        if "images" in obs_group:
                            images = decode_jpeg_bytes_dataset(obs_group["images"])
                        else:
                            raise KeyError("Neither 'images' nor 'images_jpeg' found in HDF5 file.")
                        # Actions
                        actions = f[f"data/{demo_key}/actions"][:].astype(
                            np.float32
                        )  # (episode_len, action_dim=7), float32
                        # Proprio states
                        proprio = f[f"data/{demo_key}/obs/states"][:].astype(
                            np.float32
                        )  # (episode_len, proprio_dim=9), float32
                        # language instruction]
                        command = 'push t shaped block to the location'
                        self.unique_commands.add(command)
                        num_steps = len(images)
                        # Add value function returns if applicable
                        if self.return_value_function_returns:
                            returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                        # Add entry to dataset dict
                        self.data[self.num_episodes] = dict(
                            images=images,
                            proprio=proprio,
                            actions=actions,
                            command=command,
                            num_steps=num_steps,
                            suite=os.path.relpath(file, self.data_dir).split(os.sep)[
                                0
                            ],  # Task suite folder name (e.g. libero_spatial_no_noops_rerendered)
                            returns=returns.copy() if self.return_value_function_returns else None,
                        )
                        # Update number of episodes
                        self.num_episodes += 1
                        # Update number of steps
                        self.num_steps += num_steps
        # Build mapping from global step index to episode step
        self._build_step_index_mapping()

        self.chunk_size = chunk_size

        # If applicable, load precomputed T5 text embeddings
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as file:
                self.t5_text_embeddings = pickle.load(file)

        # Calculate dataset statistics if the stats file doesn't exist
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # Normalize actions and/or proprio
        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")

            # Calculate post-normalization action statistics
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )
        # Calculate epoch structure and counts
        self._calculate_epoch_structure()
        print(f"Finished loading dataset with {self.num_episodes} episodes and {self.num_steps} steps.")
    def _calculate_epoch_structure(self):
        """Calculate epoch layout with proper scaling: demos, success rollouts, failure rollouts."""
        # Initialize rollout step counts if not available
        if not hasattr(self, "_rollout_success_total_steps"):
            self._rollout_success_total_steps = 0
        if not hasattr(self, "_rollout_failure_total_steps"):
            self._rollout_failure_total_steps = 0
        if not hasattr(self, "_rollout_total_steps"):
            self._rollout_total_steps = self._rollout_success_total_steps + self._rollout_failure_total_steps

        demo_base_count = self.num_steps
        
        result = calculate_epoch_structure(
            num_steps=demo_base_count,
            rollout_success_total_steps=self._rollout_success_total_steps,
            rollout_failure_total_steps=self._rollout_failure_total_steps,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=self.success_rollout_sampling_prob,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    def _build_step_index_mapping(self):
        """Build a mapping from global step index to (episode index, relative index within episode)."""
        self._step_to_episode_map = {}
        self._total_steps = 0

        # Reset suite mapping if it already exists
        self._suite_to_step_indices = defaultdict(list)

        for episode_idx, episode_data in self.data.items():
            num_steps = episode_data["num_steps"]
            for i in range(num_steps):
                self._step_to_episode_map[self._total_steps] = (episode_idx, i)
                self._suite_to_step_indices[episode_data["suite"]].append(self._total_steps)
                self._total_steps += 1

        # Additional bookkeeping for balanced sampling
        self._suites = list(self._suite_to_step_indices.keys())
        if len(self._suites) > 0:
            self._max_suite_len = max(len(v) for v in self._suite_to_step_indices.values())

    def _build_rollout_step_index_mapping(self):
        """Build mapping for rollout dataset with separate tracking for successful/failure episodes."""
        result = build_rollout_step_index_mapping({}, self.rollout_episode_metadata)
        self._rollout_success_step_to_episode_map = result["_rollout_success_step_to_episode_map"]
        self._rollout_failure_step_to_episode_map = result["_rollout_failure_step_to_episode_map"]
        self._rollout_success_total_steps = result["_rollout_success_total_steps"]
        self._rollout_failure_total_steps = result["_rollout_failure_total_steps"]
        self._rollout_total_steps = result["_rollout_total_steps"]

    def _load_rollout_episode_data(self, episode_metadata):
        """
        Load rollout episode data from HDF5 file using metadata.

        Args:
            episode_metadata (dict): Episode metadata containing file_path, success, etc.

        Returns:
            dict: Episode data dictionary with loaded arrays
        """
        file_path = episode_metadata["file_path"]

        with h5py.File(file_path, "r") as f:
            # Load images based on storage format
            if episode_metadata["is_jpeg"]:
                # Store raw JPEG bytes
                images = f["images"][:]
            else:
                images = f["images"][:]

            # Load actions and proprio
            actions = f["actions"][:].astype(np.float32)
            proprio = f["proprio"][:].astype(np.float32)

            # Apply normalization if needed
            if self.normalize_actions:
                actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
            if self.normalize_proprio:
                proprio = rescale_episode_data(
                    {"proprio": proprio},
                    self.dataset_stats,
                    "proprio",
                )

            # Create episode data dictionary
            episode_data = dict(
                images=images,
                proprio=proprio,
                actions=actions,
                command=episode_metadata["command"],
                num_steps=episode_metadata["num_steps"],
                success=episode_metadata["success"],
                is_jpeg=episode_metadata["is_jpeg"],
            )

            return episode_data

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        # Return pre-calculated epoch length (which already accounts for suite balancing if enabled)
        return self.epoch_length

    def __getitem__(self, idx):
        """
        Fetches images and action chunk sample by index.
        Returns action chunk rather than just single-step action.
        If the action chunk retrieval would go out of bounds, the last action is repeated however
        many times needed to fill up the chunk.

        Args:
            idx: Integer index to retrieve sample

        Returns:
            dict: Data sample: {
                video=images,
                actions=action chunk,
                t5_text_embeddings=text embedding,
                t5_text_mask=text embedding mask,
                fps=frames per second,
                padding_mask=padding mask,
                num_frames=number of frames per sequence,
                image_size=image size,
                proprio=proprio state,
                __key__=unique sample identifier,
            }
        """

        # Determine which dataset to sample from based on index ranges
        # Layout of indices within dataset: [demos] [success rollouts] [failure rollouts]
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)

        rollout_data_mask = 1 if sample_type != "demo" else 0
        rollout_data_success_mask = 1 if sample_type == "success_rollout" else 0

        if sample_type == "demo":
            # Get demonstration sample
            global_step_idx = idx % self.num_steps
            # Using global step index, get episode index and relative step index within that episode
            episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
            episode_metadata = None
            episode_data = self.data[episode_idx]
            global_rollout_idx = -1  # Not applicable for demonstration data
        elif sample_type == "success_rollout":
            # Success rollout sample
            success_idx = idx - self.adjusted_demo_count  # Index within success rollouts section
            global_rollout_idx = success_idx % self._rollout_success_total_steps
            episode_idx, relative_step_idx = self._rollout_success_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)
        else:
            # Failure rollout sample
            failure_idx = (
                idx - self.adjusted_demo_count - self.adjusted_success_rollout_count
            )  # Index within failure rollouts section
            global_rollout_idx = failure_idx % self._rollout_failure_total_steps
            episode_idx, relative_step_idx = self._rollout_failure_step_to_episode_map[global_rollout_idx]
            # Lazy load from HDF5 file
            episode_metadata = self.rollout_episode_metadata[episode_idx]
            episode_data = self._load_rollout_episode_data(episode_metadata)

        # If returning value function samples, randomly choose whether this sample is for
        # world model training or value function training
        is_world_model_sample = False
        is_value_function_sample = False
        if sample_type != "demo":
            if self.return_value_function_returns:
                p_world_model = 0.5
                if random.random() < p_world_model:
                    is_world_model_sample = True
                    is_value_function_sample = False
                else:
                    is_world_model_sample = False
                    is_value_function_sample = True
            else:
                is_world_model_sample = True
                is_value_function_sample = False

        # Calculate future frame index if needed
        future_frame_idx = relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if future_frame_idx > max_possible_idx:
            future_frame_idx = max_possible_idx

        # Handle JPEG decompression for rollout data if needed
        decompressed_images = {}
        frames_needed = {relative_step_idx, future_frame_idx}
        for frame_idx in frames_needed:
            if sample_type != "demo" and episode_data["is_jpeg"]:
                # Decompress JPEG frames
                decompressed_images[frame_idx] = decode_single_jpeg_frame(episode_data["images"][frame_idx])
                
            else:
                # Use images as-is
                decompressed_images[frame_idx] = episode_data["images"][frame_idx]

        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on

        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        first_input_image = np.expand_dims(np.zeros_like(decompressed_images[relative_step_idx]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1

        # Add proprio state if using proprio
        if self.use_proprio:
            proprio = episode_data["proprio"][relative_step_idx]
            image = decompressed_images[relative_step_idx]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(decompressed_images[relative_step_idx])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        # Add current third-person image
        if self.use_third_person_images:
            current_image = decompressed_images[relative_step_idx]
            current_image = duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add blank image for action chunk
        blank_image = np.zeros_like(decompressed_images[relative_step_idx])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        if self.predict_future_states:
            # Add future proprio
            if self.use_proprio:
                future_proprio = episode_data["proprio"][future_frame_idx]
                # Not using proprio image; proprio values will be injected into latent diffusion sequence later
                # For now just add blank image
                blank_image = np.zeros_like(decompressed_images[relative_step_idx])
                blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
                image_list.append(blank_image)
                future_proprio_latent_idx = current_sequence_idx
                current_sequence_idx += 1

            # Add future primary image
            if self.use_third_person_images:
                future_image = decompressed_images[future_frame_idx]
                future_image = duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image)
                image_list.append(future_image)
                future_image_latent_idx = current_sequence_idx
                current_sequence_idx += 1
        else: 
            future_proprio = np.zeros_like(episode_data["proprio"][relative_step_idx])
            future_proprio_latent_idx = -1
            future_image_latent_idx = -1
        # Add blank value image
        if self.return_value_function_returns:
            value_image = np.zeros_like(decompressed_images[relative_step_idx])
            value_image = duplicate_array(value_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(value_image)
            value_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # Calculate how many actions we can get from the current index
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Return the next action chunk as well
        # Calculate how many actions we can get from the current index
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # Get return for value function prediction
        if self.return_value_function_returns:
            return_timestep = future_frame_idx
            if episode_metadata is not None:
                value_function_return = episode_metadata["returns"][return_timestep]
            else:
                value_function_return = episode_data["returns"][return_timestep]
        else:
            value_function_return = float("-100")  # Just a placeholder

        # Calculate next future frame index if needed
        next_future_frame_idx = next_relative_step_idx + self.chunk_size
        max_possible_idx = episode_data["num_steps"] - 1
        if next_future_frame_idx > max_possible_idx:
            next_future_frame_idx = max_possible_idx

        # Return the next value function return as well
        if self.return_value_function_returns:
            return_timestep = next_future_frame_idx
            if episode_metadata is not None:
                next_value_function_return = episode_metadata["returns"][return_timestep]
            else:
                next_value_function_return = episode_data["returns"][return_timestep]
        else:
            next_value_function_return = float("-100")  # Just a placeholder

        sample_dict = {
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # Just copying what others have done in this codebase
            "fps": 16,  # Just set to some fixed value since we aren't generating videos anyway
            "padding_mask": torch.zeros(
                1, self.final_image_size, self.final_image_size
            ),  # Just copying what others have done in this codebase
            "image_size": self.final_image_size
            * torch.ones(
                4
            ),  # Just copying what others have done in this codebase; important because it shows up as model input
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][future_frame_idx])
            ),
            "__key__": idx,  # Unique sample identifier (required for callbacks)
            "rollout_data_mask": rollout_data_mask,
            "rollout_data_success_mask": rollout_data_success_mask,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 1 if is_value_function_sample else 0,
            "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx":  -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
        }

        return sample_dict


def create_augmentation_visualization(
    data_dir: str,
    t5_text_embeddings_path: str,
    fixed_idx: int = 100,
    num_augmentations: int = 50,
    output_dir: str = "./temp",
):
    """
    Create MP4 videos visualizing the distribution of augmentations for a fixed data point.

    Args:
        data_dir (str): Path to the dataset directory
        t5_text_embeddings_path (str): Path to T5 embeddings file
        fixed_idx (int): Index of the data point to apply augmentations to
        num_augmentations (int): Number of different augmentations to sample
        output_dir (str): Directory to save the visualization videos
    """
    print(f"\nCreating augmentation visualization with {num_augmentations} samples...")

    # Create a dataset instance with augmentations enabled
    aug_dataset = LIBERODataset(
        data_dir=data_dir,
        chunk_size=16,
        t5_text_embeddings_path=t5_text_embeddings_path,
        normalize_images=False,
        normalize_actions=True,
        use_image_aug=True,  # Enable augmentations
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        num_duplicates_per_image=1,
        use_stronger_image_aug=True,
    )

    # Collect different augmentations of the same data point
    augmented_samples = []

    print(f"Generating {num_augmentations} augmentations for data point {fixed_idx}...")
    for aug_idx in tqdm(range(num_augmentations)):
        sample = aug_dataset[fixed_idx]
        augmented_samples.append(sample)

    # Extract images from all augmented samples and organize them
    # Each sample has shape (C, T, H, W) where T is number of frames
    all_augmented_videos = []
    for sample in augmented_samples:
        video = sample["video"].permute(1, 2, 3, 0).numpy()  # (T, H, W, C)
        all_augmented_videos.append(video)

    # Stack all augmented videos: (num_augmentations, T, H, W, C)
    all_augmented_videos = np.stack(all_augmented_videos, axis=0)

    # Get dimensions
    num_augs, num_frames, height, width, channels = all_augmented_videos.shape
    print(f"Augmented video array shape: {all_augmented_videos.shape}")

    # Create video frames for each frame type
    frame_names = [
        "blank_input",
        "proprio",
        "wrist",
        "current_view",
        "future_proprio",
        "future_wrist",
        "future_view",
        "blank_action",
    ]
    if not aug_dataset.use_proprio:
        frame_names.remove("proprio")
    if not aug_dataset.use_wrist_images:
        frame_names.remove("wrist")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create MP4 videos for each frame type
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"

        # Skip blank frames for visualization
        if "blank" in frame_name:
            continue

        print(f"Creating video for {frame_name}...")

        # Extract frames for this frame type across all augmentations
        frames_for_video = []
        for aug_idx in range(num_augs):
            frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
            frames_for_video.append(frame)

        # Save as MP4 video
        video_path = os.path.join(output_dir, f"augmentation_visualization_{frame_name}.mp4")

        # Convert to uint8 if needed
        frames_array = np.stack(frames_for_video, axis=0)  # (num_augs, H, W, C)
        if frames_array.dtype != np.uint8:
            frames_array = frames_array.astype(np.uint8)

        # Save video with slower frame rate to better see the augmentations
        imageio.mimsave(video_path, frames_array, fps=5, macro_block_size=None)
        print(f"Saved augmentation visualization video: {video_path}")

    # Also create a combined video showing all frame types side by side
    print("Creating combined video with all frame types...")

    # Only use non-blank frames
    valid_frame_indices = []
    valid_frame_names = []
    for frame_idx in range(min(num_frames, len(frame_names))):
        frame_name = frame_names[frame_idx] if frame_idx < len(frame_names) else f"frame_{frame_idx}"
        if "blank" not in frame_name:
            valid_frame_indices.append(frame_idx)
            valid_frame_names.append(frame_name)

    if len(valid_frame_indices) > 0:
        combined_frames = []
        for aug_idx in range(num_augs):
            # Extract valid frames for this augmentation
            frames_to_combine = []
            for frame_idx in valid_frame_indices:
                frame = all_augmented_videos[aug_idx, frame_idx]  # (H, W, C)
                frames_to_combine.append(frame)

            # Concatenate frames horizontally
            combined_frame = np.concatenate(frames_to_combine, axis=1)  # (H, W*num_frames, C)
            combined_frames.append(combined_frame)

        # Save combined video
        combined_frames_array = np.stack(combined_frames, axis=0)  # (num_augs, H, W*num_frames, C)
        if combined_frames_array.dtype != np.uint8:
            combined_frames_array = combined_frames_array.astype(np.uint8)

        combined_video_path = os.path.join(output_dir, "augmentation_visualization_combined.mp4")
        imageio.mimsave(combined_video_path, combined_frames_array, fps=5, macro_block_size=None)
        print(f"Saved combined augmentation visualization video: {combined_video_path}")
        print(f"Combined video shows frames in order: {' | '.join(valid_frame_names)}")

    print("Augmentation visualization complete!")


if __name__ == "__main__":
    dataset = LIBERODataset(
        data_dir="users/user/libero_regen",  # Successful demos
        t5_text_embeddings_path="users/user/libero_regen/t5_embeddings.pkl",
        chunk_size=16,
        use_image_aug=True,
        use_wrist_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
        use_stronger_image_aug=True,
        rollout_data_dir="users/user/libero_regen_rollout_",  # All demo rollouts (successes + failures)
        demonstration_sampling_prob=0.5,
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.99,
    )

    # Fetch a sample
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    idx = 50
    sample = dataset[idx]
    print(f"\nImages shape, dtype: {sample['video'].shape, sample['video'].dtype}")
    print(f"Actions shape, dtype: {sample['actions'].shape, sample['actions'].dtype}")
    print(f"Actions:\n{sample['actions']}")
    print(f"T5 text embeddings shape, dtype: {sample['t5_text_embeddings'].shape, sample['t5_text_embeddings'].dtype}")
    print(f"T5 text embeddings:\n{sample['t5_text_embeddings']}")
    print(f"Unique commands: {dataset.unique_commands}")

    # Fetch more samples and save sample images
    os.makedirs("./temp", exist_ok=True)
    for _ in range(50):
        global_step_index = random.randint(0, len(dataset) - 1)
        sample = dataset[global_step_index]
        images = sample["video"].permute(1, 2, 3, 0).numpy()
        for i in range(images.shape[0]):
            img_np = images[i]
            image_path = f"./temp/video__global_step_index_{global_step_index}__is_rollout={sample['rollout_data_mask']}__global_rollout_idx={sample['global_rollout_idx']}__is_success={sample['rollout_data_success_mask']}__value_function_return={sample['value_function_return']:.4f}__frame_idx={i}.png"
            Image.fromarray(img_np).save(image_path)
            print(f"Saved image at path: {image_path}")
