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
PushT dataset with retrieval-augmented conditioning.

For each training step, retrieves the most similar state from the base demo pool
(using precomputed retrieval results) and conditions the model on:
  1. chunk_size future VIDEO FRAMES from that retrieved state
  2. chunk_size ACTIONS from that retrieved state (injected into a dedicated latent)
  3. Retrieved STATE (agent pos at start+chunk_size) injected into a dedicated latent

Image sequence layout (chunk_size=8, num_dup=4, predict_future_states=True):
  latent 0  : blank                (1 frame)    — sentinel
  latent 1-2: retrieved frame      (8 frames)   — condition (when has_ret_image)
  latent 3  : retrieved state      (4 blank)    — condition (injected, when has_ret_data)
  latent 4  : retrieved action     (4 blank)    — condition (injected, when has_ret_data)
  latent 5  : current frame        (4 frames)   — condition
  latent 6  : current state        (4 blank)    — condition (proprio injected)
  latent 7  : predicted action     (4 blank)    — generated
  latent 8  : predicted frame      (4 frames)   — generated
  latent 9  : predicted state      (4 blank)    — generated
  state_t = 10   chunk_duration = 37

Retrieval npz format (see ../tracking/retrieve_similar_state10.py):
    query_ids:  (N,)    str  "split/demo/t_last"  e.g. "tri_default_predict2_0/demo_0/7"
    match_ids:  (N, K)  str  "split/demo/t_last"  e.g. "base_0/demo_16/25"
    match_sims: (N, K)  float32  stored as -dist²  (higher = better)
"""

import json
import os
import pickle
import random
import re
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
)
from cosmos_policy.utils.utils import duplicate_array

np.set_printoptions(precision=3, linewidth=np.inf)


class PushTRetDataset(Dataset):
    """
    PushT dataset augmented with retrieved base-demo future frames, actions, and state.

    See module docstring for the full latent layout.
    """

    # Stride used when building retrieval subframes (must match retrieve_similar_flow.py)
    RETRIEVAL_STRIDE: int = 4

    def __init__(
        self,
        data_dir: str,
        retrieval_npz_path: "str | list[str]" = "",
        chunk_size: int = 8,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_wrist_images: bool = False,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        task_split: list = ["tri_default_p"],
        retrieval_source_splits: list = ["base"],  # source demo splits for retrieval (e.g. ["base", "goal_flipped"])
        return_value_function_returns: bool = True,
        predict_future_states: bool = True,
        gamma: float = 0.99,
        max_num_episodes: int = -1,
        retrieval_top_k_choice: int =3,  # 1 = best match; >1 = random among top k
        retrieval_dropout_prob: float = 0.0,  # prob of dropping ALL retrieval (zeros + mask=0)
        retrieval_image_only_dropout_prob: float = 0.0,  # prob of dropping ret images only (keep ret actions)
        ret_context_multiplier: int = 1,  # 1 = original window, 2 = 2x window
        ret_image_subsample: int = 1,  # 1 = all frames, 2 = ::2 subsampling on ret images
        ret_single_frame: bool = False,  # Use only first ret frame repeated (1 latent instead of 2)
        force_zero_ret_image: bool = False,  # Ablation: always zero out ret frames + state (keep ret action only)
        force_zero_ret_state: bool = False,  # Ablation: always zero out ret state (keep ret image + action)
        ret_action_as_target_prob: float = 0.0,  # prob of using retrieved action as GT target
        use_residual_actions: bool = False,  # predict delta = action - retrieved_action (residual learning)
        # ── Top-K episode filtering (by precomputed per-episode action error) ──
        # Either a single path (applied to all task_splits) or a list of paths
        # matching `task_split` 1-to-1 (one ranking JSON per task). The list
        # form is produced by compute_episode_action_error.py, which writes one
        # JSON per task_split (filename suffix `_<task_split>`). Top-K is
        # applied independently within each task's allowlist so every task
        # contributes at most K episodes.
        episode_allowlist_path: "str | list[str]" = "",
        episode_allowlist_top_k: int = -1,  # -1 = no top-K cap (keep all allowlisted episodes)
        # ── Auxiliary cotrain-RAG branch ─────────────────────────────────────
        # At __getitem__, with `aux_sampling_prob` we redirect to a sample
        # drawn from `extra_task_splits` episodes and condition the model on
        # that sample's OWN future (frames, actions, proprio) instead of an
        # NPZ-retrieved trajectory. No retrieval NPZ is required for aux
        # splits — self-future is used directly.
        extra_task_splits: "list[tuple[str, int]] | None" = None,
        aux_sampling_prob: float = 0.0,
        rollout_data_dir: str = "",  # unused; accepted for config-merge compatibility
        **kwargs,  # absorbs any extra kwargs injected by LazyConfig deep-merge
    ):
        assert use_wrist_images or use_third_person_images, (
            "Must use at least one of wrist or third-person images."
        )
        assert chunk_size % num_duplicates_per_image == 0, (
            f"chunk_size ({chunk_size}) must be divisible by "
            f"num_duplicates_per_image ({num_duplicates_per_image})"
        )

        self.data_dir = data_dir
        # Normalize retrieval_npz_path to a list
        if isinstance(retrieval_npz_path, str):
            self.retrieval_npz_paths = [retrieval_npz_path] if retrieval_npz_path else []
        else:
            self.retrieval_npz_paths = list(retrieval_npz_path)
        self.task_split = task_split
        self.retrieval_source_splits = retrieval_source_splits
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
        self.return_value_function_returns = return_value_function_returns
        self.predict_future_states = predict_future_states
        self.gamma = gamma
        self.retrieval_top_k_choice = retrieval_top_k_choice
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.retrieval_image_only_dropout_prob = retrieval_image_only_dropout_prob
        self.ret_context_multiplier = ret_context_multiplier
        self.ret_image_subsample = ret_image_subsample
        self.ret_single_frame = ret_single_frame
        self.force_zero_ret_image = force_zero_ret_image
        self.force_zero_ret_state = force_zero_ret_state
        self.ret_action_as_target_prob = ret_action_as_target_prob
        self.use_residual_actions = use_residual_actions
        self.episode_allowlist_path = episode_allowlist_path
        self.episode_allowlist_top_k = episode_allowlist_top_k
        self.extra_task_splits = extra_task_splits
        self.aux_sampling_prob = float(aux_sampling_prob)

        # Build per-split (suite, demo_key) allowlist from ranking JSONs.
        # `None` for a split means "no filtering for that split".
        if isinstance(episode_allowlist_path, str):
            allowlist_paths: list[str] = (
                [episode_allowlist_path] * len(task_split) if episode_allowlist_path else []
            )
        else:
            allowlist_paths = list(episode_allowlist_path)

        if allowlist_paths:
            assert len(allowlist_paths) == len(task_split), (
                f"episode_allowlist_path length ({len(allowlist_paths)}) must "
                f"match task_split length ({len(task_split)})"
            )

        split_to_allowed: dict[str, "set | None"] = {s: None for s in task_split}
        for split, path in zip(task_split, allowlist_paths):
            if not path:
                continue
            with open(path, "r") as f:
                ranking = json.load(f)
            ranked = ranking["episodes"]
            if episode_allowlist_top_k > 0:
                ranked = ranked[:episode_allowlist_top_k]
            split_to_allowed[split] = {(r["suite"], r["demo_key"]) for r in ranked}
            print(
                f"[PushTRetDataset] split={split}: keeping {len(split_to_allowed[split])} episodes "
                f"(top-{episode_allowlist_top_k if episode_allowlist_top_k > 0 else 'ALL'} "
                f"from {path})"
            )
        self._split_to_allowed = split_to_allowed

        # ── 1. Load query demos (e.g. tri_default, tri_goal) ─────────────────
        self.data: dict = {}
        self.num_episodes: int = 0
        self.num_steps: int = 0
        self.unique_commands: set = set()
        self._suite_to_step_indices: dict = {}

        all_episode_refs = []
        for split in task_split:
            split_hdf5_files = get_hdf5_files(data_dir, task_split=[split])
            if os.environ.get("DEBUGGING", "False").lower() == "true":
                split_hdf5_files = split_hdf5_files[:1]
            split_allowed = split_to_allowed.get(split)
            split_refs = []
            for file in split_hdf5_files:
                with h5py.File(file, "r") as f:
                    suite = os.path.relpath(file, self.data_dir).split(os.sep)[0]
                    sorted_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                    for demo_key in sorted_keys:
                        if split_allowed is not None and (suite, demo_key) not in split_allowed:
                            continue
                        split_refs.append((file, demo_key, suite))
            if max_num_episodes > 0 and max_num_episodes < len(split_refs):
                split_refs = split_refs[:max_num_episodes]
            all_episode_refs.extend(split_refs)

        file_to_keys: dict = defaultdict(list)
        for file, demo_key, suite in all_episode_refs:
            file_to_keys[file].append((demo_key, suite))

        for file, key_suite_pairs in tqdm(file_to_keys.items(), desc="Loading query demos"):
            with h5py.File(file, "r") as f:
                for demo_key, suite in key_suite_pairs:
                    obs_group = f[f"data/{demo_key}/obs"]
                    if "images" not in obs_group:
                        raise KeyError(f"'images' not found in {file} / {demo_key}")
                    images = decode_jpeg_bytes_dataset(obs_group["images"])
                    actions = f[f"data/{demo_key}/actions"][:].astype(np.float32)
                    # Use first 2 dims of state as proprio (agent xy)
                    proprio = f[f"data/{demo_key}/obs/states"][:][:, :2].astype(np.float32)
                    command = "push t shaped block to the location"
                    self.unique_commands.add(command)
                    num_steps = len(images)
                    returns = (
                        compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                        if self.return_value_function_returns
                        else None
                    )
                    self.data[self.num_episodes] = dict(
                        images=images,
                        proprio=proprio,
                        actions=actions,
                        command=command,
                        num_steps=num_steps,
                        suite=suite,
                        demo_key=demo_key,
                        returns=returns.copy() if returns is not None else None,
                    )
                    self.num_episodes += 1
                    self.num_steps += num_steps

        # ── 1b. Load extra_task_splits demos for aux branch (self-future) ───
        # Strict suite match: '^{prefix}_\d+$' (so 'rot15' does not pick up
        # 'rot150'). max_eps<=0 means "all matched episodes for this prefix".
        self._aux_split_counts: dict = {}
        if self.extra_task_splits:
            print(f"Loading extra_task_splits for aux branch: {self.extra_task_splits}")
            aux_file_to_keys: dict = defaultdict(list)
            for prefix, max_eps in self.extra_task_splits:
                suite_pattern = re.compile(rf"^{re.escape(prefix)}_\d+$")
                suite_names = sorted(os.listdir(data_dir))
                matched_suites = [
                    s for s in suite_names
                    if suite_pattern.match(s) and os.path.isdir(os.path.join(data_dir, s))
                ]
                prefix_refs: list = []
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
                                prefix_refs.append((fp, dk, suite))
                if max_eps is not None and max_eps > 0 and len(prefix_refs) > max_eps:
                    prefix_refs = prefix_refs[:max_eps]
                for fp, dk, suite in prefix_refs:
                    aux_file_to_keys[fp].append((dk, suite))
                self._aux_split_counts[prefix] = len(prefix_refs)
                print(
                    f"[PushTRetDataset] aux split '{prefix}': {len(prefix_refs)} episodes "
                    f"from {len(matched_suites)} suite folder(s)"
                )

            for file, key_suite_pairs in tqdm(aux_file_to_keys.items(), desc="Loading aux demos"):
                with h5py.File(file, "r") as f:
                    for demo_key, suite in key_suite_pairs:
                        obs_group = f[f"data/{demo_key}/obs"]
                        if "images" not in obs_group:
                            raise KeyError(f"'images' not found in {file} / {demo_key}")
                        images = decode_jpeg_bytes_dataset(obs_group["images"])
                        actions = f[f"data/{demo_key}/actions"][:].astype(np.float32)
                        proprio = f[f"data/{demo_key}/obs/states"][:][:, :2].astype(np.float32)
                        command = "push t shaped block to the location"
                        self.unique_commands.add(command)
                        num_steps = len(images)
                        returns = (
                            compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
                            if self.return_value_function_returns
                            else None
                        )
                        self.data[self.num_episodes] = dict(
                            images=images,
                            proprio=proprio,
                            actions=actions,
                            command=command,
                            num_steps=num_steps,
                            suite=suite,
                            demo_key=demo_key,
                            returns=returns.copy() if returns is not None else None,
                            is_aux=True,
                        )
                        self.num_episodes += 1
                        self.num_steps += num_steps

        self._build_step_index_mapping()

        # Partition step indices so the 70% path samples only tri data and the
        # 30% aux path samples only extra_task_splits data. When no aux data
        # was loaded, _tri_step_indices covers everything (identity to old
        # behavior).
        self._tri_step_indices: list = []
        self._aux_step_indices: list = []
        for global_idx, (ep_idx, _) in self._step_to_episode_map.items():
            if self.data[ep_idx].get("is_aux", False):
                self._aux_step_indices.append(global_idx)
            else:
                self._tri_step_indices.append(global_idx)
        # Aux is a stochastic augmentation; epoch length should track the
        # primary (tri) pool so training iterates it once per epoch.
        if self._aux_step_indices:
            self.num_steps = len(self._tri_step_indices)
            print(
                f"[PushTRetDataset] tri steps: {len(self._tri_step_indices)}, "
                f"aux steps: {len(self._aux_step_indices)}, "
                f"aux_sampling_prob: {self.aux_sampling_prob}"
            )

        # ── 2. Load retrieval source data (e.g. base, goal_flipped) ──────────
        print(f"Loading retrieval source data for splits: {retrieval_source_splits} ...")
        # {(source_suite, demo_key): {"images": ndarray, "actions": ndarray, "proprio": ndarray}}
        self.base_data: dict = {}
        source_hdf5_files = get_hdf5_files(data_dir, task_split=retrieval_source_splits)
        for file in tqdm(source_hdf5_files, desc="Loading retrieval source images+actions+proprio"):
            source_suite = os.path.relpath(file, self.data_dir).split(os.sep)[0]
            with h5py.File(file, "r") as f:
                sorted_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                for demo_key in sorted_keys:
                    imgs = decode_jpeg_bytes_dataset(f[f"data/{demo_key}/obs/images"])
                    acts = f[f"data/{demo_key}/actions"][:].astype(np.float32)
                    prop = f[f"data/{demo_key}/obs/states"][:][:, :2].astype(np.float32)
                    self.base_data[(source_suite, demo_key)] = {"images": imgs, "actions": acts, "proprio": prop}
        print(f"  loaded {len(self.base_data)} retrieval source demos")

        # ── 3. Load retrieval lookup (merge all npz files) ─────────────────
        self._retrieval_lookup: dict = {}
        for npz_path in self.retrieval_npz_paths:
            print(f"Loading retrieval results from {npz_path} ...")
            self._build_retrieval_lookup(npz_path)

        # ── 4. Normalization & statistics ─────────────────────────────────────
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)

        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        # ── 4b. Compute delta statistics BEFORE normalizing actions ──────────
        self.delta_stats = None
        if self.use_residual_actions:
            self.delta_stats = self._load_or_compute_delta_statistics()

        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        self._calculate_epoch_structure()
        print(f"Finished loading {self.num_episodes} episodes / {self.num_steps} steps.")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_or_compute_delta_statistics(self) -> dict:
        """Compute min/max of raw deltas (gt_action - ret_action) across all
        training samples using top-1 retrieval.  Cached to disk as
        ``delta_dataset_statistics.json`` next to the regular stats file."""
        delta_stats_path = os.path.join(self.data_dir, "delta_dataset_statistics.json")
        if os.path.exists(delta_stats_path):
            with open(delta_stats_path, "r") as f:
                js = json.load(f)
            print(f"Loaded delta statistics from: {delta_stats_path}")
            return {k: np.array(v) for k, v in js.items()}

        print("Computing delta (gt - ret) statistics over the dataset (top-1 retrieval) ...")
        all_deltas = []
        for ep_idx, ep_data in self.data.items():
            suite = ep_data["suite"]
            demo_key = ep_data["demo_key"]
            raw_actions = ep_data["actions"]  # still raw at this point
            for step_idx in range(ep_data["num_steps"]):
                # Look up retrieval match (same search logic as _get_retrieved_data)
                matches = None
                for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
                    key = (suite, demo_key, step_idx + offset)
                    if key in self._retrieval_lookup:
                        matches = self._retrieval_lookup[key]
                        break
                if matches is None:
                    continue
                m_suite, m_demo, m_start, _ = matches[0]  # top-1
                base_entry = self.base_data.get((m_suite, m_demo))
                if base_entry is None:
                    continue
                # Raw action chunk (gt)
                action_chunk = get_action_chunk_with_padding(
                    raw_actions, step_idx, self.chunk_size, ep_data["num_steps"],
                )
                # Raw retrieved actions
                base_acts = base_entry["actions"]
                T = len(base_acts)
                ret_actions = np.stack(
                    [base_acts[max(0, min(m_start + i, T - 1))] for i in range(self.chunk_size)]
                )
                all_deltas.append(action_chunk - ret_actions)

        all_deltas = np.concatenate(all_deltas, axis=0)  # (N*chunk, action_dim)
        delta_stats = {
            "delta_actions_min": all_deltas.min(axis=0),
            "delta_actions_max": all_deltas.max(axis=0),
        }
        # Save to disk
        js = {k: v.tolist() for k, v in delta_stats.items()}
        with open(delta_stats_path, "w") as f:
            json.dump(js, f, indent=4)
        print(f"Delta statistics saved to: {delta_stats_path}")
        print(f"  delta_min={delta_stats['delta_actions_min']}, delta_max={delta_stats['delta_actions_max']}")
        return delta_stats

    def _build_step_index_mapping(self):
        self._step_to_episode_map: dict = {}
        self._total_steps: int = 0
        self._suite_to_step_indices = defaultdict(list)
        for ep_idx, ep_data in self.data.items():
            for i in range(ep_data["num_steps"]):
                self._step_to_episode_map[self._total_steps] = (ep_idx, i)
                self._suite_to_step_indices[ep_data["suite"]].append(self._total_steps)
                self._total_steps += 1
        self._suites = list(self._suite_to_step_indices.keys())

    def _calculate_epoch_structure(self):
        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=1.0,
            success_rollout_sampling_prob=0.0,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = 0
        self.adjusted_failure_rollout_count = 0
        self.epoch_length = result["epoch_length"]

    def _build_retrieval_lookup(self, npz_path: str):
        """
        Merge entries from npz into self._retrieval_lookup:
            (suite, demo_key, subframe_start) →
                [(source_suite, source_demo, source_start, sim), ...]  ordered best-first
        """
        data = np.load(npz_path, allow_pickle=True)
        query_ids = data["query_ids"]    # (N,)   str
        match_ids = data["match_ids"]    # (N, K) str
        match_sims = data["match_sims"]  # (N, K) float32  (higher = better)

        count = 0
        for qi, qid in enumerate(query_ids):
            parts = qid.split("/")
            suite, demo_key, start = parts[0], parts[1], int(parts[2])
            matches = []
            for ki in range(match_ids.shape[1]):
                mid = str(match_ids[qi, ki])
                mparts = mid.split("/")
                m_suite, m_demo, m_start = mparts[0], mparts[1], int(mparts[2])
                matches.append((m_suite, m_demo, m_start, float(match_sims[qi, ki])))
            self._retrieval_lookup[(suite, demo_key, start)] = matches
            count += 1

        print(f"  added {count:,} entries (total: {len(self._retrieval_lookup):,})")

    @property
    def n_ret_frames(self) -> int:
        """Number of retrieved image frames after context expansion and subsampling."""
        return self.chunk_size * self.ret_context_multiplier // self.ret_image_subsample

    @property
    def n_ret_actions(self) -> int:
        """Number of retrieved action steps (full context window, no subsampling)."""
        return self.chunk_size * self.ret_context_multiplier

    def _get_retrieved_data(
        self, suite: str, demo_key: str, step_idx: int
    ) -> tuple:
        """
        Look up the retrieval result for (suite, demo_key, step_idx) and return
        retrieved future frames, actions, and state (proprio at start+chunk_size).

        Returns:
            frames:  ndarray (n_ret_frames, H, W, 3) uint8
            actions: ndarray (n_ret_actions, action_dim) float32, normalized to [-1, 1]
            ret_proprio: ndarray (proprio_dim,) float32, normalized if normalize_proprio
        Falls back to zeros if no retrieval entry is found.
        """
        # Search nearest retrieval key (t_last-based indices, spaced by stride=2)
        matches = None
        for offset in [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8]:
            key = (suite, demo_key, step_idx + offset)
            if key in self._retrieval_lookup:
                matches = self._retrieval_lookup[key]
                break

        action_dim = self.data[0]["actions"].shape[-1]
        proprio_dim = self.data[0]["proprio"].shape[-1]
        raw_img_size = self.data[0]["images"][0].shape[0]  # actual H before preprocess resize
        zero_frames = np.zeros((self.n_ret_frames, raw_img_size, raw_img_size, 3), dtype=np.uint8)
        zero_actions = np.zeros((self.n_ret_actions, action_dim), dtype=np.float32)
        zero_proprio = np.zeros(proprio_dim, dtype=np.float32)

        if matches is None:
            return zero_frames, zero_actions, zero_proprio, zero_actions.copy()

        m_suite, m_demo, m_start, _ = random.choice(matches[: self.retrieval_top_k_choice])

        base_entry = self.base_data.get((m_suite, m_demo))
        if base_entry is None:
            return zero_frames, zero_actions, zero_proprio, zero_actions.copy()

        base_imgs = base_entry["images"]
        base_acts = base_entry["actions"]
        base_proprio = base_entry["proprio"]
        T = len(base_imgs)

        # Context window: when multiplier>1, shift start back by (mult-1)*chunk/2
        ctx_mult = self.ret_context_multiplier
        ret_chunk = self.chunk_size * ctx_mult
        window_start = m_start - (ctx_mult - 1) * self.chunk_size // 2

        # Collect all frames and actions from the extended window (clip to valid range)
        all_frames = [base_imgs[max(0, min(window_start + i, T - 1))] for i in range(ret_chunk)]
        all_acts_raw = np.stack(
            [base_acts[max(0, min(window_start + i, T - 1))] for i in range(ret_chunk)], axis=0
        )

        # Subsample images (e.g. ::2)
        frames = all_frames[:: self.ret_image_subsample]

        # Normalize retrieved actions with the same stats as query actions
        all_acts = all_acts_raw.copy()
        if self.normalize_actions:
            a_min = self.dataset_stats["actions_min"]
            a_max = self.dataset_stats["actions_max"]
            all_acts = 2.0 * ((all_acts - a_min) / (a_max - a_min + 1e-8)) - 1.0

        # Retrieved state: proprio at the matched frame (m_start = t_last)
        ret_proprio_idx = min(m_start, T - 1)
        ret_proprio = base_proprio[ret_proprio_idx].copy()
        if self.normalize_proprio:
            p_min = self.dataset_stats["proprio_min"]
            p_max = self.dataset_stats["proprio_max"]
            ret_proprio = 2.0 * ((ret_proprio - p_min) / (p_max - p_min + 1e-8)) - 1.0

        return (
            np.stack(frames, axis=0),
            all_acts.astype(np.float32),
            ret_proprio.astype(np.float32),
            all_acts_raw.astype(np.float32),
        )

    def _get_self_future_data(self, ep_idx: int, step_idx: int) -> tuple:
        """Aux-branch analog of _get_retrieved_data: read conditioning from the
        *same* trajectory's future instead of an NPZ-matched other trajectory.

        ep_data["actions"] and ep_data["proprio"] are already normalized by
        rescale_data at load time, matching the normalization _get_retrieved_data
        applies inline. The raw-actions return slot is zeroed — aux configs run
        with use_residual_actions=False and never consume it.
        """
        ep_data = self.data[ep_idx]
        T = ep_data["num_steps"]
        own_imgs = ep_data["images"]
        own_acts = ep_data["actions"]
        own_proprio = ep_data["proprio"]

        ctx_mult = self.ret_context_multiplier
        ret_chunk = self.chunk_size * ctx_mult
        window_start = step_idx - (ctx_mult - 1) * self.chunk_size // 2

        all_frames = [own_imgs[max(0, min(window_start + i, T - 1))] for i in range(ret_chunk)]
        frames = all_frames[:: self.ret_image_subsample]
        ret_frames = np.stack(frames, axis=0)

        ret_actions = np.stack(
            [own_acts[max(0, min(window_start + i, T - 1))] for i in range(ret_chunk)],
            axis=0,
        ).astype(np.float32)

        proprio_idx = min(max(0, step_idx), T - 1)
        ret_proprio = own_proprio[proprio_idx].astype(np.float32).copy()

        action_dim = own_acts.shape[-1]
        zero_raw = np.zeros((self.n_ret_actions, action_dim), dtype=np.float32)

        return ret_frames, ret_actions, ret_proprio, zero_raw

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Aux branch: with aux_sampling_prob, redirect to a step from
        # extra_task_splits and feed self-future as conditioning. Backward
        # compatible: when aux_sampling_prob=0 or no aux data, this reduces to
        # `idx % self.num_steps` over the tri pool.
        use_aux = (
            self.aux_sampling_prob > 0.0
            and len(self._aux_step_indices) > 0
            and random.random() < self.aux_sampling_prob
        )
        if use_aux:
            global_step_idx = random.choice(self._aux_step_indices)
        else:
            tri_n = len(self._tri_step_indices)
            global_step_idx = self._tri_step_indices[idx % tri_n] if tri_n else idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]
        is_aux_sample = bool(episode_data.get("is_aux", False))

        # Current and future frame indices
        future_frame_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        current_image = episode_data["images"][relative_step_idx]
        future_image = episode_data["images"][future_frame_idx]

        # ── Retrieve data ─────────────────────────────────────────────────────
        suite = episode_data["suite"]
        demo_key = episode_data["demo_key"]
        if is_aux_sample:
            ret_frames, ret_actions, ret_proprio, ret_actions_raw = self._get_self_future_data(
                episode_idx, relative_step_idx
            )
        else:
            ret_frames, ret_actions, ret_proprio, ret_actions_raw = self._get_retrieved_data(
                suite, demo_key, relative_step_idx
            )
        # normalize ret_actions and ret_proprio
        # Retrieval dropout
        has_ret_data = 1
        has_ret_image = 1
        has_current_image = 1
        if self.retrieval_dropout_prob > 0 and random.random() < self.retrieval_dropout_prob:
            ret_frames = np.zeros_like(ret_frames)
            ret_actions = np.zeros_like(ret_actions)
            ret_proprio = np.zeros_like(ret_proprio)
            has_ret_data = 0
            has_ret_image = 0
        elif self.retrieval_image_only_dropout_prob > 0 and random.random() < self.retrieval_image_only_dropout_prob:
            has_current_image = 0
            current_image = np.zeros_like(current_image)

        # ── Build image sequence ───────────────────────────────────────────────
        # Layout: blank | ret_frame | ret_state | ret_action | cur_frame | cur_state | pred_action | pred_frame | pred_state
        image_list = []
        current_sequence_idx = 0
        blank_dup = duplicate_array(np.zeros_like(current_image), total_num_copies=self.num_duplicates_per_image)

        # Default sentinel values
        proprio = np.zeros_like(episode_data["proprio"][relative_step_idx])
        future_proprio = np.zeros_like(episode_data["proprio"][relative_step_idx])
        current_proprio_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_image_latent_idx = -1
        retrieved_state_latent_idx = -1

        # [blank input] — 1 frame (latent 0)
        image_list.append(np.expand_dims(np.zeros_like(current_image), axis=0))
        current_sequence_idx += 1

        # [retrieved frame] — latents 1-2 (chunk_size frames), or 1 latent if ret_single_frame
        if self.force_zero_ret_image:
            retrieved_video_end_latent_idx = -1
            retrieved_video_start_latent_idx = -1
            pass
        elif self.ret_single_frame:
            # Use only first retrieved frame, repeated like observation
            image_list.append(duplicate_array(ret_frames[0], total_num_copies=self.num_duplicates_per_image))
            retrieved_video_start_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            retrieved_video_end_latent_idx = current_sequence_idx
        else:
            image_list.append(ret_frames)  # (n_ret_frames, H, W, 3)
            retrieved_video_start_latent_idx = current_sequence_idx
            num_ret_latents = self.n_ret_frames // self.num_duplicates_per_image
            current_sequence_idx += num_ret_latents
            retrieved_video_end_latent_idx = current_sequence_idx

        # [retrieved state] — latent 3 (NEW)
        if self.force_zero_ret_state or self.force_zero_ret_image:
            retrieved_state_latent_idx = -1
        else:
            image_list.append(blank_dup.copy())
            retrieved_state_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # [retrieved action] — latent 4
        image_list.append(blank_dup.copy())
        retrieved_action_latent_idx = current_sequence_idx
        current_sequence_idx += 1


        # [current frame] — latent 5
        if self.use_third_person_images:
            cur_pixels = current_image if has_current_image else np.zeros_like(current_image)
            image_list.append(duplicate_array(cur_pixels, total_num_copies=self.num_duplicates_per_image))
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        
        # [current state/proprio] — latent 6
        if self.use_proprio:
            proprio = episode_data["proprio"][relative_step_idx] if has_current_image else np.zeros_like(episode_data["proprio"][relative_step_idx])
            image_list.append(blank_dup.copy())
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1


        # [predicted action] — latent 7
        image_list.append(blank_dup.copy())
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1

        # [predicted frame + predicted state] — latents 8-9 (if predict_future_states)
        if self.predict_future_states:
            if self.use_third_person_images:
                fut_img = duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image)
                image_list.append(fut_img)
                future_image_latent_idx = current_sequence_idx
                current_sequence_idx += 1
            if self.use_proprio:
                future_proprio = episode_data["proprio"][future_frame_idx]
                image_list.append(blank_dup.copy())
                future_proprio_latent_idx = current_sequence_idx
                current_sequence_idx += 1

        # ── Preprocess all frames together ────────────────────────────────────
        images = np.concatenate(image_list, axis=0)  # (total_frames, H, W, 3)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # ── Action chunk ──────────────────────────────────────────────────────
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )
        next_relative_step_idx = min(relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # ── Optionally use retrieved action as GT target ──────────────────────
        if (
            has_ret_data
            and self.ret_action_as_target_prob > 0
            and random.random() < self.ret_action_as_target_prob
        ):
            action_chunk = ret_actions[: self.chunk_size].copy()

        # ── Residual learning: predict delta in RAW space, normalize with delta stats ──
        if self.use_residual_actions and has_ret_data:
            # action_chunk is normalized → unnormalize to raw
            a_min = self.dataset_stats["actions_min"]
            a_max = self.dataset_stats["actions_max"]
            raw_action_chunk = 0.5 * (action_chunk + 1.0) * (a_max - a_min) + a_min
            # Compute raw delta
            raw_delta = raw_action_chunk - ret_actions_raw[: self.chunk_size]
            # Normalize delta with delta-specific stats
            d_min = self.delta_stats["delta_actions_min"]
            d_max = self.delta_stats["delta_actions_max"]
            action_chunk = 2.0 * ((raw_delta - d_min) / (d_max - d_min + 1e-8)) - 1.0

        # ── Value function returns ────────────────────────────────────────────
        if self.return_value_function_returns:
            value_function_return = episode_data["returns"][future_frame_idx]
            next_future_frame_idx = min(next_relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1)
            next_value_function_return = episode_data["returns"][next_future_frame_idx]
        else:
            value_function_return = float("-100")
            next_value_function_return = float("-100")

        return {
            "video": images,
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[episode_data["command"]]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio if self.use_proprio else np.zeros_like(episode_data["proprio"][relative_step_idx]),
            "future_proprio": (
                future_proprio
                if self.use_proprio
                else np.zeros_like(episode_data["proprio"][relative_step_idx])
            ),
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            # Retrieval-specific
            "retrieved_video_start_latent_idx": retrieved_video_start_latent_idx,
            "retrieved_video_end_latent_idx": retrieved_video_end_latent_idx,
            "retrieved_action_latent_idx": retrieved_action_latent_idx,
            "retrieved_actions": ret_actions,  # (n_ret_actions, action_dim) float32, normalized
            "retrieved_proprio": ret_proprio,  # (proprio_dim,) float32, normalized
            "retrieved_state_latent_idx": retrieved_state_latent_idx,
            "has_ret_data": has_ret_data,
            "has_ret_image": has_ret_image,
            "has_current_image": has_current_image,
        }
