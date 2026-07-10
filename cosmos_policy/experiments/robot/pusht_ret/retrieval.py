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
retrieval.py

Pool loading and 3-stage retrieval for PushT retrieval-augmented policy.
Uses 10-dim state features from GT states in the dataset (obs/states).

GT states format: [agent_x, agent_y, block_x, block_y, block_angle]
  - Positions in sim coordinates (0-512)
  - Angle in radians

Three-stage retrieval:
  Stage 1: filter to N_DEMO_FILTER demos by initial block position L2.
  Stage 2: hard position gate on end-of-window block position.
  Stage 3: rank by 10-dim state feature L2.

Feature (10 dims):
  block position (x, y)        : end-of-window x W_BLOCK_POS      2 dims
  agent position (x, y)        : end-of-window x W_AGENT_POS      2 dims
    --block_rel 사용 시: block 좌표계 기준 상대 좌표 (translate + rotate by -yaw)
  block angle (sin2t, cos2t)   : end-of-window x W_YAW            2 dims
  block velocity (dx, dy)      : mean finite diff x W_BLOCK_VEL   2 dims
  agent velocity (dx, dy)      : mean finite diff x W_AGENT_VEL   2 dims

All positions are normalized by SIM_SCALE (512) to [0, 1].
"""

import os
import re
from collections import defaultdict

import h5py
import numpy as np

from cosmos_policy.datasets.dataset_utils import decode_jpeg_bytes_dataset, get_hdf5_files


# ── Pool resolution helpers ──────────────────────────────────────────────────
# Pool dir name patterns scanned by the auto-resolver.
#   plain     : rot{ANGLE}_{i}                e.g. rot60_0
#   flipcolor : rot{ANGLE}_flipcolor_{i}      e.g. rot60_flipcolor_0
_POOL_PATTERNS: dict[str, list[re.Pattern]] = {
    "plain":     [re.compile(r"^rot(-?\d+)_(\d+)$")],
    "flipcolor": [re.compile(r"^rot(-?\d+)_flipcolor_(\d+)$")],
    "both":      [re.compile(r"^rot(-?\d+)_(\d+)$"),
                  re.compile(r"^rot(-?\d+)_flipcolor_(\d+)$")],
}


def _scan_rotation_pools(data_dir: str, pattern: str = "plain") -> dict[int, list[str]]:
    """Scan ``data_dir`` for rotation pool subdirectories grouped by angle.

    Returns ``{angle_deg: [dir_name, ...]}`` with dir names sorted.
    """
    if pattern not in _POOL_PATTERNS:
        raise ValueError(
            f"Unknown pool_pattern={pattern!r}; choose from {list(_POOL_PATTERNS)}"
        )
    regexes = _POOL_PATTERNS[pattern]

    pools: dict[int, list[str]] = defaultdict(list)
    if not os.path.isdir(data_dir):
        return pools
    for name in sorted(os.listdir(data_dir)):
        for r in regexes:
            m = r.match(name)
            if m:
                pools[int(m.group(1))].append(name)
                break
    return pools


def _angular_distance_deg(a: float, b: float) -> float:
    """Circular angular distance in degrees, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


_VC_ANGLE_RE = re.compile(r"^tri_rot(-?\d+)$")

# Non-rot-named pools that still represent a fixed goal angle in the dataset.
# Per dataset convention: base goal is at +45°, goal_flipped at -45°.
_NON_ROT_POOL_ANGLES: dict[str, int] = {
    "base":         45,
    "goal_flipped": -45,
}


def _vc_target_angle(visual_config: str) -> int | None:
    """Target rotation angle encoded in a ``tri_rot{N}`` visual_config name."""
    m = _VC_ANGLE_RE.match(visual_config)
    return int(m.group(1)) if m else None


def _group_explicit_rot_pools(
    explicit: list[str], pool_pattern: str
) -> dict[int, list[str]]:
    """Group dirs in ``explicit`` by goal-angle (degrees).

    Recognizes ``rot{N}_{i}`` directly via ``pool_pattern`` regexes, and
    treats ``base_*`` / ``goal_flipped_*`` as fixed-angle pools per
    ``_NON_ROT_POOL_ANGLES``.
    """
    regexes = _POOL_PATTERNS.get(pool_pattern, _POOL_PATTERNS["plain"])
    pools: dict[int, list[str]] = defaultdict(list)
    for name in explicit:
        matched = False
        for r in regexes:
            m = r.match(name)
            if m:
                pools[int(m.group(1))].append(name)
                matched = True
                break
        if matched:
            continue
        for prefix, angle in _NON_ROT_POOL_ANGLES.items():
            if name == prefix or name.startswith(f"{prefix}_"):
                pools[angle].append(name)
                break
    return pools


def resolve_retrieval_split(
    data_dir: str,
    visual_config: str,
    goal_angle: float | None,
    explicit_split: str | None,
    k: int,
    fallback_map: dict[str, list[str]],
    pool_pattern: str = "plain",
) -> list[str]:
    """Resolve which pool dirs to feed into ``PushTRetrieval(split=...)``.

    Priority:
      1. ``explicit_split`` — comma-separated dir names from bash (cumulative
         across stages). Filter that pool down to the natural mapping for
         ``visual_config``:
           - tri_default      → base_*           (always)
           - tri_goal_flipped → goal_flipped_*   (always)
           - tri_rot{N}       → rot{N}_* if present in explicit (exact match),
                                else the K nearest angle pools in explicit
                                (rot/base/flipped) by circular angular distance.
         If no natural mapping applies and no angle pools are present in
         explicit, return the explicit list as-is.
      2. ``goal_angle`` — pick the K nearest available rotation pools by
         circular angular distance. Exact-match (≤0.5°) short-circuits to K=1.
      3. ``fallback_map[visual_config]`` — original hard-coded mapping.
    """
    if explicit_split:
        explicit = [s.strip() for s in explicit_split.split(",") if s.strip()]
        natural = fallback_map.get(visual_config)
        if natural:
            explicit_set = set(explicit)
            intersection = [d for d in natural if d in explicit_set]
            if intersection:
                return intersection
        # tri_rot{N}: natural rot pool absent → take K nearest angle pools
        # (rot/base/flipped) from the explicit cumulative pool.
        target_angle = _vc_target_angle(visual_config)
        if target_angle is not None:
            rot_pools = _group_explicit_rot_pools(explicit, pool_pattern)
            if rot_pools:
                ranked = sorted(
                    rot_pools.keys(),
                    key=lambda a: _angular_distance_deg(a, float(target_angle)),
                )
                chosen_angles = ranked[: max(1, k)]
                chosen: list[str] = []
                for a in chosen_angles:
                    chosen.extend(rot_pools[a])
                return chosen
        return explicit

    if goal_angle is not None:
        pools = _scan_rotation_pools(data_dir, pattern=pool_pattern)
        if not pools:
            raise ValueError(
                f"No rotation pools matching pattern={pool_pattern!r} found in "
                f"{data_dir}; cannot auto-resolve goal_angle={goal_angle}."
            )
        ranked = sorted(pools.keys(), key=lambda a: _angular_distance_deg(a, float(goal_angle)))
        # Exact-match short-circuit: if a pool exists at the goal angle itself,
        # use only that one. Fall back to K nearest only when no exact pool exists.
        if _angular_distance_deg(ranked[0], float(goal_angle)) < 0.5:
            chosen_angles = [ranked[0]]
        else:
            chosen_angles = ranked[: max(1, k)]
        chosen: list[str] = []
        for a in chosen_angles:
            chosen.extend(pools[a])
        return chosen

    if visual_config not in fallback_map:
        raise KeyError(
            f"visual_config={visual_config!r} has no entry in fallback retrieval "
            f"split map. Pass --retrieval_pool_split or --goal_angle to override."
        )
    return fallback_map[visual_config]


class PushTRetrieval:
    WINDOW_SIZE   = 8
    STRIDE        = 2
    SIM_SCALE     = 512.0   # PushT sim coordinate range

    # 10-dim feature weights
    W_BLOCK_POS   = 2.0
    W_AGENT_POS   = 2.5 #1.5
    W_YAW         = 1.5
    W_BLOCK_VEL   = 1.0 #0.1
    W_AGENT_VEL   = 1.0

    # stage-1/2 gates
    N_DEMO_FILTER = 100
    POS_THRESHOLD = 0.15
    MIN_POS_POOL  = 20

    def __init__(self, data_dir: str, chunk_size: int = 8, split="base",
                 block_rel: bool = False,
                 ret_context_multiplier: int = 1, ret_image_subsample: int = 1):
        self.chunk_size = chunk_size
        self.split = list(split) if isinstance(split, (list, tuple)) else [split]
        self.block_rel = block_rel
        self.ret_context_multiplier = ret_context_multiplier
        self.ret_image_subsample = ret_image_subsample
        mode = "block-relative" if block_rel else "absolute"
        print(f"Loading retrieval pool from GT states (split={self.split}, agent_pos: {mode}, "
              f"ctx_mult={ret_context_multiplier}, img_subsample={ret_image_subsample}) ...")
        self._load_pool(data_dir)

    # ── Pool loading ─────────────────────────────────────────────────────────

    # Velocity is computed over the last VEL_WINDOW+1 frames (VEL_WINDOW diffs),
    # matching retrieve_similar_state10.py which uses STRIDE=2 for the same purpose.
    VEL_WINDOW = 2

    @staticmethod
    def _build_feature(block_pos_seq, agent_pos_seq, block_angle_seq,
                       t, window, S,
                       w_block_pos, w_agent_pos, w_yaw, w_block_vel, w_agent_vel,
                       block_rel=False, vel_window=2):
        """10-dim feature from GT state sequences. Positions normalized by S."""
        t_last = min(t + window - 1, len(block_pos_seq) - 1)

        block_pos = (block_pos_seq[t_last] / S) * w_block_pos          # (2,)
        yaw = float(block_angle_seq[t_last])

        if block_rel:
            # agent pos → block 좌표계 (translate + rotate by -yaw)
            rel = agent_pos_seq[t_last] / S - block_pos_seq[t_last] / S
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            local_x =  rel[0] * cos_y + rel[1] * sin_y
            local_y = -rel[0] * sin_y + rel[1] * cos_y
            agent_pos = np.array([local_x, local_y],
                                 dtype=np.float32) * w_agent_pos       # (2,)
        else:
            agent_pos = (agent_pos_seq[t_last] / S) * w_agent_pos      # (2,)

        yaw_feat = np.array([np.sin(yaw), np.cos(yaw)],
                            dtype=np.float32) * w_yaw                   # (2,)

        # Velocity: mean finite diff over last vel_window frames
        # (matches retrieve_similar_state10.py: STRIDE=2 → last 3 frames, 2 diffs)
        vel_start = max(t, t_last - vel_window)
        if t_last > vel_start:
            block_vel = np.mean(
                np.diff(block_pos_seq[vel_start:t_last + 1] / S, axis=0), axis=0
            ).astype(np.float32)
            agent_vel = np.mean(
                np.diff(agent_pos_seq[vel_start:t_last + 1] / S, axis=0), axis=0
            ).astype(np.float32)
        else:
            block_vel = np.zeros(2, dtype=np.float32)
            agent_vel = np.zeros(2, dtype=np.float32)

        return np.concatenate([
            block_pos, agent_pos, yaw_feat,
            block_vel * w_block_vel, agent_vel * w_agent_vel,
        ]).astype(np.float32)   # (10,)

    def _load_pool(self, data_dir: str):
        """Load GT states + images from dataset HDF5 files for the retrieval pool."""
        hdf5_files = get_hdf5_files(data_dir, task_split=self.split)

        subframes = []
        self._base_data: dict = {}

        for file in sorted(hdf5_files):
            suite = os.path.relpath(file, data_dir).split(os.sep)[0]
            with h5py.File(file, "r") as f:
                sorted_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
                for demo_key in sorted_keys:
                    grp = f[f"data/{demo_key}"]

                    # Load images, actions, and proprio for retrieval output
                    imgs = decode_jpeg_bytes_dataset(grp["obs/images"])
                    acts = grp["actions"][:].astype(np.float32)
                    prop = grp["obs/states"][:][:, :2].astype(np.float32)
                    self._base_data[(suite, demo_key)] = {"images": imgs, "actions": acts, "proprio": prop}

                    # Load GT states: [agent_x, agent_y, block_x, block_y, block_angle]
                    states = grp["obs/states"][:].astype(np.float32)
                    T = len(states)
                    agent_pos_seq = states[:, :2]    # (T, 2) agent (x, y)
                    block_pos_seq = states[:, 2:4]   # (T, 2) block (x, y)
                    block_angle_seq = states[:, 4]   # (T,)

                    # Early-frame subframes (t_last=0..WINDOW_SIZE-2)
                    # Matches retrieve_similar_state10.py pool so eval pool
                    # covers the same range as the training NPZ pool.
                    for t_last_early in range(min(self.WINDOW_SIZE - 1, T)):
                        w = t_last_early + 1
                        feat = self._build_feature(
                            block_pos_seq, agent_pos_seq, block_angle_seq,
                            0, w, self.SIM_SCALE,
                            self.W_BLOCK_POS, self.W_AGENT_POS, self.W_YAW,
                            self.W_BLOCK_VEL, self.W_AGENT_VEL,
                            block_rel=self.block_rel,
                            vel_window=self.VEL_WINDOW,
                        )
                        subframes.append({
                            "split": suite, "demo": demo_key, "start": 0,
                            "t_last": t_last_early,
                            "block_x0": float(block_pos_seq[0, 0]),
                            "block_y0": float(block_pos_seq[0, 1]),
                            "end_x":   float(block_pos_seq[t_last_early, 0]),
                            "end_y":   float(block_pos_seq[t_last_early, 1]),
                            "feat":    feat,
                        })

                    # Standard sliding-window subframes
                    for t in range(0, T - self.WINDOW_SIZE + 1, self.STRIDE):
                        t_last = min(t + self.WINDOW_SIZE - 1, T - 1)
                        feat = self._build_feature(
                            block_pos_seq, agent_pos_seq, block_angle_seq,
                            t, self.WINDOW_SIZE, self.SIM_SCALE,
                            self.W_BLOCK_POS, self.W_AGENT_POS, self.W_YAW,
                            self.W_BLOCK_VEL, self.W_AGENT_VEL,
                            block_rel=self.block_rel,
                            vel_window=self.VEL_WINDOW,
                        )
                        subframes.append({
                            "split": suite, "demo": demo_key, "start": t,
                            "t_last": t_last,
                            "block_x0": float(block_pos_seq[0, 0]),
                            "block_y0": float(block_pos_seq[0, 1]),
                            "end_x":   float(block_pos_seq[t_last, 0]),
                            "end_y":   float(block_pos_seq[t_last, 1]),
                            "feat":    feat,
                        })

        self._subframes = subframes
        n_demos = len(set((sf["split"], sf["demo"]) for sf in subframes))
        print(f"  {len(subframes):,} subframes from {n_demos} demos")

        self._end_pos = np.array(
            [[sf["end_x"], sf["end_y"]] for sf in subframes], dtype=np.float32
        ) / self.SIM_SCALE  # normalized (N, 2)
        self._feat = np.stack([sf["feat"] for sf in subframes])  # (N, 10)

        # Per-demo index mapping (for stage-1 filtering)
        demo_indices: dict = defaultdict(list)
        for i, sf in enumerate(subframes):
            demo_indices[(sf["split"], sf["demo"])].append(i)
        self._demo_keys    = sorted(demo_indices.keys())
        self._demo_indices = demo_indices
        # Initial block position of each demo (normalized)
        self._demo_init_block_pos = np.array([
            [subframes[demo_indices[k][0]]["block_x0"],
             subframes[demo_indices[k][0]]["block_y0"]]
            for k in self._demo_keys
        ], dtype=np.float32) / self.SIM_SCALE

        print(f"  {len(self._base_data)} demos with images loaded")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_retrieved_data(
        self,
        agent_pos: np.ndarray,
        block_pos: np.ndarray,
        block_angle: float,
        block_pos_history: np.ndarray | None = None,
        agent_pos_history: np.ndarray | None = None,
    ) -> tuple:
        """
        3-stage retrieval using 10-dim state feature from gt pose.
        Returns (ret_frames, ret_actions, ret_proprio).

        agent_pos, block_pos: raw sim coordinates (x, y), range ~0-512.
        block_angle: raw physics angle (radians).
        block_pos_history: (T, 2) recent block positions (x, y), for velocity.
        agent_pos_history: (T, 2) recent agent positions (x, y), for velocity.
        """
        S = self.SIM_SCALE
        print(block_pos, agent_pos, block_angle)
        block_x = float(block_pos[0]) / S
        block_y = float(block_pos[1]) / S
        agent_x = float(agent_pos[0]) / S
        agent_y = float(agent_pos[1]) / S
        yaw = float(block_angle)

        # ── Build 10-dim query feature (same normalization as pool) ──
        block_pos_feat = np.array([block_x, block_y], dtype=np.float32) * self.W_BLOCK_POS

        if self.block_rel:
            # agent pos → block 좌표계 (translate + rotate by -yaw)
            rel_x, rel_y = agent_x - block_x, agent_y - block_y
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            local_x =  rel_x * cos_y + rel_y * sin_y
            local_y = -rel_x * sin_y + rel_y * cos_y
            agent_pos_feat = np.array([local_x, local_y],
                                      dtype=np.float32) * self.W_AGENT_POS
        else:
            agent_pos_feat = np.array([agent_x, agent_y], dtype=np.float32) * self.W_AGENT_POS

        yaw_feat = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32) * self.W_YAW

        # Velocity: use only last VEL_WINDOW+1 frames (matching training pool features)
        vw = self.VEL_WINDOW
        if block_pos_history is not None and len(block_pos_history) >= 2:
            bph = block_pos_history[-(vw + 1):]  # last 3 frames → 2 diffs
            block_vel = np.mean(
                np.diff(bph / S, axis=0), axis=0
            ).astype(np.float32)
        else:
            block_vel = np.zeros(2, dtype=np.float32)

        if agent_pos_history is not None and len(agent_pos_history) >= 2:
            aph = agent_pos_history[-(vw + 1):]
            agent_vel = np.mean(
                np.diff(aph / S, axis=0), axis=0
            ).astype(np.float32)
        else:
            agent_vel = np.zeros(2, dtype=np.float32)

        q_feat = np.concatenate([
            block_pos_feat, agent_pos_feat, yaw_feat,
            block_vel * self.W_BLOCK_VEL, agent_vel * self.W_AGENT_VEL,
        ]).astype(np.float32)   # (10,)

        q_pos = np.array([block_x, block_y], dtype=np.float32)

        # Stage 1: top-N demos by initial block position
        cent_d2   = ((self._demo_init_block_pos - q_pos) ** 2).sum(axis=1)
        top_demos = {self._demo_keys[i] for i in np.argsort(cent_d2)[:self.N_DEMO_FILTER]}
        sub_idx   = np.array([i for dk in top_demos for i in self._demo_indices[dk]])

        # Stage 3: 10-dim feature L2
        dists  = ((self._feat[sub_idx] - q_feat) ** 2).sum(axis=1)
        best_i = int(sub_idx[np.argmin(dists)])
        sf     = self._subframes[best_i]
        key    = (sf["split"], sf["demo"])

        # Context window parameters
        ctx_mult = self.ret_context_multiplier
        subsample = self.ret_image_subsample
        ret_chunk = self.chunk_size * ctx_mult
        n_ret_frames = ret_chunk // subsample

        entry = self._base_data.get(key)
        if entry is None:
            return (
                np.zeros((n_ret_frames, 128, 128, 3), dtype=np.uint8),
                np.zeros((ret_chunk, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )
        imgs, acts, prop = entry["images"], entry["actions"], entry["proprio"]
        T  = len(imgs)
        t_last = min(sf["t_last"], T - 1)

        all_frames  = [imgs[max(0, min(t_last + i, T - 1))] for i in range(ret_chunk)]
        all_actions = [acts[max(0, min(t_last + i, T - 1))] for i in range(ret_chunk)]

        # Retrieved state: proprio at matched frame (t_last)
        ret_proprio = prop[min(t_last, T - 1)].copy()

        # Subsample images (e.g. ::2)
        frames = all_frames[::subsample]
        return np.stack(frames), np.stack(all_actions), ret_proprio
