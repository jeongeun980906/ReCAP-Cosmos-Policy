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
retrieval_cumulative.py

Cumulative EMA retrieval: maintain an exponential moving average of trajectory
distances for every pool subframe. At each query step, each pool entry's
score blends its historical consistency (via same-demo predecessor) with
the current-frame distance.

Algorithm:
  cum(p, 0) = dist²(q[0], p)
  cum(p, i) = min(
      γ · cum(pred(p), i-1) + (1-γ) · dist²(q[i], p),   # continuation
      dist²(q[i], p) + switch_cost                        # fresh start
  )
  pred(p) = same demo, start − chunk_size

Same interface as PushTRetrieval.get_retrieved_data(), but maintains state
across calls within an episode. Call reset() between episodes.
"""

from collections import defaultdict

import numpy as np

from .retrieval import PushTRetrieval


class PushTCumulativeRetrieval(PushTRetrieval):
    """Cumulative EMA retrieval with top-N diverse tracks.

    At each retrieval call:
      1. Compute dist² from current query to every pool subframe.
      2. Update cumulative score: min(EMA continuation, fresh start + cost).
      3. Pick top-N matches from distinct demos based on cum_dist.
      4. Return frames/actions from the best (rank-0) match.
    """

    # Override weights to match tuned values
    W_BLOCK_POS = 2.0
    W_AGENT_POS = 3.0
    W_YAW       = 1.5
    W_BLOCK_VEL = 1.0
    W_AGENT_VEL = 1.0

    def __init__(self, data_dir: str, chunk_size: int = 8, split: str = "base",
                 block_rel: bool = False,
                 ret_context_multiplier: int = 1, ret_image_subsample: int = 1,
                 top_n: int = 3, gamma: float = 0.95, switch_cost: float = 0.05):
        self.top_n = top_n
        self.gamma = gamma
        self.switch_cost = switch_cost
        super().__init__(data_dir, chunk_size, split, block_rel,
                         ret_context_multiplier, ret_image_subsample)

        # Fast lookup: (split, demo, t_last) -> pool index
        self._key_to_idx: dict = {}
        for i, sf in enumerate(self._subframes):
            self._key_to_idx[(sf["split"], sf["demo"], sf["t_last"])] = i

        # Predecessor: same demo, t_last − chunk_size
        self._pred_idx = np.full(len(self._subframes), -1, dtype=np.int64)
        for i, sf in enumerate(self._subframes):
            pred_key = (sf["split"], sf["demo"], sf["t_last"] - chunk_size)
            if pred_key in self._key_to_idx:
                self._pred_idx[i] = self._key_to_idx[pred_key]
        self._has_pred = self._pred_idx >= 0

        print(f"  CumulativeRetrieval: top_n={top_n}, gamma={gamma}, "
              f"switch_cost={switch_cost}")
        self.reset()

    def reset(self):
        """Clear cumulative state. Call between episodes."""
        self._cum_dist = None
        self._step = 0
        self._retrieval_count = 0
        self._follow_count = 0  # not used here, kept for interface compat

    def get_retrieved_data(
        self,
        agent_pos: np.ndarray,
        block_pos: np.ndarray,
        block_angle: float,
        block_pos_history: np.ndarray | None = None,
        agent_pos_history: np.ndarray | None = None,
    ) -> tuple:
        q_feat = self._build_query_feature(
            agent_pos, block_pos, block_angle,
            block_pos_history, agent_pos_history,
        )

        # Current dist² to all pool entries
        cur_dist = np.sum((self._feat - q_feat) ** 2, axis=1)

        if self._step == 0:
            self._cum_dist = cur_dist.copy()
        else:
            # Continuation: γ * cum_dist[pred] + (1-γ) * cur_dist
            continued = np.full(len(self._subframes), np.inf, dtype=np.float32)
            continued[self._has_pred] = (
                self.gamma * self._cum_dist[self._pred_idx[self._has_pred]]
                + (1.0 - self.gamma) * cur_dist[self._has_pred]
            )
            # Fresh start (+ optional penalty)
            fresh = cur_dist + self.switch_cost if self.switch_cost > 0 else cur_dist
            self._cum_dist = np.minimum(continued, fresh)

        self._step += 1
        self._retrieval_count += 1

        # Top-N diverse matches from cum_dist
        order = np.argsort(self._cum_dist)
        matches = []
        used_demos = set()
        for idx in order:
            sf = self._subframes[idx]
            dk = (sf["split"], sf["demo"])
            if dk in used_demos:
                continue
            matches.append((sf["split"], sf["demo"], sf["t_last"]))
            used_demos.add(dk)
            if len(matches) >= self.top_n:
                break

        # Return frames/actions from rank-0 match
        return self._get_frames_actions(matches[0])

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_query_feature(self, agent_pos, block_pos, block_angle,
                             block_pos_history, agent_pos_history):
        """Build 10-dim query feature from current gt pose (same normalization as pool)."""
        S = self.SIM_SCALE
        bx = float(block_pos[0]) / S
        by = float(block_pos[1]) / S
        ax = float(agent_pos[0]) / S
        ay = float(agent_pos[1]) / S
        yaw = float(block_angle)

        block_pos_feat = np.array([bx, by], np.float32) * self.W_BLOCK_POS

        if self.block_rel:
            rel_x, rel_y = ax - bx, ay - by
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            local_x =  rel_x * cos_y + rel_y * sin_y
            local_y = -rel_x * sin_y + rel_y * cos_y
            agent_pos_feat = np.array([local_x, local_y], np.float32) * self.W_AGENT_POS
        else:
            agent_pos_feat = np.array([ax, ay], np.float32) * self.W_AGENT_POS

        yaw_feat = np.array([np.sin(yaw), np.cos(yaw)], np.float32) * self.W_YAW

        vw = self.VEL_WINDOW
        if block_pos_history is not None and len(block_pos_history) >= 2:
            bph = block_pos_history[-(vw + 1):]
            block_vel = np.mean(np.diff(bph / S, axis=0), axis=0).astype(np.float32)
        else:
            block_vel = np.zeros(2, np.float32)

        if agent_pos_history is not None and len(agent_pos_history) >= 2:
            aph = agent_pos_history[-(vw + 1):]
            agent_vel = np.mean(np.diff(aph / S, axis=0), axis=0).astype(np.float32)
        else:
            agent_vel = np.zeros(2, np.float32)

        return np.concatenate([
            block_pos_feat, agent_pos_feat, yaw_feat,
            block_vel * self.W_BLOCK_VEL, agent_vel * self.W_AGENT_VEL,
        ]).astype(np.float32)

    def _get_frames_actions(self, match):
        """Extract frames, actions, and proprio for a matched subframe."""
        key = (match[0], match[1])
        entry = self._base_data.get(key)

        ctx_mult = self.ret_context_multiplier
        subsample = self.ret_image_subsample
        ret_chunk = self.chunk_size * ctx_mult
        n_ret_frames = ret_chunk // subsample

        if entry is None:
            return (
                np.zeros((n_ret_frames, 128, 128, 3), dtype=np.uint8),
                np.zeros((ret_chunk, 2), dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )

        imgs, acts, prop = entry["images"], entry["actions"], entry["proprio"]
        T = len(imgs)
        t_last = min(match[2], T - 1)  # match[2] is t_last directly

        all_frames  = [imgs[max(0, min(t_last + i, T - 1))] for i in range(ret_chunk)]
        all_actions = [acts[max(0, min(t_last + i, T - 1))] for i in range(ret_chunk)]
        ret_proprio = prop[min(t_last, T - 1)].copy()

        frames = all_frames[::subsample]
        return np.stack(frames), np.stack(all_actions), ret_proprio
