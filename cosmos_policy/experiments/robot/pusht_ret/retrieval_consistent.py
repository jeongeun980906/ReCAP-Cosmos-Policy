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
retrieval_consistent.py

Consistent retrieval: follow matched demo trajectory, re-retrieve only when
the distance grows too large. Produces top-N diverse tracks.

Same interface as PushTRetrieval.get_retrieved_data(), but maintains state
across calls within an episode. Call reset() between episodes.
"""

from collections import defaultdict

import numpy as np

from .retrieval import PushTRetrieval


class PushTConsistentRetrieval(PushTRetrieval):
    """Follow + re-retrieve retrieval with top-N diverse tracks.

    At each retrieval call:
      1. Try to continue each track in its current demo (advance by chunk_size).
      2. If distance² > threshold or demo ends → re-retrieve from full pool,
         excluding demos used by other tracks (diversity).
      3. Return frames/actions from the best (rank-0) track.
    """

    # Override weights to match tuned values for consistent retrieval
    W_BLOCK_POS = 1.0
    W_AGENT_POS = 2.0
    W_YAW       = 1.5
    W_BLOCK_VEL = 1.0
    W_AGENT_VEL = 1.0

    def __init__(self, data_dir: str, chunk_size: int = 8, split: str = "base",
                 block_rel: bool = False,
                 ret_context_multiplier: int = 1, ret_image_subsample: int = 1,
                 top_n: int = 3, dist_threshold: float = 0.3):
        self.top_n = top_n
        self.dist_threshold = dist_threshold
        super().__init__(data_dir, chunk_size, split, block_rel,
                         ret_context_multiplier, ret_image_subsample)

        # Fast lookup: (split, demo, t_last) -> pool index
        self._key_to_idx: dict = {}
        for i, sf in enumerate(self._subframes):
            self._key_to_idx[(sf["split"], sf["demo"], sf["t_last"])] = i

        # Demo -> pool indices for fast exclusion masking
        self._demo_to_pool_idx: dict = defaultdict(list)
        for i, sf in enumerate(self._subframes):
            self._demo_to_pool_idx[(sf["split"], sf["demo"])].append(i)

        print(f"  ConsistentRetrieval: top_n={top_n}, threshold={dist_threshold}")
        self.reset()

    def reset(self):
        """Clear tracking state. Call between episodes."""
        self._currents = [None] * self.top_n   # (split, demo, t_last) per track
        self._retrieval_count = 0
        self._follow_count = 0

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

        dists = None  # lazy: computed once, shared across tracks
        sf_matches = []

        for rank in range(self.top_n):
            do_retrieve = self._currents[rank] is None

            if not do_retrieve:
                cur = self._currents[rank]
                continued = (cur[0], cur[1], cur[2] + self.chunk_size)
                if continued in self._key_to_idx:
                    cont_feat = self._feat[self._key_to_idx[continued]]
                    cont_dist = float(np.sum((q_feat - cont_feat) ** 2))
                    if cont_dist <= self.dist_threshold:
                        self._currents[rank] = continued
                        sf_matches.append(continued)
                        self._follow_count += 1
                        continue
                do_retrieve = True

            if do_retrieve:
                if dists is None:
                    dists = np.sum((self._feat - q_feat) ** 2, axis=1)

                # Exclude demos used by other tracks + already picked this step
                excluded = set()
                for other in range(self.top_n):
                    if other != rank and self._currents[other] is not None:
                        excluded.add((self._currents[other][0], self._currents[other][1]))
                for prev_m in sf_matches:
                    excluded.add((prev_m[0], prev_m[1]))

                if excluded:
                    masked = dists.copy()
                    for dk in excluded:
                        for idx in self._demo_to_pool_idx.get(dk, []):
                            masked[idx] = np.inf
                else:
                    masked = dists

                best_idx = int(np.argmin(masked))
                if masked[best_idx] == np.inf:
                    best_idx = int(np.argmin(dists))

                sf = self._subframes[best_idx]
                self._currents[rank] = (sf["split"], sf["demo"], sf["t_last"])
                sf_matches.append(self._currents[rank])
                self._retrieval_count += 1

        # Return frames/actions from rank-0 track
        return self._get_frames_actions(self._currents[0])

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
