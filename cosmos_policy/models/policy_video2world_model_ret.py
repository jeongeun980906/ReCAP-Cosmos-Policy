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
Cosmos Policy Video2World Model with Retrieval Conditioning.

Extends CosmosPolicyVideo2WorldModel to treat the retrieved section of the video
(images, proprio, actions) as additional conditioning inputs (mask=1, latent injection).

Does NOT modify any base source files.

Video tensor layout built by the retrieval dataset:
  slot 0              : blank frame
  slots 1 .. ret_end  : retrieved section (ret wrist/left/right images, proprio, actions)
  slots ret_end+1 ..  : original current + future observations

Additional data_batch keys provided by the retrieval dataset:
  retrieved_proprio           (B, ret_prop_dim)           retrieved proprio at end of chunk
  retrieved_actions           (B, chunk_size, action_dim) retrieved action chunk
  has_ret_data                (B,)                        1 = valid retrieval, 0 = fallback zeros
  ret_wrist_latent_idx        (B,)                        latent slot for retrieved wrist camera
  ret_left_latent_idx         (B,)                        latent slot for retrieved left camera
  ret_right_latent_idx        (B,)                        latent slot for retrieved right camera
  retrieved_state_latent_idx  (B,)                        latent slot for retrieved proprio  (-1 if unused)
  retrieved_action_latent_idx (B,)                        latent slot for retrieved actions
"""

import torch

from cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldConfig,
    CosmosPolicyVideo2WorldModel,
)


class CosmosPolicyRetVideo2WorldConfig(CosmosPolicyVideo2WorldConfig):
    """Config for retrieval-conditioned policy model. Inherits all base settings."""
    pass


class CosmosPolicyRetVideo2WorldModel(CosmosPolicyVideo2WorldModel):
    """
    Extends CosmosPolicyVideo2WorldModel with retrieval conditioning.

    Retrieved camera images are already encoded into x0 as part of the video tensor.
    This class additionally:
      1. Sets condition_video_input_mask=1 for all retrieved latent slots so the
         diffusion model treats them as clean conditioning frames (not noised).
      2. Injects retrieved_proprio and retrieved_actions into their latent slots via
         the same replace_latent_with_* mechanism used for current proprio/actions.

    Training path  : get_data_and_condition() override.
    Inference path : get_x0_fn_from_batch() override.

    Note on in-place sharing:
      replace_latent_with_* modifies x0 in-place and returns it. Since
      condition.gt_frames references the same tensor as latent_state, injections
      applied to condition.gt_frames in get_data_and_condition() are automatically
      visible in get_x0_fn_from_batch() when it calls self.get_data_and_condition().
      The conditioning mask is the only thing that must be patched separately in
      the inference path.
    """

    def __init__(self, config: CosmosPolicyRetVideo2WorldConfig):
        super().__init__(config)

    # ── training ──────────────────────────────────────────────────────────────

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        self._apply_ret_conditioning(condition, data_batch)
        return raw_state, latent_state, condition

    # ── inference ─────────────────────────────────────────────────────────────

    def get_x0_fn_from_batch(self, data_batch, guidance, **kwargs):
        """
        Calls parent, then patches the conditioning mask inside the returned
        x0_fn closure to mark retrieved latent slots as conditioning frames.

        Latent injection (retrieved_proprio, retrieved_actions) is already handled
        because get_x0_fn_from_batch internally calls self.get_data_and_condition(),
        which is our overridden version that injects into condition.gt_frames in-place.
        """
        result = super().get_x0_fn_from_batch(data_batch, guidance, **kwargs)
        x0_fn = result[0] if isinstance(result, tuple) else result

        # Find the Video2WorldCondition object inside x0_fn's closure and patch the mask.
        condition = _find_condition_in_closure(x0_fn)
        if condition is not None:
            self._apply_ret_mask(condition, data_batch)

        return result

    # ── shared helpers ────────────────────────────────────────────────────────

    def _apply_ret_conditioning(self, condition, data_batch):
        """Apply both mask and latent injection for retrieved data. Used in training."""
        self._apply_ret_mask(condition, data_batch)
        self._inject_ret_latents(condition, data_batch)

    def _apply_ret_mask(self, condition, data_batch):
        """
        Set condition_video_input_mask=1 for all retrieved latent slots
        (slots 1 through retrieved_action_latent_idx inclusive).

        Only applied to samples where has_ret_data == 1.
        """
        if "retrieved_action_latent_idx" not in data_batch:
            return

        mask = condition.condition_video_input_mask_B_C_T_H_W  # (B, 1, T, H', W')
        B = mask.shape[0]
        batch_indices = torch.arange(B, device=mask.device)

        has_ret = data_batch.get("has_ret_data", torch.ones(B, device=mask.device)).bool()
        has_ret_b = has_ret.view(B, 1, 1, 1)

        ret_end = data_batch["retrieved_action_latent_idx"]
        ret_end_val = int(ret_end[0]) if ret_end.dim() > 0 else int(ret_end)

        for slot in range(1, ret_end_val + 1):
            orig = mask[batch_indices, :, slot, :, :]  # (B, 1, H', W')
            mask[batch_indices, :, slot, :, :] = torch.where(
                has_ret_b.expand_as(orig),
                torch.ones_like(orig),
                orig,
            )

    def _inject_ret_latents(self, condition, data_batch):
        """
        Inject retrieved_proprio and retrieved_actions into condition.gt_frames.
        Only called during training (get_data_and_condition path).
        For inference the injection flows in via the in-place latent_state modification.
        """
        if "retrieved_proprio" in data_batch and "retrieved_state_latent_idx" in data_batch:
            if torch.all(data_batch["retrieved_state_latent_idx"] != -1):
                condition.gt_frames = replace_latent_with_proprio(
                    condition.gt_frames,
                    data_batch["retrieved_proprio"].to(condition.gt_frames.dtype),
                    proprio_indices=data_batch["retrieved_state_latent_idx"],
                    encoder=getattr(self, 'proprio_encoder', None),
                )

        if "retrieved_actions" in data_batch and "retrieved_action_latent_idx" in data_batch:
            if torch.all(data_batch["retrieved_action_latent_idx"] != -1):
                condition.gt_frames = replace_latent_with_action_chunk(
                    condition.gt_frames,
                    data_batch["retrieved_actions"].to(condition.gt_frames.dtype),
                    action_indices=data_batch["retrieved_action_latent_idx"],
                    encoder=getattr(self, 'action_encoder', None),
                )


# ── utility ───────────────────────────────────────────────────────────────────

def _find_condition_in_closure(fn):
    """
    Walk fn's closure cells to find the Video2WorldCondition object
    (identified by having condition_video_input_mask_B_C_T_H_W attribute).
    Returns the object, or None if not found.
    """
    for cell in fn.__closure__ or []:
        try:
            obj = cell.cell_contents
        except ValueError:
            continue
        if hasattr(obj, "condition_video_input_mask_B_C_T_H_W"):
            return obj
    return None
