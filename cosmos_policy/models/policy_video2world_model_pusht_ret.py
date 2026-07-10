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
Cosmos Policy models with PushT-retrieval conditioning.

Sequence layout (chunk_size=8, num_dup=4, with predict_future_states=True):
  latent 0     : blank input          (1 frame)               — sentinel
  latent 1     : current frame        (4 frames → 1 latent)   — condition
  latent 2     : current state        (4 frames → 1 latent)   — condition (proprio injected)
  latent 3     : retrieved action     (4 frames → 1 latent)   — condition (injected)
  latents 4-5  : retrieved frame      (8 frames → 2 latents)  — condition
  latent 6     : retrieved state      (4 frames → 1 latent)   — condition (injected)
  latent 7     : predicted action     (4 frames → 1 latent)   — generated
  latent 8     : predicted frame      (4 frames → 1 latent)   — generated
  latent 9     : predicted state      (4 frames → 1 latent)   — generated
  ─────────────────────────────────────────────────────
  state_t = 10   chunk_duration = 37

Conditioning strategy:
  - min_num_conditional_frames=3 covers latents 0-2 (blank + cur_frame + cur_state)
  - _apply_ret_mask additionally conditions: latent 3 (ret_action),
    latents 4-5 (ret_frame), latent 6 (ret_state)
  - Latent 7 (predicted action) is GENERATED at both training and inference.

Two variants:
  CosmosPolicyPushTRetModel                          - EDM / WAN2.1 backbone
  CosmosPolicyPushTRetModelRectifiedFlow             - Rectified Flow / Predict2.5 backbone
"""

import torch

from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk, replace_latent_with_proprio
from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldModel,
    CosmosPolicyVideo2WorldConfig,
)
from cosmos_policy.models.policy_video2world_model_rectified_flow import (
    CosmosPolicyVideo2WorldModelRectifiedFlow,
    CosmosPolicyVideo2WorldConfigRectifiedFlow,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _apply_ret_mask(condition, data_batch):
    """
    Condition the retrieval slots and the current-state slots in the mask.

    Retrieval slots are conditioned only when retrieval data is present:
      - has_ret_data=1 → ret_action, ret_state get mask=1
      - has_ret_image=1 → ret_video gets mask=1
      - has_ret_data=0 (dropout) → retrieval slots get mask=0 (unconditioned)
    Current-state slots (proprio, image) are always conditioned.
    """
    if "retrieved_video_start_latent_idx" not in data_batch:
        return
    mask = condition.condition_video_input_mask_B_C_T_H_W   # (B, 1, T, H', W')

    # Per-sample retrieval presence: (B, 1, 1, 1, 1)
    if "has_ret_data" in data_batch:
        has_ret = data_batch["has_ret_data"].float().reshape(-1, 1, 1, 1, 1).to(mask.device)
    else:
        has_ret = torch.ones(mask.shape[0], 1, 1, 1, 1, device=mask.device)

    if "has_ret_image" in data_batch:
        has_ret_image = data_batch["has_ret_image"].float().reshape(-1, 1, 1, 1, 1).to(mask.device)
    else:
        has_ret_image = has_ret

    # Retrieved video — conditioned when has_ret_image=1
    ret_start = int(data_batch["retrieved_video_start_latent_idx"].flatten()[0])
    ret_end   = int(data_batch["retrieved_video_end_latent_idx"].flatten()[0])
    if ret_start != -1 and ret_end != -1:
        mask[:, :, ret_start:ret_end, :, :] = has_ret_image

    # Retrieved action — conditioned when has_ret_data=1
    if "retrieved_action_latent_idx" in data_batch:
        idx = int(data_batch["retrieved_action_latent_idx"].flatten()[0])
        if idx != -1:
            mask[:, :, idx, :, :] = has_ret[:, :, 0, :, :]

    # Current proprio — always conditioned
    if "current_proprio_latent_idx" in data_batch:
        idx = int(data_batch["current_proprio_latent_idx"].flatten()[0])
        if idx != -1:
            mask[:, :, idx, :, :] = 1.0

    # Current image — always conditioned
    if "current_image_latent_idx" in data_batch:
        idx = int(data_batch["current_image_latent_idx"].flatten()[0])
        if idx != -1:
            mask[:, :, idx, :, :] = 1.0

    # Retrieved state — conditioned when has_ret_data=1
    if "retrieved_state_latent_idx" in data_batch:
        idx = int(data_batch["retrieved_state_latent_idx"].flatten()[0])
        if idx != -1:
            mask[:, :, idx, :, :] = has_ret[:, :, 0, :, :]


def _inject_retrieved_actions(x0, data_batch):
    """Inject retrieved actions into x0 at the retrieved_action_latent_idx slot.

    Skips injection when has_ret_data=0 (retrieval dropped).
    """
    if "retrieved_actions" not in data_batch or "retrieved_action_latent_idx" not in data_batch:
        return
    ret_action_idx = data_batch["retrieved_action_latent_idx"]
    if torch.all(ret_action_idx == -1):
        return
    if "has_ret_data" in data_batch and torch.all(data_batch["has_ret_data"] == 0):
        return
    replace_latent_with_action_chunk(
        x0,
        data_batch["retrieved_actions"].to(x0.dtype),
        action_indices=ret_action_idx,
    )


def _inject_retrieved_state(x0, data_batch):
    """Inject retrieved state (proprio) into x0 at the retrieved_state_latent_idx slot.

    Skips injection when has_ret_data=0 (retrieval dropped).
    """
    if "retrieved_proprio" not in data_batch or "retrieved_state_latent_idx" not in data_batch:
        return
    ret_state_idx = data_batch["retrieved_state_latent_idx"]
    if torch.all(ret_state_idx == -1):
        return
    if "has_ret_data" in data_batch and torch.all(data_batch["has_ret_data"] == 0):
        return
    replace_latent_with_proprio(
        x0,
        data_batch["retrieved_proprio"].to(x0.dtype),
        proprio_indices=ret_state_idx,
    )


def _find_condition_in_closure(fn):
    """Walk fn's closure to find the Video2WorldCondition (has condition_video_input_mask_B_C_T_H_W)."""
    for cell in (fn.__closure__ or []):
        try:
            obj = cell.cell_contents
        except ValueError:
            continue
        if hasattr(obj, "condition_video_input_mask_B_C_T_H_W"):
            return obj
    return None


# ── EDM / WAN2.1 ──────────────────────────────────────────────────────────────

class CosmosPolicyPushTRetConfig(CosmosPolicyVideo2WorldConfig):
    """Config for PushT-retrieval EDM policy model. Inherits all base settings."""
    pass


class CosmosPolicyPushTRetModel(CosmosPolicyVideo2WorldModel):
    """EDM policy model with retrieval conditioning. See module docstring."""

    def __init__(self, config: CosmosPolicyPushTRetConfig):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        _apply_ret_mask(condition, data_batch)
        _inject_retrieved_actions(condition.gt_frames, data_batch)
        _inject_retrieved_state(condition.gt_frames, data_batch)
        _inject_retrieved_actions(latent_state, data_batch)
        _inject_retrieved_state(latent_state, data_batch)
        return raw_state, latent_state, condition

    def get_x0_fn_from_batch(self, data_batch, guidance, **kwargs):
        result = super().get_x0_fn_from_batch(data_batch, guidance, **kwargs)
        x0_fn = result[0] if isinstance(result, tuple) else result
        condition = _find_condition_in_closure(x0_fn)
        if condition is not None:
            _apply_ret_mask(condition, data_batch)
        return result


# ── Rectified Flow / Predict2.5 ───────────────────────────────────────────────

class CosmosPolicyPushTRetConfigRectifiedFlow(CosmosPolicyVideo2WorldConfigRectifiedFlow):
    """Config for PushT-retrieval RF policy model. Inherits all base settings."""
    pass


class CosmosPolicyPushTRetModelRectifiedFlow(CosmosPolicyVideo2WorldModelRectifiedFlow):
    """Rectified Flow policy model with retrieval conditioning. See module docstring."""

    def __init__(self, config: CosmosPolicyPushTRetConfigRectifiedFlow):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        _apply_ret_mask(condition, data_batch)
        # Inject into both gt_frames (conditioning) and latent_state (x0 for velocity target).
        # They are different tensors (bf16 vs fp32) because set_video_condition() converts dtype.
        _inject_retrieved_actions(condition.gt_frames, data_batch)
        _inject_retrieved_state(condition.gt_frames, data_batch)
        _inject_retrieved_actions(latent_state, data_batch)
        _inject_retrieved_state(latent_state, data_batch)
        return raw_state, latent_state, condition

    def get_velocity_fn_from_batch(self, data_batch, guidance=1.5, **kwargs):
        result = super().get_velocity_fn_from_batch(data_batch, guidance, **kwargs)
        condition = _find_condition_in_closure(result)
        if condition is not None:
            _apply_ret_mask(condition, data_batch)
        return result
