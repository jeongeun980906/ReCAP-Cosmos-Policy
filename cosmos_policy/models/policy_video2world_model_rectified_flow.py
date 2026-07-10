# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------------
# Modifications Copyright (c) 2026 Jeongeun Park et al. (ReCAP).
# This file is derived from NVIDIA Cosmos Policy
# (https://github.com/NVlabs/cosmos-policy) and was modified for the ReCAP
# project (https://github.com/jeongeun980906/ReCAP-Cosmos-Policy).
# Modifications are released under the Apache License, Version 2.0. See NOTICE.md.
# -----------------------------------------------------------------------------

"""
Cosmos Policy Rectified Flow Models - extends Text2WorldModelRectifiedFlow with policy-specific functionality.

This file contains:
- CosmosPolicyModelConfigRectifiedFlow: Config for the T2V policy base model
- CosmosPolicyDiffusionModelRectifiedFlow: T2V policy base model (no video conditioning)
- CosmosPolicyVideo2WorldConfigRectifiedFlow: Config extending base with video conditioning
- CosmosPolicyVideo2WorldModelRectifiedFlow: Full policy model with video conditioning

IMPORTANT: CosmosPolicyVideo2WorldModelRectifiedFlow inherits from CosmosPolicyDiffusionModelRectifiedFlow
(not Video2WorldModelRectifiedFlow) to ensure it gets all the policy-specific functionality
(training_step, compute_loss, etc.)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy._src.imaginaire.utils.context_parallel import broadcast_split_tensor, cat_outputs_cp
from cosmos_policy._src.predict2.conditioner import DataType
from cosmos_policy._src.predict2.models.text2world_model_rectified_flow import (
    Text2WorldModelRectifiedFlow,
    Text2WorldModelRectifiedFlowConfig,
)
from cosmos_policy._src.predict2.models.video2world_model import NUM_CONDITIONAL_FRAMES_KEY, ConditioningStrategy
from cosmos_policy.conditioner import Text2WorldCondition
from cosmos_policy.config.conditioner.video2world_conditioner import Video2WorldCondition
from cosmos_policy.models.policy_text2world_model import (
    LatentExtraction,
    LatentProjection,
    inject_action_noise_into_epsilon,
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

LOG_200 = math.log(200)
LOG_100000 = math.log(100000)


# =============================================================================
# Base Policy Model (T2V / no video conditioning)
# =============================================================================


@attrs.define(slots=False)
class CosmosPolicyModelConfigRectifiedFlow(Text2WorldModelRectifiedFlowConfig):
    """
    Extended config for Cosmos Policy diffusion model using rectified flow.
    Also adds policy-specific parameters for loss masking and action prediction.
    """

    sde: LazyDict = L(HybridEDMSDE)(
        # Note: Most of these values get overridden later in the experiment configs
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
        hybrid_sigma_distribution=True,
        uniform_lower=1.0,
        uniform_upper=85.0,
    )

    # Whether to use loss masking to separate action, future state, and value prediction
    mask_loss_for_action_future_state_prediction: bool = False
    # Whether to use loss masking on value prediction during policy predictions
    mask_value_prediction_loss_for_policy_prediction: bool = False
    # Whether to mask out some inputs (current state and action) during future state value prediction
    mask_current_state_action_for_value_prediction: bool = False
    # Whether to mask out some inputs (future state) during Q(s,a) prediction
    mask_future_state_for_qvalue_prediction: bool = False

    # Action loss multiplier (if greater than 1, upweights loss on predicting actions relative to other losses)
    action_loss_multiplier: int = 1

    # Latent projection settings — replace tiling with learned MLP encoder/decoder
    use_action_projection: bool = False
    use_proprio_projection: bool = False
    projection_hidden_dim: int = 256
    action_dim: int = 7  # Per-step action dimensionality (e.g., 2 for PushT, 7 for LIBERO)
    proprio_dim: int = 2  # Proprioception dimensionality
    chunk_size: int = 8  # Action chunk size (number of action steps per chunk)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert not (
            self.mask_loss_for_action_future_state_prediction and self.mask_value_prediction_loss_for_policy_prediction
        ), (
            "Cannot enable both mask_loss_for_action_future_state_prediction and mask_value_prediction_loss_for_policy_prediction!"
        )


class CosmosPolicyDiffusionModelRectifiedFlow(Text2WorldModelRectifiedFlow):
    """
    Cosmos Policy Diffusion Model using rectified flow - extends Text2WorldModelRectifiedFlow
    with policy-specific functionality.

    Adds support for:
    - Action chunk prediction and injection
    - Proprioception (proprio) prediction and injection
    - Value function prediction
    - Loss masking for different prediction types (action, future state, value)
    - Multi-component loss tracking
    """

    def __init__(self, config: CosmosPolicyModelConfigRectifiedFlow):
        super().__init__(config)
        self.config: CosmosPolicyModelConfigRectifiedFlow = config

        # Cosmos Policy SDE and Sampler
        self.sde = lazy_instantiate(config.sde)
        self.sampler = CosmosPolicySampler()

        # Optional MLP encoder/decoder for action/proprio latent injection
        if config.use_action_projection:
            action_input_dim = config.chunk_size * config.action_dim
            self.action_encoder = LatentProjection(
                action_input_dim, hidden_dim=config.projection_hidden_dim,
            )
            self.action_decoder = LatentExtraction(
                action_input_dim, hidden_dim=config.projection_hidden_dim,
            )
        if config.use_proprio_projection:
            self.proprio_encoder = LatentProjection(
                config.proprio_dim, hidden_dim=config.projection_hidden_dim,
            )
            self.proprio_decoder = LatentExtraction(
                config.proprio_dim, hidden_dim=config.projection_hidden_dim,
            )

    def _apply_ret_loss_mask(
        self,
        final_mask_B_T: torch.Tensor,
        data_batch: Optional[Dict] = None,
    ) -> Optional[torch.Tensor]:
        """Subclass hook to zero out retrieval-augmented latent slots from the
        loss mask so the backbone doesn't spend capacity reconstructing
        retrieved frames / actions that are only needed as conditioning input.

        Default: no-op (returns the input mask unchanged via ``None``).
        Subclasses returning a *new* tensor (must not be the same object) opt
        into having the mask applied to the velocity loss.
        """
        return None

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Performs a single training step using rectified flow with policy-specific extensions.
        """
        self._update_train_stats(data_batch)

        # Obtain text embeddings online if configured
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        # ── DEBUG: dump first 10 iterations line-by-line ─────────────────────
        import os as _os_dbg
        _dbg_on = int(_os_dbg.environ.get("DEBUG_TRAINING_VERBOSE", "0"))
        _dbg_tag = _os_dbg.environ.get("DEBUG_TRAINING_TAG", "model")
        # ─────────────────────────────────────────────────────────────────────

        # Get the input data and condition
        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)

        if _dbg_on and iteration < 10:
            import torch as _t
            x0 = x0_B_C_T_H_W
            print(f"\n══════ [{_dbg_tag}] iter={iteration} ══════", flush=True)
            print(f"  x0 shape={tuple(x0.shape)} dtype={x0.dtype} range=[{x0.min().item():.3f}, {x0.max().item():.3f}]", flush=True)
            print(f"  x0 per-T-slot mean: {[round(x0[:, :, ti].mean().item(), 3) for ti in range(x0.shape[2])]}", flush=True)
            print(f"  x0 per-T-slot std : {[round(x0[:, :, ti].std().item(), 3) for ti in range(x0.shape[2])]}", flush=True)
            print(f"  condition type={type(condition).__name__}", flush=True)
            mask = getattr(condition, "condition_video_input_mask_B_C_T_H_W", None)
            if mask is not None:
                print(f"  mask shape={tuple(mask.shape)} per-T: {mask[0, 0, :, 0, 0].int().tolist()}", flush=True)
            gt = getattr(condition, "gt_frames", None)
            if gt is not None:
                print(f"  gt_frames shape={tuple(gt.shape)} range=[{gt.min().item():.3f}, {gt.max().item():.3f}]", flush=True)
                print(f"  gt_frames per-T mean: {[round(gt[:, :, ti].mean().item(), 3) for ti in range(gt.shape[2])]}", flush=True)
            ai = data_batch.get("action_latent_idx", None)
            if ai is not None:
                print(f"  action_latent_idx: {ai.flatten().tolist()}", flush=True)
            actions = data_batch.get("actions", None)
            if actions is not None:
                print(f"  actions(target) shape={tuple(actions.shape)} range=[{actions.min().item():.3f}, {actions.max().item():.3f}]", flush=True)
            ret_a = data_batch.get("retrieved_actions", None)
            if ret_a is not None:
                print(f"  retrieved_actions shape={tuple(ret_a.shape)} range=[{ret_a.min().item():.3f}, {ret_a.max().item():.3f}]", flush=True)

        # Sample N(0,1) noise and training time for rectified flow
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        # Broadcast and split for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )

        # Get discrete timesteps and sigmas for rectified flow
        timesteps_B = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas_B = self.rectified_flow.get_sigmas(timesteps_B, self.tensor_kwargs_fp32)
        timesteps_B_T = rearrange(timesteps_B, "b -> b 1")
        sigmas_B_T = rearrange(sigmas_B, "b -> b 1")

        output_batch, velocity_loss = self.compute_loss_rectified_flow(
            x0_B_C_T_H_W,
            condition,
            epsilon_B_C_T_H_W,
            timesteps_B_T,
            sigmas_B_T,
            action_chunk=data_batch["actions"],
            action_indices=data_batch["action_latent_idx"],
            proprio=data_batch["proprio"],
            current_proprio_indices=data_batch["current_proprio_latent_idx"],
            future_proprio=data_batch["future_proprio"],
            future_proprio_indices=data_batch["future_proprio_latent_idx"],
            future_wrist_image_indices=data_batch["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=(
                data_batch["future_wrist_image2_latent_idx"] if "future_wrist_image2_latent_idx" in data_batch else None
            ),
            future_image_indices=data_batch["future_image_latent_idx"],
            future_image2_indices=(
                data_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in data_batch else None
            ),
            rollout_data_mask=data_batch["rollout_data_mask"],
            world_model_sample_mask=data_batch["world_model_sample_mask"],
            value_function_sample_mask=data_batch["value_function_sample_mask"],
            value_function_return=data_batch["value_function_return"],
            value_indices=data_batch["value_latent_idx"],
            data_batch=data_batch,
        )

        velocity_loss = velocity_loss.mean()

        if _dbg_on and iteration < 10:
            import torch as _t
            xt_dbg = output_batch.get("xt", None)
            x0_dbg = output_batch.get("x0", None)
            sig_dbg = output_batch.get("sigma", None)
            pred_dbg = output_batch.get("model_pred", None)
            if pred_dbg is not None and x0_dbg is not None:
                vt_target = -1 * (pred_dbg - 0)  # placeholder if not stored
                # actual vt = noise - x0 in cosmos
                pass
            if sig_dbg is not None:
                print(f"  sigmas: {sig_dbg.flatten().tolist()[:4]}", flush=True)
            if xt_dbg is not None:
                print(f"  xt per-T mean: {[round(xt_dbg[:, :, ti].mean().item(), 3) for ti in range(xt_dbg.shape[2])]}", flush=True)
            if pred_dbg is not None:
                print(f"  pred per-T mean: {[round(pred_dbg[:, :, ti].mean().item(), 3) for ti in range(pred_dbg.shape[2])]}", flush=True)
            print(f"  velocity_loss (final): {velocity_loss.item():.6f}", flush=True)

        return output_batch, velocity_loss

    def compute_loss_rectified_flow(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        sigmas_B_T: torch.Tensor,
        action_chunk: torch.Tensor,
        action_indices: torch.Tensor,
        proprio: torch.Tensor,
        current_proprio_indices: torch.Tensor,
        future_proprio: torch.Tensor,
        future_proprio_indices: torch.Tensor,
        future_wrist_image_indices: torch.Tensor,
        future_wrist_image2_indices: Optional[torch.Tensor],
        future_image_indices: torch.Tensor,
        future_image2_indices: Optional[torch.Tensor],
        rollout_data_mask: torch.Tensor,
        world_model_sample_mask: torch.Tensor,
        value_function_sample_mask: torch.Tensor,
        value_function_return: torch.Tensor,
        value_indices: torch.Tensor,
        data_batch: Optional[Dict] = None,
    ):
        """
        Compute velocity matching loss with policy-specific functionality.
        """
        condition.orig_x0_B_C_T_H_W = x0_B_C_T_H_W.clone()
        batch_indices = torch.arange(x0_B_C_T_H_W.shape[0], device=x0_B_C_T_H_W.device)
        C_latent, H_latent, W_latent = x0_B_C_T_H_W.shape[1], x0_B_C_T_H_W.shape[3], x0_B_C_T_H_W.shape[4]

        # Action injection
        x0_B_C_T_H_W = replace_latent_with_action_chunk(
            x0_B_C_T_H_W, action_chunk, action_indices=action_indices,
            encoder=getattr(self, 'action_encoder', None),
        )
        # Proprio injection
        if torch.all(current_proprio_indices != -1):
            x0_B_C_T_H_W = replace_latent_with_proprio(
                x0_B_C_T_H_W, proprio, proprio_indices=current_proprio_indices,
                encoder=getattr(self, 'proprio_encoder', None),
            )
        # Future proprio injection
        if torch.all(future_proprio_indices != -1):
            x0_B_C_T_H_W = replace_latent_with_proprio(
                x0_B_C_T_H_W, future_proprio, proprio_indices=future_proprio_indices,
                encoder=getattr(self, 'proprio_encoder', None),
            )
        # Value injection (only if value slot is present; value_indices=-1 means no value slot)
        if torch.all(value_indices != -1):
            x0_B_C_T_H_W[batch_indices, :, value_indices, :, :] = (
                value_function_return.reshape(-1, 1, 1, 1).expand(-1, C_latent, H_latent, W_latent).to(x0_B_C_T_H_W.dtype)
            )

        # Action noise in action space: replace epsilon's action slots with encoder(action_noise)
        action_encoder = getattr(self, 'action_encoder', None)
        if action_encoder is not None:
            epsilon_B_C_T_H_W = inject_action_noise_into_epsilon(
                epsilon_B_C_T_H_W, action_indices,
                self.config.action_dim, self.config.chunk_size, action_encoder,
            )

        # Get interpolation for rectified flow
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas_B_T)

        # Get velocity prediction from the model
        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            condition=condition,
        )

        # Get time-based loss weights
        time_weights_B = self.rectified_flow.train_time_weight(timesteps_B_T, self.tensor_kwargs_fp32)

        # Construct loss mask
        B, T = x0_B_C_T_H_W.shape[0], x0_B_C_T_H_W.shape[2]
        device = timesteps_B_T.device
        final_mask_B_T = torch.ones((B, T), dtype=torch.long, device=device)

        # Mask for value prediction input masking
        if (
            self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
        ):
            mask_B_T = torch.ones((B, T), dtype=torch.long, device=device)
            value_idx_B = ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(device)
            if torch.any(value_idx_B):
                value_batch_indices = torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[value_batch_indices, :] = 0
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Mask for action/future state prediction separation
        if self.config.mask_loss_for_action_future_state_prediction:
            mask_B_T = torch.zeros((B, T), dtype=torch.long, device=device)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(device)
            if torch.any(demo_idx_B):
                demo_batch_indices = torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(device)
            if torch.any(world_idx_B):
                world_batch_indices = torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                if torch.all(future_image_indices != -1):
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(future_image2_indices != -1):
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(future_wrist_image2_indices != -1):
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
            value_idx_B = ((rollout_data_mask == 1) & (value_function_sample_mask == 1)).to(torch.long).to(device)
            if torch.any(value_idx_B):
                value_batch_indices = torch.nonzero(value_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[value_batch_indices, value_indices[value_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Mask for value prediction loss during policy prediction
        if self.config.mask_value_prediction_loss_for_policy_prediction:
            assert value_function_sample_mask.sum() == 0, (
                "No value function samples should be present when mask_value_prediction_loss_for_policy_prediction==True!"
            )
            mask_B_T = torch.zeros((B, T), dtype=torch.long, device=device)
            demo_idx_B = (rollout_data_mask == 0).to(torch.long).to(device)
            if torch.any(demo_idx_B):
                demo_batch_indices = torch.nonzero(demo_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                mask_B_T[demo_batch_indices, action_indices[demo_batch_indices]] = 1
                if torch.all(future_image_indices != -1):
                    mask_B_T[demo_batch_indices, future_image_indices[demo_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(future_image2_indices != -1):
                    mask_B_T[demo_batch_indices, future_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):
                    mask_B_T[demo_batch_indices, future_wrist_image_indices[demo_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(future_wrist_image2_indices != -1):
                    mask_B_T[demo_batch_indices, future_wrist_image2_indices[demo_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):
                    mask_B_T[demo_batch_indices, future_proprio_indices[demo_batch_indices]] = 1
            world_idx_B = (rollout_data_mask == 1) & (world_model_sample_mask == 1).to(torch.long).to(device)
            if torch.any(world_idx_B):
                world_batch_indices = torch.nonzero(world_idx_B, as_tuple=False).squeeze(-1).to(torch.long).to(device)
                if torch.all(future_image_indices != -1):
                    mask_B_T[world_batch_indices, future_image_indices[world_batch_indices]] = 1
                if future_image2_indices is not None and torch.all(future_image2_indices != -1):
                    mask_B_T[world_batch_indices, future_image2_indices[world_batch_indices]] = 1
                if torch.all(future_wrist_image_indices != -1):
                    mask_B_T[world_batch_indices, future_wrist_image_indices[world_batch_indices]] = 1
                if future_wrist_image2_indices is not None and torch.all(future_wrist_image2_indices != -1):
                    mask_B_T[world_batch_indices, future_wrist_image2_indices[world_batch_indices]] = 1
                if torch.all(future_proprio_indices != -1):
                    mask_B_T[world_batch_indices, future_proprio_indices[world_batch_indices]] = 1
            final_mask_B_T = final_mask_B_T * mask_B_T

        # Upweight action loss by multiplier
        if self.config.action_loss_multiplier != 1:
            final_mask_B_T[batch_indices, action_indices] = final_mask_B_T[batch_indices, action_indices] * int(
                self.config.action_loss_multiplier
            )

        # Subclass hook (RAG): zero-out ret latent slots from reconstruction
        # loss so cosmos backbone capacity isn't spent re-creating retrieved
        # frames / retrieved actions that are only needed as conditioning
        # input. Default no-op.
        ret_mask_applied = False
        new_final_mask = self._apply_ret_loss_mask(final_mask_B_T, data_batch)
        if new_final_mask is not None and new_final_mask is not final_mask_B_T:
            final_mask_B_T = new_final_mask
            ret_mask_applied = True

        # Compute velocity matching loss
        velocity_mse_B_C_T_H_W = (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2

        # Apply time-based weighting
        time_weights_expanded = rearrange(time_weights_B, "b 1 -> b 1 1 1 1")
        weighted_velocity_loss_B_C_T_H_W = velocity_mse_B_C_T_H_W * time_weights_expanded

        # Apply the loss mask
        if (
            self.config.mask_loss_for_action_future_state_prediction
            or self.config.mask_current_state_action_for_value_prediction
            or self.config.mask_future_state_for_qvalue_prediction
            or self.config.action_loss_multiplier != 1
            or ret_mask_applied
        ):
            weighted_velocity_loss_B_C_T_H_W = weighted_velocity_loss_B_C_T_H_W * rearrange(
                final_mask_B_T, "b t -> b 1 t 1 1"
            )

        # Per-component losses for logging
        if torch.all(future_image_indices != -1):
            future_image_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_image_indices, :, :]
            )
            future_image_diff_demo = future_image_diff[rollout_data_mask == 0]
            future_image_diff_world_model = future_image_diff[world_model_sample_mask == 1]
            demo_sample_future_image_mse_loss = (future_image_diff_demo**2).mean()
            demo_sample_future_image_l1_loss = torch.abs(future_image_diff_demo).mean()
            world_model_sample_future_image_mse_loss = (future_image_diff_world_model**2).mean()
            world_model_sample_future_image_l1_loss = torch.abs(future_image_diff_world_model).mean()
            all_samples_future_image_mse_loss = (future_image_diff**2).mean()
            all_samples_future_image_l1_loss = torch.abs(future_image_diff).mean()
        else:
            demo_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        if torch.all(future_wrist_image_indices != -1):
            future_wrist_image_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_wrist_image_indices, :, :]
            )
            future_wrist_image_diff_demo = future_wrist_image_diff[rollout_data_mask == 0]
            future_wrist_image_diff_world_model = future_wrist_image_diff[world_model_sample_mask == 1]
            demo_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_demo**2).mean()
            demo_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_demo).mean()
            world_model_sample_future_wrist_image_mse_loss = (future_wrist_image_diff_world_model**2).mean()
            world_model_sample_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff_world_model).mean()
            all_samples_future_wrist_image_mse_loss = (future_wrist_image_diff**2).mean()
            all_samples_future_wrist_image_l1_loss = torch.abs(future_wrist_image_diff).mean()
        else:
            demo_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_wrist_image_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        if torch.all(future_proprio_indices != -1):
            future_proprio_diff = (
                vt_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
                - vt_pred_B_C_T_H_W[batch_indices, :, future_proprio_indices, :, :]
            )
            future_proprio_diff_demo = future_proprio_diff[rollout_data_mask == 0]
            future_proprio_diff_world_model = future_proprio_diff[world_model_sample_mask == 1]
            demo_sample_future_proprio_mse_loss = (future_proprio_diff_demo**2).mean()
            demo_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_demo).mean()
            world_model_sample_future_proprio_mse_loss = (future_proprio_diff_world_model**2).mean()
            world_model_sample_future_proprio_l1_loss = torch.abs(future_proprio_diff_world_model).mean()
            all_samples_future_proprio_mse_loss = (future_proprio_diff**2).mean()
            all_samples_future_proprio_l1_loss = torch.abs(future_proprio_diff).mean()
        else:
            demo_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            demo_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            world_model_sample_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_mse_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)
            all_samples_future_proprio_l1_loss = torch.tensor(float("nan"), device=x0_B_C_T_H_W.device)

        action_diff = (
            vt_B_C_T_H_W[batch_indices, :, action_indices, :, :]
            - vt_pred_B_C_T_H_W[batch_indices, :, action_indices, :, :]
        )
        action_diff_demo = action_diff[rollout_data_mask == 0]
        demo_sample_action_mse_loss = (action_diff_demo**2).mean()
        demo_sample_action_l1_loss = torch.abs(action_diff_demo).mean()
        all_samples_action_mse_loss = (action_diff**2).mean()
        all_samples_action_l1_loss = torch.abs(action_diff).mean()

        value_diff = (
            vt_B_C_T_H_W[batch_indices, :, value_indices, :, :]
            - vt_pred_B_C_T_H_W[batch_indices, :, value_indices, :, :]
        )
        value_diff_demo = value_diff[rollout_data_mask == 0]
        value_diff_world_model = value_diff[world_model_sample_mask == 1]
        value_diff_value_function = value_diff[value_function_sample_mask == 1]
        demo_sample_value_mse_loss = (value_diff_demo**2).mean()
        demo_sample_value_l1_loss = torch.abs(value_diff_demo).mean()
        world_model_sample_value_mse_loss = (value_diff_world_model**2).mean()
        world_model_sample_value_l1_loss = torch.abs(value_diff_world_model).mean()
        value_function_sample_value_mse_loss = (value_diff_value_function**2).mean()
        value_function_sample_value_l1_loss = torch.abs(value_diff_value_function).mean()
        all_samples_value_mse_loss = (value_diff**2).mean()
        all_samples_value_l1_loss = torch.abs(value_diff).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "timesteps": timesteps_B_T,
            "sigmas": sigmas_B_T,
            "condition": condition,
            "vt_pred": vt_pred_B_C_T_H_W,
            "vt_target": vt_B_C_T_H_W,
            "velocity_mse_loss": velocity_mse_B_C_T_H_W.mean(),
            "edm_loss": weighted_velocity_loss_B_C_T_H_W.mean(),
            "velocity_loss_per_frame": torch.mean(weighted_velocity_loss_B_C_T_H_W, dim=[1, 3, 4]),
            "demo_sample_action_mse_loss": demo_sample_action_mse_loss,
            "demo_sample_action_l1_loss": demo_sample_action_l1_loss,
            "demo_sample_future_proprio_mse_loss": demo_sample_future_proprio_mse_loss,
            "demo_sample_future_proprio_l1_loss": demo_sample_future_proprio_l1_loss,
            "demo_sample_future_wrist_image_mse_loss": demo_sample_future_wrist_image_mse_loss,
            "demo_sample_future_wrist_image_l1_loss": demo_sample_future_wrist_image_l1_loss,
            "demo_sample_future_image_mse_loss": demo_sample_future_image_mse_loss,
            "demo_sample_future_image_l1_loss": demo_sample_future_image_l1_loss,
            "demo_sample_value_mse_loss": demo_sample_value_mse_loss,
            "demo_sample_value_l1_loss": demo_sample_value_l1_loss,
            "world_model_sample_future_proprio_mse_loss": world_model_sample_future_proprio_mse_loss,
            "world_model_sample_future_proprio_l1_loss": world_model_sample_future_proprio_l1_loss,
            "world_model_sample_future_wrist_image_mse_loss": world_model_sample_future_wrist_image_mse_loss,
            "world_model_sample_future_wrist_image_l1_loss": world_model_sample_future_wrist_image_l1_loss,
            "world_model_sample_future_image_mse_loss": world_model_sample_future_image_mse_loss,
            "world_model_sample_future_image_l1_loss": world_model_sample_future_image_l1_loss,
            "world_model_sample_value_mse_loss": world_model_sample_value_mse_loss,
            "world_model_sample_value_l1_loss": world_model_sample_value_l1_loss,
            "value_function_sample_value_mse_loss": value_function_sample_value_mse_loss,
            "value_function_sample_value_l1_loss": value_function_sample_value_l1_loss,
        }
        return output_batch, weighted_velocity_loss_B_C_T_H_W

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float | None = None,
        use_variance_scale: bool = False,
        return_orig_clean_latent_frames: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate samples using rectified flow with Cosmos Policy extensions.
        """
        # Default shift to model config value (e.g. 5.0 for predict2.5, 1.0 for predict2)
        if shift is None:
            shift = float(self.config.shift)
        print(f"Using shift={shift} for rectified flow sampling (guidance={guidance}, use_variance_scale={use_variance_scale})")
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        orig_clean_latent_frames = None
        if return_orig_clean_latent_frames:
            _, orig_clean_latent_frames, _ = self.get_data_and_condition(data_batch)

        if use_variance_scale:
            torch.manual_seed(seed)
            shift_variance_scale = torch.rand(1).item() * 4.0 + 3.0
            effective_shift = shift * shift_variance_scale / 5.0
        else:
            effective_shift = shift

        noise = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.tensor_kwargs["device"],
            seed,
        )

        # Action noise in action space: replace action slots in initial noise
        # with encoder(action_space_noise) to match training distribution
        action_encoder = getattr(self, 'action_encoder', None)
        if action_encoder is not None and "action_latent_idx" in data_batch:
            action_indices = data_batch["action_latent_idx"]
            if torch.all(action_indices != -1):
                noise = inject_action_noise_into_epsilon(
                    noise, action_indices,
                    self.config.action_dim, self.config.chunk_size, action_encoder,
                )

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        # sample_clean: run (num_steps-1) ODE steps + 1 final clean x0 prediction
        # Mirrors CosmosPolicySampler behaviour for the EDM model.
        sample_clean = num_steps > 1
        n_ode_steps = num_steps - 1 if sample_clean else num_steps

        self.sample_scheduler.set_timesteps(
            n_ode_steps,
            device=self.tensor_kwargs["device"],
            shift=effective_shift,
            use_kerras_sigma=self.config.use_kerras_sigma_at_inference,
        )

        timesteps = self.sample_scheduler.timesteps
        velocity_fn = self.get_velocity_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())

        latents = noise

        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        # Final clean step: directly predict x0 at the last scheduled sigma.
        # x0 = xt - sigma * velocity  (rectified-flow identity)
        if sample_clean:
            t_clean = timesteps[-1]
            velocity_clean = velocity_fn(noise, latents, t_clean.unsqueeze(0).unsqueeze(0))
            sigma_clean = t_clean.float() / 1000.0
            latents = latents - sigma_clean * velocity_clean

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        if return_orig_clean_latent_frames:
            return latents, orig_clean_latent_frames
        else:
            return latents


# =============================================================================
# Video2World Policy Model (with video conditioning)
# =============================================================================


@attrs.define(slots=False)
class CosmosPolicyVideo2WorldConfigRectifiedFlow(CosmosPolicyModelConfigRectifiedFlow):
    """
    Extended config for Cosmos Policy Video2World model using rectified flow.
    Adds video conditioning parameters on top of CosmosPolicyModelConfigRectifiedFlow.
    """

    min_num_conditional_frames: int = 1
    max_num_conditional_frames: int = 2
    conditional_frame_timestep: float = -1.0  # Noise level for conditional frames; -1 means not effective
    conditioning_strategy: str = str(ConditioningStrategy.FRAME_REPLACE)
    denoise_replace_gt_frames: bool = True
    conditional_frames_probs: Optional[Dict[int, float]] = None
    cond_dropout_warmup_steps: int = 0  # 0 = no warmup dropout

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert self.conditioning_strategy in [
            str(ConditioningStrategy.FRAME_REPLACE),
        ]


class CosmosPolicyVideo2WorldModelRectifiedFlow(CosmosPolicyDiffusionModelRectifiedFlow):
    """
    Cosmos Policy Video2World Model using rectified flow.

    Extends CosmosPolicyDiffusionModelRectifiedFlow with video conditioning:
    - Video frame conditioning via gt_frames
    - Policy-specific mask manipulation for world model/value function
    - Input masking for different prediction modes (V(s'), Q(s,a))
    """

    def __init__(self, config: CosmosPolicyVideo2WorldConfigRectifiedFlow):
        super().__init__(config)
        self.config: CosmosPolicyVideo2WorldConfigRectifiedFlow = config

    def _mask_latent_frame(
        self,
        condition_mask: Tensor,
        batch_indices: Tensor,
        latent_idx: Tensor,
        mask_value: float,
        sample_mask: Tensor | None = None,
    ) -> None:
        """Apply mask to a specific latent frame index."""
        if sample_mask is not None:
            condition_mask[batch_indices, :, latent_idx, :, :] = torch.where(
                sample_mask[:, :, 0, :, :].bool(),
                torch.full_like(condition_mask[batch_indices, :, latent_idx, :, :], mask_value),
                condition_mask[batch_indices, :, latent_idx, :, :],
            )
        else:
            condition_mask[batch_indices, :, latent_idx, :, :] = mask_value

    def _apply_current_state_action_masks(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
        sample_mask: Tensor | None = None,
    ) -> None:
        """Mask out current state and action for V(s') prediction."""
        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]
        batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)

        if torch.all(data_batch["current_proprio_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_proprio_latent_idx"],
                0,
                sample_mask,
            )
        if torch.all(data_batch["current_wrist_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_wrist_image_latent_idx"],
                0,
                sample_mask,
            )
        if "current_wrist_image2_latent_idx" in data_batch and torch.all(
            data_batch["current_wrist_image2_latent_idx"] != -1
        ):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_wrist_image2_latent_idx"],
                0,
                sample_mask,
            )
        if torch.all(data_batch["current_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_image_latent_idx"],
                0,
                sample_mask,
            )
        if "current_image2_latent_idx" in data_batch and torch.all(data_batch["current_image2_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["current_image2_latent_idx"],
                0,
                sample_mask,
            )
        self._mask_latent_frame(
            condition.condition_video_input_mask_B_C_T_H_W,
            batch_indices,
            data_batch["action_latent_idx"],
            0,
            sample_mask,
        )

    def _apply_future_state_masks(
        self,
        condition: Video2WorldCondition,
        data_batch: dict[str, torch.Tensor],
        sample_mask: Tensor | None = None,
    ) -> None:
        """Mask out future state for Q(s, a) prediction."""
        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]
        batch_indices = torch.arange(B, device=condition.condition_video_input_mask_B_C_T_H_W.device)

        if torch.all(data_batch["future_proprio_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_proprio_latent_idx"],
                0,
                sample_mask,
            )
        if torch.all(data_batch["future_wrist_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_wrist_image_latent_idx"],
                0,
                sample_mask,
            )
        if "future_wrist_image2_latent_idx" in data_batch and torch.all(
            data_batch["future_wrist_image2_latent_idx"] != -1
        ):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_wrist_image2_latent_idx"],
                0,
                sample_mask,
            )
        if torch.all(data_batch["future_image_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_image_latent_idx"],
                0,
                sample_mask,
            )
        if "future_image2_latent_idx" in data_batch and torch.all(data_batch["future_image2_latent_idx"] != -1):
            self._mask_latent_frame(
                condition.condition_video_input_mask_B_C_T_H_W,
                batch_indices,
                data_batch["future_image2_latent_idx"],
                0,
                sample_mask,
            )

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, Video2WorldCondition]:
        """
        Extended get_data_and_condition with video conditioning and policy-specific logic.
        """
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)

        # Set video conditioning
        condition = condition.set_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        if "rollout_data_mask" in data_batch:
            world_model_sample_mask = data_batch["world_model_sample_mask"]
            value_function_sample_mask = data_batch["value_function_sample_mask"]
            H_latent, W_latent = condition.condition_video_input_mask_B_C_T_H_W.shape[-2:]
            world_model_sample_mask = (
                world_model_sample_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)
            value_function_sample_mask = (
                value_function_sample_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                .expand(-1, 1, 1, H_latent, W_latent)
            ).to(condition.condition_video_input_mask_B_C_T_H_W.dtype)

            batch_indices = torch.arange(world_model_sample_mask.shape[0], device=world_model_sample_mask.device)

            # World model: set action frame mask to 1
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, data_batch["action_latent_idx"], :, :] = (
                world_model_sample_mask[:, :, 0, :, :]
            )

            # Value function: set all frames to 1 except the value frame
            T = condition.condition_video_input_mask_B_C_T_H_W.shape[2]
            value_mask_all_frames = value_function_sample_mask.expand(-1, -1, T, -1, -1)
            condition.condition_video_input_mask_B_C_T_H_W = torch.where(
                value_mask_all_frames.bool(),
                torch.ones_like(condition.condition_video_input_mask_B_C_T_H_W),
                condition.condition_video_input_mask_B_C_T_H_W,
            )
            if torch.all(data_batch["value_latent_idx"] != -1):
                condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, data_batch["value_latent_idx"], :, :] = (
                    torch.where(
                        value_function_sample_mask[:, :, 0, :, :].bool(),
                        torch.zeros_like(
                            condition.condition_video_input_mask_B_C_T_H_W[
                                batch_indices, :, data_batch["value_latent_idx"], :, :
                            ]
                        ),
                        condition.condition_video_input_mask_B_C_T_H_W[
                            batch_indices, :, data_batch["value_latent_idx"], :, :
                        ],
                    )
                )

            if self.config.mask_current_state_action_for_value_prediction:
                self._apply_current_state_action_masks(condition, data_batch, sample_mask=value_function_sample_mask)
            if self.config.mask_future_state_for_qvalue_prediction:
                self._apply_future_state_masks(condition, data_batch, sample_mask=value_function_sample_mask)

            condition.orig_gt_frames = condition.gt_frames.clone()
            condition.gt_frames = replace_latent_with_action_chunk(
                condition.gt_frames, data_batch["actions"], action_indices=data_batch["action_latent_idx"]
            )

        # Add proprio to gt_frames
        if "proprio" in data_batch and torch.all(data_batch["current_proprio_latent_idx"] != -1):
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["proprio"],
                proprio_indices=data_batch["current_proprio_latent_idx"],
            )
        if "future_proprio" in data_batch and torch.all(data_batch["future_proprio_latent_idx"] != -1):
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["future_proprio"],
                proprio_indices=data_batch["future_proprio_latent_idx"],
            )

        # Add value to gt_frames
        if torch.all(data_batch["value_latent_idx"] != -1) and "value_function_return" in data_batch:
            batch_indices = torch.arange(condition.gt_frames.shape[0], device=condition.gt_frames.device)
            _, C_latent, _, H_latent, W_latent = condition.gt_frames.shape
            condition.gt_frames[batch_indices, :, data_batch["value_latent_idx"], :, :] = (
                data_batch["value_function_return"]
                .reshape(-1, 1, 1, 1)
                .expand(-1, C_latent, H_latent, W_latent)
                .to(condition.gt_frames.dtype)
            )

        return raw_state, latent_state, condition

    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Text2WorldCondition,
    ) -> torch.Tensor:
        """Denoise with video conditioning support."""
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                timesteps_B_1_1_1_1 = timesteps_B_T.view(timesteps_B_T.shape[0], 1, 1, 1, 1)
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_1_1_1_1 * (
                    1 - condition_video_mask_B_1_T_1_1
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze(dim=(1, 3, 4))

        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            **condition.to_dict(),
        ).float()

        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """Generates a callable velocity function from data batch for inference."""
        del guidance, is_negative_prompt  # No CFG for cosmos policy

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        condition, _ = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)

        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)

        condition.orig_gt_frames = condition.gt_frames.clone()

        B = condition.condition_video_input_mask_B_C_T_H_W.shape[0]

        if "proprio" in data_batch and torch.all(data_batch["current_proprio_latent_idx"] != -1):
            proprio = data_batch["proprio"]
            current_proprio_latent_idx = data_batch["current_proprio_latent_idx"]
            batch_indices = torch.arange(B, device=proprio.device)
            condition.condition_video_input_mask_B_C_T_H_W[batch_indices, :, current_proprio_latent_idx, :, :] = 1
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames, proprio, proprio_indices=current_proprio_latent_idx,
                encoder=getattr(self, 'proprio_encoder', None),
            )

        if (
            "mask_current_state_action_for_value_prediction" in data_batch
            and data_batch["mask_current_state_action_for_value_prediction"]
        ):
            self._apply_current_state_action_masks(condition, data_batch, sample_mask=None)

        if (
            "mask_future_state_for_qvalue_prediction" in data_batch
            and data_batch["mask_future_state_for_qvalue_prediction"]
        ):
            self._apply_future_state_masks(condition, data_batch, sample_mask=None)

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.denoise(noise, noise_x, timestep, condition)

        return velocity_fn
