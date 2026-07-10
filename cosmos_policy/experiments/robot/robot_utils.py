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

"""Utils for evaluating robot policies in various environments."""

import logging
import os
import time
from typing import Any, List, Optional, Tuple

import imageio
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "cosmos": 224,
}


def get_image_resize_size(model_family: str) -> int:
    """
    Get image resize dimensions for a specific model (assumes square images).

    Args:
        model_family: Model family name (e.g., "cosmos")

    Returns:
        int: Image resize dimension

    Raises:
        ValueError: If model family is not supported
    """
    if model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {model_family}")

    return MODEL_IMAGE_SIZES[model_family]


def setup_logging(
    cfg: Any,
    task_identifier: str,
    log_dir: str,
    run_id_note: Optional[str] = None,
    use_wandb: bool = True,
    wandb_entity: str = "nvidia-dir",
    wandb_project: str = "cosmos_policy_eval",
    extra_wandb_tags: Optional[list] = None,
) -> Tuple[Any, str, str]:
    """
    Set up logging to file and optionally to wandb.

    Args:
        cfg: Configuration object with model parameters
        task_identifier: Task/suite identifier (e.g., "libero_spatial", "PnPCounterToCab")
        log_dir: Local directory for log files
        run_id_note: Optional note to append to run ID
        use_wandb: Whether to log to Weights & Biases
        wandb_entity: WandB entity name
        wandb_project: WandB project name
        extra_wandb_tags: Optional additional tags for WandB run

    Returns:
        Tuple of (log_file, local_log_filepath, run_id)
    """
    # Create run ID
    data_collection = getattr(cfg, "data_collection", False)
    model_family = getattr(cfg, "model_family", "cosmos")

    if data_collection:
        run_id = f"DATA_COLLECTION-{task_identifier}-{DATE_TIME}"
    else:
        run_id = f"ENV_EVAL-{task_identifier}-{model_family}-{DATE_TIME}"

    if run_id_note is not None:
        run_id += f"--{run_id_note}"

    # Set up local logging
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if use_wandb:
        tags = [task_identifier]
        if extra_wandb_tags:
            tags.extend(extra_wandb_tags)

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_id,
            tags=tags,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """
    Log a message to console and optionally to a log file.

    Args:
        message: Message to log
        log_file: Optional file handle to write to
    """
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def save_rollout_video_with_future_image_predictions(
    rollout_images: List[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    chunk_size: int,
    num_open_loop_steps: int,
    output_dir: Optional[str] = None,
    rollout_wrist_images: Optional[List[np.ndarray]] = None,
    future_primary_image_predictions: Optional[List[np.ndarray]] = None,
    future_wrist_image_predictions: Optional[List[np.ndarray]] = None,
    show_diff: bool = False,
    log_file=None,
):
    """Saves an MP4 replay of an episode with future image predictions shown alongside real frames.

    Args:
        rollout_images: List of primary camera images from the episode.
        idx: Episode index.
        success: Whether the episode was successful.
        task_description: Text description of the task.
        chunk_size: Number of timesteps per future prediction (used for labeling).
        num_open_loop_steps: How many steps each predicted future image spans.
        output_dir: Directory where the MP4 file will be saved.
        rollout_wrist_images: Optional list of wrist camera images.
        future_primary_image_predictions: Optional list of predicted future primary images.
        future_wrist_image_predictions: Optional list of predicted future wrist images.
        show_diff: If True, add a difference column between real and predicted images.
        log_file: Optional file handle for logging the save path.
    """
    if not future_primary_image_predictions and not future_wrist_image_predictions:
        return None

    if output_dir is None:
        output_dir = os.path.join(".", "rollouts", DATE)
    os.makedirs(output_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:35]
    mp4_path = os.path.join(
        output_dir,
        f"{DATE_TIME}--with_future_img--episode={idx}--success={success}--task={processed_task_description}.mp4",
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)

    # Determine availability of predictions
    has_primary_predictions = future_primary_image_predictions is not None and len(future_primary_image_predictions) > 0
    has_wrist_replay = rollout_wrist_images is not None
    has_wrist_predictions = has_wrist_replay and (
        future_wrist_image_predictions is not None and len(future_wrist_image_predictions) > 0
    )

    # Determine target dimensions to use for resizing
    if has_primary_predictions:
        target_h, target_w, c = future_primary_image_predictions[0].shape
    elif has_wrist_predictions:
        target_h, target_w, c = future_wrist_image_predictions[0].shape
    else:
        target_h, target_w, c = rollout_images[0].shape

    # Define text parameters
    text_height = 60
    font_size = 18

    # Define column labels
    if show_diff:
        column_labels = []
        if has_wrist_replay:
            column_labels.append("replay wrist")
        if has_wrist_predictions:
            column_labels.append("future wrist")
            column_labels.append("wrist difference")
        column_labels.append("replay primary")
        if has_primary_predictions:
            column_labels.append("future primary")
            column_labels.append("primary difference")
    else:
        column_labels = []
        if has_wrist_replay:
            column_labels.append("real wrist image")
        if has_wrist_predictions:
            column_labels.append("predicted wrist image")
        column_labels.append("real current image")
        if has_primary_predictions:
            column_labels.append("predicted future image")
    num_columns = len(column_labels)

    image_iterator = zip(rollout_images, rollout_wrist_images) if has_wrist_replay else rollout_images

    for i, image_data in enumerate(image_iterator):
        if has_wrist_replay:
            img, wrist_img = image_data
        else:
            img = image_data

        # Resize rollout images to match target dimensions if needed
        if img.shape[:2] != (target_h, target_w):
            img = np.array(Image.fromarray(img).resize((target_w, target_h), Image.LANCZOS))
        if has_wrist_replay and wrist_img.shape[:2] != (target_h, target_w):
            wrist_img = np.array(Image.fromarray(wrist_img).resize((target_w, target_h), Image.LANCZOS))

        # Select future prediction for this timestep
        future_idx = i // num_open_loop_steps
        future_img = None
        future_wrist_img = None
        if has_primary_predictions:
            future_img = future_primary_image_predictions[min(future_idx, len(future_primary_image_predictions) - 1)]
        if has_wrist_predictions:
            future_wrist_img = future_wrist_image_predictions[min(future_idx, len(future_wrist_image_predictions) - 1)]

        # Compute difference images if requested
        if show_diff:
            if has_primary_predictions and future_img is not None:
                primary_diff = np.clip(np.abs(img.astype(np.float32) - future_img.astype(np.float32)), 0, 255).astype(np.uint8)
            if has_wrist_predictions and future_wrist_img is not None:
                wrist_diff = np.clip(np.abs(wrist_img.astype(np.float32) - future_wrist_img.astype(np.float32)), 0, 255).astype(np.uint8)

        # Build combined image
        combined_img = np.zeros((target_h, target_w * num_columns, c), dtype=np.uint8)
        col = 0
        if show_diff:
            if has_wrist_replay:
                combined_img[:, target_w * col: target_w * (col + 1)] = wrist_img; col += 1
            if has_wrist_predictions and future_wrist_img is not None:
                combined_img[:, target_w * col: target_w * (col + 1)] = future_wrist_img; col += 1
                combined_img[:, target_w * col: target_w * (col + 1)] = wrist_diff; col += 1
            combined_img[:, target_w * col: target_w * (col + 1)] = img; col += 1
            if has_primary_predictions and future_img is not None:
                combined_img[:, target_w * col: target_w * (col + 1)] = future_img; col += 1
                combined_img[:, target_w * col: target_w * (col + 1)] = primary_diff; col += 1
        else:
            if has_wrist_replay:
                combined_img[:, target_w * col: target_w * (col + 1)] = wrist_img; col += 1
            if has_wrist_predictions and future_wrist_img is not None:
                combined_img[:, target_w * col: target_w * (col + 1)] = future_wrist_img; col += 1
            combined_img[:, target_w * col: target_w * (col + 1)] = img; col += 1
            if has_primary_predictions and future_img is not None:
                combined_img[:, target_w * col: target_w * (col + 1)] = future_img; col += 1

        # Build text label area
        text_area = np.ones((text_height, target_w * num_columns, 3), dtype=np.uint8) * 255
        text_img = Image.fromarray(text_area)
        draw = ImageDraw.Draw(text_img)
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans", font_size)
            except IOError:
                font = ImageFont.load_default()

        for col_idx, label in enumerate(column_labels):
            x_pos = col_idx * target_w + target_w // 2
            text_width = draw.textlength(label, font=font)
            draw.text((x_pos - text_width // 2, 8), label, font=font, fill=(0, 0, 0))
            if ("predicted" in label) or ("future" in label):
                k_text = f"(K={chunk_size} timesteps)"
                k_text_width = draw.textlength(k_text, font=font)
                draw.text((x_pos - k_text_width // 2, 35), k_text, font=font, fill=(0, 0, 0))

        timestep_text = f"t = {i}"
        timestep_text_width = draw.textlength(timestep_text, font=font)
        center_x = (target_w * num_columns) // 2
        draw.text((center_x - timestep_text_width // 2, 36), timestep_text, font=font, fill=(255, 0, 0))

        video_writer.append_data(np.vstack((np.array(text_img), combined_img)))

    video_writer.close()
    log_message(f"Saved rollout MP4 at path {mp4_path}", log_file)
    log_file.flush()
    return mp4_path
        
