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
Important constants for Cosmos Policy training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.

Adapted from: https://github.com/user/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import sys

# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 9,
}

ROBOCASA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 32,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 9,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
}

PUSHT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 2,
    "PROPRIO_DIM": 2,
}

# RoboTwin: bimanual absolute end-effector control in native format, no conversion.
# Per arm: xyz(3) + quat_wxyz(4) + gripper(1) = 8 dim. Two arms = 16 dim.
ROBOTWIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,
    "ACTION_DIM": 16,
    "PROPRIO_DIM": 16,
}

# RoboTwin (qpos): joint-space control. 6-DOF aloha-agilex bimanual.
# Per arm: arm(6) + gripper(1) = 7 dim. Two arms = 14 dim.
# Layout: [L_arm(6), L_grip(1), R_arm(6), R_grip(1)] — what
# RoboTwin's take_action(..., action_type='qpos') consumes.
ROBOTWIN_QPOS_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
}

# Real-robot bg2 bimanual (ai_worker FFW BG2): single third-eye camera,
# absolute end-effector pose in rot6d. Per arm: xyz(3) + rot6d(6) + grip(1) = 10
# dim. Two arms = 20 dim.
# Layout: [L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)] —
# matches RealRobotDataset / convert_real_robot_npz_to_h5.py STATE_ACTION_LAYOUT.
REAL_ROBOT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 20,
    "PROPRIO_DIM": 20,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    # Check robotwin_qpos before robotwin: the qpos sentinel contains "robotwin"
    # as a substring. Then check robotwin before aloha: the RoboTwin bimanual
    # embodiment name ("aloha-agilex") contains "aloha" as a substring.
    # Check real_robot/realrobot first so config names like
    # "cosmos_predict2_2b_480p_realrobot_bg2" route here rather than the
    # default LIBERO fallback.
    if "realrobot" in cmd_args or "real_robot" in cmd_args:
        return "REAL_ROBOT"
    elif "robotwin_qpos" in cmd_args:
        return "ROBOTWIN_QPOS"
    elif "robotwin" in cmd_args:
        return "ROBOTWIN"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "robocasa" in cmd_args:
        return "ROBOCASA"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "pusht" in cmd_args:
        return "PUSHT"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ROBOCASA":
    constants = ROBOCASA_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "PUSHT":
    constants = PUSHT_CONSTANTS
elif ROBOT_PLATFORM == "ROBOTWIN":
    constants = ROBOTWIN_CONSTANTS
elif ROBOT_PLATFORM == "ROBOTWIN_QPOS":
    constants = ROBOTWIN_QPOS_CONSTANTS
elif ROBOT_PLATFORM == "REAL_ROBOT":
    constants = REAL_ROBOT_CONSTANTS
# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print("If needed, manually set the correct constants in `projects/cosmos/cosmos_policy/constants.py`!")
