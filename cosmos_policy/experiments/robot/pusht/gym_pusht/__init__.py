# SPDX-License-Identifier: MIT
#
# PushT environment adapted from gym-pusht (https://github.com/huggingface/gym-pusht)
# and Diffusion Policy (https://github.com/real-stanford/diffusion_policy), both MIT
# licensed. Vendored and adapted for the ReCAP project. See NOTICE.md for attribution.

from gymnasium.envs.registration import register

register(
    id="gym_pusht/PushT-v0",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)
