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

"""Debug-sample callback that always writes a LOCAL mp4 of input vs output.

Subclasses ``EveryNDrawSample`` (which already runs generate_samples_from_batch
+ decode and builds ``to_show = [generated(per guidance), GT]``) and overrides
``run_save`` to dump the full video to ``{job}/EgoVerseDebugSample/`` locally —
top rows = generated, bottom row = GT input layout, columns = batch.

Wire it into an experiment via trainer.callbacks; fires every ``every_n`` steps.
"""
from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from cosmos_policy._src.imaginaire.visualize.video import save_img_or_video
from cosmos_policy._src.predict2.callbacks.every_n_draw_sample import EveryNDrawSample


class EgoVerseDebugSample(EveryNDrawSample):
    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        # to_show: list of [b, c, t, h, w] — generated(per guidance) then GT.
        # Save the full video locally under a distinct *_video name (the real debug
        # artifact), then defer to the parent for its resized jpg, whose path is
        # returned so the parent's wandb.Image logging stays valid.
        stacked = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0   # [n, b, c, t, h, w]
        if stacked.shape[3] > 1:   # multi-frame -> dump mp4 (rows=guidance+GT, cols=batch)
            vid = rearrange(stacked, "n b c t h w -> c t (n h) (b w)")
            save_img_or_video(vid, f"{self.local_dir}/{base_fp_wo_ext}_video", fps=self.fps)
        return super().run_save(to_show, batch_size, base_fp_wo_ext)
