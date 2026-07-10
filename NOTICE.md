# Attribution

This repository is a fork / derivative work and is distributed under the Apache
License 2.0 (see [LICENSE](LICENSE)). It combines code from the upstream projects below
with original additions (the PushT environment integration and the retrieval-augmented
"PushT-RAG" dataset, policy model, and evaluation).

## Upstream sources

- **Cosmos Policy** — https://github.com/NVlabs/cosmos-policy
  Policy framework (training/eval harness, base policy models, config system).

- **Cosmos-Predict2.5** — https://github.com/nvidia-cosmos/cosmos-predict2.5
  Video-world backbone under `cosmos_policy/_src/` (Predict2.5 rectified-flow models,
  tokenizers, schedulers). Backbone weights are downloaded from the public
  `nvidia/Cosmos-Predict2.5-2B` Hugging Face repo at runtime.

Copyright for the upstream code remains with NVIDIA CORPORATION & AFFILIATES.
See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for third-party license texts.

## Third-party components

- **gym-pusht / Diffusion Policy** (both MIT) — the PushT simulation environment under
  `cosmos_policy/experiments/robot/pusht/gym_pusht/` is adapted from
  [gym-pusht](https://github.com/huggingface/gym-pusht) and
  [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
- **pymunk** (MIT, © 2007–2016 Victor Blomqvist) — `gym_pusht/envs/pymunk_override.py`
  retains its original upstream copyright header.

## Per-file attribution headers

Every source file carries a header indicating its provenance:

- **NVIDIA-authored files** keep their original NVIDIA SPDX Apache-2.0 header. Files we
  modified additionally carry a `Modifications Copyright (c) 2026 Jeongeun Park et al.
  (ReCAP)` notice appended below the original header (per Apache-2.0 §4(b)).
- **ReCAP-authored files** (new in this fork) carry an
  `SPDX-FileCopyrightText: Copyright (c) 2026 Jeongeun Park et al. (ReCAP)` Apache-2.0 header.
- **Third-party files** retain / reference their original license (MIT for gym-pusht,
  Diffusion Policy, and pymunk).

## Original additions in this fork

Copyright (c) 2026 Jeongeun Park et al. (ReCAP), released under Apache-2.0:

- PushT environment + evaluation: `cosmos_policy/experiments/robot/pusht/`
- Retrieval-augmented (RAG) dataset / model / eval:
  - `cosmos_policy/datasets/pusht_dataset_ret.py`
  - `cosmos_policy/models/policy_video2world_model_pusht_ret.py`
  - `cosmos_policy/experiments/robot/pusht_ret/`
- PushT experiment configs: `cosmos_policy/config/experiment/pusht_experiment_configs.py`

## Citation

If you use this code, please cite ReCAP:

```bibtex
@article{park2026retrieve,
  title={Retrieve, Don't Retrain: Extending Vision Language Action Models to New Tasks at Test Time},
  author={Park, Jeongeun and Park, Juhan and Kim, Taekyung and Choi, Sungjoon and Han, Dongyoon and Yun, Sangdoo},
  journal={arXiv preprint arXiv:2606.15631},
  year={2026}
}
```

and the upstream Cosmos Policy work it builds on:

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
