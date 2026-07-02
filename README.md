# ReCAP-RAG-Policy — Retrieval-Augmented Cosmos Policy on PushT

Retrieval-augmented visuomotor policy for the **PushT** benchmark, built on
[Cosmos Policy](https://github.com/NVlabs/cosmos-policy) and the
[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) video backbone.

At each step the policy retrieves the most similar state from a demonstration pool and
conditions a rectified-flow video-world model on the retrieved **future frames**,
**action chunk**, and **state**; it then predicts a **residual (delta)** on top of the
retrieved action chunk. See [NOTICE.md](NOTICE.md) for attribution and
[DATA_MANIFEST.md](DATA_MANIFEST.md) for the exact HF file lists.

> **Project page:** https://recap-robot.github.io/

> Released artifacts:
> - **Checkpoint** (model repo): [`Jeongeun/ReCAP-Cosmos2.5-pusht`](https://huggingface.co/Jeongeun/ReCAP-Cosmos2.5-pusht)
> - **Dataset** (dataset repo): [`Jeongeun/ReCAP-Cosmos2.5-pusht`](https://huggingface.co/datasets/Jeongeun/ReCAP-Cosmos2.5-pusht)

## 1. Setup

Follow [SETUP.md](SETUP.md), then install PushT deps (Python 3.10):

```bash
uv sync --extra cu128 --group pusht
```

## 2. Download dataset & checkpoint (Hugging Face)

```bash
pip install -U huggingface_hub          # provides the `hf` CLI
export BASE_DATASETS_DIR=/path/to/data  # dataset root

# Dataset  (success_only/ pools + stats + retrieval npz)
hf download Jeongeun/ReCAP-Cosmos2.5-pusht --repo-type dataset \
  --local-dir "$BASE_DATASETS_DIR/PushT-Cosmos-Policy"

# Checkpoint  (model_000007000.pt + bundled stats)
hf download Jeongeun/ReCAP-Cosmos2.5-pusht --repo-type model \
  --local-dir ./checkpoints
```

## 3. Evaluation

Runs the residual RAG policy across 9 visual configs (live retrieval from the demo pool —
no precomputed retrieval files needed at eval time):

```bash
CKPT=./checkpoints/model_000007000.pt BASE_DATASETS_DIR=$BASE_DATASETS_DIR ./eval_pusht_rag.sh
```

<details><summary>Single config, manually</summary>

```bash
DATA=$BASE_DATASETS_DIR/PushT-Cosmos-Policy/success_only
uv run --extra cu128 --group pusht --python 3.10 \
  -m cosmos_policy.experiments.robot.pusht_ret.run_eval \
  --config cosmos_predict2p5_2b_480p_pusht_ret_top100_residual_inference_only \
  --ckpt_path ./checkpoints/model_000007000.pt \
  --config_file cosmos_policy/config/config.py \
  --t5_text_embeddings_path "$DATA/t5_embeddings.pkl" \
  --dataset_stats_path      "$DATA/dataset_statistics.json" \
  --retrieval_data_dir      "$DATA" \
  --use_residual_actions True --delta_stats_path "$DATA/delta_dataset_statistics.json" \
  --visual_config tri_default \
  --num_trials 50 --chunk_size 8 --num_open_loop_steps 8 \
  --num_denoising_steps_action 5 --predict_future_states True --seed 42
```
</details>

Success = ≥85% block coverage within 300 steps. Each episode saves an MP4
(real | generated-future | retrieved).

**Results** (50 trials/config, iter 7000, seed 42):

| visual_config | success | | visual_config | success |
|---|---|---|---|---|
| tri_default      | **60.0%** | | tri_rot15  | 36.0% |
| tri_rot30        | 48.0% | | tri_rot60  | 34.0% |
| tri_rot-30       | 44.0% | | tri_rot0   | 28.0% |
| tri_goal_flipped | 40.0% | | tri_rot-60 | 28.0% |
|                  |       | | tri_rot-15 | 26.0% |

## 4. Training

Fine-tunes the residual RAG policy (8×80 GB GPUs, 7000 iters). Requires the training
splits + precomputed retrieval `.npz` (included in the dataset repo; see DATA_MANIFEST.md):

```bash
BASE_DATASETS_DIR=$BASE_DATASETS_DIR ./train_and_eval_pusht_rag.sh   # train → DCP→.pt → eval
```

Key config: `cosmos_predict2p5_2b_480p_pusht_ret_top100_residual`
(`cosmos_policy/config/experiment/pusht_experiment_configs.py`) — top-100 episodes/task,
top-1 retrieval, residual delta target, `state_t=10`, `chunk_size=8`. The Predict2.5 video
backbone weights download automatically from the public `nvidia/Cosmos-Predict2.5-2B` HF repo.

---

<sub>The remainder of this README is the upstream Cosmos Policy documentation, preserved.</sub>

---

# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

<p align="center">
  <a href="https://arxiv.org/abs/2601.16163">Paper</a>&nbsp | <a href="https://research.nvidia.com/labs/dir/cosmos-policy/">Project Website</a>&nbsp | 🤗 <a href="https://huggingface.co/collections/nvidia/cosmos-policy">Models & Training Data</a>&nbsp | <a href="https://youtu.be/V2qdFD9n5BM">Summary Video</a>
</p>

## System Requirements

Inference with base Cosmos Policy only (i.e., no model-based planning):
* 1 GPU with 6.8 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with 8.9 GB VRAM for RoboCasa sim benchmark tasks
* 1 GPU with 6.0 GB VRAM for ALOHA robot tasks

Inference with Cosmos Policy + model-based planning (best-of-N search) on ALOHA robot tasks:
* Minimum (serial inference): 1 GPU with 10.0 GB VRAM
* Recommended (parallel inference): N GPUs with 10.0 GB VRAM each

Training:
* Generally, it is recommended to have at least 1 node of 8 80GB GPUs. For the experiments in the Cosmos Policy paper, we used 8 80GB GPUs (H100s) for 48 hours for small-scale ALOHA robot data fine-tuning (<200 demos), 32 80GB GPUs (H100s) for 48 hours for RoboCasa training (1200 demos), and 64 80GB GPUs (H100s) for 48 hours for LIBERO training (2000 demos). If you have fewer GPUs, you can use gradient accumulation to increase total batch size, which we found leads to faster convergence than taking more gradient steps with a smaller batch size.

## Quick Start

First, set up a Docker container following the instructions in [SETUP.md](SETUP.md).

Then, inside the Docker container, enter a Python shell via: `uv run --extra cu128 --group libero --python 3.10 python`.

Then, run the Python code below to generate (1) robot actions, (2) predicted future state (represented by robot proprioception and future image observations), and (3) future state value (expected cumulative rewards):

```python
import pickle
import torch
from PIL import Image
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    load_dataset_stats,
    init_t5_text_embeddings_cache,
    get_t5_embedding_from_cache,
)

# Instantiate config (see PolicyEvalConfig in cosmos_policy/experiments/robot/libero/run_libero_eval.py for definitions)
cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only",
    ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=16,
    num_open_loop_steps=16,
    trained_with_image_aug=True,
    use_jpeg_compression=True,
    flip_images=True,  # Only for LIBERO; images render upside-down
    num_denoising_steps_action=5,
    num_denoising_steps_future_state=1,
    num_denoising_steps_value=1,
)
# Load dataset stats for action/proprio scaling
dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
# Initialize T5 text embeddings cache
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
# Load model
model, cosmos_config = get_model(cfg)
# Load sample observation:
#   observation (dict): {
#     "primary_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "proprio": robot proprioceptive state,
#   }
with open("cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl", "rb") as file:
    observation = pickle.load(file)
    task_description = "put both the alphabet soup and the tomato sauce in the basket"
# Generate robot actions, future state (proprio + images), and value
action_return_dict = get_action(
    cfg,
    model,
    dataset_stats,
    observation,
    task_description,
    num_denoising_steps_action=cfg.num_denoising_steps_action,
    generate_future_state_and_value_in_parallel=True,
)
# Print actions
print(f"Generated action chunk: {action_return_dict['actions']}")
# Save future image predictions (third-person image and wrist image)
img_path1, img_path2 = "future_image.png", "future_wrist_image.png"
Image.fromarray(action_return_dict['future_image_predictions']['future_image']).save(img_path1)
Image.fromarray(action_return_dict['future_image_predictions']['future_wrist_image']).save(img_path2)
print(f"Saved future image predictions to:\n\t{img_path1}\n\t{img_path2}")
# Print value
print(f"Generated value: {action_return_dict['value_prediction']}")
```

If you run into runtime errors, you may need to enter the Python shell via `uv run   --extra cu128   --group libero   --python 3.10   python` before running the code above.

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the environment.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ROBOCASA.md](ROBOCASA.md) for fine-tuning/evaluating on RoboCasa simulation benchmark tasks.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

If you run into any issues, please open a new GitHub issue. For critical blocking issues, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2601.16163):

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
