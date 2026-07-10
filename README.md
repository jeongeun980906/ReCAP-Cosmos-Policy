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

## Built on

This repository is a fork of **[Cosmos Policy](https://github.com/NVlabs/cosmos-policy)**
(Apache-2.0) and uses the **[Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)**
video backbone. On top of it we add the PushT environment integration and the
retrieval-augmented **ReCAP** dataset, policy model, and evaluation. Files we added or
modified carry per-file attribution headers; see **[NOTICE.md](NOTICE.md)** for the full
attribution and licensing details.

For base Cosmos Policy usage (LIBERO / RoboCasa / ALOHA fine-tuning and evaluation, the
base-policy quick start, and system requirements), refer to the upstream
[Cosmos Policy](https://github.com/NVlabs/cosmos-policy) repository.

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

This work builds on Cosmos Policy; please also cite it:

```bibtex
@article{kim2026cosmos,
  title={Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
  author={Kim, Moo Jin and Gao, Yihuai and Lin, Tsung-Yi and Lin, Yen-Chen and Ge, Yunhao and Lam, Grace and Liang, Percy and Song, Shuran and Liu, Ming-Yu and Finn, Chelsea and Gu, Jinwei},
  journal={arXiv preprint arXiv:2601.16163},
  year={2026}
}
```
