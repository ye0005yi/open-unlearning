# TFU — Test-time Unlearning

> This project is built on top of [OpenUnlearning](https://github.com/locuslab/open-unlearning).
> See [OPENUNLEARNING_README.md](OPENUNLEARNING_README.md) for full framework documentation, installation, and baseline methods.

---

## Overview

**TFU** is an **inference-time** unlearning approach that requires **no retraining**. It modifies output logits at inference time by combining the main model with a retrieval-augmented helper model:

```
final_logits = w * logits + (1 - w) * RAG_logits
```

The helper model receives retrieved context (via FAISS) related to the forget set.  
The hyperparameter `w` controls unlearning strength, and three **activation methods** control how `w` adapts per query based on the retrieval score:

| Activation method | Rule |
|---|---|
| `similarity` | `w = 1` if `score < threshold`, else `w = score * (model.w - 1) + 1` |
| `static` | `w = 1` if `score < threshold`, else `w = model.w` |
| `naive` | `w = model.w` always |

---

## Setup

Follow the [installation instructions](OPENUNLEARNING_README.md#installation) in the OpenUnlearning README. No extra dependencies are required for TFU.

```bash
pip install .[lm_eval,dev]
python setup_data.py --eval
```

---

## Reproducing Experiments

### TOFU Benchmark

Model: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` (pretrained target, loaded automatically).  
Config: `configs/experiment/eval/tfu/default.yaml`

**Retain baseline** (upper-bound reference):
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default task_name=EVAL_LLAMA_1B_RETAIN
```

**TFU — similarity activation** (recommended):
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default \
  model.w=3 tfu.activation_method=similarity task_name=EVAL_LLAMA_1B_TFU_sim_w3
```

**TFU — static activation**:
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default \
  model.w=3 tfu.activation_method=static task_name=EVAL_LLAMA_1B_TFU_sta_w3
```

**TFU — naive activation**:
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default \
  model.w=3 tfu.activation_method=naive task_name=EVAL_LLAMA_1B_TFU_nav_w3
```

To sweep over different `w` values, change `model.w=<value>` accordingly. Results are saved under `saves/eval/<task_name>/`.

---

### MUSE Benchmark

Config: `configs/experiment/eval/tfu/muse.yaml`

Set `data_split` to `News` or `Books`. The target model should be the MUSE-finetuned model (set `model.model_args.pretrained_model_name_or_path` to the relevant checkpoint).

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/muse \
  model.w=2 tfu.activation_method=similarity data_split=News task_name=EVAL_TFU_MUSE_NEWS
```

---

## Key Files

| Path | Description |
|---|---|
| `src/model/tfu.py` | Core TFU model implementation |
| `configs/experiment/eval/tfu/` | Experiment configs for TOFU and MUSE |
| `scripts/` | Bulk experiment scripts used to generate paper results |
