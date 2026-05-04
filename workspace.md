server:
    ssh -p 30740 app@10.97.176.242
work_dir:
    /data/tfu_jx/

---

# TFU Comprehensive Test Plan (Final v2)

## Objective
Compare TFU (Task-Free Unlearning) against existing unlearning methods across multiple dimensions: models, datasets/benchmarks, and metrics. Demonstrate TFU's advantages as a training-free, inference-time approach.

---

## Decisions Summary

- **Grouping**: By benchmark (TOFU -> MUSE -> WMDP)
- **Model families**: 2+ families per benchmark (Llama + Qwen + Zephyr)
- **DPO**: TOFU only (requires idk dataset)
- **RMU**: TOFU + WMDP only (not pre-validated on MUSE)
- **WMDP**: Include and debug; use Zephyr-7b (pre-configured) as primary
- **TFU `w` sweep**: {0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 4.0, 5.0}
- **Threshold sweep**: {0.55, 0.65, 0.75, 0.85} (similarity/static only)
- **Metrics**: Enable MIA attacks + exact_memorization (extended set)

---

## Method x Benchmark Compatibility Matrix

| Method | TOFU | MUSE | WMDP | Constraint |
|--------|:----:|:----:|:----:|-----------|
| **Retain** (baseline) | Y | Y | Y | Train from scratch on retain set |
| **TFU** (ours) | Y | Y | Y | No training needed |
| GradAscent | Y | Y | Y | Generic |
| GradDiff | Y | Y | Y | Generic |
| NPO | Y | Y | Y | Generic |
| SimNPO | Y | Y | Y | Generic |
| DPO (IdkDPO) | Y | N | N | Requires idk paired data (TOFU only) |
| RMU | Y | N | Y | Layer-targeting needs tuning; validated on TOFU+WMDP |
| UNDIAL | Y | Y | Y | Generic |

---

## Models Selected (2+ families per benchmark)

| Model | Size | Family | TOFU | MUSE | WMDP |
|-------|------|--------|------|------|------|
| Llama-3.2-1B-Instruct | 1B | Llama | Pretrained on HF | Needs finetune | Secondary |
| Llama-3.1-8B-Instruct | 8B | Llama | Pretrained on HF | Needs finetune | -- |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen | Needs finetune | Needs finetune | -- |
| Zephyr-7b-beta | 7B | Mistral | -- | -- | Pre-configured (primary) |

> Llama-2-7b excluded. Zephyr is Mistral-family, adding a 3rd family for WMDP.

### TFU Helper Model Selection

TFU requires a **helper model** (`tfu.help_model`) to produce alternative logits.
Formula: `final_logits = w * main_logits + (1 - w) * helper_logits`

**HARD CONSTRAINT**: Main and helper must have **identical vocab size** (logits are added element-wise).
The assertion `assert ret_ori_unignore.shape == ret_enh_unignore.shape` in tfu.py enforces this.

#### Vocab size by family (cross-family is IMPOSSIBLE):

| Family | Vocab Size | Models Available | Can Cross With |
|--------|-----------|-----------------|---------------|
| Llama-3.x | 128,256 | 1B, 3B, 8B Instruct | Only other Llama-3.x |
| Qwen2.5 | 151,936 | 1.5B, 3B, 7B Instruct | Only other Qwen2.5 |
| Mistral/Zephyr | 32,000 | 7B only | None (no smaller Mistral exists) |

**Conclusion**:
- Llama + Qwen: CANNOT cross (128K vs 152K vocab mismatch)
- Zephyr + smaller helper: NOT POSSIBLE (no small Mistral-family model available; 7B is the only option)
- Within Llama-3.x: CAN cross sizes (1B, 3B, 8B all share tokenizer)
- Within Qwen2.5: CAN cross sizes (1.5B, 3B, 7B all share tokenizer)

#### Valid helper model options (helper must be <= main size):

| Main Model (finetuned) | Helper Option | Type | Feasible |
|------------------------|--------------|------|:--------:|
| tofu_Llama-3.2-1B_full | `meta-llama/Llama-3.2-1B-Instruct` | Same-size (default) | Y |
| tofu_Llama-3.1-8B_full | `meta-llama/Llama-3.1-8B-Instruct` | Same-size (default) | Y |
| tofu_Llama-3.1-8B_full | `meta-llama/Llama-3.2-1B-Instruct` | Smallest helper (saves most memory) | Y |
| tofu_Qwen2.5-1.5B_full | `Qwen/Qwen2.5-1.5B-Instruct` | Same-size only (no smaller Qwen) | Y |
| Zephyr-7b (WMDP) | `HuggingFaceH4/zephyr-7b-beta` | Same-model only (no smaller Mistral) | Y |

#### Suggested helper model experiments:

1. **Same-family same-size** (default, Priority 1):
   - Helper = base instruct model (no finetune knowledge).
   - Example: main = tofu_Llama-1B_full, helper = Llama-1B-Instruct
   - Already implemented in `configs/experiment/eval/tfu/default.yaml`

2. **Same-family smaller** (Priority 3, Llama-8B main only):
   - main = tofu_Llama-8B_full, helper = Llama-3.2-1B-Instruct (or 3B)
   - Saves ~7B parameters worth of memory during inference.
   - Tests if a smaller model provides sufficient "clean" signal.
   - Only viable for Llama-8B (Qwen-1.5B and Zephyr-7b have no smaller same-family model).

#### CLI override to change helper:
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  model.w=3 \
  tfu.help_model.pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct" \
  task_name=EVAL_8B_TFU_helper_1B
```

---

## TFU Hyperparameter Sweep

### Weight `w` sweep (12 values)
```
w in {0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0, 4.0, 5.0}
```
Formula: `final_logits = w * logits_main + (1 - w) * logits_helper`

### Activation threshold sweep (4 values, similarity/static only)
```
threshold in {0.55, 0.65, 0.75, 0.85}
```
- **similarity**: `w = 1` if `RAG_score <= threshold` else `w = RAG_score * (model.w - 1) + 1`
- **static**: `w = 1` if `RAG_score <= threshold` else `w = model.w`
- **naive**: threshold has no effect (always `w = model.w`)

**Note**: Threshold bug fixed in commit 536ef21 (local). Push to server before running.

### TFU Variants
| Variant | Helper Model | Activation | Sweep |
|---------|-------------|-----------|-------|
| TFU-RAG-similarity | Base LLM | similarity | w x threshold = 12x4 = 48 combos |
| TFU-RAG-static | Base LLM | static | w x threshold = 12x4 = 48 combos |
| TFU-RAG-naive | Base LLM | naive | w only = 12 combos |
| TFU-Finetuned | Finetuned on forget | naive | w only = 12 combos |

**Total TFU eval runs per model per split**: 120 (48+48+12+12)

---

## Execution Groups

### Group 1: TOFU Benchmark

**Families**: Llama (1B, 8B) + Qwen (1.5B)
**Splits**: forget01, forget05, forget10
**Methods**: Retain, TFU(x4 variants), GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL

#### Status: What's Done vs Not Done

| Item | Llama-3.2-1B | Llama-3.1-8B | Qwen2.5-1.5B |
|------|:---:|:---:|:---:|
| Pretrained model (full) | DONE (HF) | DONE (HF) | NOT DONE (finetune needed) |
| Retain model | DONE (HF) | DONE (HF) | NOT DONE (finetune needed) |
| Baseline eval logs | DONE (`setup_data.py --eval`) | DONE (`setup_data.py --eval`) | NOT DONE |
| TFU-RAG eval (forget10, w=3) | DONE (sim/static/naive) | NOT DONE | NOT DONE |
| TFU-Finetuned eval | PARTIAL | NOT DONE | NOT DONE |
| TFU w sweep (12 values) | PARTIAL (w=1.5,2,2.5,3 naive only) | NOT DONE | NOT DONE |
| TFU threshold sweep | NOT DONE | NOT DONE | NOT DONE |
| GradAscent/GradDiff/NPO/SimNPO/RMU | DONE (repro numbers in docs/repro.md) | NOT DONE | NOT DONE |
| DPO (IdkDPO) | DONE (repro numbers) | NOT DONE | NOT DONE |
| UNDIAL | NOT DONE | NOT DONE | NOT DONE |

#### Experiment Tree

```
TOFU Benchmark
|-- Llama-3.2-1B-Instruct [pretrained: open-unlearning/tofu_Llama-3.2-1B-Instruct_full]
|   |-- Baselines (eval only -- use HF models + setup_data.py --eval)
|   |   |-- Finetuned (full) -- DONE
|   |   +-- Retain (retain90/95/99) -- DONE
|   |-- Traditional Methods (8 methods x 3 splits = 24 training runs)
|   |   |-- GradAscent x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   |-- GradDiff x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   |-- NPO x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   |-- SimNPO x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   |-- DPO x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   |-- RMU x {forget01, forget05, forget10} -- DONE (repro), re-eval with extended metrics
|   |   +-- UNDIAL x {forget01, forget05, forget10} -- NOT DONE
|   +-- TFU (eval only -- 120 runs per split x 3 splits = 360 eval runs)
|       |-- TFU-RAG-similarity x 12 w values x 4 thresholds -- PARTIAL (w=3 only, threshold=0.55)
|       |-- TFU-RAG-static x 12 w values x 4 thresholds -- PARTIAL (w=3 only, threshold=0.55)
|       |-- TFU-RAG-naive x 12 w values -- PARTIAL (w=1.5,2,2.5,3 done)
|       +-- TFU-Finetuned x 12 w values -- NOT DONE
|
|-- Llama-3.1-8B-Instruct [pretrained: open-unlearning/tofu_Llama-3.1-8B-Instruct_full]
|   |-- Baselines -- DONE (HF models available)
|   |-- Traditional Methods (24 training runs) -- ALL NOT DONE
|   +-- TFU (360 eval runs) -- ALL NOT DONE
|
+-- Qwen2.5-1.5B-Instruct [NO pretrained -- needs full pipeline]
    |-- Step 0: Finetune on TOFU full -> create base model -- NOT DONE
    |-- Step 0b: Finetune on retain splits -> retain baselines -- NOT DONE
    |-- Traditional Methods (24 training runs) -- ALL NOT DONE
    +-- TFU (360 eval runs) -- ALL NOT DONE
```

**Total for Group 1**: 
- Finetuning: 0 (Llama) + ~4 (Qwen: full + retain90/95/99) = 4 runs
- Unlearning training: 24 x 3 models = 72 runs (Llama-1B repro exists, need re-eval w/ extended metrics)
- TFU eval: 360 x 3 models = 1080 runs (cheap, eval-only)

---

### Group 2: MUSE Benchmark

**Families**: Llama (1B, 8B) + Qwen (1.5B)
**Splits**: News, Books
**Methods**: Retain, TFU(x4), GradAscent, GradDiff, NPO, SimNPO, UNDIAL (no DPO, no RMU)

#### Status: What's Done vs Not Done

| Item | Llama-3.2-1B | Llama-3.1-8B | Qwen2.5-1.5B |
|------|:---:|:---:|:---:|
| MUSE finetuned model | NOT DONE (config exists: finetune/muse/1b.yaml) | NOT DONE | NOT DONE |
| Retain model | NOT DONE | NOT DONE | NOT DONE |
| TFU eval | NOT DONE (config: eval/tfu/muse.yaml, untested) | NOT DONE | NOT DONE |
| Traditional methods | NOT DONE | NOT DONE | NOT DONE |

> Note: MUSE has existing Llama-2-7b-hf results in docs/repro.md but we can't use Llama-2.
> All MUSE experiments for our selected models require finetuning from scratch.
> Config `configs/experiment/finetune/muse/1b.yaml` exists for Llama-3.2-1B.

#### Experiment Tree

```
MUSE Benchmark
|-- Prerequisites (finetuning for ALL models -- no pretrained available)
|   |-- Llama-3.2-1B: Finetune on News(full+retain) + Books(full+retain) = 4 runs
|   |-- Llama-3.1-8B: Finetune on News(full+retain) + Books(full+retain) = 4 runs
|   +-- Qwen2.5-1.5B: Finetune on News(full+retain) + Books(full+retain) = 4 runs
|
|-- Per model x per split (News, Books):
|   |-- Baselines: Finetuned, Retain
|   |-- Traditional Methods (5 methods per split = 5 training runs)
|   |   |-- GradAscent
|   |   |-- GradDiff
|   |   |-- NPO
|   |   |-- SimNPO
|   |   +-- UNDIAL
|   +-- TFU (120 eval runs per split: 48+48+12+12)
|
+-- Status: ENTIRELY NOT DONE (except config skeleton exists)
```

**Total for Group 2**:
- Finetuning: 12 runs (3 models x 2 splits x 2 variants: full+retain)
- Unlearning training: 30 runs (3 models x 2 splits x 5 methods)
- TFU eval: 720 runs (3 models x 2 splits x 120 combos)

---

### Group 3: WMDP Benchmark

**Families**: Zephyr-7b (Mistral, primary) + Llama-3.2-1B (secondary)
**Splits**: cyber, bio
**Methods**: Retain, TFU(x4), GradAscent, GradDiff, NPO, SimNPO, RMU, UNDIAL (no DPO)

#### Status: What's Done vs Not Done

| Item | Zephyr-7b | Llama-3.2-1B |
|------|:---:|:---:|
| Pre-configured experiment | DONE (configs/experiment/unlearn/wmdp/default.yaml) | NOT DONE (need to adapt) |
| WMDP data downloaded | NOT DONE (need `setup_data.py --wmdp`) | Same |
| Pipeline verified | NOT DONE ("might have issues") | NOT DONE |
| TFU eval | NOT DONE | NOT DONE |
| Traditional methods | NOT DONE | NOT DONE |

#### Experiment Tree

```
WMDP Benchmark [experimental -- debug as needed]
|-- Prerequisites
|   |-- Download WMDP data: python setup_data.py --wmdp
|   |-- Verify Zephyr-7b pipeline end-to-end (existing config)
|   |-- Adapt config for Llama-3.2-1B (if Zephyr works)
|   +-- Debug any pipeline issues
|
|-- Zephyr-7b-beta [PRIMARY -- pre-configured]
|   |-- Cyber Split
|   |   |-- Baselines: Original, Retain
|   |   |-- Traditional Methods (6): GradAscent, GradDiff, NPO, SimNPO, RMU, UNDIAL
|   |   +-- TFU (120 eval runs)
|   +-- Bio Split
|       +-- (same as Cyber)
|
+-- Llama-3.2-1B-Instruct [SECONDARY -- adapt from Zephyr config]
    +-- (same structure, if Zephyr pipeline works)

Status: ENTIRELY NOT DONE -- experimental, may require debugging
```

**Total for Group 3**:
- Unlearning training: 12 runs per model x 2 models = 24 runs
- TFU eval: 120 per split x 2 splits x 2 models = 480 runs
- Note: Uses lm-eval-harness (MMLU, wmdp_cyber, wmdp_bio), different eval system

---

## Extended Metrics (Enabled)

### TOFU (all enabled)
| Metric | Category | Status |
|--------|----------|--------|
| forget_quality | Unlearning effectiveness | Default |
| model_utility | Utility preservation | Default |
| forget_truth_ratio | Unlearning effectiveness | Default |
| privleak | Privacy | Default |
| extraction_strength | Privacy | Default |
| exact_memorization | Memorization | **NEW -- enable** |
| mia_min_k_plus_plus | MIA attack | **NEW -- enable** |
| mia_min_k | MIA attack | **NEW -- enable** |
| mia_loss | MIA attack | **NEW -- enable** |
| mia_zlib | MIA attack | **NEW -- enable** |
| mia_gradnorm | MIA attack | **NEW -- enable** |
| mia_reference | MIA attack | **NEW -- enable** |

### MUSE (all enabled)
| Metric | Category | Status |
|--------|----------|--------|
| forget_knowmem_ROUGE | Knowledge forgetting | Default |
| forget_verbmem_ROUGE | Verbatim forgetting | Default |
| retain_knowmem_ROUGE | Utility preservation | Default |
| privleak | Privacy | Default |
| extraction_strength | Privacy | **NEW -- enable** |
| exact_memorization | Memorization | **NEW -- enable** |
| mia_min_k_plus_plus | MIA attack | **NEW -- enable** |
| mia_min_k | MIA attack | **NEW -- enable** |
| mia_loss | MIA attack | **NEW -- enable** |
| mia_zlib | MIA attack | **NEW -- enable** |
| mia_gradnorm | MIA attack | **NEW -- enable** |
| mia_reference | MIA attack | **NEW -- enable** |

### WMDP
| Metric | Category |
|--------|----------|
| wmdp_cyber | Forgetting (accuracy) |
| wmdp_bio | Forgetting (accuracy) |
| mmlu | Utility preservation |
| gsm8k | Utility preservation (optional) |

---

## Analysis Perspectives (Post-Experiment)

### A. TFU Advantage: Zero Training Cost
| | TFU | Traditional Methods |
|---|---|---|
| Training time | **0** | Hours (10 epochs x data size) |
| GPU during unlearning | **None** | Full training GPU |
| Tune at inference time | **Yes** (adjust w, threshold) | No (retrain needed) |
| Hyperparameter sensitivity | Sweep at eval time (cheap) | Retrain per config (expensive) |

### B. Per-Benchmark Comparison Tables (template -- fill after experiments)

**[TOFU forget10, Llama-3.2-1B-Instruct]**
| Method | forget_quality^ | model_utility^ | truth_ratio^ | privleak->0 | extraction_v | exact_mem_v |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Retain (gold) | 1.0 | 0.59 | 0.63 | ~0 | ? | ? |
| TFU-RAG-sim (best w) | ? | ? | ? | ? | ? | ? |
| TFU-Finetuned (best w) | ? | ? | ? | ? | ? | ? |
| GradAscent | 0.27* | 0.33* | 0.59* | ? | ? | ? |
| GradDiff | 0.77* | 0.43* | 0.57* | ? | ? | ? |
| NPO | 0.92* | 0.56* | 0.66* | ? | ? | ? |
| SimNPO | 0.58* | 0.46* | 0.55* | ? | ? | ? |
| DPO | 0.01* | 0.51* | 0.60* | ? | ? | ? |
| RMU | 0.16* | 0.55* | 0.70* | ? | ? | ? |
| UNDIAL | ? | ? | ? | ? | ? | ? |

`* from docs/repro.md -- need re-eval with extended metrics`

### C. Cross-Model Robustness (Llama-1B vs Llama-8B vs Qwen-1.5B)
- Does TFU scale consistently across model sizes AND families?
- Do traditional methods degrade differently per family?

### D. Cross-Dataset Generalization (TOFU vs MUSE vs WMDP)
- Same method, different tasks: who generalizes best?

### E. Forget Ratio Sensitivity (forget01 vs forget05 vs forget10)
- TFU w is tunable per ratio; traditional methods use fixed training

### F. TFU Hyperparameter Analysis
- Optimal w per model/benchmark
- Threshold sensitivity (similarity vs static)
- Pareto frontier: forgetting vs utility at different w values
- w sweep curve: plot metrics vs w for each activation method

---

## Execution Checklist (per group)

### Step-by-step execution order:

1. **Setup**: `python setup_data.py --eval` (baselines) + `--wmdp` (for G3)
2. **Finetuning** (if needed): Create target + retain baselines for Qwen/MUSE
3. **Unlearning Training**: Run each traditional method
4. **TFU Eval**: Run full w x threshold sweep (no training)
5. **Extended Eval**: Re-evaluate existing models with MIA + exact_memorization
6. **Compare**: Generate per-benchmark comparison tables + cross-benchmark analysis

---

## Resource Reuse

| Resource | How to Get | What it Provides |
|----------|-----------|-----------------|
| Eval logs | `python setup_data.py --eval` | Retain/finetuned model baselines for TOFU+MUSE |
| TOFU Llama pretrained | HuggingFace `open-unlearning/` | Skip finetuning for Llama TOFU |
| Repro results | `docs/repro.md` + HF eval dataset | Reference numbers (need re-eval with extended metrics) |
| WMDP data | `python setup_data.py --wmdp` | WMDP corpus (password: wmdpcorpora) |
| TFU partial results | `sh_extra.md` | Llama-1B TFU w=1.5,2,2.5,3 naive results on TOFU forget10 |

---
---

# EXECUTION PLAN (Step-by-Step)

## Context for Cold Start (Read this if you are a new agent/session)

### What is this project?
`open-unlearning` is a framework for LLM unlearning research. It trains LLMs to "forget"
specific data while retaining general knowledge. Built on HuggingFace Transformers + Hydra configs.

### What is TFU?
TFU (Task-Free Unlearning) is OUR method. Unlike traditional methods that require retraining,
TFU works at inference time by composing logits from two models:
```
final_logits = w * main_model_logits + (1 - w) * helper_model_logits
```
The main model is the finetuned model (has memorized data). The helper provides "clean" logits.
No gradient updates needed — just change `w` or `threshold` at eval time.

### Key entry points
- `python src/train.py --config-name=train.yaml` — finetuning (create base models)
- `python src/train.py --config-name=unlearn.yaml` — unlearning training (traditional methods)
- `python src/eval.py --config-name=eval.yaml` — evaluation (all methods including TFU)

### How Hydra configs compose
Commands override configs hierarchically:
```bash
python src/eval.py --config-name=eval.yaml \    # base config
  experiment=eval/tofu/default \                 # loads configs/experiment/eval/tofu/default.yaml (overrides model, data, eval)
  model=Llama-3.2-1B-Instruct \                  # loads configs/model/Llama-3.2-1B-Instruct.yaml
  model.model_args.pretrained_model_name_or_path=... \  # overrides single field
  task_name=MY_TASK                              # output saved to saves/eval/MY_TASK/ or saves/unlearn/MY_TASK/
```
The `experiment=` override sets model, data, eval, and trainer defaults. Individual field overrides
take priority over experiment defaults.

### Output locations
- Eval results: `saves/eval/<task_name>/` → contains `TOFU_EVAL.json` (raw) + `TOFU_SUMMARY.json` (aggregated)
- Unlearn models: `saves/unlearn/<task_name>/` → full model checkpoint
- Finetune models: `saves/finetune/<task_name>/` → full model checkpoint
- MUSE uses `MUSE_EVAL.json` / `MUSE_SUMMARY.json` instead

### retain_logs_path (IMPORTANT)
Many metrics (forget_quality, privleak) compare the evaluated model against a reference (retain model).
This is passed as `retain_logs_path=saves/eval/<retain_task>/TOFU_EVAL.json`.
- If omitted: metrics that need it will either error or return null
- Must point to the EVAL.json (not SUMMARY) of the retain model for the same split
- Mapping: forget10 → retain90, forget05 → retain95, forget01 → retain99

### Git info
- Branch: `tfu`
- Remote: check with `git remote -v` on server (URL not stored here for security)
- Key commit: 536ef21 (activation_threshold fix)

### HuggingFace access
- `meta-llama/` models require HF token: `huggingface-cli login`
- `open-unlearning/` models are public (no auth needed)
- `Qwen/` models are public
- `HuggingFaceH4/zephyr-7b-beta` is public

### GPU / Memory guidance
| Task | Model Size | Min GPU Memory | Recommended |
|------|-----------|---------------|-------------|
| Eval (TFU, single model) | 1B | 8GB | 1x 24GB GPU |
| Eval (TFU, main+helper same size) | 1B+1B | 12GB | 1x 24GB GPU |
| Eval (TFU, 8B+8B) | 8B+8B | 40GB | 1x 48GB or 2x 24GB |
| Eval (TFU, 8B+1B helper) | 8B+1B | 24GB | 1x 48GB GPU |
| Training (unlearn/finetune) | 1B | 16GB | 1x 24GB GPU |
| Training (unlearn/finetune) | 8B | 48GB+ | 2x L40s + DeepSpeed ZeRO3 |

For multi-GPU training (8B models), prepend with accelerate:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port 29500 \
  src/train.py --config-name=unlearn.yaml ...
```

### DPO special case
DPO uses a DIFFERENT experiment config than other methods:
```bash
# Other methods:
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default trainer=GradAscent ...

# DPO (requires idk data):
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/idk trainer=DPO ...
```
The `idk` experiment loads `TOFU_QA_forget_idk` dataset (paired original + "I don't know" answers).
Requires `data/idk.jsonl` to exist (from `setup_data.py --idk`).

### TFU eval config defaults (IMPORTANT for overrides)
`configs/experiment/eval/tfu/default.yaml` HARDCODES:
- `model: Llama-3.2-1B-Instruct`
- `model.model_args.pretrained_model_name_or_path: open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- `tfu.help_model.pretrained_model_name_or_path: meta-llama/Llama-3.2-1B-Instruct`

For 8B runs, you MUST override ALL THREE:
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default \
  model=Llama-3.1-8B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
  model.w=3 ...
```

### Naming convention: TOFU vs MUSE config parameters
- TOFU uses: `forget_split=forget10`, `retain_split=retain90`, `holdout_split=holdout10`
- MUSE uses: `data_split=News` or `data_split=Books`
- WMDP uses: data split is implicit in the JSONL files (cyber-forget-corpus, bio-forget-corpus)

### Disk space estimate
- 1B model checkpoint: ~5GB
- 8B model checkpoint: ~16GB
- Eval results (JSON): ~1MB per task
- Full Group 1 (all checkpoints): ~500GB (72 training runs x avg 7GB)
- Tip: Delete intermediate checkpoints, keep only final: `trainer.args.save_strategy=no` or delete `checkpoint-*/`

---

## Conventions

### Status Markers
Each step has a status field. Update after completion:
- `[ ]` = Not started
- `[~]` = In progress
- `[x]` = Completed
- `[!]` = Failed / blocked (see notes)
- `[S]` = Skipped (with reason)

### Checkpoint Format
After completing a step, record:
```
STATUS: [x] DONE
DATE: YYYY-MM-DD
OUTPUT: path/to/result or summary
NOTES: any observations
```

### Issue Resolution Workflow
When encountering an error:
1. Document the error in the step's NOTES field
2. Attempt to fix locally (on dev machine)
3. Ask user to review the fix
4. Commit the fix to git
5. Sync to server: `git push` then `git pull` on server
6. If fix affects previous results, mark affected steps with `[!]` and note which need re-run
7. Resume from checkpoint

---

## 0. Environment Setup & Init

Run these steps on a FRESH server or after a crash. Skip if already set up.

### 0.1 Server Access
```bash
ssh -p 30740 app@10.97.176.242
cd /data/tfu_jx/
```
STATUS: [ ]

### 0.2 Clone / Pull Latest Code
```bash
# Fresh clone:
git clone <repo_url> /data/tfu_jx/
cd /data/tfu_jx/
git checkout tfu

# Or if repo exists, pull latest:
cd /data/tfu_jx/
git pull origin tfu
```
STATUS: [ ]

### 0.3 Python Environment
```bash
# Create env (if fresh):
conda create -n unlearn python=3.10 -y
conda activate unlearn
pip install -r requirements.txt

# Or activate existing:
conda activate unlearn
```
STATUS: [ ]

### 0.4 Verify GPU Access
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```
Expected: CUDA available, >= 1 GPU
STATUS: [ ]

### 0.5 Download Baseline Eval Logs
```bash
python setup_data.py --eval
```
Expected: `saves/eval/` populated with pre-computed baselines
STATUS: [ ]

### 0.6 Download WMDP Data (for Group 3)
```bash
python setup_data.py --wmdp
```
Expected: `data/wmdp/wmdp-corpora/` with cyber/bio JSONL files
STATUS: [ ]

### 0.7 Download IDK Data (for DPO method)
```bash
python setup_data.py --idk
```
Expected: `data/idk.jsonl`
STATUS: [ ]

### 0.8 Verify TFU Threshold Fix is Present
```bash
grep "self.activation_threshold" src/model/tfu.py | head -5
```
Expected: Should see `self.activation_threshold = threshold` in `set_activation()`
If NOT present: pull latest from tfu branch (commit 536ef21 has the fix)
STATUS: [ ]

---

## 1. Backup & Restore Procedures

### 1.1 What to Backup

| Directory | Contents | Priority |
|-----------|----------|----------|
| `saves/eval/` | All evaluation results (JSON) | HIGH -- expensive to regenerate |
| `saves/unlearn/` | Unlearned model checkpoints | MEDIUM -- can retrain but costs GPU hours |
| `saves/finetune/` | Finetuned model checkpoints | MEDIUM -- same |
| `configs/` | Any modified configs | HIGH -- small files, easy to lose changes |

### 1.2 Backup Script (run periodically or after each major step)
```bash
#!/bin/bash
# backup.sh -- run from /data/tfu_jx/
BACKUP_DIR="/data/tfu_jx_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup eval results (most important, smallest)
cp -r saves/eval/ $BACKUP_DIR/eval/ 2>/dev/null

# Backup configs
cp -r configs/ $BACKUP_DIR/configs/

# List model checkpoints (too large to copy every time, just record paths)
find saves/unlearn/ -name "*.json" -o -name "config.json" | sort > $BACKUP_DIR/unlearn_manifest.txt
find saves/finetune/ -name "*.json" -o -name "config.json" | sort > $BACKUP_DIR/finetune_manifest.txt

echo "Backup saved to: $BACKUP_DIR"
```

### 1.3 Pull Results to Local Machine
```bash
# From local machine -- pull eval results
scp -P 30740 -r app@10.97.176.242:/data/tfu_jx/saves/eval/ ./saves/eval_server/

# Pull specific experiment result
scp -P 30740 app@10.97.176.242:/data/tfu_jx/saves/eval/<task_name>/ ./saves/eval_server/<task_name>/
```

### 1.4 Restore After Crash
```bash
# 1. Re-setup environment (Section 0)
# 2. Restore eval results from backup:
cp -r /data/tfu_jx_backup/<latest>/eval/ /data/tfu_jx/saves/eval/
# 3. Check this execution plan -- find last [x] step, resume from next [ ] step
```

---

## 2. Enable Extended Metrics

Before running any evaluations, enable MIA + exact_memorization in configs.

### 2.1 Enable Extended TOFU Metrics
Edit `configs/eval/tofu.yaml` -- uncomment these metrics:
```yaml
metrics:
  # ... existing defaults ...
  exact_memorization:
  mia_min_k_plus_plus:
  mia_min_k:
  mia_loss:
  mia_zlib:
  mia_gradnorm:
  mia_reference:
```
STATUS: [ ]
NOTES: Check exact syntax by reading current tofu.yaml. If unsure, ask user.

### 2.2 Enable Extended MUSE Metrics
Edit `configs/eval/muse.yaml` -- uncomment these metrics:
```yaml
metrics:
  # ... existing defaults ...
  extraction_strength:
  exact_memorization:
  mia_min_k_plus_plus:
  mia_min_k:
  mia_loss:
  mia_zlib:
  mia_gradnorm:
  mia_reference:
```
STATUS: [ ]

### 2.3 Commit & Sync Metric Config Changes
```bash
# On local machine:
git add configs/eval/tofu.yaml configs/eval/muse.yaml
git commit -m "feat: enable extended metrics (MIA attacks + exact_memorization)"
git push origin tfu

# On server:
cd /data/tfu_jx/ && git pull origin tfu
```
STATUS: [ ]

---

## GROUP 1: TOFU Benchmark

---

### G1-Phase-A: Llama-3.2-1B-Instruct (Pretrained Available)

#### G1-A-1: Verify Baseline Models Accessible
```bash
# Test that HF model loads:
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('open-unlearning/tofu_Llama-3.2-1B-Instruct_full', torch_dtype='auto')
print(f'Loaded: {model.config._name_or_path}, params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')
"
```
Expected: Model loads, ~1.2B params
STATUS: [ ]

#### G1-A-2: Evaluate Baselines with Extended Metrics
```bash
# Finetuned (full) model -- extended metrics
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  task_name=G1_baseline_llama1b_full

# Retain model
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  task_name=G1_baseline_llama1b_retain90
```
Expected: `saves/eval/G1_baseline_llama1b_full/TOFU_EVAL.json` and `*_SUMMARY.json`
STATUS: [ ]
BACKUP: `cp -r saves/eval/G1_baseline_* /data/tfu_jx_backup/`

#### G1-A-3: TFU RAG Naive -- w sweep (forget10)
Run all 12 w values. No threshold needed for naive.
```bash
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tfu/default \
    model.w=$w \
    tfu.activation_method=naive \
    forget_split=forget10 \
    task_name=G1_tfu_llama1b_naive_w${w}_forget10
done
```
Expected: 12 result folders in saves/eval/
STATUS: [ ]
BACKUP: `cp -r saves/eval/G1_tfu_llama1b_naive_* /data/tfu_jx_backup/`

#### G1-A-4: TFU RAG Similarity -- w x threshold sweep (forget10)
```bash
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  for th in 0.55 0.65 0.75 0.85; do
    python src/eval.py --config-name=eval.yaml \
      experiment=eval/tfu/default \
      model.w=$w \
      tfu.activation_method=similarity \
      tfu.activation_threshold=$th \
      forget_split=forget10 \
      task_name=G1_tfu_llama1b_sim_w${w}_th${th}_forget10
  done
done
```
Expected: 48 result folders
STATUS: [ ]
BACKUP: After completion

#### G1-A-5: TFU RAG Static -- w x threshold sweep (forget10)
```bash
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  for th in 0.55 0.65 0.75 0.85; do
    python src/eval.py --config-name=eval.yaml \
      experiment=eval/tfu/default \
      model.w=$w \
      tfu.activation_method=static \
      tfu.activation_threshold=$th \
      forget_split=forget10 \
      task_name=G1_tfu_llama1b_static_w${w}_th${th}_forget10
  done
done
```
Expected: 48 result folders
STATUS: [ ]

#### G1-A-6: TFU Finetuned Helper -- prepare helper model
```bash
# Finetune Llama-1B on forget10 data (creates the helper)
python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/forget10 \
  task_name=G1_helper_llama1b_forget10
```
Expected: Model saved to `saves/finetune/G1_helper_llama1b_forget10/`
STATUS: [ ]

#### G1-A-7: TFU Finetuned -- w sweep (forget10)
```bash
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tfu/finetuned \
    model.w=$w \
    tfu.help_model.pretrained_model_name_or_path=./saves/finetune/G1_helper_llama1b_forget10 \
    forget_split=forget10 \
    task_name=G1_tfu_llama1b_finetuned_w${w}_forget10
done
```
Expected: 12 result folders
STATUS: [ ]

#### G1-A-8: Traditional Methods -- UNDIAL (forget10)
(Other methods exist from repro; UNDIAL is the only one NOT done)
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=UNDIAL \
  forget_split=forget10 \
  retain_split=retain90 \
  task_name=G1_undial_llama1b_forget10
```
Expected: Model saved to `saves/unlearn/G1_undial_llama1b_forget10/`
STATUS: [ ]

#### G1-A-8b: Traditional Methods -- DPO (forget10)
DPO uses a DIFFERENT experiment config (requires idk data):
```bash
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/idk \
  trainer=DPO \
  forget_split=forget10 \
  retain_split=retain90 \
  task_name=G1_dpo_llama1b_forget10
```
Prerequisite: `data/idk.jsonl` must exist (step 0.7).
Expected: Model saved to `saves/unlearn/G1_dpo_llama1b_forget10/`
STATUS: [ ]

#### G1-A-9: Evaluate UNDIAL Model (forget10)
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=./saves/unlearn/G1_undial_llama1b_forget10 \
  retain_logs_path=saves/eval/G1_baseline_llama1b_retain90/TOFU_EVAL.json \
  task_name=G1_eval_undial_llama1b_forget10
```
STATUS: [ ]

#### G1-A-10: Re-evaluate Existing Repro Models with Extended Metrics (forget10)
The repro script (`scripts/tofu_unlearn.sh`) saves models with task_names like:
`tofu_Llama-3.2-1B-Instruct_<method>_forget10` (check actual names on server with `ls saves/unlearn/`)

If repro model checkpoints exist on server:
```bash
# First, check what's actually there:
ls saves/unlearn/ | grep -i "llama.*1b.*forget10"

# Then evaluate each. Adjust paths to match actual saved names.
# Template (replace <ACTUAL_PATH> with what you find):
for method in GradAscent GradDiff NPO SimNPO RMU; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=./saves/unlearn/<ACTUAL_PATH_FOR_${method}> \
    retain_logs_path=saves/eval/G1_baseline_llama1b_retain90/TOFU_EVAL.json \
    task_name=G1_eval_${method}_llama1b_forget10
done

# DPO separately (different source path):
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model.model_args.pretrained_model_name_or_path=./saves/unlearn/<ACTUAL_DPO_PATH> \
  retain_logs_path=saves/eval/G1_baseline_llama1b_retain90/TOFU_EVAL.json \
  task_name=G1_eval_DPO_llama1b_forget10
```

**If repro checkpoints DO NOT exist**, you must re-train them first:
```bash
# Re-train all methods (from scripts/tofu_unlearn.sh pattern):
for method in GradAscent GradDiff NPO SimNPO RMU; do
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=$method \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=G1_${method}_llama1b_forget10
done

# For 8B models, use accelerate:
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml \
#   src/train.py --config-name=unlearn.yaml ...
```

**UNCERTAINTY**: Repro model checkpoints may not exist on server. Options:
- (a) Check `ls saves/unlearn/` first — maybe they're already there from prior runs
- (b) Re-train from scratch using commands above (costs GPU hours)
- (c) Use `setup_data.py --eval` downloaded results if they include model checkpoints (unlikely — probably eval logs only)

**Ask user which option applies before proceeding.**
STATUS: [ ]

#### G1-A-11: Expand to forget01 and forget05
Repeat G1-A-3 through G1-A-10 with `forget_split=forget01` and `forget_split=forget05`.
Replace `forget10` with `forget01`/`forget05` in task_name and config overrides.
Also update retain_split (forget01->retain99, forget05->retain95).

**Key differences per split:**
- retain_logs_path must point to the correct retain model eval:
  - forget10 → `G1_baseline_llama1b_retain90/TOFU_EVAL.json`
  - forget05 → need retain95 baseline (eval `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95`)
  - forget01 → need retain99 baseline (eval `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99`)
- TFU-Finetuned needs a SEPARATE helper model per split:
  - forget10 helper: `python src/train.py ... experiment=finetune/tofu/forget10 task_name=G1_helper_llama1b_forget10`
  - forget05 helper: finetune on forget05 data (need config for this — may need `data/datasets@data.train=TOFU_QA_forget05`)
  - forget01 helper: finetune on forget01 data

```bash
# Step 1: Create retain baselines for each split
for retain in retain95 retain99; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_${retain} \
    task_name=G1_baseline_llama1b_${retain}
done

# Step 2: Create helper models for TFU-Finetuned per split
for fsplit in forget01 forget05; do
  python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/forget10 \
    "data/datasets@data.train=TOFU_QA_${fsplit}" \
    task_name=G1_helper_llama1b_${fsplit}
done

# Step 3: Run all TFU sweeps per split
for split in forget01 forget05; do
  case $split in
    forget01) retain=retain99 ;;
    forget05) retain=retain95 ;;
  esac

  # TFU naive sweep (12 runs)
  for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
    python src/eval.py --config-name=eval.yaml \
      experiment=eval/tfu/default \
      model.w=$w \
      tfu.activation_method=naive \
      forget_split=$split \
      task_name=G1_tfu_llama1b_naive_w${w}_${split}
  done

  # TFU similarity sweep (48 runs)
  for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
    for th in 0.55 0.65 0.75 0.85; do
      python src/eval.py --config-name=eval.yaml \
        experiment=eval/tfu/default \
        model.w=$w \
        tfu.activation_method=similarity \
        tfu.activation_threshold=$th \
        forget_split=$split \
        task_name=G1_tfu_llama1b_sim_w${w}_th${th}_${split}
    done
  done

  # TFU static sweep (48 runs) -- same as similarity but tfu.activation_method=static

  # TFU finetuned sweep (12 runs)
  for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
    python src/eval.py --config-name=eval.yaml \
      experiment=eval/tfu/finetuned \
      model.w=$w \
      tfu.help_model.pretrained_model_name_or_path=./saves/finetune/G1_helper_llama1b_${split} \
      forget_split=$split \
      task_name=G1_tfu_llama1b_finetuned_w${w}_${split}
  done

  # UNDIAL training
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=UNDIAL \
    forget_split=$split \
    retain_split=$retain \
    task_name=G1_undial_llama1b_${split}

  # UNDIAL eval
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=./saves/unlearn/G1_undial_llama1b_${split} \
    retain_logs_path=saves/eval/G1_baseline_llama1b_${retain}/TOFU_EVAL.json \
    task_name=G1_eval_undial_llama1b_${split}

  # DPO training (different experiment config!)
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/idk \
    trainer=DPO \
    forget_split=$split \
    retain_split=$retain \
    task_name=G1_dpo_llama1b_${split}
done
```
STATUS: [ ]

---

### G1-Phase-B: Llama-3.1-8B-Instruct (Pretrained Available)

Same structure as Phase-A but with 8B model. Key differences:
- Model path: `open-unlearning/tofu_Llama-3.1-8B-Instruct_full`
- More GPU memory needed (may need to adjust batch size)
- Can also test smaller helper (1B) for TFU

#### G1-B-1: Verify 8B Model Loads
```bash
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('open-unlearning/tofu_Llama-3.1-8B-Instruct_full', torch_dtype='auto')
print(f'Loaded: params={sum(p.numel() for p in model.parameters())/1e9:.1f}B')
"
```
STATUS: [ ]

#### G1-B-2: Baselines with Extended Metrics
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.1-8B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  task_name=G1_baseline_llama8b_full

python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.1-8B-Instruct \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
  task_name=G1_baseline_llama8b_retain90
```
STATUS: [ ]

#### G1-B-3 to G1-B-11: Same as G1-A-3 to G1-A-11
Replace all occurrences:
- `llama1b` -> `llama8b`
- Model path -> 8B variant
- Add `model=Llama-3.1-8B-Instruct` override
- TFU config: update `tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct`

**Additional for 8B**: TFU with smaller (1B) helper:
```bash
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tfu/default \
    model=Llama-3.1-8B-Instruct \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
    model.w=$w \
    tfu.activation_method=naive \
    tfu.help_model.pretrained_model_name_or_path=meta-llama/Llama-3.2-1B-Instruct \
    forget_split=forget10 \
    task_name=G1_tfu_llama8b_naive_helper1b_w${w}_forget10
done
```
STATUS: [ ]

---

### G1-Phase-C: Qwen2.5-1.5B-Instruct (Needs Finetuning from Scratch)

#### G1-C-1: Finetune Qwen on TOFU Full Dataset
```bash
python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=Qwen2.5-1.5B-Instruct \
  task_name=G1_finetune_qwen1.5b_full
```
**UNCERTAINTY**: The finetune config defaults to Llama. May need to adjust:
- learning rate, batch size, or other hyperparams for Qwen
- model config override syntax

If errors occur: document, fix locally, commit, sync, resume.
STATUS: [ ]

#### G1-C-2: Finetune Qwen on Retain Splits
```bash
for split in retain90 retain95 retain99; do
  python src/train.py --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=Qwen2.5-1.5B-Instruct \
    "data/datasets@data.train=TOFU_QA_${split}" \
    task_name=G1_finetune_qwen1.5b_${split}
done
```
STATUS: [ ]

#### G1-C-3: Evaluate Qwen Baselines
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Qwen2.5-1.5B-Instruct \
  model.model_args.pretrained_model_name_or_path=./saves/finetune/G1_finetune_qwen1.5b_full \
  task_name=G1_baseline_qwen1.5b_full

python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Qwen2.5-1.5B-Instruct \
  model.model_args.pretrained_model_name_or_path=./saves/finetune/G1_finetune_qwen1.5b_retain90 \
  task_name=G1_baseline_qwen1.5b_retain90
```
STATUS: [ ]

#### G1-C-4: TFU Eval on Qwen (all sweeps)
**UNCERTAINTY**: TFU currently uses `TFULlamaForCausalLM` which extends `LlamaForCausalLM`.
Qwen2.5 uses `Qwen2ForCausalLM` -- TFU may NOT work out of the box with Qwen.

**Options**:
- (a) Create `TFUQwen2ForCausalLM` class (mirrors TFULlama but extends Qwen2)
- (b) Refactor TFU to be model-agnostic (use a mixin or wrapper)
- (c) Skip Qwen TFU and only run traditional methods on Qwen

**Ask user which option to take before proceeding.**
STATUS: [ ]

#### G1-C-5 to G1-C-11: Traditional Methods on Qwen
Same as G1-A-8 through G1-A-10 but with Qwen model overrides.
```bash
# UNDIAL example:
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  model=Qwen2.5-1.5B-Instruct \
  model.model_args.pretrained_model_name_or_path=./saves/finetune/G1_finetune_qwen1.5b_full \
  trainer=UNDIAL \
  forget_split=forget10 \
  retain_split=retain90 \
  task_name=G1_undial_qwen1.5b_forget10
```
STATUS: [ ]

---

### G1-Phase-D: Aggregate & Compare Results

#### G1-D-1: Collect All TOFU Results into Summary Table
```bash
# List all completed eval results:
ls -d saves/eval/G1_* | wc -l

# Generate summary table with links to result folders:
python -c "
import json, glob, os

rows = []
for path in sorted(glob.glob('saves/eval/G1_*/TOFU_SUMMARY.json')):
    folder = os.path.dirname(path)
    task_name = os.path.basename(folder)
    with open(path) as f:
        metrics = json.load(f)
    rows.append({
        'task_name': task_name,
        'result_folder': folder,
        **metrics
    })

# Write as JSON (machine-readable)
with open('saves/eval/G1_TOFU_ALL_RESULTS.json', 'w') as f:
    json.dump(rows, f, indent=2)

# Write as CSV (spreadsheet-friendly, includes folder path)
import csv
if rows:
    keys = rows[0].keys()
    with open('saves/eval/G1_TOFU_ALL_RESULTS.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

print(f'Aggregated {len(rows)} experiments')
print(f'JSON: saves/eval/G1_TOFU_ALL_RESULTS.json')
print(f'CSV:  saves/eval/G1_TOFU_ALL_RESULTS.csv')
"
```
STATUS: [ ]
BACKUP: `cp saves/eval/G1_TOFU_ALL_RESULTS.* /data/tfu_jx_backup/`

#### G1-D-2: Generate Comparison Table (Markdown with Result Links)
```bash
python -c "
import json, os

with open('saves/eval/G1_TOFU_ALL_RESULTS.json') as f:
    rows = json.load(f)

# Key metrics to display
METRICS = ['forget_quality', 'model_utility', 'forget_truth_ratio', 'privleak', 'extraction_strength', 'exact_memorization']

# Group by model and split
from collections import defaultdict
grouped = defaultdict(list)
for r in rows:
    # Parse task_name to extract model, method, split
    # Convention: G1_<type>_<model>_<method>_<params>_<split>
    grouped[r['task_name']] = r

# Generate markdown table
header = '| task_name | result_folder | ' + ' | '.join(METRICS) + ' |'
sep = '|' + '---|' * (len(METRICS) + 2)
lines = [header, sep]
for r in rows:
    vals = [str(r.get(m, 'N/A')) for m in METRICS]
    lines.append(f\"| {r['task_name']} | {r['result_folder']} | {' | '.join(vals)} |\")

table = '\n'.join(lines)
with open('saves/eval/G1_TOFU_COMPARISON.md', 'w') as f:
    f.write('# Group 1: TOFU Benchmark Results\n\n')
    f.write('Each row links to the full result folder containing TOFU_EVAL.json and TOFU_SUMMARY.json.\n\n')
    f.write(table)
    f.write('\n')

print('Table written to: saves/eval/G1_TOFU_COMPARISON.md')
print(f'Total rows: {len(rows)}')
"
```
Expected output: `saves/eval/G1_TOFU_COMPARISON.md` with format:
```
| task_name | result_folder | forget_quality | model_utility | ... |
|---|---|---|---|---|
| G1_tfu_llama1b_naive_w3_forget10 | saves/eval/G1_tfu_llama1b_naive_w3_forget10 | 0.85 | 0.58 | ... |
| G1_eval_GradAscent_llama1b_forget10 | saves/eval/G1_eval_GradAscent_llama1b_forget10 | 0.27 | 0.33 | ... |
```
STATUS: [ ]

---

## GROUP 2: MUSE Benchmark

---

### G2-Phase-A: Llama-3.2-1B-Instruct on MUSE

#### G2-A-1: Finetune Llama-1B on MUSE-News (full)
```bash
python src/train.py --config-name=train.yaml \
  experiment=finetune/muse/1b \
  data_split=News \
  task_name=G2_finetune_llama1b_muse_news_full
```
**UNCERTAINTY**: Check if `configs/experiment/finetune/muse/1b.yaml` needs edits for News split.
STATUS: [ ]

#### G2-A-2: Finetune Llama-1B on MUSE-News (retain)
```bash
python src/train.py --config-name=train.yaml \
  experiment=finetune/muse/1b \
  data_split=News \
  "data/datasets@data.train=MUSE_retain" \
  task_name=G2_finetune_llama1b_muse_news_retain
```
**UNCERTAINTY**: Exact config override for retain-only finetuning on MUSE may differ.
Check `configs/data/datasets/MUSE_*.yaml` for correct dataset name.
STATUS: [ ]

#### G2-A-3: Finetune Llama-1B on MUSE-Books (full + retain)
```bash
python src/train.py --config-name=train.yaml \
  experiment=finetune/muse/1b \
  data_split=Books \
  task_name=G2_finetune_llama1b_muse_books_full

python src/train.py --config-name=train.yaml \
  experiment=finetune/muse/1b \
  data_split=Books \
  "data/datasets@data.train=MUSE_retain" \
  task_name=G2_finetune_llama1b_muse_books_retain
```
STATUS: [ ]

#### G2-A-4: Evaluate MUSE Baselines
```bash
# News finetuned
python src/eval.py --config-name=eval.yaml \
  experiment=eval/muse/default \
  model.model_args.pretrained_model_name_or_path=./saves/finetune/G2_finetune_llama1b_muse_news_full \
  data_split=News \
  task_name=G2_baseline_llama1b_muse_news_full

# News retain
python src/eval.py --config-name=eval.yaml \
  experiment=eval/muse/default \
  model.model_args.pretrained_model_name_or_path=./saves/finetune/G2_finetune_llama1b_muse_news_retain \
  data_split=News \
  task_name=G2_baseline_llama1b_muse_news_retain
```
Repeat for Books split.
STATUS: [ ]

#### G2-A-5: Traditional Methods on MUSE-News (Llama-1B)
```bash
for method in GradAscent GradDiff NPO SimNPO UNDIAL; do
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=./saves/finetune/G2_finetune_llama1b_muse_news_full \
    trainer=$method \
    data_split=News \
    task_name=G2_${method}_llama1b_muse_news
done
```
STATUS: [ ]

#### G2-A-6: Evaluate Traditional Methods on MUSE-News
```bash
for method in GradAscent GradDiff NPO SimNPO UNDIAL; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/muse/default \
    model.model_args.pretrained_model_name_or_path=./saves/unlearn/G2_${method}_llama1b_muse_news \
    data_split=News \
    retain_logs_path=saves/eval/G2_baseline_llama1b_muse_news_retain/MUSE_EVAL.json \
    task_name=G2_eval_${method}_llama1b_muse_news
done
```
STATUS: [ ]

#### G2-A-7: TFU on MUSE-News (Llama-1B)
```bash
# Naive w sweep
for w in 0.5 1.0 1.25 1.5 1.75 2.0 2.25 2.50 2.75 3.0 4.0 5.0; do
  python src/eval.py --config-name=eval.yaml \
    experiment=eval/tfu/muse \
    model.model_args.pretrained_model_name_or_path=./saves/finetune/G2_finetune_llama1b_muse_news_full \
    model.w=$w \
    tfu.activation_method=naive \
    data_split=News \
    task_name=G2_tfu_llama1b_naive_w${w}_muse_news
done

# Similarity and Static sweeps (same pattern as G1-A-4/5)
```
**UNCERTAINTY**: TFU MUSE config (`eval/tfu/muse.yaml`) may need adjustment for:
- `pretrained_model_name_or_path` (should point to MUSE finetuned model)
- `tfu.help_model` (should be base Llama-1B-Instruct)
- FAISS dataset source (MUSE forget data instead of TOFU)

Review `configs/experiment/eval/tfu/muse.yaml` and adjust if needed before running.
STATUS: [ ]

#### G2-A-8 to G2-A-12: Repeat for Books split
Same as G2-A-5 to G2-A-7 but with `data_split=Books` and Books model paths.
STATUS: [ ]

---

### G2-Phase-B: Llama-3.1-8B-Instruct on MUSE
Same structure as Phase-A but 8B model. Higher GPU cost.
STATUS: [ ]

### G2-Phase-C: Qwen2.5-1.5B-Instruct on MUSE
Same structure. Same TFU compatibility uncertainty as G1-C-4.
STATUS: [ ]

### G2-Phase-D: Aggregate MUSE Results
Same pattern as G1-D-1/2 but with `MUSE_SUMMARY.json` files.
```bash
python -c "
import json, glob, os, csv

rows = []
for path in sorted(glob.glob('saves/eval/G2_*/MUSE_SUMMARY.json')):
    folder = os.path.dirname(path)
    task_name = os.path.basename(folder)
    with open(path) as f:
        metrics = json.load(f)
    rows.append({'task_name': task_name, 'result_folder': folder, **metrics})

with open('saves/eval/G2_MUSE_ALL_RESULTS.json', 'w') as f:
    json.dump(rows, f, indent=2)
if rows:
    with open('saves/eval/G2_MUSE_ALL_RESULTS.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(f'Aggregated {len(rows)} MUSE experiments')
"
```
Output: `saves/eval/G2_MUSE_ALL_RESULTS.json`, `.csv`
STATUS: [ ]

---

## GROUP 3: WMDP Benchmark

---

### G3-Phase-A: Pipeline Verification (Zephyr-7b)

#### G3-A-1: Verify WMDP Data Exists
```bash
ls data/wmdp/wmdp-corpora/
# Expected: cyber-forget-corpus.jsonl, cyber-retain-corpus.jsonl,
#           bio-forget-corpus.jsonl, bio-retain-corpus.jsonl
```
STATUS: [ ]

#### G3-A-2: Test WMDP Pipeline End-to-End (single method, small run)
```bash
# Quick sanity check with 1 epoch:
python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/wmdp/default \
  trainer=GradAscent \
  trainer.args.num_train_epochs=1 \
  task_name=G3_test_zephyr_gradascent_1epoch
```
If this fails: document error, fix, commit, sync, resume.
STATUS: [ ]

#### G3-A-3: Test WMDP Evaluation
```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/wmdp/default \
  model.model_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta \
  task_name=G3_test_eval_zephyr_baseline
```
Expected: MMLU, wmdp_cyber, wmdp_bio accuracy scores
STATUS: [ ]

#### G3-A-4: Full Traditional Methods (Zephyr-7b)
```bash
for method in GradAscent GradDiff NPO SimNPO RMU UNDIAL; do
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/wmdp/default \
    trainer=$method \
    task_name=G3_${method}_zephyr_cyber

  python src/eval.py --config-name=eval.yaml \
    experiment=eval/wmdp/default \
    model.model_args.pretrained_model_name_or_path=./saves/unlearn/G3_${method}_zephyr_cyber \
    task_name=G3_eval_${method}_zephyr_cyber
done
```
**UNCERTAINTY**: WMDP default config may train on cyber only. Need to check if
bio requires separate runs or if both splits are handled together.
STATUS: [ ]

#### G3-A-5: TFU on WMDP (Zephyr-7b)
**UNCERTAINTY**: TFU uses `TFULlamaForCausalLM` which extends LlamaForCausalLM.
Zephyr-7b is Mistral-based (`MistralForCausalLM`), NOT Llama.

**Options**:
- (a) Create `TFUMistralForCausalLM` class
- (b) Refactor TFU to be architecture-agnostic
- (c) Use only Llama-1B for WMDP TFU experiments (skip Zephyr TFU)

**Ask user which option to take before proceeding.**
STATUS: [ ]

---

### G3-Phase-B: Llama-3.2-1B on WMDP (Secondary)

#### G3-B-1: Adapt WMDP Config for Llama-1B
Create or modify config to use Llama-3.2-1B-Instruct instead of Zephyr.
```bash
# May need to adjust:
# - model config override
# - RMU module_regex (layer targeting differs between architectures)
# - learning rate / batch size
```
**UNCERTAINTY**: RMU's `module_regex: model.layers.7` is architecture-specific.
Need to verify correct layer naming for Llama-3.2-1B.
STATUS: [ ]

#### G3-B-2 onwards: Same pattern as G3-A-4 but for Llama-1B
STATUS: [ ]

### G3-Phase-C: Aggregate WMDP Results
Same pattern but WMDP uses lm-eval output format. Check output file name (may not be `WMDP_SUMMARY.json`).
```bash
# Find what output files exist:
ls saves/eval/G3_*/ | head -20
# Then aggregate similarly to G1-D-1, adapting for the actual file format.
```
STATUS: [ ]

---

## CROSS-GROUP ANALYSIS (After All Groups Complete)

### X-1: Generate Cross-Benchmark Summary
Aggregate best TFU config per benchmark vs traditional methods.
Produce a single master table with columns:
```
| method | model | benchmark | split | best_w | best_threshold | forget_metric | utility_metric | result_folder |
```
```bash
python -c "
import json

# Load all group results
all_results = []
for f in ['saves/eval/G1_TOFU_ALL_RESULTS.json', 'saves/eval/G2_MUSE_ALL_RESULTS.json']:
    try:
        with open(f) as fh:
            all_results.extend(json.load(fh))
    except FileNotFoundError:
        print(f'Skipping {f} (not found)')

with open('saves/eval/MASTER_ALL_RESULTS.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f'Master table: {len(all_results)} total experiments')
print('Output: saves/eval/MASTER_ALL_RESULTS.json')
"
```
STATUS: [ ]

### X-2: Generate Pareto Frontier Plots
For each model/benchmark: plot forgetting metric vs utility metric, one point per method.
STATUS: [ ]

### X-3: Generate w Sensitivity Curves
For each TFU variant: plot metric vs w value.
STATUS: [ ]

### X-4: Write Final Comparison Report
STATUS: [ ]

---

## QUICK REFERENCE: Manual Continuation

If LLM is unavailable, user can continue manually:

1. **Find last checkpoint**: Search this file for last `[x]` status
2. **Find next step**: The next `[ ]` step after the last `[x]`
3. **Run the command**: Copy-paste the bash command from that step
4. **Check output**: Verify expected files exist
5. **Mark complete**: Update status to `[x]` with date
6. **Backup**: Run backup script after major steps

### Useful commands for debugging:
```bash
# Check what's been completed:
ls saves/eval/ | grep "G1_" | wc -l
ls saves/unlearn/ | grep "G1_" | wc -l

# Check a specific result:
cat saves/eval/<task_name>/TOFU_SUMMARY.json | python -m json.tool

# Re-run a failed eval (just re-run the command, eval is idempotent)

# Check GPU usage:
nvidia-smi

# Kill a hung process:
pkill -f "python src/train.py"
```

---

## UNCERTAINTIES LOG

Track unresolved questions here. Each needs user decision before proceeding.

| ID | Step | Question | Options | Decision | Date |
|----|------|----------|---------|----------|------|
| U1 | G1-A-10 | Repro model checkpoints -- exist on server? | (a) `ls saves/unlearn/` to check (b) retrain (c) use HF eval logs only | PENDING | |
| U2 | G1-C-4 | TFU + Qwen compatibility (TFULlamaForCausalLM) | (a) create TFUQwen2 class (b) refactor to agnostic (c) skip Qwen TFU | PENDING | |
| U3 | G3-A-5 | TFU + Zephyr/Mistral compatibility | (a) create TFUMistral class (b) refactor (c) skip Zephyr TFU | PENDING | |
| U4 | G3-A-4 | WMDP bio vs cyber: separate or combined training? | Check config | PENDING | |
| U5 | G1-A-11 | Finetune helper for forget01/forget05 -- does `experiment=finetune/tofu/forget10` accept arbitrary data override? | Test with `data/datasets@data.train=TOFU_QA_forget05` | PENDING | |
| U6 | G2-A-1 | MUSE finetune config (`finetune/muse/1b.yaml`) -- does it handle News/Books via `data_split` param or need separate configs? | Read the yaml first | PENDING | |
