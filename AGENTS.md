# AGENTS.md - open-unlearning

A unified framework for LLM unlearning benchmarking with pluggable methods, datasets, metrics, and evaluators.

## Quick Reference

```bash
# Install
pip install .[lm_eval,dev]

# Lint / format
make quality        # check (ruff)
make style          # auto-fix (ruff)

# Train / unlearn
python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent task_name=MY_TASK

# Evaluate
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default task_name=MY_EVAL

# Data setup
python setup_data.py --eval
```

## Architecture

### Registry Pattern

All major components use a registry. To add a new component, implement the class and register it:

| Registry | Location | Examples |
|----------|----------|----------|
| `TRAINER_REGISTRY` | `src/trainer/` | GradAscent, GradDiff, NPO, DPO, RMU, CEU, ... |
| `DATASET_REGISTRY` | `src/data/` | QADataset, PretrainingDataset, ForgetRetainDataset |
| `COLLATOR_REGISTRY` | `src/data/collators.py` | DataCollatorForSupervisedDataset |
| `MODEL_REGISTRY` | `src/model/` | ProbeModelForCausalLM, TFULlamaForCausalLM |
| `EVALUATOR_REGISTRY` | `src/evals/` | TOFUEvaluator, MUSEEvaluator, LMEvalEvaluator |

### Entry Points

- `src/train.py` - Training and unlearning pipeline
- `src/eval.py` - Evaluation pipeline

### Configuration (Hydra)

All configs are in `configs/` with a hierarchical YAML structure. Experiments compose model + trainer + data + eval configs:

```
configs/
  train.yaml, eval.yaml, unlearn.yaml     # top-level
  experiment/                               # full experiment configs
    unlearn/{tofu,muse,wmdp}/
    finetune/{tofu,muse}/
    eval/{tofu,muse,wmdp,tfu}/
  trainer/                                  # method hyperparams
  model/                                    # model specs
  data/                                     # dataset definitions
  eval/                                     # evaluator configs
```

Override any config value via CLI: `python src/train.py ... trainer=GradDiff model.w=3`

### Naming Conventions

- Methods: CamelCase (`GradAscent`, `TFULlamaForCausalLM`)
- Benchmarks: UPPERCASE (`TOFU`, `MUSE`, `WMDP`)
- Metrics: snake_case (`extraction_strength`, `forget_truth_ratio`)
- Dataset splits: `forget10/50/90`, `retain90/95/99`, `eval`, `full`

## TFU (Test-time Fine-tuning Unlearning)

TFU is on the `tfu` branch. It works at inference time by combining logits from the main model and a helper model:

```
final_logits = w * logits + (1 - w) * help_logits
```

- **RAG-based**: helper logits come from FAISS retrieval context (`configs/experiment/eval/tfu/default.yaml`)
- **Fine-tuned**: helper logits come from a model fine-tuned on forget data (`configs/experiment/eval/tfu/finetuned.yaml`)
- **Activation methods**: `similarity`, `static`, `naive` - control how `w` adapts based on retrieval score
- Core implementation: `src/model/tfu.py`
- Practical usage examples: [sh_extra.md](sh_extra.md)

## Contributing New Methods

Follow the template in `community/methods/template/`. See [docs/contributing.md](docs/contributing.md) and [docs/components.md](docs/components.md) for details on adding trainers, datasets, metrics, or models.

## Key Documentation

- [README.md](README.md) - Overview, quickstart, component tables
- [docs/experiments.md](docs/experiments.md) - Running experiments
- [docs/hydra.md](docs/hydra.md) - Hydra config tutorial
- [docs/evaluation.md](docs/evaluation.md) - Metrics and evaluation
- [docs/repro.md](docs/repro.md) - Baseline reproduction results
