## TFU

### Eval on retain original
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default task_name=EVAL_LLAMA_1B_RETAIN
```
```
{
    "extraction_strength": 0.05994265088359635,
    "forget_Q_A_Prob": 0.11614424599101766,
    "forget_Q_A_ROUGE": 0.37904325318231613,
    "forget_truth_ratio": 0.6274000611015691,
    "model_utility": 0.590838646120328,
    "privleak": 23.394999995321008
}
```

### Method 1 RAG-based
Eval TFU with RAG, ```final_logits = w * logits + (1 - w) * RAG_logits```.
There are 3 activation_method options: similarity, static, naive:
#### Similarity: w.r.t ```w = 1``` if ```RAG_score < 0.55``` else ```w = RAG_score * (model.w - 1) + 1```
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default model.w=3 tfu.activation_method=similarity task_name=EVAL_LLAMA_1B_TFU_act_sim_w3
```
```
{
    "extraction_strength": 0.08150354650809964,
    "forget_Q_A_Prob": 0.14329776889418555,
    "forget_Q_A_ROUGE": 0.3472742549348874,
    "forget_truth_ratio": 0.39785632997451265,
    "model_utility": 0.5994075065906783,
    "privleak": 28.979999994204004
}
```
#### Static: w.r.t ```w = 1``` if ```RAG_score < 0.55``` else ```w = model.w```
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default model.w=3 tfu.activation_method=static task_name=EVAL_LLAMA_1B_TFU_act_sta_w3
```
```
{
    "extraction_strength": 0.05682896705450961,
    "forget_Q_A_Prob": 0.0569404341051488,
    "forget_Q_A_ROUGE": 0.2907486786893424,
    "forget_truth_ratio": 0.3621175437699432,
    "model_utility": 0.6003274515846244,
    "privleak": 73.19124998536175
}
```
#### Naive: w.r.t ```w = model.w```
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/default model.w=3 tfu.activation_method=naive task_name=EVAL_LLAMA_1B_TFU_act_nav_w3
```
```
# w = 1.5
{
    "extraction_strength": 0.23837091182905598,
    "forget_Q_A_Prob": 0.6210900763049721,
    "forget_Q_A_ROUGE": 0.5750646389196481,
    "forget_truth_ratio": 0.4518670760954884,
    "model_utility": 0.5861980035254123,
    "privleak": -97.93874998041225
}
# w = 2
{
    "extraction_strength": 0.10238732467852961,
    "forget_Q_A_Prob": 0.28764765362255273,
    "forget_Q_A_ROUGE": 0.42176321884822554,
    "forget_truth_ratio": 0.43547580504476197,
    "model_utility": 0.5397761832677919,
    "privleak": -88.31624998233679
}
# w = 2.5
{
    "extraction_strength": 0.06549367296520915,
    "forget_Q_A_Prob": 0.12288217329380131,
    "forget_Q_A_ROUGE": 0.33837595002750787,
    "forget_truth_ratio": 0.39906461495518136,
    "model_utility": 0.4843262639719512,
    "privleak": -66.91999998661598
}
# w = 3
{
    "extraction_strength": 0.05682896705450961,
    "forget_Q_A_Prob": 0.0569404341051488,
    "forget_Q_A_ROUGE": 0.2907486786893424,
    "forget_truth_ratio": 0.3621175437699432,
    "model_utility": 0.42851771149668133,
    "privleak": -43.75124999124974
}
```
### Method 2 RAG-based
#### Finetune llama on forget10, model saved to saves/finetune/SAMPLE_TRAIN_forget
```bash
python src/train.py --config-name=train.yaml experiment=finetune/tofu/forget10 task_name=SAMPLE_TRAIN_forget
```
#### Eval tfu w.r.t finetuned model, final logits = w*logits + (1-w)*logits
note: using above model as help_model, see configs/experiment/eval/tfu/finetuned.yaml help_model.pretrained_model_name_or_path: "./saves/finetune/SAMPLE_TRAIN_forget"
```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tfu/finetuned model.w=3 task_name=EVAL_LLAMA_1B_TFU_finetuned_w3
```