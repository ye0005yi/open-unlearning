import logging
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from data.utils import preprocess_chat_instance, IGNORE_INDEX
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.base import unlearning_metric

logger = logging.getLogger("evaluator")


def _build_mc_sample(tokenizer, template_args, question, choice, max_length):
    """Format a single MC question+choice as a sample with process dict for TFU."""
    tokenized = preprocess_chat_instance(
        tokenizer, template_args, [question], [choice], max_length, False
    )
    process = {
        "type": "qa",
        "func": preprocess_chat_instance,
        "tokenizer": tokenizer,
        "template_args": template_args,
        "question": [question],
        "answer": [choice],
        "predict_with_generate": False,
    }
    tokenized["process"] = process
    return tokenized


def _compute_choice_logprob(logits, labels):
    """Compute average log-probability of labeled (non-ignored) tokens."""
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
    mask = shifted_labels != IGNORE_INDEX
    if mask.sum() == 0:
        return float('-inf')
    target_log_probs = torch.gather(
        log_probs, dim=-1, index=shifted_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    masked_log_probs = target_log_probs[mask]
    return masked_log_probs.mean().item()


@unlearning_metric(name="mc_accuracy")
def mc_accuracy(model, **kwargs):
    """Multiple-choice accuracy with TFU composition active.

    Loads WMDP MC questions, formats each question+choice as a batch with
    process dict (enabling TFU retrieval), computes log-likelihood per choice,
    and reports accuracy.
    """
    tokenizer = kwargs["tokenizer"]
    template_args = kwargs["template_args"]
    hf_dataset = kwargs["hf_dataset"]
    hf_subset = kwargs["hf_subset"]
    max_length = kwargs.get("max_length", 2048)
    cache_dir = kwargs.get("cache_dir", None)

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, index=None)

    if not hasattr(model, '_data') or model._data is None:
        model._data = True
    if not hasattr(model, '_collators') or model._collators is None:
        model._collators = collator

    ds = load_dataset(hf_dataset, hf_subset, split="test", cache_dir=cache_dir)

    correct = 0
    total = 0
    value_by_index = {}

    for idx, item in enumerate(tqdm(ds, desc=f"MC Accuracy ({hf_subset})")):
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]

        choice_logprobs = []
        for choice in choices:
            sample = _build_mc_sample(tokenizer, template_args, question, choice, max_length)
            batch = collator([sample])
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v
                     for k, v in batch.items()}

            with torch.no_grad():
                output = model(**batch)

            logprob = _compute_choice_logprob(output.logits, batch["labels"])
            choice_logprobs.append(logprob)

        predicted = int(np.argmax(choice_logprobs))
        is_correct = predicted == answer_idx
        if is_correct:
            correct += 1
        total += 1
        value_by_index[idx] = {"predicted": predicted, "correct": answer_idx, "hit": is_correct}

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"MC Accuracy ({hf_subset}): {accuracy:.4f} ({correct}/{total})")
    return {"agg_value": accuracy, "value_by_index": value_by_index}
