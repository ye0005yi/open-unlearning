import os
import hashlib
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from data import get_unlearn_data, get_collators

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators
from transformers import AutoModelForCausalLM

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger("eval")

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    tfu_cfg = cfg.get('tfu', None)
    if tfu_cfg is not None:
        FAISS_model = tfu_cfg.get('FAISS_model', None)
        if FAISS_model is not None:
            data_cfg = cfg.get('data', None)
            assert data_cfg != None
            cache_key = hashlib.md5((OmegaConf.to_yaml(data_cfg) + FAISS_model).encode()).hexdigest()
            cache_dir = os.path.join(".cache", "faiss", cache_key)
            embeddings = HuggingFaceEmbeddings(model_name=FAISS_model)
            if os.path.exists(cache_dir):
                logger.info(f"Loading cached FAISS index from {cache_dir}")
                db = FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
            else:
                logger.info("Building FAISS index from scratch (will be cached for reuse)")
                data = get_unlearn_data(data_cfg, tokenizer=tokenizer, template_args=template_args)
                data = [Document(page_content=d) for d in data]
                db = FAISS.from_documents(data, embeddings)
                os.makedirs(cache_dir, exist_ok=True)
                db.save_local(cache_dir)
                logger.info(f"Saved FAISS index to {cache_dir}")
            #retriever = db.as_retriever(search_type=tfu_cfg.get('search_type'), search_kwargs={"k": int(tfu_cfg.get('search_topk'))})
            #retriever = RunnableLambda(lambda q: db.similarity_search_with_score(q, k=int(tfu_cfg.get('search_topk'))))
            retriever = RunnableLambda(lambda q: db.similarity_search_with_relevance_scores(q, k=int(tfu_cfg.get('search_topk'))))
            model.retriever = retriever

        model.help_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = tfu_cfg.get('help_model').get('pretrained_model_name_or_path'),
            attn_implementation = tfu_cfg.get('help_model').get('attn_implementation'),
            torch_dtype = tfu_cfg.get('help_model').get('torch_dtype'))
        model.help_model.to(model.device)
        model.set_activation(tfu_cfg.get('activation_method'), tfu_cfg.get('activation_threshold'))
        if tfu_cfg.get('enhanced_max_length', None) is not None:
            model.enhanced_max_length = tfu_cfg.get('enhanced_max_length')

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
