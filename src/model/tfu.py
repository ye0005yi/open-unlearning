from transformers import AutoConfig, LlamaForCausalLM
import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import copy

from data.utils import preprocess_chat_instance, preprocess_pretraining_instance
from data.utils import IGNORE_INDEX

from langchain.globals import set_debug

set_debug(False)
logger = logging.getLogger("model")


class TFULlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.w = 1.
        self.w_adj = torch.tensor(1)
        self.gen_mode = False
        self.gen_past_key_values = None
        self.activations = {
            'naive': self.activation_naive,
            'static':self.activation_static,
            'similarity': self.activation_similarity
        }
        self.activation_method = "similarity"
        self.activation_threshold = 0.55
        self.enhanced_max_length = getattr(config, 'max_position_embeddings', 4096)
    
    def set_activation(self, method, threshold):
        if method not in self.activations:
            logger.error(f'{method} not supported, supported list: {list(self.activations.keys())}')
            return
        logger.info(f'Set activation_method to {method}, threshold: {threshold}, previous is {self.activation_method}, threshold: {self.activation_threshold}.')
        self.activation_method = method
        self.activation_threshold = threshold

    def activation_naive(self, scores):
        self.w_adj = scores.to(device=self.device, copy=True)
        self.w_adj[:] = self.w
        return
    
    def activation_static(self, scores):
        self.w_adj = scores.to(device=self.device, copy=True)
        self.w_adj[scores <= self.activation_threshold] = 0
        self.w_adj[scores > self.activation_threshold] = self.w - 1
        self.w_adj += 1
        return

    def activation_similarity(self, scores):
        self.w_adj = scores.to(device=self.device, copy=True)
        self.w_adj[scores <= self.activation_threshold] = 0
        self.w_adj[scores > self.activation_threshold] *= self.w - 1
        self.w_adj += 1
        return
    
    def adjust_w(self, scores):
        self.activations[self.activation_method](scores)
        return  

    def _construct_enhanced(self, question, answer, enhanced_list, process):
        temp_en = '\n'.join(enhanced_list)
        temp_en = f" Use the following context to help:\n{temp_en}\n"
        enhanced_max_length = self.enhanced_max_length
        if process['type'] == 'qa':
            template_args = copy.deepcopy(process['template_args'])
            if template_args["apply_chat_template"]:
                system_prompt = template_args.get('system_prompt', "\nYou are a helpful assistant.")
                system_prompt += temp_en
                template_args.update({'system_prompt': system_prompt})
            else:
                system_prompt_with_special_tokens = f"<|system|>\nYou are a helpful assistant.{temp_en}<|end|>\n"
                template_args.update({'system_prompt_with_special_tokens': system_prompt_with_special_tokens})
            process_func = process['func']
            tokenized_data = process_func(
                process['tokenizer'],
                template_args,
                [question],
                [answer],
                enhanced_max_length,
                process['predict_with_generate']
            )
        elif process['type'] == 'pretraining':
            prefix = temp_en + "\n\n" + question
            text = answer
            process_func = process['func']
            tokenized_data = process_func(
                process['tokenizer'],
                prefix,
                text,
                enhanced_max_length,
                process['predict_with_generate'],
                process['insert_space']
            )
        else:
            raise NotImplementedError(f"Process type {process['type']} not supported.")

        return tokenized_data

    def __construct_enhanced_ids(self, process):
        if process[0]['type'] == 'qa':
            questions = [i['question'][0] for i in process]
            answers = [i['answer'][0] for i in process]
        elif process[0]['type'] == 'pretraining':
            questions = [i['prefix'] for i in process]
            answers = [i['text'] for i in process]
        enhanced_ori = self.retriever.batch(questions)
        enhanced = [[i.page_content for i, _ in bchs] for bchs in enhanced_ori]
        enhanced_scores = [torch.mean(torch.tensor([score for _, score in bchs])) for bchs in enhanced_ori]
        enhanced_scores = torch.tensor(enhanced_scores)
        self.adjust_w(enhanced_scores)

        #print([ (q, e[0][0], e[0][1]) for q, e in zip(questions, enhanced_ori)])
        batch_enhanced = [self._construct_enhanced(q, a, enhanced_list, p) for q, a, enhanced_list, p in zip(questions, answers, enhanced, process)]

        batch_enhanced = self._collators(batch_enhanced)
        batch_enhanced = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch_enhanced.items()}
        return batch_enhanced

    def _construct_enhanced_ids(self, process, input_ids, kwargs):
        if getattr(self, "retriever", None):
            return self.__construct_enhanced_ids(process)
        elif input_ids is not None:
            batch_sz = len(input_ids)
            self.adjust_w(torch.ones(batch_sz))
            return {'input_ids': input_ids, 'attention_mask': kwargs['attention_mask']}
        else:
            batch_sz = len(kwargs['input_ids'])
            self.adjust_w(torch.ones(batch_sz))
            return kwargs

    def _get_enhanced_logits(self, batch_enhanced, mask):
        is_all_one = torch.all(torch.abs(self.w_adj - 1.0) < 1e-6)
        if is_all_one:
            return None
        ret_enh = self.help_model(**batch_enhanced)
        ret_enh_unignore = ret_enh.logits[mask]
        if batch_enhanced.get('use_cache', False):
            self.gen_past_key_values = ret_enh.past_key_values
        return ret_enh_unignore
    
    def _compose_logits(self, ret_ori, ret_enh_unignore, mask):
        is_all_one = torch.all(torch.abs(self.w_adj - 1.0) < 1e-6)
        if is_all_one:
            return ret_ori
        ret_ori_unignore = ret_ori.logits[mask]
        tmp_adj = self.w_adj
        tmp_adj = tmp_adj.unsqueeze(-1)
        if isinstance(mask, tuple):
            tmp_adj = tmp_adj.unsqueeze(-1)
        else:
            tmp_adj = tmp_adj.expand(mask.shape)[mask][:, None]
        ret_ori.logits[mask] = tmp_adj * ret_ori_unignore + (1 - tmp_adj) * ret_enh_unignore
        return ret_ori

    def forward(self, *args, **kwargs):
        process = kwargs.pop('process', None)
        if not self.gen_mode and process != None and self._data is not None:
            batch_enhanced = self._construct_enhanced_ids(process, None, kwargs)
            ret_enh_unignore = self._get_enhanced_logits(batch_enhanced, batch_enhanced['labels'] != IGNORE_INDEX)
        if self.gen_mode:
            if self.first_time == 0:
                self.gen_past_key_values = DynamicCache()
                enhanced_ids = self.batch_enhanced_ids
            else:
                self.batch_enhanced_attention_mask = torch.cat([self.batch_enhanced_attention_mask, self.pre_cal_append_att_mask], dim=-1)
                assert self.gen_past_key_values != None
                enhanced_ids = kwargs['input_ids'][:, -1:]
            batch_enhanced = {'input_ids': enhanced_ids, 'attention_mask': self.batch_enhanced_attention_mask, 'use_cache': True, 'past_key_values':self.gen_past_key_values}
            ret_enh_unignore = self._get_enhanced_logits(batch_enhanced, (slice(None), slice(-1, None), slice(None)))
            self.first_time += 1

        ret_ori = super().forward(*args, **kwargs)

        if not self.gen_mode and process != None and self._data is not None:
            ret_ori = self._compose_logits(ret_ori, ret_enh_unignore, kwargs['labels'] != IGNORE_INDEX)
        if self.gen_mode:
            ret_ori = self._compose_logits(ret_ori, ret_enh_unignore, (slice(None), slice(-1, None), slice(None)))

        return ret_ori
    
    def generate(self, *args, **kwargs):
        input_ids = args[0]
        _ = kwargs.pop('input_ids', None)
        labels = kwargs.pop('labels', None)
        process = kwargs.pop('process', None)

        if process != None and self._data is not None:
            self.gen_mode = True
            self.first_time = 0
            tmp_enhanced = self._construct_enhanced_ids(process, input_ids, kwargs)
            self.batch_enhanced_ids = tmp_enhanced['input_ids']
            self.batch_enhanced_attention_mask = tmp_enhanced['attention_mask']
            # pre calcualted mask to append batch_size * 1, looks like [[1], [1], ..., [1]]
            self.pre_cal_append_att_mask = torch.ones(len(self.batch_enhanced_ids), 1).to(self.device)

        ret = super().generate(*args, **kwargs)

        if process != None and self._data is not None:
            self.gen_mode = False
            self.first_time = 0
            self.batch_enhanced_ids = None
            self.batch_enhanced_attention_mask = None
            self.gen_past_key_values = None
        return ret


def _make_tfu_class(base_cls):
    """Create a TFU variant for any CausalLM base class by copying TFU methods."""
    import types

    new_cls = type(f"TFU{base_cls.__name__}", (base_cls,), {})

    for k, v in TFULlamaForCausalLM.__dict__.items():
        if k.startswith('__') and k != '__init__':
            continue
        if isinstance(v, types.FunctionType):
            if v.__closure__ and '__class__' in v.__code__.co_freevars:
                idx = v.__code__.co_freevars.index('__class__')
                new_closure = list(v.__closure__)
                new_closure[idx] = types.CellType(new_cls)
                v = types.FunctionType(
                    v.__code__, v.__globals__, v.__name__,
                    v.__defaults__, tuple(new_closure))
            setattr(new_cls, k, v)

    return new_cls


try:
    from transformers import Qwen2ForCausalLM
    TFUQwen2ForCausalLM = _make_tfu_class(Qwen2ForCausalLM)
except ImportError:
    pass

try:
    from transformers import MistralForCausalLM
    TFUMistralForCausalLM = _make_tfu_class(MistralForCausalLM)
except ImportError:
    pass