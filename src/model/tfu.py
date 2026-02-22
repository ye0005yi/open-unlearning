from transformers import AutoConfig, LlamaForCausalLM 
import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import copy

from data.utils import preprocess_chat_instance
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
        self.w_adj[scores <= 0.55] = 0
        self.w_adj[scores > 0.55] = self.w - 1
        self.w_adj += 1
        return

    def activation_similarity(self, scores):
        self.w_adj = scores.to(device=self.device, copy=True)
        self.w_adj[scores <= 0.55] = 0
        self.w_adj[scores > 0.55] *= self.w - 1
        self.w_adj += 1
        return
    
    def adjust_w(self, scores):
        self.activations[self.activation_method](scores)
        return  

    def _construct_enhanced(self, question, answer, enhanced_list):
        temp_en = '\n'.join(enhanced_list)
        temp_en = f" Use the following context to help:\n{temp_en}\n"
        template_args = copy.deepcopy(self._data.template_args)
        if template_args["apply_chat_template"]:
            system_prompt = template_args.get('system_prompt', "\nYou are a helpful assistant.")
            system_prompt += temp_en
            template_args.update({'system_prompt': system_prompt})
        else:
            system_prompt_with_special_tokens = f"<|system|>\nYou are a helpful assistant.{temp_en}<|end|>\n"
            template_args.update({'system_prompt_with_special_tokens': system_prompt_with_special_tokens})

        tokenized_data = preprocess_chat_instance(
            self._data.tokenizer,
            template_args,
            [question],
            [answer],
            self._data.max_length,
            self._data.predict_with_generate,
        )

        return tokenized_data

    def _construct_enhanced_ids(self, questions, answers):
        questions = [i[0] for i in questions]
        answers = [i[0] for i in answers]
        enhanced_ori = self.retriever.batch(questions)
        enhanced = [[i.page_content for i, _ in bchs] for bchs in enhanced_ori]
        enhanced_scores = [torch.mean(torch.tensor([score for _, score in bchs])) for bchs in enhanced_ori]
        enhanced_scores = torch.tensor(enhanced_scores)
        self.adjust_w(enhanced_scores)

        #print([ (q, e[0][0], e[0][1]) for q, e in zip(questions, enhanced_ori)])
        batch_enhanced = [self._construct_enhanced(q, a, enhanced_list) for q, a, enhanced_list in zip(questions, answers, enhanced)]

        batch_enhanced = self._collators(batch_enhanced)
        batch_enhanced = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch_enhanced.items()}
        return batch_enhanced

    def _get_enhanced_logits(self, batch_enhanced, mask):
        is_all_one = torch.all(torch.abs(self.w_adj - 1.0) < 1e-6)
        if is_all_one:
            return None
        #ret_enh = super().forward(**batch_enhanced)
        ret_enh = self.help_model(**batch_enhanced)
        ret_enh_unignore = ret_enh.logits[mask]
        if batch_enhanced.get('use_cache', False):
            self.past_key_values = ret_enh.past_key_values
        return ret_enh_unignore
    
    def _compose_logits(self, ret_ori, ret_enh_unignore, mask):
        is_all_one = torch.all(torch.abs(self.w_adj - 1.0) < 1e-6)
        if is_all_one:
            return ret_ori
        ret_ori_unignore = ret_ori.logits[mask]
        assert ret_ori_unignore.shape == ret_enh_unignore.shape
        tmp_adj = self.w_adj
        tmp_adj = tmp_adj.unsqueeze(-1)
        if isinstance(mask, tuple):
            tmp_adj = tmp_adj.unsqueeze(-1)
        else:
            tmp_adj = tmp_adj.expand(mask.shape)[mask][:, None]
        ret_ori.logits[mask] *= tmp_adj
        ret_ori.logits[mask] += (1 - tmp_adj) * ret_enh_unignore
        return ret_ori

    def forward(self, *args, **kwargs):
        questions = kwargs.pop('questions', None)
        answers = kwargs.pop('answers', None)
        if not self.gen_mode and questions != None and self._data is not None:
            batch_enhanced = self._construct_enhanced_ids(questions, answers)
            ret_enh_unignore = self._get_enhanced_logits(batch_enhanced, batch_enhanced['labels'] != IGNORE_INDEX)
        if self.gen_mode:
            if self.first_time == 0:
                self.past_key_values = DynamicCache()
                enhanced_ids = self.batch_enhanced_ids
            else:
                #self.batch_enhanced_ids = torch.cat([self.batch_enhanced_ids, kwargs['input_ids'][:, -1:]], dim=-1)
                self.batch_enhanced_attention_mask = torch.cat([self.batch_enhanced_attention_mask, self.pre_cal_append_att_mask], dim=-1)
                assert self.past_key_values != None
                enhanced_ids = kwargs['input_ids'][:, -1:]
            batch_enhanced = {'input_ids': enhanced_ids, 'attention_mask': self.batch_enhanced_attention_mask, 'use_cache': True, 'past_key_values':self.past_key_values}
            #print(self._data.tokenizer.decode(enhanced_ids[0]), end='')
            ret_enh_unignore = self._get_enhanced_logits(batch_enhanced, (slice(None), slice(-1, None), slice(None))) #[:, -1:, :]
            self.first_time += 1
            #print(self._data.tokenizer.decode(kwargs['input_ids'][0]), end='')

        ret_ori = super().forward(*args, **kwargs)

        if not self.gen_mode and questions != None and self._data is not None:
            ret_ori = self._compose_logits(ret_ori, ret_enh_unignore, kwargs['labels'] != IGNORE_INDEX)
        if self.gen_mode:
            ret_ori = self._compose_logits(ret_ori, ret_enh_unignore, (slice(None), slice(-1, None), slice(None))) #[:, -1:, :]

        #print(f"Allocated8: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        return ret_ori
    
    def generate(self, *args, **kwargs):
        input_ids = args[0]
        _ = kwargs.pop('input_ids', None)
        labels = kwargs.pop('labels', None)
        questions = kwargs.pop('questions', None)
        answers = kwargs.pop('answers', None)

        if questions != None and self._data is not None:
            self.gen_mode = True
            self.first_time = 0
            tmp_enhanced = self._construct_enhanced_ids(questions, answers)
            self.batch_enhanced_ids = tmp_enhanced['input_ids']
            self.batch_enhanced_attention_mask = tmp_enhanced['attention_mask']
            # pre calcualted mask to append batch_size * 1, looks like [[1], [1], ..., [1]]
            self.pre_cal_append_att_mask = torch.ones(len(self.batch_enhanced_ids), 1).to(self.device)
            print("---------------- generate --------------")
        ret = super().generate(*args, **kwargs)

        if questions != None and self._data is not None:
            self.gen_mode = False
            self.first_time = 0
            self.batch_enhanced_ids = None
            self.batch_enhanced_attention_mask = None
            self.gen_past_key_values = None
        return ret