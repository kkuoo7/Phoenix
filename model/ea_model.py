import copy
import json
import time

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .configs import EConfig

from typing import Optional
from .utils import *
from HASS.evaluation.collapse_collector import CollapseCollector



class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            setattr(self.ea_layer, '_diff_device', True)
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            setattr(self.ea_layer, '_diff_device', False)
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @property
    def device(self):
        return self.base_model.lm_head.weight.device

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        if base_model_path is None:
            raise ValueError("base_model_path cannot be None")
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type=='Qwen2ForCausalLM':
            base_model=KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        if ea_model_path is None:
            raise ValueError("ea_model_path cannot be None")
        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )



        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1




        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            output_hidden_states=False
    ):
        """
        이 함수는 Speculative Decoding의 검증 단계와 초기화 단계에서 모두 사용됩니다.
        따라서 드래프트 모델과 타겟 모델 모두 호출할 수 있어야 한다. 

        EaModel은 기본적으로 타겟 모델(base_model)을 사용한 연산을 수행한다. 
        드래프트 모델을 사용한 연산은 eagenerate 함수 내에서 별도로 처리된다. 
        """

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True, # KV 캐시
                output_hidden_states=output_hidden_states, # 중요 
             )
            # 마지막 레이어의 히든벡터 
            hidden_states = outputs.hidden_states[-1] if output_hidden_states else None

            # 최종 로짓 
            logits = outputs.logits 

            # 반환 로직을 코드 베이스의 일관성을 유지하며며 수정 
            if output_orig: 
                return logits, hidden_states
            else: 
                return logits, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            hidden_state_collector: Optional[CollapseCollector] = None,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=int(top_k))
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask,tree_position_ids, _, draft_hidden_state, _  = initialize_tree(
            input_ids, self, past_key_values, logits_processor, output_hidden_states=True
        )
        new_token = 0
        accept_length_list = []

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, target_hidden_states = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                output_hidden_states=True
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # 조건부 피처 수집 로직 
            if hidden_state_collector:
                # accepted token (draft feature)
                for i in range(accept_length):
                    feature_index = retrieve_indices[best_candidate, i + 1].item()
                    hidden_state_collector.add(state='speculated"', hidden_state=draft_hidden_state[:, feature_index:feature_index+1, :])
                # bonus token (accept_length 이후 첫 토큰) 피처 추출
                if (accept_length + 1) < retrieve_indices.shape[1]:
                    feature_index = retrieve_indices[best_candidate, accept_length + 1].item()
                    hidden_state_collector.add(state='speculated', hidden_state=draft_hidden_state[:, feature_index:feature_index+1, :])
                elif (accept_length + 1) == retrieve_indices.shape[1]:
                    # all accepted 상황: 실제 생성 토큰의 target_hidden_state 저장
                    if target_hidden_states is not None:
                        hidden_state_collector.add(state='speculated', hidden_state=target_hidden_states[:, -1:, :])
                    else:
                        print(f"[DEBUG][WARN] target_hidden_states is None at step {idx}. Skipping hidden state collection.")
            

            try:
                accept_length_list.append(accept_length.item())
            except:
                accept_length_list.append(accept_length)


            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                target_hidden_states,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx, accept_length_list


    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
            hidden_state_collector: Optional[CollapseCollector] = None,

    ):

        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=int(top_k))
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        new_token = 0

        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(input_ids, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            
            if hidden_state_collector:
                if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                    penultimate_feature = outputs.hidden_states[-1][:, -1:, :]
                    hidden_state_collector.add(
                        state='baseline_accepted',
                        hidden_state=penultimate_feature
                    )
                else:
                    print(f"Warning: outputs.hidden_states is None or empty at step {idx}. Skipping hidden state collection.")
                    # Fallback: use the last hidden state from the model output
                    if hasattr(outputs, 'last_hidden_state'):
                        penultimate_feature = outputs.last_hidden_state[:, -1:, :]
                        hidden_state_collector.add(
                            state='baseline_accepted',
                            hidden_state=penultimate_feature
                        )
            
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    # @torch.no_grad()
    # def ea_generate(
    #         self,
    #         input_ids,
    #         temperature=0.0,
    #         top_p=0.0,
    #         top_k=0.0,
    #         max_new_tokens=512,
    #         max_length=2048,
    #         log=False,
    #         is_llama3=False,

    # ):
    #     if is_llama3:
    #         stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     max_length=max_length-self.ea_layer.total_tokens-10

    #     if temperature > 1e-5:
    #         logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=int(top_k))
    #     else:
    #         logits_processor = None
    #     #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    #     # Avoid modifying the input_ids in-place

    #     padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
    #     input_ids = input_ids.clone()
    #     self.ea_layer.reset_kv()



    #     # Initialize the past key and value states
    #     if hasattr(self, "past_key_values"):
    #         past_key_values = self.past_key_values
    #         past_key_values_data = self.past_key_values_data
    #         current_length_data = self.current_length_data
    #         # Reset the past key and value states
    #         current_length_data.zero_()
    #     else:
    #         (
    #             past_key_values,
    #             past_key_values_data,
    #             current_length_data,
    #         ) = initialize_past_key_values(self.base_model)
    #         self.past_key_values = past_key_values
    #         self.past_key_values_data = past_key_values_data
    #         self.current_length_data = current_length_data

    #     input_len = input_ids.shape[1]
    #     reset_tree_mode(self)
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
    #         input_ids, self, past_key_values, logits_processor
    #     )
    #     new_token = 0

    #     for idx in range(max_length):
    #         #with Timer("all"):
    #         self.base_model.model.tree_mask = tree_mask

    #         draft_tokens=draft_tokens.to(input_ids.device)
    #         #with Timer("tree_decoding"):
    #         logits, hidden_state_new, outputs = tree_decoding(
    #             self,
    #             draft_tokens,
    #             past_key_values,
    #             tree_position_ids,
    #             input_ids,
    #             retrieve_indices,
    #         )
    #         #retrieve_indices=tree_buffers["retrieve_indices"]
    #         #logits = logits[0, retrieve_indices]
    #         draft_tokens=torch.cat((draft_tokens,padding),dim=1)
    #         candidates=draft_tokens[0,retrieve_indices]
    #         best_candidate, accept_length, sample_p = evaluate_posterior(
    #             logits, candidates, logits_processor
    #         )
    #         # print(accept_length)
    #         #with Timer("update_inference_inputs"):
    #         input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
    #             input_ids,
    #             candidates,
    #             best_candidate,
    #             accept_length,
    #             retrieve_indices,
    #             logits_processor,
    #             new_token,
    #             past_key_values_data,
    #             current_length_data,
    #             self,
    #             hidden_state_new,
    #             sample_p
    #         )

    #         yield input_ids

    #         if is_llama3:
    #             if stop_token_id in input_ids[0, input_len:].tolist():
    #                 break

    #         if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
    #             break
    #         if new_token > max_new_tokens:
    #             break
    #         if input_ids.shape[1] > max_length:
    #             break


    # @torch.no_grad()
    # def naive_generate(
    #         self,
    #         input_ids,
    #         temperature=0.0,
    #         top_p=0.0,
    #         top_k=0.0,
    #         max_new_tokens=512,
    #         max_length=2048,
    #         log=False,
    #         is_llama3=False,

    # ):
    #     if is_llama3:
    #         stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #     max_length = max_length - self.ea_layer.total_tokens - 10

    #     if temperature > 1e-5:
    #         logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=int(top_k))
    #     else:
    #         logits_processor = None
    #     # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    #     # Avoid modifying the input_ids in-place

    #     padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
    #     input_ids = input_ids.clone()
    #     self.ea_layer.reset_kv()

    #     # Initialize the past key and value states
    #     if hasattr(self, "past_key_values"):
    #         past_key_values = self.past_key_values
    #         past_key_values_data = self.past_key_values_data
    #         current_length_data = self.current_length_data
    #         # Reset the past key and value states
    #         current_length_data.zero_()
    #     else:
    #         (
    #             past_key_values,
    #             past_key_values_data,
    #             current_length_data,
    #         ) = initialize_past_key_values(self.base_model)
    #         self.past_key_values = past_key_values
    #         self.past_key_values_data = past_key_values_data
    #         self.current_length_data = current_length_data

    #     input_len = input_ids.shape[1]
    #     reset_tree_mode(self)
    #     outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
    #     new_token = 0


    #     for idx in range(max_length):
    #         if logits_processor is not None:
    #             logits = outputs.logits[:, -1]
    #             logits = logits_processor(input_ids.to(torch.long), logits)
    #             probabilities = torch.nn.functional.softmax(logits, dim=-1)
    #             input_id = torch.multinomial(probabilities, 1)
    #         else:
    #             input_id = outputs.logits[:, -1:].argmax(dim=-1)

    #         outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
    #         input_ids = torch.cat([input_ids, input_id], dim=-1)
    #         new_token += 1

    #         yield input_ids



    #         if is_llama3:
    #             if stop_token_id in input_ids[0, input_len:].tolist():
    #                 break

    #         if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
    #             break
    #         if new_token > max_new_tokens:
    #             break
    #         if input_ids.shape[1] > max_length:
    #             break



