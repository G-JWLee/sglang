"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Radix attention. Replace FlashInfer to torch.zeros_like for debugging"""

import os
import time
from typing import Optional
import warnings

import torch
from flashinfer.cascade import merge_state
from torch import nn

from sglang.global_config import global_config
from sglang.srt.layers.decode_attention import decode_attention_fwd
from sglang.srt.layers.extend_attention import extend_attention_fwd
from sglang.srt.model_executor.forward_batch_info import ForwardMode, InputMetadata
from sglang.srt.model_executor.model_runner import global_server_args_dict

from .radix_attention import RadixAttention as SRTRadixAttention

class RadixAttention(SRTRadixAttention):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        sliding_window_size: Optional[int] = None,
        logit_cap: int = -1,
        v_head_dim: int = -1,
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
        )
        
        self.checkout_metadata = False
        self.last_metadata = None
        self.using_cached_metadata = False
        self.cached_metadata = None

    def extend_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        # using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        prefill_wrapper_paged = input_metadata.flashinfer_prefill_wrapper_paged
        if self.sliding_window_size != -1:
            prefill_wrapper_paged = prefill_wrapper_paged[0]
        else:
            if isinstance(prefill_wrapper_paged, list):
                prefill_wrapper_paged = prefill_wrapper_paged[1]

        if not input_metadata.flashinfer_use_ragged:
            if k is not None:
                assert v is not None
                self.store_kv_cache(k, v, input_metadata)

            o = torch.zeros_like(q.contiguous().view(-1, self.tp_q_head_num, self.head_dim))
        else:
            if input_metadata.triton_max_seq_len == 0:
                t_start = time.time()
                input_metadata.triton_max_seq_len = torch.max(input_metadata.seq_lens).item()
                elapsed_item = time.time() - t_start
                print(f'RadixAttention: Seq len calculated {input_metadata.triton_max_seq_len}, took {elapsed_item} ms')
            
            o = torch.zeros_like(q.contiguous().view(-1, self.tp_q_head_num, self.head_dim))
            
            if input_metadata.total_num_tokens >= global_config.layer_sync_threshold:
                torch.cuda.synchronize()

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def decode_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        decode_wrapper = input_metadata.flashinfer_decode_wrapper
        if self.sliding_window_size != -1:
            decode_wrapper = decode_wrapper[0]
        else:
            if isinstance(decode_wrapper, list):
                decode_wrapper = decode_wrapper[1]

        if k is not None:
            assert v is not None
            self.store_kv_cache(k, v, input_metadata)

        query = q.contiguous().view(-1, self.tp_q_head_num, self.head_dim)
        k_cache, v_cache = input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id)

        o = torch.zeros_like(query)

        return o.view(-1, self.tp_q_head_num * self.head_dim)
