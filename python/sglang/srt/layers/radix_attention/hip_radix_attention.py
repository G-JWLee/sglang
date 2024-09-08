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

"""Radix attention."""

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

from hip import paged_hip_attention, HiPAttentionArgs

class HiPAttentionEnvs:
    def __init__(self):
        self.refresh_interval = int(os.getenv('HIP_REFRESH_INTERVAL', '8'))
        self.hip_dense_layers = os.getenv('HIP_DENSE_LAYERS', '0,1,2')
        try:
            t = int(self.hip_dense_layers)
            self.hip_dense_layers = list(range(t))
            warnings.warn(
                f'You gave single integer ({t}) for hip dense layers. '
                'From HiP 1.1, this changed into list of integers, e.g., `0,1,2` '
                'Are you sure about this?'
            )
        except: 
            self.hip_dense_layers = [int(i) for i in self.hip_dense_layers.split(',')]
        
        self.hip_k = int(os.getenv('HIP_K', '512'))
        self.hip_bq = int(os.getenv('HIP_BQ', '64'))
        self.hip_bsq = int(os.getenv('HIP_BSQ', '2'))
        self.hip_bk = int(os.getenv('HIP_BK', '2'))
        self.hip_bsk = int(os.getenv('HIP_BSK', '1'))
        
        self.hip_prefill_k = int(os.getenv('HIP_PREFILL_K', self.hip_k))
        self.hip_prefill_bq = int(os.getenv('HIP_PREFILL_BQ', self.hip_bq))
        self.hip_prefill_bsq = int(os.getenv('HIP_PREFILL_BSQ', self.hip_bsq))
        self.hip_prefill_bk = int(os.getenv('HIP_PREFILL_BK', self.hip_bk))
        self.hip_prefill_bsk = int(os.getenv('HIP_PREFILL_BSK', self.hip_bsk))
        
        self.hip_sw = int(os.getenv('HIP_SW', '256'))
        self.hip_nsink = int(os.getenv('HIP_NSINK', '16'))
        
        self.hip_sample_method = os.getenv('HIP_SAMPLE_METHOD', 'center')
        
        self.hip_seq_threshold = int(os.getenv('HIP_SEQ_THRESH', '-1'))
        
        self.hip_offload = os.getenv('HIP_OFFLOAD', '0') == '1'
        
        print(self.decode_config())
        print(self.prefill_config())
        print(self.hip_dense_layers)
    
    def decode_config(self):
        hip_kwargs = {
            'mask_k': self.hip_k,
            'block_size_q': self.hip_bq,
            'block_stride_q': self.hip_bsq,
            'block_size_k': self.hip_bk,
            'block_stride_k': self.hip_bsk,
            'sample_method': self.hip_sample_method,
            'sliding_window_size': self.hip_sw,
            'sink_token_size': self.hip_nsink,
        }
        return hip_kwargs
    
    def prefill_config(self):
        hip_prefill_kwargs = self.decode_config()
        hip_prefill_kwargs.update({
            'mask_k': self.hip_prefill_k,
            'block_size_q': self.hip_prefill_bq,
            'block_stride_q': self.hip_prefill_bsq,
            'block_size_k': self.hip_prefill_bk,
            'block_stride_k': self.hip_prefill_bsk,
            'num_dense_queries': self.hip_seq_threshold,
        })
        return hip_prefill_kwargs

envs = HiPAttentionEnvs()

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

            o = prefill_wrapper_paged.forward(
                q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                causal=True,
                sm_scale=self.scaling,
                window_left=self.sliding_window_size,
                logits_soft_cap=self.logit_cap,
            )
        else:
            if input_metadata.triton_max_seq_len == 0:
                t_start = time.time()
                input_metadata.triton_max_seq_len = torch.max(input_metadata.seq_lens).item()
                elapsed_item = time.time() - t_start
                print(f'RadixAttention: Seq len calculated {input_metadata.triton_max_seq_len}, took {elapsed_item} ms')
            
            # start = torch.cuda.Event(True)
            # start.record()
            
            if  (self.layer_id in envs.hip_dense_layers) or\
                (input_metadata.batch_size > 1) or\
                (
                    (input_metadata.triton_max_seq_len < (20 * 1024)) or\
                    (q.shape[0] < 512)
                ):
                o1, s1 = (
                    input_metadata.flashinfer_prefill_wrapper_ragged.forward_return_lse(
                        q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                        k.contiguous().view(-1, self.tp_k_head_num, self.head_dim),
                        v.contiguous().view(-1, self.tp_v_head_num, self.head_dim),
                        causal=True,
                        sm_scale=self.scaling,
                        logits_soft_cap=self.logit_cap,
                    )
                )

                if input_metadata.extend_no_prefix:
                    o = o1
                else:
                    o2, s2 = prefill_wrapper_paged.forward_return_lse(
                        q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                        input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id),
                        causal=False,
                        sm_scale=self.scaling,
                        logits_soft_cap=self.logit_cap,
                    )

                    o, _ = merge_state(o1, s1, o2, s2)

                self.store_kv_cache(k, v, input_metadata)
            else:
                warnings.warn('HiP attention is used in prompting!', stacklevel=0)
                
                k_cache, v_cache = input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id)
                
                o, _ = forward_paged_hip(
                    q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                    k_cache, 
                    v_cache,
                    input_metadata.positions,
                    input_metadata.seq_lens,
                    input_metadata.req_to_token_pool.req_to_token,
                    input_metadata.req_pool_indices,
                    self.scaling,
                    input_metadata.batch_size,
                    is_prefill=True,
                )
            
            # end = torch.cuda.Event(True)
            # end.record()
            
            if input_metadata.total_num_tokens >= global_config.layer_sync_threshold:
                torch.cuda.synchronize()
            
            # end.synchronize()
            # print(start.elapsed_time(end), input_metadata.triton_max_seq_len)

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

        if self.layer_id in envs.hip_dense_layers:
            o = decode_wrapper.forward(
                query,
                (k_cache, v_cache),
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
            )
        else:
            warnings.warn('HiP attention is used in decoding!', stacklevel=0)
            metadata = None
            if self.using_cached_metadata:
                metadata = self.cached_metadata
            
            o, metadata = forward_paged_hip(
                query, 
                k_cache, 
                v_cache, 
                input_metadata.positions,
                input_metadata.seq_lens,
                input_metadata.req_to_token_pool.req_to_token,
                input_metadata.req_pool_indices,
                self.scaling,
                input_metadata.batch_size,
                cached_metadata=metadata,
            )
            
            if self.checkout_metadata:
                self.last_metadata = metadata

        return o.view(-1, self.tp_q_head_num * self.head_dim)

def forward_paged_hip(
    query: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sm_scale: float,
    batch_size: int,
    cached_metadata = None,
    is_prefill: bool = False,
):
    N, HEAD, HID = query.shape
    TDST = N // batch_size
    query = query.view(batch_size, TDST, HEAD, HID)
    
    N_PAGE, HEAD_KV, _HID = k_cache.shape
    assert v_cache.shape == k_cache.shape
    assert _HID == HID
    k_cache = k_cache.view(N_PAGE, 1, HEAD_KV, HID)
    v_cache = v_cache.view(N_PAGE, 1, HEAD_KV, HID)
    
    block_table = req_to_tokens.index_select(
        dim=0, index=req_pool_indices
    )
    _BSZ, MODEL_SEQ_LEN = block_table.shape
    assert batch_size == _BSZ
    
    cache_seq_lens = seq_lens
    position_ids = positions.view(batch_size, TDST) + 1 # BUG(HJ): this naming is wrong... this should be key_seq_lens
    
    args = HiPAttentionArgs(
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cache_seq_lens=cache_seq_lens,
        position_ids=position_ids,
        **(envs.decode_config() if not is_prefill else envs.prefill_config()),
    )
    
    context, metadata = paged_hip_attention(
        query,
        previous_mask_metadata=cached_metadata,
        softmax_scale=sm_scale,
        args=args,
    )
    
    return context.view(N, HEAD, HID), metadata