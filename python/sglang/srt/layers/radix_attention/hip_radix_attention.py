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
from typing import List, Optional
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
        self.hip_dense_layers: List[int] = os.getenv('HIP_DENSE_LAYERS', '0,1,2')
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
        self.hip_nsink = int(os.getenv('HIP_NSINK', '256'))
        
        self.hip_sample_method = os.getenv('HIP_SAMPLE_METHOD', 'center')
        
        self.hip_seq_threshold = int(os.getenv('HIP_SEQ_THRESH', '-1'))
        
        self.hip_offload = os.getenv('HIP_OFFLOAD', '0') == '1'
        assert not self.hip_offload, "working on..."
        
        self.hip_decode_always_dense = os.getenv('HIP_DECODE_ALWAYS_DENSE', '0') == '1'
        self.hip_prefill_always_dense = os.getenv('HIP_PREFILL_ALWAYS_DENSE', '0') == '1'
        
        self.hip_prefill_dense_threshold = int(os.getenv('HIP_PREFILL_DENSE_THRESHOLD', '8192'))
        self.hip_decode_dense_threshold = int(os.getenv('HIP_DECODE_DENSE_THRESHOLD', '8192'))
        self.hip_decode_dense_batch_token_threshold = int(os.getenv('HIP_DECODE_DENSE_BATCH_SIZE_THRESHOLD', f'{32 * 8192}')) # batch token per GPU
        
        self.hip_extend = os.getenv('HIP_EXTEND', '0') == '1'
        self.hip_extend_context_length = int(os.getenv('HIP_EXTEND_CONTEXT_LENGTH', '32768'))
        
        print(self.decode_config())
        print(self.prefill_config())
        print('dense layers =', self.hip_dense_layers)
        print(f'is decode dense? = {self.hip_decode_always_dense}, is prefill dense? = {self.hip_prefill_always_dense}')
    
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
        rope = None,
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
        self.force_dense = False
        self.rope = rope
        if rope is not None:
            cos_sin = self.rope.cos_sin_cache
            cos, sin = cos_sin.chunk(2, dim=-1)
            self.rope_cos = cos.repeat(1, 2)
            self.rope_sin = sin.repeat(1, 2)
        else:
            self.rope_cos = self.rope_sin = None

    def extend_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        # using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        prefill_wrapper_paged = input_metadata.flashinfer_prefill_wrapper_paged
        if self.sliding_window_size != -1:
            prefill_wrapper_paged = prefill_wrapper_paged[0]
        else:
            if isinstance(prefill_wrapper_paged, list):
                prefill_wrapper_paged = prefill_wrapper_paged[1]

        if input_metadata.triton_max_seq_len == 0:
            t_start = time.time()
            input_metadata.triton_max_seq_len = torch.max(input_metadata.seq_lens).item()
            elapsed_item = time.time() - t_start
            if elapsed_item >= 1:
                print(f'RadixAttention: Seq len calculated {input_metadata.triton_max_seq_len}, took {elapsed_item} ms')
        
        is_dense_layer = self.layer_id in envs.hip_dense_layers
        require_dense = (
            is_dense_layer or\
            envs.hip_prefill_always_dense or\
            self.force_dense or\
            any(map(lambda x: x <= envs.hip_prefill_dense_threshold, input_metadata.extend_prefix_lens_cpu))
        )
        if  require_dense and (not envs.hip_extend):
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
            
            if k is not None:
                assert v is not None
                self.store_kv_cache(k, v, input_metadata)
                
            # print(input_metadata.extend_seq_lens_cpu)
            
            k_cache, v_cache = input_metadata.token_to_kv_pool.get_kv_buffer(self.layer_id)
            
            q_reshaped = q.contiguous().view(-1, self.tp_q_head_num, self.head_dim)
            o = torch.empty_like(q_reshaped)
            start_len = 0
            decoding_reqs = []
            decoding_reqs_poistions = []
            for idx_batch, seq_len in enumerate(input_metadata.extend_seq_lens_cpu):
                if seq_len == 0:
                    decoding_reqs.append(idx_batch)
                    decoding_reqs_poistions.append(start_len)
                else:
                    o_req, _ = self.forward_paged_hip(
                        query=q_reshaped[start_len:start_len+seq_len],
                        k_cache=k_cache, 
                        v_cache=v_cache,
                        positions=input_metadata.positions[start_len:start_len+seq_len],
                        seq_lens=input_metadata.seq_lens[idx_batch:idx_batch+1],
                        req_to_tokens=input_metadata.req_to_token_pool.req_to_token,
                        req_pool_indices=input_metadata.req_pool_indices[idx_batch:idx_batch+1],
                        sm_scale=self.scaling, 
                        batch_size=1,
                        is_prefill=True,
                        is_dense=is_dense_layer,
                    )
                    o[start_len:start_len+seq_len] = o_req
                start_len += seq_len
            assert len(decoding_reqs) == 0
            
        if  (input_metadata.total_num_tokens >= global_config.layer_sync_threshold) and\
            (self.layer_id % 2 == 1):
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

        is_dense_layer = (self.layer_id in envs.hip_dense_layers)
        require_dense = (
            is_dense_layer or\
            envs.hip_decode_always_dense or\
            self.force_dense
        )
        if  require_dense and (not envs.hip_extend):
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
            
            o, metadata = self.forward_paged_hip(
                query=query, 
                sm_scale=self.scaling,
                batch_size=input_metadata.batch_size,
                
                k_cache=k_cache, 
                v_cache=v_cache, 
                
                positions=input_metadata.positions,
                seq_lens=input_metadata.seq_lens,
                req_to_tokens=input_metadata.req_to_token_pool.req_to_token,
                req_pool_indices=input_metadata.req_pool_indices,
                
                cached_metadata=metadata,
                is_prefill=False,
                is_dense=is_dense_layer,
            )
            
            if self.checkout_metadata:
                self.last_metadata = metadata

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def forward_paged_hip(
        self, 
        
        query: torch.Tensor,
        sm_scale: float,
        batch_size: int,
        
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor,
        
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_tokens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        
        cached_metadata = None,
        is_prefill: bool = False,
        is_dense: bool = False,
    ):
        N, HEAD, HID = query.shape
        TDST = N // batch_size
        
        is_decode = TDST == 1
        
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
        position_ids = positions.view(batch_size, TDST) + 1 # BUG(-): this naming is wrong... this should be key_seq_lens
        
        args = HiPAttentionArgs(
            k_cache=k_cache,
            v_cache=v_cache,
            block_table=block_table,
            cache_seq_lens=cache_seq_lens,
            position_ids=position_ids,
            **(envs.decode_config() if not is_prefill else envs.prefill_config()),
        )
        
        DEBUG_NAN = (os.getenv('DEBUG_NAN', '0') == '1') and not torch.cuda.is_current_stream_capturing()
        
        if DEBUG_NAN:
            passed_query = torch.logical_or(torch.isnan(query), torch.isinf(query))
        
        if (not envs.hip_extend):
            context, metadata = paged_hip_attention(
                query,
                previous_mask_metadata=cached_metadata,
                softmax_scale=sm_scale,
                args=args,
            )
        else:
            from hip.models.hip_attention.attention2_draft_sampling_extend import dual_stage_quadratic_hip_attention
            IS_GEMMA = os.getenv('IS_GEMMA', '0') == '1'
            
            cos = self.rope_cos # type: torch.Tensor
            sin = self.rope_sin # type: torch.Tensor
            
            mask_k_1k = int(os.getenv('HIP_DRAFT_MASK_K_1K', '2'))
            scan_k_1k = int(os.getenv('HIP_DRAFT_SCAN_K_1K', '16'))
            
            # print(query.dtype) # bfloat16
            context, metadata = dual_stage_quadratic_hip_attention(
                (query * sm_scale).to(query.dtype), 
                None, 
                None, 
                args=HiPAttentionArgs(
                    k_cache=args.k_cache.view(torch.uint8) if args.k_cache.dtype == torch.float8_e5m2 else args.k_cache,
                    v_cache=args.v_cache.view(torch.uint8) if args.v_cache.dtype == torch.float8_e5m2 else args.v_cache,
                    block_table=args.block_table,
                    cache_seq_lens=args.cache_seq_lens,
                    position_ids=args.position_ids - 1,
                    
                    mask_k=128, # control quadratic cost
                    block_size_q=32 if IS_GEMMA else 64,
                    block_stride_q=2 if IS_GEMMA else 1,
                    block_size_k=32 if IS_GEMMA else 64, # BLOCK_CHUNK
                    block_stride_k=2 if IS_GEMMA else 1,
                    
                    sliding_window_size=1024 if (not is_dense) else 4096,
                    sink_token_size=256 if (not is_dense) else 256,
                    
                    using_extend=True,
                    need_apply_rope=True,
                    rope_cos=cos,
                    rope_sin=sin,
                    
                    logit_softcap=args.logit_softcap,
                ),
                second_stage_k=mask_k_1k*1024 if (not is_dense) else mask_k_1k*2*1024,
                stages=[ # control linear cost
                    # (2, 64, 32768),
                    # (1, 8, 8192),
                    
                    (1, 32, 32768),
                    (1, 1, 8192),
                    # (32, scan_k_1k*1024 if (not is_dense) else scan_k_1k*4*1024),
                ] if (not is_dense) else [
                    # (2, 64, 65536),
                    # (1, 8, 16384),
                    
                    (1, 32, 65536),
                    (1, 1, 16384),
                ],
                scan_stride=1,
                scan_block_stride_q=-1,
                model_context_length=envs.hip_extend_context_length,
                scan_early_terminate=1,
                stage_early_terminate=1,
                cached_metadata=cached_metadata,
                # scan_extend_backend='dynamic_extend' if (not is_dense) else 'streaming',
                scan_extend_backend='streaming' if self.layer_id < 3 else 'relative',
                sa_extend_backend='dynamic_extend',
                # block_sparse_block_size_q=32,
            )
            context = context.to(query.dtype)
        
        if DEBUG_NAN:
            passed_context = torch.logical_or(torch.isnan(context), torch.isinf(context))
            torch.cuda.synchronize()
            assert not passed_query.any().item(), f'{passed_query} {passed_query.shape}'
            assert not passed_context.any().item(), f'{passed_context} {passed_context.shape}'
        
        return context.view(N, HEAD, HID), metadata