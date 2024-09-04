import argparse
import time
import torch
import wikipediaapi

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import List, Union, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.server import Runtime
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

DEFAULT_PROMPTS = []

def reset_prompts(batch_size=16, word_count=32000):
    DEFAULT_PROMPTS.clear()
    wiki = wikipediaapi.Wikipedia(
        'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)',
        language='en', 
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    
    keywords = [
        'Korea', 'Japan', 'China', 'Google', 'Meta_Platforms', 'Microsoft', 'Amazon_(company)', 
        'Transformer', 'Llama', 'Cat', 'Dog', 'Apple', 'United_States'
    ] * 100
    keywords = keywords[:batch_size]
    for keyword in keywords:
        page = wiki.page(keyword)
        words = page.text.split()
        while len(words) < word_count:
            words.extend(words[:])
        DEFAULT_PROMPTS.append(' '.join(words[:word_count]) + ' ')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_tokens', default=64, type=int)
    parser.add_argument('--word_count', default=32000, type=int)
    parser.add_argument('--model', default='llama3.1_8b_instruct', type=str)
    
    args = parser.parse_args()
    
    return args

def get_dtype_str(torch_dtype):
    if torch_dtype is torch.float16:
        return "float16"
    else:
        raise NotImplementedError()

@dataclass
class ModelOutput:
    output_strs: List[str] = None
    output_ids: List[int] = None
    top_input_logprobs: List[torch.Tensor] = None
    top_output_logprobs: List[torch.Tensor] = None
    embed_logits: List[torch.Tensor] = None

class SRTRunner:
    def __init__(
        self,
        model_path,
        torch_dtype,
        tp_size=1,
        port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    ):
        self.runtime = Runtime(
            model_path=model_path,
            tp_size=tp_size,
            dtype=get_dtype_str(torch_dtype),
            port=port,
            mem_fraction_static=0.7,
            trust_remote_code=False,
        )

    def forward(
        self,
        prompts: Union[List[str], List[torch.Tensor]] = DEFAULT_PROMPTS,
        max_new_tokens=64,
    ):
        # the return value contains logprobs from prefill
        output_strs = []
        sampling_params = {
            "max_new_tokens": max_new_tokens, 
            "temperature": 0
        }
        
        response = self.runtime.generate(
            prompts,
            sampling_params=sampling_params,
        )
        response = json.loads(response)
        
        for r in response:
            output_strs.append(r['text'])

        return ModelOutput(
            output_strs=output_strs
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.runtime.shutdown()
        del self.runtime


def main():
    args = parse_args()
    
    reset_prompts(
        batch_size=args.batch_size,
        word_count=args.word_count,
    )
    
    t = time.time()
    
    with SRTRunner(
        model_path={
            'llama3.1_8b': 'meta-llama/Meta-Llama-3.1-8B',
            'llama3.1_8b_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        }[args.model],
        torch_dtype=torch.float16,
        tp_size=torch.cuda.device_count(),
    ) as runner:
        out = runner.forward(
            max_new_tokens=args.max_tokens,
        )
        lines = out.output_strs
        print(*lines, sep='\n------\n')
    
    print(f'took end-to-end {time.time() - t:.3f} sec')

if __name__ == '__main__':
    main()