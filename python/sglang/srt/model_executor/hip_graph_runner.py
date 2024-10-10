import torch
from typing import Dict, List
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner, ScheduleBatch
from sglang.srt.layers.radix_attention.hip_radix_attention import RadixAttention
from sglang.srt.layers.radix_attention.hip_radix_attention import envs as hip_envs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

class HiPGraphRunner:
    def __init__(
        self,
        model_runner: "ModelRunner",
        max_batch_size_to_capture: int,
        use_torch_compile: bool, # False
        disable_padding: bool, # False
    ):
        self.model_runner = model_runner
        self.max_batch_to_capture = max_batch_size_to_capture
        self.use_torch_compile = use_torch_compile
        self.disable_padding = disable_padding
        self.batch_sizes = []
        
        self.runner_refresh = {}
        self.runner_cached = {}
        self.runner_dense = {}
        
        self.step = 0
        self.refresh_interval = 8
        self.pool = None
    
    def can_run(self, batch_size: int):
        if self.disable_padding:
            return batch_size in self.batch_sizes
        else:
            return batch_size <= self.max_batch_to_capture
    
    def capture_runners(self, batch_size_list: List[int]):
        self.batch_sizes = batch_size_list
        
        for bsz in list(sorted(batch_size_list, reverse=True)):
            runner_refresh = CudaGraphRunner(
                self.model_runner, 
                max_batch_size_to_capture=bsz,
                use_torch_compile=self.use_torch_compile,
                disable_padding=self.disable_padding,
            )
            runner_cached = CudaGraphRunner(
                self.model_runner, 
                max_batch_size_to_capture=bsz,
                use_torch_compile=self.use_torch_compile,
                disable_padding=self.disable_padding,
            )
            runner_dense = CudaGraphRunner(
                self.model_runner,
                max_batch_size_to_capture=bsz,
                use_torch_compile=self.use_torch_compile,
                disable_padding=self.disable_padding,
            )
            
            for module in self.model_runner.model.modules():
                if isinstance(module, RadixAttention):
                    module.checkout_metadata = False
                    module.last_metadata = None
                    module.using_cached_metadata = False
                    module.cached_metadata = None
                    module.force_dense = True
            
            runner_dense.graph_memory_pool = runner_refresh.graph_memory_pool
            runner_dense.capture([bsz])
            self.pool = runner_dense.graph_memory_pool
            
            for module in self.model_runner.model.modules():
                if isinstance(module, RadixAttention):
                    module.checkout_metadata = True
                    module.last_metadata = None
                    module.using_cached_metadata = False
                    module.cached_metadata = None
                    module.force_dense = False
            
            runner_refresh.graph_memory_pool = self.pool
            runner_refresh.capture([bsz])
            self.pool = runner_refresh.graph_memory_pool
            
            for module in self.model_runner.model.modules():
                if isinstance(module, RadixAttention):
                    module.checkout_metadata = False
                    module.using_cached_metadata = True
                    module.cached_metadata = module.last_metadata
                    module.last_metadata = None
                    module.force_dense = False
            
            runner_cached.graph_memory_pool = self.pool
            runner_cached.capture([bsz])
            self.pool = runner_cached.graph_memory_pool
            
            for module in self.model_runner.model.modules():
                if isinstance(module, RadixAttention):
                    module.checkout_metadata = False
                    module.last_metadata = None
                    module.using_cached_metadata = False
                    module.cached_metadata = None
                    module.force_dense = False
            
            self.runner_refresh[bsz] = runner_refresh
            self.runner_cached[bsz] = runner_cached
            self.runner_dense[bsz] = runner_dense
    
    def capture(self, batch_size_list: List[int]):
        try:
            self.capture_runners(batch_size_list)
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )
            
    def reset_step(self):
        self.step = 0
    
    def replay(self, batch: ScheduleBatch):
        def get_graph(graphs: Dict[int, CudaGraphRunner]) -> CudaGraphRunner:
            if self.disable_padding:
                return graphs[batch.batch_size()]
            else:
                graph = None
                for bsz in graphs:
                    if bsz >= batch.batch_size():
                        graph = graphs[bsz]
                assert graph is not None
                return graph
        
        mean_seq_len = sum(map(lambda req: len(req.fill_ids), batch.reqs)) / len(batch.reqs)
        
        too_short_sequence = mean_seq_len <= hip_envs.hip_decode_dense_threshold
        too_small_batch = (
            batch.batch_size() * mean_seq_len) <\
            (hip_envs.hip_decode_dense_batch_token_threshold * self.model_runner.server_args.tp_size
        )
        
        if too_short_sequence or too_small_batch:
            # print('dense step', too_short_sequence, too_small_batch)
            out = get_graph(self.runner_dense).replay(batch)
            # NOTE: do not proceed step
        elif (self.step % self.refresh_interval) == 0:
            out = get_graph(self.runner_refresh).replay(batch)
            self.step += 1
        else:
            out = get_graph(self.runner_cached).replay(batch)
            self.step += 1
        return out