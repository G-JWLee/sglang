import warnings
import os
from .radix_attention import RadixAttention as SRTRadixAttention
from .hip_radix_attention import RadixAttention as HiPRadixAttention
from .zero_radix_attention import RadixAttention as ZeroRadixAttention

METHOD = os.getenv('SRT_ATTENTION_BACKEND', 'SRT')

if METHOD == 'HIP_ATTN':
    RadixAttention = HiPRadixAttention
elif METHOD == 'ZERO':
    RadixAttention = ZeroRadixAttention
elif METHOD == 'SRT':
    RadixAttention = SRTRadixAttention
else:
    raise Exception()

warnings.warn(f'Attention backend is {METHOD}')