from .radix_attention import RadixAttention as SRTRadixAttention
from .hip_radix_attention import RadixAttention as HiPRadixAttention

METHOD = 'HIP_ATTN'

if METHOD == 'HIP_ATTN':
    RadixAttention = HiPRadixAttention
elif METHOD == 'SRT':
    RadixAttention = SRTRadixAttention
else:
    raise Exception()