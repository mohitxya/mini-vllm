import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varien_func, flash_attn_with_kvcache
from minivllm.utils.context import get_context


@triton.jit
def storeKvCacheKernel(): 
    pass
def storeKvCache():
    pass

class Attention(nn.Module): 
    def __init__(): 
        pass
    def forward(): 
        pass
