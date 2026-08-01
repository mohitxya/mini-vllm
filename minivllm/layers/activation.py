import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):
    @torch.compile
    def forward():
        x, y = x.chunk(2,-1)
        return F.silu(x)*y
