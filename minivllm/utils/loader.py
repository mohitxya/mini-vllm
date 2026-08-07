import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    