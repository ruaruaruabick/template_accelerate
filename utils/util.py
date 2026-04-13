import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

def mul(*args):
    res = 1
    for arg in args:
        res *= arg
    return res

OmegaConf.register_new_resolver("intdiv", lambda x, y: int(x / y),replace=True)
OmegaConf.register_new_resolver("intmul", lambda *args: int(mul(*args)),replace=True)

def load_yaml(path):
    return OmegaConf.load(path)

def load_json(file_path,use_split=False, index=None):
    if use_split:
        assert index is not None
        file_path = file_path.replace('.json', f'_{index}.json')
        if index == 0:
            print('###################Split Training Data###################')
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data 

def compute_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024
def init_weights(module, ):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.elementwise_affine:
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
def init_weights_zero(module, ):
    if hasattr(module, 'weight') and module.weight is not None:
        torch.nn.init.zeros_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.zeros_(module.bias)
        
def get_mask_from_lengths(lengths, max_len=None):
    #[True,True,True,True,True,False,False,False,False,False,False,]
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(
        0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return ~mask