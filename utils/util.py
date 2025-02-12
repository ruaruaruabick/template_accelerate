import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_yaml(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(module):
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