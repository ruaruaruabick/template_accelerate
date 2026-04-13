import contextlib
from collections import OrderedDict
import torch

# 可以简化为只合并最后几个检查点
class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        self.model = model
        
        #register model parameters
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone().float()
    
    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data.float() + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            dtype = param.data.dtype
            if name in self.shadow:
                self.original[name] = param.data
                param.data = self.shadow[name].to(dtype)
            else:
                print(f"Warning: {name} not found in shadow")
    
    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.original
            param.data = self.original[name]
        self.original = {}
        
    @contextlib.contextmanager
    def average_parameters(self):
        self.apply_shadow()
        yield
        self.restore()
    
    def state_dict(self):
        return OrderedDict([('shadow', self.shadow)])
    
    def load_state_dict(self, state_dict):
        for name, param in state_dict['shadow'].items():
            self.shadow[name] = param
        
    def to(self, device):
        for name, param in self.shadow.items():
            self.shadow[name] = param.to(device)