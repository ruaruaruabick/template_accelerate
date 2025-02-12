import math
class CosineLRScheduler(object):
    def __init__(self, init_lr, last_lr, warmup_steps, max_steps, num_gpus):
        self.init_lr = init_lr
        self.last_lr = last_lr
        self.warmup_steps = warmup_steps * num_gpus
        self.max_steps = max_steps * num_gpus
        
        
    def __call__(self, step):
        if step < self.warmup_steps:
            return (step+1) / self.warmup_steps
        else:
            cosine_rate = 0.5 * ( math.cos((step - self.warmup_steps) /(self.max_steps - self.warmup_steps) * math.pi) + 1)
            return (cosine_rate * (self.init_lr - self.last_lr)+self.last_lr) / self.init_lr