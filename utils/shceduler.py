import math
class CosineLRScheduler(object):
    #accelerate.split_batch=False(default)需要乘gpu数
    def __init__(self, init_lr, last_lr, num_warmup_steps, num_training_steps, num_gpus):
        self.init_lr = init_lr
        self.last_lr = last_lr
        self.warmup_steps = num_warmup_steps * num_gpus
        self.max_steps = num_training_steps * num_gpus
        
        
    def __call__(self, step):
        if step < self.warmup_steps:
            return (step+1) / self.warmup_steps
        else:
            cosine_rate = 0.5 * ( math.cos((step - self.warmup_steps) /(self.max_steps - self.warmup_steps) * math.pi) + 1)
            return (cosine_rate * (self.init_lr - self.last_lr)+self.last_lr) / self.init_lr
        
class WSDLRScheduler(object):
    def __init__(self, init_lr, num_warmup_steps, num_training_steps, num_gpus, num_decay_steps, **kwargs):
        self.init_lr = init_lr
        self.warmup_steps = num_warmup_steps * num_gpus
        self.max_steps = num_training_steps * num_gpus
        self.decay_steps = int(self.max_steps * num_decay_steps)
        self.stable_steps = self.max_steps - self.warmup_steps - self.decay_steps
        
    def __call__(self, step):
        if step < self.warmup_steps:
            return (step+1) / self.warmup_steps
        elif step < self.warmup_steps + self.stable_steps:
            return 1.0
        else:
            decay_process = 1.0 - (step - self.warmup_steps - self.stable_steps) / self.decay_steps
            return decay_process