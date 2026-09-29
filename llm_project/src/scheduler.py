import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_schedule(optimizer, warmup_steps, max_steps, min_lr_ratio):
    def f(step):
        if step < warmup_steps: return max(1e-8, step / max(1, warmup_steps))
        progress=min(1.0,(step-warmup_steps)/max(1,max_steps-warmup_steps))
        return min_lr_ratio + (1-min_lr_ratio)*0.5*(1+math.cos(math.pi*progress))
    return LambdaLR(optimizer, f)
