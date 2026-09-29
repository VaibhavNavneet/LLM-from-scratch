import math
import torch

@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches=20, amp_dtype=None):
    was_training=model.training; model.eval(); losses=[]
    use_amp=device.type=='cuda' and amp_dtype is not None
    for i,(x,y) in enumerate(loader):
        if i>=max_batches: break
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=use_amp): _,loss=model(x,y)
        losses.append(loss.item())
    if was_training: model.train()
    mean=sum(losses)/max(1,len(losses)); return mean, math.exp(min(mean,20))
