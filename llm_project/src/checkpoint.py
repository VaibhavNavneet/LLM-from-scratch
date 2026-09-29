import os, random
import numpy as np
import torch

def save_checkpoint(path, model, optimizer, scheduler, scaler, step, tokens_seen, best_val_loss, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"scaler_state_dict":scaler.state_dict() if scaler else None,"step":step,"tokens_seen":tokens_seen,"best_val_loss":best_val_loss,"config":config.to_dict(),"random_states":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}},path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    ckpt=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(ckpt["model_state_dict"])
    if optimizer: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"): scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt
