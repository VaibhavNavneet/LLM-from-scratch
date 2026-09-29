import argparse, csv, math, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.config import Config
from src.dataset import create_loader
from src.model import GPTModel
from src.scheduler import cosine_schedule
from src.checkpoint import load_checkpoint, save_checkpoint
from src.tokenizer import BPETokenizer
from src.utils import count_parameters, set_seed

root=Path(__file__).parents[1]; parser=argparse.ArgumentParser(); parser.add_argument('--max-steps',type=int); parser.add_argument('--eval-interval',type=int); parser.add_argument('--checkpoint-interval',type=int); parser.add_argument('--resume'); args=parser.parse_args(); cfg=Config();
if args.max_steps: cfg.max_steps=args.max_steps
if args.eval_interval: cfg.eval_interval=args.eval_interval
if args.checkpoint_interval: cfg.checkpoint_interval=args.checkpoint_interval
set_seed(cfg.seed)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=GPTModel(cfg).to(device)
print('device:',device,'parameters:',count_parameters(model)); print('effective tokens/update:',cfg.effective_tokens_per_update)
train=create_loader(root/'data/processed/train.bin',cfg.context_length,cfg.micro_batch_size,True); val=create_loader(root/'data/processed/val.bin',cfg.context_length,cfg.micro_batch_size,False)
if cfg.max_steps is None:
    cfg.max_steps=max(1,math.ceil(len(train.dataset.tokens)*cfg.epochs/cfg.effective_tokens_per_update))
    cfg.warmup_steps=min(cfg.warmup_steps,max(1,cfg.max_steps//10))
print('planned optimizer steps:',cfg.max_steps,'epochs:',cfg.epochs)
opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,betas=(cfg.beta1,cfg.beta2),weight_decay=cfg.weight_decay)
sched=cosine_schedule(opt,cfg.warmup_steps,cfg.max_steps,cfg.min_learning_rate/cfg.learning_rate)
use_amp=device.type=='cuda'; amp_dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
scaler=torch.amp.GradScaler('cuda',enabled=use_amp and amp_dtype==torch.float16)
start_step=0; tokens_seen=0; best=float('inf')
if args.resume:
    ckpt=load_checkpoint(args.resume,model,opt,sched,scaler,device); start_step=ckpt['step']; tokens_seen=ckpt.get('tokens_seen',0); best=ckpt.get('best_val_loss',float('inf')); print('resumed from step',start_step)
def evaluate():
    model.eval(); losses=[]; it=iter(val)
    with torch.no_grad():
        for _ in range(min(cfg.eval_batches,len(val))):
            x,y=next(it); x,y=x.to(device),y.to(device)
            with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=use_amp): _,loss=model(x,y)
            losses.append(loss.item())
    model.train(); mean=sum(losses)/max(1,len(losses)); return mean,math.exp(min(mean,20))
logpath=root/'experiments/metrics.csv'; logpath.parent.mkdir(exist_ok=True); log=logpath.open('w',newline='',encoding='utf-8'); writer=csv.DictWriter(log,fieldnames=['step','tokens_seen','train_loss','validation_loss','perplexity','learning_rate','gradient_norm','tokens_per_second']); writer.writeheader()
it=iter(train); start=time.time(); model.train()
for step in range(start_step+1,cfg.max_steps+1):
    opt.zero_grad(set_to_none=True); total=0.0
    for _ in range(cfg.gradient_accumulation_steps):
        try:x,y=next(it)
        except StopIteration: it=iter(train); x,y=next(it)
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); tokens_seen+=x.numel()
        with torch.autocast(device_type='cuda',dtype=amp_dtype,enabled=use_amp): _,loss=model(x,y); scaled=loss/cfg.gradient_accumulation_steps
        if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss at step {step}: {loss.item()}')
        scaler.scale(scaled).backward(); total+=loss.item()
    scaler.unscale_(opt); grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip)
    if not torch.isfinite(grad_norm): raise FloatingPointError(f'non-finite gradient at step {step}')
    scaler.step(opt); scaler.update(); sched.step()
    if step==1 or step%cfg.eval_interval==0 or step==cfg.max_steps:
        vl,pp=evaluate(); elapsed=max(time.time()-start,1e-6); lr=opt.param_groups[0]['lr']; writer.writerow({'step':step,'tokens_seen':tokens_seen,'train_loss':total/cfg.gradient_accumulation_steps,'validation_loss':vl,'perplexity':pp,'learning_rate':lr,'gradient_norm':float(grad_norm),'tokens_per_second':tokens_seen/elapsed}); log.flush(); print(f'step {step} train {total/cfg.gradient_accumulation_steps:.4f} val {vl:.4f} ppl {pp:.2f} lr {lr:.2e}')
        if vl<best: best=vl; save_checkpoint(root/'checkpoints/best.pt',model,opt,sched,scaler,step,tokens_seen,best,cfg)
    if step%cfg.checkpoint_interval==0 or step==cfg.max_steps: save_checkpoint(root/'checkpoints/latest.pt',model,opt,sched,scaler,step,tokens_seen,best,cfg)
log.close()
