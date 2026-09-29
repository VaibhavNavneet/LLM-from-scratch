"""Minimal supervised instruction fine-tuning entry point.

JSONL records must contain instruction, response, and optional input fields.
Only response tokens contribute to the loss; this is intentionally separate
from pretraining so the base-model checkpoint remains untouched.
"""
import json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.config import Config
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.checkpoint import load_checkpoint, save_checkpoint
root=Path(__file__).parents[1]; c=Config(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tok=BPETokenizer(root/'data/tokenizer/tokenizer.json'); model=GPTModel(c).to(device); load_checkpoint(root/'checkpoints/best.pt',model,device=device); opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
data=Path(sys.argv[1]) if len(sys.argv)>1 else root/'data/raw/instructions.jsonl'; model.train()
for epoch in range(1):
    for line in data.open(encoding='utf-8'):
        rec=json.loads(line); prompt=f"### Instruction:\n{rec['instruction']}\n"+(f"### Input:\n{rec['input']}\n" if rec.get('input') else '')+'### Response:\n'; full=prompt+rec['response']; p=tok.encode(prompt); ids=tok.encode(full)[:c.context_length];
        if len(ids)<2: continue
        x=torch.tensor([ids[:-1]],device=device); y=torch.tensor([ids[1:]],device=device); y[:,:max(0,len(p)-1)]=-100; _,loss=model(x,y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
save_checkpoint(root/'checkpoints/instruction_finetuned.pt',model,opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1),None,0,0,float('nan'),c); print('saved instruction-finetuned checkpoint')
