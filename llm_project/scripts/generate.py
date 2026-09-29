import sys
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.config import Config
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.checkpoint import load_checkpoint
from src.generation import generate
root=Path(__file__).parents[1]; cfg=Config(); tok=BPETokenizer(root/'data/tokenizer/tokenizer.json'); model=GPTModel(cfg); load_checkpoint(root/'checkpoints/best.pt',model); model.eval()
for prompt in ['The future of artificial intelligence','Once upon a time','Machine learning is','India is','Scientists discovered']:
 print('\n'+prompt+'\n'+generate(model,tok,prompt,80,temperature=0.8,top_k=40))
