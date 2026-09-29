import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.config import Config
from src.dataset import create_loader
from src.model import GPTModel
from src.tokenizer import BPETokenizer
from src.checkpoint import load_checkpoint
from src.evaluation import evaluate_loss
root=Path(__file__).parents[1]; c=Config(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=GPTModel(c).to(device); load_checkpoint(root/'checkpoints/best.pt',m,device=device)
loader=create_loader(root/'data/processed/val.bin',c.context_length,c.micro_batch_size,False); loss,ppl=evaluate_loss(m,loader,device,c.eval_batches); print(f'validation loss: {loss:.4f}\nperplexity: {ppl:.3f}')
