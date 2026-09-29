import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import torch
from src.config import Config
from src.model import GPTModel

def test_tiny_overfit_loss_drops():
    torch.manual_seed(4)
    c=Config(vocab_size=32,context_length=8,emb_dim=32,n_layers=1,n_heads=4,ffn_dim=64,dropout=0)
    m=GPTModel(c); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    x=torch.tensor([[1,2,3,4,5,6,7,8]]*4); y=torch.roll(x,-1,1)
    with torch.no_grad(): _, first=m(x,y)
    for _ in range(35):
        opt.zero_grad(); _,loss=m(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    assert loss.item() < first.item() * 0.35
