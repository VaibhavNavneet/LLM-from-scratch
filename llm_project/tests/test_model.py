import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from src.config import Config
from src.model import GPTModel, CausalSelfAttention
from src.utils import count_parameters

def test_shapes_and_tying():
    c=Config(vocab_size=100,context_length=16,emb_dim=32,n_layers=2,n_heads=4,ffn_dim=64)
    m=GPTModel(c); x=torch.randint(0,c.vocab_size,(2,16)); logits,loss=m(x,x)
    assert logits.shape==(2,16,c.vocab_size) and loss.ndim==0
    assert m.lm_head.weight is m.tok_emb.weight
    assert count_parameters(m)['trainable']>0

def test_causal_attention():
    c=Config(vocab_size=20,context_length=8,emb_dim=16,n_layers=1,n_heads=4,ffn_dim=32,dropout=0)
    a=CausalSelfAttention(c).eval(); x=torch.randn(1,8,16); _,w=a(x,True)
    assert torch.allclose(w[0,:,0,1:],torch.zeros_like(w[0,:,0,1:]))
