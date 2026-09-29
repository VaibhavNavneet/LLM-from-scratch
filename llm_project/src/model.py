import math
import torch
from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__(); assert cfg.emb_dim % cfg.n_heads == 0
        self.n_heads, self.head_dim = cfg.n_heads, cfg.emb_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)
        self.proj = nn.Linear(cfg.emb_dim, cfg.emb_dim); self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask", torch.triu(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool), 1))
    def forward(self, x, return_attention=False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(b,t,self.n_heads,self.head_dim).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:t,:t], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1); out = self.dropout(weights) @ v
        out = self.proj(out.transpose(1,2).contiguous().view(b,t,c))
        return (out, weights) if return_attention else out

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.net = nn.Sequential(nn.Linear(cfg.emb_dim,cfg.ffn_dim), nn.GELU(), nn.Linear(cfg.ffn_dim,cfg.emb_dim), nn.Dropout(cfg.dropout))
    def forward(self,x): return self.net(x)

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.norm1=nn.LayerNorm(cfg.emb_dim); self.attn=CausalSelfAttention(cfg); self.norm2=nn.LayerNorm(cfg.emb_dim); self.mlp=MLP(cfg)
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.emb_dim); self.pos_emb=nn.Embedding(cfg.context_length,cfg.emb_dim); self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)]); self.norm=nn.LayerNorm(cfg.emb_dim); self.lm_head=nn.Linear(cfg.emb_dim,cfg.vocab_size,bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("mlp.net.2.weight") or name.endswith("attn.proj.weight"): nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        b,t=idx.shape
        if t > self.cfg.context_length: raise ValueError("sequence exceeds context_length")
        pos=torch.arange(t,device=idx.device); x=self.drop(self.tok_emb(idx)+self.pos_emb(pos))
        for block in self.blocks: x=block(x)
        logits=self.lm_head(self.norm(x)); loss=None
        if targets is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits, loss
