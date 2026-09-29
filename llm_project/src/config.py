from dataclasses import asdict, dataclass

@dataclass
class Config:
    vocab_size: int = 16_000
    context_length: int = 256
    emb_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ffn_dim: int = 1_024
    dropout: float = 0.1
    qkv_bias: bool = False
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 100
    max_steps: int | None = None
    epochs: float = 3.0
    gradient_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    seed: int = 1337
    @property
    def effective_tokens_per_update(self):
        return self.micro_batch_size * self.gradient_accumulation_steps * self.context_length
    def to_dict(self):
        return asdict(self)
