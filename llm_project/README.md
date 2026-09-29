# From-Scratch Small GPT Language Model

An educational, end-to-end implementation of a decoder-only Transformer language model. The project trains its tokenizer and model weights from scratch on user-provided text. It does not use GPT-2, pretrained LLM weights, or the pretrained GPT-2 `tiktoken` vocabulary.

The goal is a small model whose complete data, model, optimization, evaluation, and generation pipeline can be understood and explained during an interview.

## What the project does

```text
Raw text
   │
   ├── cleaning and document split
   ├── train/validation split
   ├── corpus-trained BPE tokenizer
   ├── uint16 binary token files
   └── memory-mapped training dataset
          │
          ▼
   GPT-style decoder Transformer
          │
          ├── next-token pretraining
          ├── validation loss and perplexity
          ├── resumable checkpoints
          ├── text generation
          └── optional instruction fine-tuning
```

## Architecture

```text
Token IDs
   │
Token embedding + learned positional embedding
   │
Dropout
   │
6 × Transformer block:
    LayerNorm → causal multi-head self-attention → residual
    LayerNorm → GELU feed-forward network → residual
   │
Final LayerNorm
   │
Tied language-model head
   │
Vocabulary logits
```

The default model configuration is:

| Component | Value |
|---|---:|
| Vocabulary | 16,000 BPE tokens |
| Context length | 256 tokens |
| Embedding dimension | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 1,024 |
| Dropout | 0.1 |
| Query/key/value bias | Disabled |
| Parameters | 8.896M trainable parameters |

The LM head shares the same parameter object as the token embedding, reducing memory and parameter duplication.

## Repository layout

```text
llm_project/
├── data/
│   ├── raw/              # Input .txt documents
│   ├── processed/        # train.bin and val.bin
│   └── tokenizer/        # tokenizer.json
├── src/
│   ├── config.py         # Central hyperparameter configuration
│   ├── tokenizer.py      # Corpus-trained BPE tokenizer
│   ├── dataset.py        # Memory-mapped causal LM dataset
│   ├── model.py          # Attention, blocks, and GPT model
│   ├── scheduler.py      # Warmup and cosine decay
│   ├── checkpoint.py     # Save/load/resume support
│   ├── evaluation.py     # Validation loss and perplexity
│   ├── generation.py     # Greedy and sampling generation
│   └── utils.py          # Seeds, parameter counts, JSON utilities
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── finetune.py
├── tests/
├── checkpoints/
├── experiments/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer
- Hugging Face `tokenizers`
- NumPy
- Pytest
- CUDA is optional; CPU fallback is supported

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

Run these commands from the `llm_project` directory:

```bash
python scripts/train_tokenizer.py
python scripts/prepare_data.py
pytest -q
python scripts/train.py
python scripts/evaluate.py
python scripts/generate.py
```

All `.txt` files in `data/raw/` are included. Add more documents there before retraining the tokenizer and regenerating the binary datasets.

## Data preparation

The preprocessing pipeline:

1. Reads all raw `.txt` files.
2. Splits documents using blank lines.
3. Assigns 90% of documents to training and 10% to validation.
4. Adds `<bos>` and `<eos>` around each document.
5. Encodes the documents with the locally trained BPE tokenizer.
6. Writes `train.bin` and `val.bin` as `uint16` arrays.

The vocabulary size is 16,000, so `uint16` safely stores every token ID. The training dataset uses NumPy memory mapping and returns:

```python
x = tokens[i : i + context_length]
y = tokens[i + 1 : i + context_length + 1]
```

This implements standard next-token prediction without creating one PyTorch tensor per training example.

## Tokenizer

The tokenizer is a corpus-trained byte-level BPE tokenizer with these special tokens:

```text
<pad>  <bos>  <eos>  <unk>
```

It is saved to:

```text
data/tokenizer/tokenizer.json
```

The tokenizer API is:

```python
from src.tokenizer import BPETokenizer

tokenizer = BPETokenizer("data/tokenizer/tokenizer.json")
ids = tokenizer.encode("Hello, world!", add_special_tokens=True)
text = tokenizer.decode(ids)
```

No pretrained tokenizer vocabulary is loaded.

## Training

The objective is causal language modeling:

```text
input:  t1 t2 t3 t4
target: t2 t3 t4 t5
```

Training uses:

- AdamW
- Learning rate `3e-4`
- Weight decay `0.1`
- Betas `(0.9, 0.95)`
- Linear warmup followed by cosine decay
- Minimum learning rate `3e-5`
- Gradient clipping at `1.0`
- Configurable gradient accumulation
- BF16 AMP when supported, otherwise FP16 on CUDA
- FP32 fallback on CPU
- Fixed random seed for reproducibility

The effective tokens per optimizer update are:

```text
micro_batch_size × gradient_accumulation_steps × context_length
```

The default is `8 × 1 × 256 = 2,048 tokens/update`.

Training automatically derives the number of optimizer steps from the processed training-token count and defaults to three corpus passes. Override this for experiments:

```bash
python scripts/train.py --max-steps 1000
python scripts/train.py --eval-interval 50 --checkpoint-interval 250
```

Resume from a checkpoint:

```bash
python scripts/train.py --resume checkpoints/latest.pt
```

## Checkpoints and metrics

Checkpoints contain:

- Model parameters
- Optimizer state
- Scheduler state
- AMP scaler state
- Current step
- Tokens seen
- Best validation loss
- Configuration
- Python, NumPy, and PyTorch random states

Files are written to:

```text
checkpoints/best.pt
checkpoints/latest.pt
```

Training metrics are written to:

```text
experiments/metrics.csv
```

Recorded metrics include step, tokens seen, training loss, validation loss, perplexity, learning rate, gradient norm, and tokens per second.

Perplexity is calculated as:

```python
perplexity = exp(validation_loss)
```

## Text generation

`generate.py` generates text from fixed prompts. The generation function supports:

- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p/nucleus sampling
- EOS stopping
- Context-window cropping

Example use:

```python
from src.generation import generate

text = generate(
    model,
    tokenizer,
    "The future of artificial intelligence",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
)
```

## Instruction fine-tuning

Instruction fine-tuning is performed only after pretraining. Prepare a JSONL file with this format:

```json
{"instruction":"Explain gradient descent.","input":"","response":"Gradient descent is an optimization method..."}
```

Run:

```bash
python scripts/finetune.py data/raw/instructions.jsonl
```

The prompt format is:

```text
### Instruction:
...
### Input:
...
### Response:
...
```

Only response tokens contribute to the supervised loss. The pretrained base checkpoint is loaded first, and the fine-tuned checkpoint is saved as `checkpoints/instruction_finetuned.pt`.

## Tests

Run:

```bash
pytest -q
```

The tests cover:

- Token shifting and sequence length
- Causal attention masking
- Forward-pass output shapes
- Loss computation
- Gradient flow
- Weight tying
- Tiny-batch loss reduction during overfitting

## Current dataset status

The supplied general-knowledge corpus is domain-specific. The current file in this repository measured approximately 171K BPE tokens with the project tokenizer. This is suitable for debugging and smoke tests, but not for meaningful large-scale pretraining.

For the intended ~1.31M-token run, place the updated corpus in `data/raw/`, verify its token count, then rerun tokenizer training and data preparation. The pipeline automatically adapts its number of training steps to the resulting token count.

The model should not be compared with ChatGPT or other large language models. Its purpose is educational experimentation and understanding the complete training process.

## Reproducibility and experiment reporting

For each serious run, record:

- Corpus files and token count
- Git/project revision
- Configuration values
- Device and GPU model
- Batch size and accumulation steps
- Training duration
- Tokens per second
- Final validation loss
- Final perplexity
- Generation samples

The seed is stored in `src/config.py` and included in checkpoints. Exact reproducibility can still vary across hardware and CUDA kernels.

## Limitations and future work

This implementation intentionally does not include:

- Distributed training, DDP, FSDP, or tensor parallelism
- KV-cache generation
- FlashAttention dependency
- Large-scale instruction alignment
- Human preference optimization
- Safety evaluation or deployment serving

Potential future improvements include a larger and more diverse corpus, packed document boundaries, better instruction data, KV caching, fused attention, experiment visualization, and multi-GPU training after correctness has been established.
