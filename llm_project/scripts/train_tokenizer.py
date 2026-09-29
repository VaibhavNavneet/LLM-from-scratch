import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.config import Config
from src.tokenizer import BPETokenizer

root=Path(__file__).parents[1]; cfg=Config(); sources=sorted(root.glob('data/raw/*.txt'))
if not sources: raise FileNotFoundError('no .txt files found in data/raw')
tok=BPETokenizer.train(sources, root/'data/tokenizer/tokenizer.json', cfg.vocab_size)
print(f"trained vocabulary: {len(tok)}")
