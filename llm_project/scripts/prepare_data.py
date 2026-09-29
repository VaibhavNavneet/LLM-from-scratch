import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.config import Config
from src.tokenizer import BPETokenizer

root=Path(__file__).parents[1]; tok=BPETokenizer(root/'data/tokenizer/tokenizer.json'); sources=sorted(root.glob('data/raw/*.txt'))
if not sources: raise FileNotFoundError('no .txt files found in data/raw')
text='\n\n'.join(p.read_text(encoding='utf-8') for p in sources)
docs=[d.strip() for d in re.split(r'\n\s*\n',text) if d.strip()]
cut=max(1,int(len(docs)*.9)); train_docs, val_docs=docs[:cut], docs[cut:]
def encode(ds):
    ids=[]
    for d in ds: ids.extend(tok.encode(d, add_special_tokens=True))
    return np.asarray(ids,dtype=np.uint16)
out=root/'data/processed'; out.mkdir(parents=True,exist_ok=True)
for name, ds in [('train',train_docs),('val',val_docs)]:
    arr=encode(ds); arr.tofile(out/f'{name}.bin'); print(name, len(arr), 'tokens', len(ds), 'documents')
