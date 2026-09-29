import numpy as np, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.dataset import MemmapDataset

def test_shift(tmp_path):
    p=tmp_path/'x.bin'; np.arange(20,dtype=np.uint16).tofile(p); x,y=MemmapDataset(p,4)[3]
    assert x.tolist()==[3,4,5,6] and y.tolist()==[4,5,6,7]
