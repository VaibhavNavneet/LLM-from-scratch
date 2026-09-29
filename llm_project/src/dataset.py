import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    def __init__(self, path, context_length):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.context_length = context_length
    def __len__(self): return max(0, len(self.tokens) - self.context_length)
    def __getitem__(self, i):
        x = np.asarray(self.tokens[i:i+self.context_length], dtype=np.int64)
        y = np.asarray(self.tokens[i+1:i+self.context_length+1], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def create_loader(path, context_length, batch_size, shuffle=True, workers=0):
    return DataLoader(MemmapDataset(path, context_length), batch_size=batch_size,
                      shuffle=shuffle, drop_last=True, num_workers=workers,
                      pin_memory=torch.cuda.is_available())
