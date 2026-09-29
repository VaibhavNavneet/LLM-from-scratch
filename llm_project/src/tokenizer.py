from pathlib import Path
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

class BPETokenizer:
    def __init__(self, path=None, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer.from_file(str(path))
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.unk_id = self.tokenizer.token_to_id("<unk>")
    @classmethod
    def train(cls, files, output, vocab_size=16_000):
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                      special_tokens=SPECIAL, show_progress=True)
        tok.train([str(Path(f)) for f in files], trainer)
        Path(output).parent.mkdir(parents=True, exist_ok=True); tok.save(str(output))
        return cls(tokenizer=tok)
    def encode(self, text, add_special_tokens=False):
        ids = self.tokenizer.encode(text).ids
        return ([self.bos_id] + ids + [self.eos_id]) if add_special_tokens else ids
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)
    def __len__(self): return self.tokenizer.get_vocab_size()
