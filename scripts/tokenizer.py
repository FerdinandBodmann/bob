"""A tiny character-level tokenizer: full functionality, zero downloads.

Swappable for a HF `tokenizers` BPE later — same encode/decode surface.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars=None):
        self.chars = sorted(set(chars)) if chars else []
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def train_from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"chars": self.chars}))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["chars"])
