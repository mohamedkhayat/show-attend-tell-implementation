import json
from collections import Counter
import torch


class Vocabulary:
    def __init__(self, min_freq=5, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        # Special tokens
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()

    def build_vocab(self, captions):
        """Build vocabulary from captions"""
        # Count word frequencies
        for caption in captions:
            words = caption.lower().split()
            self.word_count.update(words)

        # Add words above min_freq
        words = [
            word for word, freq in self.word_count.items() if freq >= self.min_freq
        ]
        words = words[: self.max_vocab_size - len(self.word2idx)]

        for idx, word in enumerate(words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, caption, max_length=20):
        """Encode caption to indices"""
        words = caption.lower().split()
        indices = [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]

        # Add <start> and <end>
        indices = [self.word2idx["<start>"]] + indices[:max_length - 2] + [self.word2idx["<end>"]]
        # Pad to max_length
        indices += [self.word2idx["<pad>"]] * (max_length - len(indices))
        return torch.tensor(indices[:max_length])

    def decode(self, indices):
        """Decode indices to text"""
        words = []
        start_idx = self.word2idx["<start>"]
        end_idx = self.word2idx["<end>"]
        pad_idx = self.word2idx["<pad>"]

        for idx in indices:
            # Support torch tensors, numpy scalars, and plain ints.
            if hasattr(idx, "item"):
                idx = int(idx.item())
            else:
                idx = int(idx)

            if idx == end_idx:
                break
            if idx in (pad_idx, start_idx):
                continue

            if idx != pad_idx:
                words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def save(self, filepath):
        """Save vocabulary to JSON"""
        with open(filepath, "w") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word,
                    "word_count": dict(self.word_count),
                },
                f,
            )

    @classmethod
    def load(cls, filepath):
        """Load vocabulary from JSON"""
        vocab = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
            vocab.word2idx = data["word2idx"]
            # JSON object keys are strings, convert back to integer ids.
            vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
            vocab.word_count = Counter(data["word_count"])
        return vocab
