"""Training-only vocabulary builder and serializer for BiLSTM-CRF.

Ensures deterministic mapping, UNK handling, and zero leakage from validation/test.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from src.utils.io import digest, read_json, write_json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    """Deterministic vocabulary mapping tokens to IDs with UNK and PAD support."""

    def __init__(
        self,
        token2id: Optional[Dict[str, int]] = None,
        min_frequency: int = 1,
        source_split: str = "train_clean",
    ):
        self.min_frequency = min_frequency
        self.source_split = source_split

        if token2id is not None:
            self.token2id = dict(token2id)
            self.id2token = {v: k for k, v in self.token2id.items()}
        else:
            self.token2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
            self.id2token = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert list of token strings to token IDs."""
        return [self.token2id.get(token, UNK_ID) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert list of token IDs back to token strings."""
        return [self.id2token.get(i, UNK_TOKEN) for i in ids]

    @classmethod
    def build_from_records(
        cls,
        records: Iterable[dict],
        min_frequency: int = 1,
        source_split: str = "train_clean",
    ) -> "Vocabulary":
        """Build vocabulary strictly from training records."""
        counter = Counter()
        for row in records:
            split = row.get("source_split")
            if split is not None and split not in ("train", "train_clean"):
                raise ValueError(f"Vocabulary must only be built from train_clean records, got {split}")
            counter.update(row["tokens"])

        # Deterministic ordering: descending frequency, then lexicographical by token
        sorted_tokens = sorted(
            [tok for tok, count in counter.items() if count >= min_frequency],
            key=lambda t: (-counter[t], t),
        )

        token2id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token in sorted_tokens:
            if token not in token2id:
                token2id[token] = len(token2id)

        vocab = cls(token2id=token2id, min_frequency=min_frequency, source_split=source_split)
        return vocab

    def save(self, path: Path) -> dict:
        """Save vocabulary to JSON with checksum."""
        data = {
            "source_split": self.source_split,
            "min_frequency": self.min_frequency,
            "vocab_size": len(self.token2id),
            "pad_token": PAD_TOKEN,
            "pad_id": PAD_ID,
            "unk_token": UNK_TOKEN,
            "unk_id": UNK_ID,
            "token2id": self.token2id,
        }
        data["vocab_sha256"] = digest(self.token2id)
        write_json(path, data)
        return data

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        data = read_json(path)
        token2id = data["token2id"]
        return cls(
            token2id=token2id,
            min_frequency=data.get("min_frequency", 1),
            source_split=data.get("source_split", "train_clean"),
        )

    def evaluate_unknown_rate(self, records: Iterable[dict]) -> dict:
        """Compute total tokens, unknown tokens, and unknown rate on an evaluation set."""
        total_tokens = 0
        unknown_tokens = 0
        for row in records:
            tokens = row["tokens"]
            total_tokens += len(tokens)
            unknown_tokens += sum(1 for t in tokens if t not in self.token2id)
        unk_rate = (unknown_tokens / total_tokens) if total_tokens > 0 else 0.0
        return {
            "total_tokens": total_tokens,
            "unknown_tokens": unknown_tokens,
            "unknown_token_rate": unk_rate,
        }
