"""PyTorch Dataset and DataLoader collation for BiLSTM-CRF."""
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.bilstm_vocabulary import PAD_ID, Vocabulary
from src.models.bilstm_crf import LABEL2ID


class NERDataset(Dataset):
    """Sequence labeling dataset mapping tokens to IDs and BIO tags to tag IDs."""

    def __init__(self, records: List[dict], vocab: Vocabulary):
        self.samples = []
        for i, row in enumerate(records):
            tokens = row.get("tokens")
            labels = row.get("labels")
            if not tokens or not labels:
                raise ValueError(f"Empty tokens or labels in record index {i}")
            if len(tokens) != len(labels):
                raise ValueError(f"Token/label length mismatch in record index {i}")
            
            input_ids = vocab.encode(tokens)
            tag_ids = [LABEL2ID[label] for label in labels]

            self.samples.append({
                "tokens": tokens,
                "labels": labels,
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
                "length": len(tokens),
                "source_index": row.get("source_index", i),
                "source_split": row.get("source_split", "unknown"),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_ner_batch(batch: List[dict]) -> dict:
    """Collate variable length sequences with padding and boolean mask."""
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)

    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    tag_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    lengths = torch.empty(batch_size, dtype=torch.long)

    tokens_list = []
    labels_list = []
    source_indices = []

    for b, item in enumerate(batch):
        seq_len = item["length"]
        input_ids[b, :seq_len] = item["input_ids"]
        tag_ids[b, :seq_len] = item["tag_ids"]
        mask[b, :seq_len] = True
        lengths[b] = seq_len
        tokens_list.append(item["tokens"])
        labels_list.append(item["labels"])
        source_indices.append(item["source_index"])

    return {
        "input_ids": input_ids,
        "tag_ids": tag_ids,
        "mask": mask,
        "lengths": lengths,
        "tokens": tokens_list,
        "labels": labels_list,
        "source_indices": source_indices,
    }


def get_dataloader(
    records: List[dict],
    vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with variable-length batch collation."""
    dataset = NERDataset(records, vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ner_batch,
        num_workers=num_workers,
    )
