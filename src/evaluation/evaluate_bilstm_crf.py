"""Strict entity evaluation module for PyTorch BiLSTM-CRF.

Evaluates on validation_clean; test splits are rejected by sealed evaluation policy.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from src.data.bilstm_dataset import get_dataloader
from src.data.bilstm_vocabulary import Vocabulary
from src.evaluation.metrics import StrictMetrics
from src.models.bilstm_crf import BiLSTM_CRF, ID2LABEL


def evaluate_model(
    model: BiLSTM_CRF,
    records: List[dict],
    vocab: Vocabulary,
    device: torch.device,
    batch_size: int = 32,
    split_name: str = "validation_clean",
) -> dict:
    """Evaluate BiLSTM-CRF model on a dataset using strict entity metrics.

    Args:
        model: BiLSTM_CRF model instance.
        records: List of record dictionaries with 'tokens' and 'labels'.
        vocab: Training vocabulary instance.
        device: PyTorch device (MPS or CPU).
        batch_size: Evaluation batch size.
        split_name: Split identifier (must not be test_clean or official_test).

    Returns:
        Dictionary of strict entity and token metrics.
    """
    if "test" in split_name.lower():
        raise ValueError(f"Sealed evaluation policy forbids evaluating on {split_name} during development")

    model.eval()
    loader = get_dataloader(records, vocab, batch_size=batch_size, shuffle=False)
    metrics = StrictMetrics()
    latencies_ms = []

    started_total = time.perf_counter()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            gold_labels = batch["labels"]

            batch_loss = model(input_ids, tags=tag_ids, mask=mask, reduction="mean")
            total_loss += batch_loss.item()
            num_batches += 1

            batch_start = time.perf_counter()
            predicted_paths = model.decode(input_ids, mask=mask)
            batch_time = time.perf_counter() - batch_start

            per_seq_ms = (batch_time / len(predicted_paths)) * 1000.0
            latencies_ms.extend([per_seq_ms] * len(predicted_paths))

            for gold, pred_ids in zip(gold_labels, predicted_paths):
                pred_labels = [ID2LABEL[pid] for pid in pred_ids]
                metrics.add(gold, pred_labels)

    total_time = time.perf_counter() - started_total
    result = metrics.result()
    result.update({
        "records_evaluated": len(records),
        "split_evaluated": split_name,
        "evaluation_seconds": total_time,
        "loss": (total_loss / max(1, num_batches)) if num_batches else 0.0,
        "median_latency_ms": float(np.median(latencies_ms)) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    })
    return result
