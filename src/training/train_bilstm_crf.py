"""Controlled PyTorch BiLSTM-CRF training pipeline for Hindi NER."""
import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional
import torch
import yaml
from src.data.audit import Statistics
from src.data.bilstm_dataset import get_dataloader
from src.data.bilstm_vocabulary import Vocabulary
from src.data.deterministic_sampler import create_manifest
from src.data.loader import records
from src.evaluation.evaluate_bilstm_crf import evaluate_model
from src.models.bilstm_crf import BiLSTM_CRF
from src.utils.environment import environment
from src.utils.io import checksums, digest, file_hash, read_json, write_json, verify_checksums
from src.utils.neural_resource_monitor import NeuralResourceTracker, get_device, system_diagnostics
from src.utils.resource_monitor import limits
from src.utils.torch_reproducibility import set_seed


def train_bilstm_crf(
    size: int,
    config_path: str = "configs/bilstm_crf.yaml",
    seed: Optional[int] = None,
) -> dict:
    """Train BiLSTM-CRF model for a specific sample size (1k, 10k, 50k, 100k) and seed."""
    if size not in (1000, 10000, 50000, 100000):
        raise ValueError("Only 1k, 10k, 50k, and 100k training sizes are authorised")

    config = yaml.safe_load(Path(config_path).read_text())
    if seed is not None:
        config["experiment"]["seed"] = seed
    else:
        seed = config["experiment"]["seed"]

    data_config = yaml.safe_load(Path(config["data"]["data_config"]).read_text())
    report_data_dir = Path(data_config["derived_report_dir"])
    version = data_config["derived_version"]

    manifest = read_json(report_data_dir / f"{version}_manifest.json")
    derived_dir = Path(data_config["derived_dir"])

    # Verify derived training and validation files exist and match checksums
    train_clean_path = derived_dir / "train_clean.jsonl"
    val_clean_path = derived_dir / "validation_clean.jsonl"
    if not train_clean_path.exists() or not val_clean_path.exists():
        raise FileNotFoundError("Clean training/validation datasets not found")

    manifest_dir = Path("experiments/bilstm_crf/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Sample manifest selection:
    # Repeated 50k runs MUST reuse the exact same 50k seed-42 sample manifest
    if size == 50000:
        sample_path = manifest_dir / "train_050k_seed42.json"
        if not sample_path.exists():
            sample = create_manifest(
                sample_path,
                train_clean_path,
                50000,
                42,
                manifest["derived_checksum"]["combined_sha256"],
                manifest["clean_statistics"]["train_clean"],
            )
        else:
            sample = read_json(sample_path)
    elif size == 100000:
        # Controlled 100k run uses nested seed 42 sample
        sample_path = manifest_dir / "train_100k_seed42.json"
        sample = create_manifest(
            sample_path,
            train_clean_path,
            100000,
            42,
            manifest["derived_checksum"]["combined_sha256"],
            manifest["clean_statistics"]["train_clean"],
        )
        # Verify strict nesting: 100k sample must contain all 50k sample indices
        sample_50k = read_json(manifest_dir / "train_050k_seed40.json" if (manifest_dir / "train_050k_seed40.json").exists() else manifest_dir / "train_050k_seed42.json")
        if not set(sample_50k["source_indices"]).issubset(set(sample["source_indices"])):
            raise ValueError("100k sample does not strictly nest the 50k training sample")
    else:
        sample_path = manifest_dir / f"train_{size//1000:03d}k_seed42.json"
        sample = create_manifest(
            sample_path,
            train_clean_path,
            size,
            42,
            manifest["derived_checksum"]["combined_sha256"],
            manifest["clean_statistics"]["train_clean"],
        )

    # Set seeds across Python, NumPy, PyTorch CPU, and MPS
    seed_info = set_seed(seed)

    # Device selection
    device = get_device(
        preferred=config["resources"]["preferred_device"],
        allow_cpu_fallback=config["resources"]["allow_cpu_fallback"],
    )
    print(f"[{size} records | Seed {seed}] Active device: {device}", flush=True)

    # Load sampled training records
    selected_indices = set(sample["source_indices"])
    train_records = []
    train_stats = Statistics()
    for row in records(train_clean_path):
        if row["source_index"] in selected_indices:
            train_records.append(row)
            train_stats.add(row)

    if len(train_records) != size or train_stats.result() != sample["sample_statistics"]:
        raise ValueError("Loaded training sample does not match sample manifest statistics")

    # Load validation records
    val_records = list(records(val_clean_path))

    # Build vocabulary strictly from training records
    vocab = Vocabulary.build_from_records(
        train_records,
        min_frequency=config["vocabulary"]["min_frequency"],
        source_split="train_clean",
    )
    unk_stats = vocab.evaluate_unknown_rate(val_records)
    print(f"[{size} records | Seed {seed}] Vocab size: {len(vocab)}, Val UNK rate: {unk_stats['unknown_token_rate']:.4%}", flush=True)

    # For 50k runs, verify vocabulary matches baseline 50k vocabulary
    if size == 50000:
        baseline_vocab_path = Path("models/bilstm_crf/bilstm_crf_050k_seed42_1c3a3bd180d5/vocabulary.json")
        if baseline_vocab_path.exists():
            baseline_vocab = Vocabulary.load(baseline_vocab_path)
            if len(vocab) != len(baseline_vocab) or vocab.token2id != baseline_vocab.token2id:
                raise ValueError("50k vocabulary does not match approved baseline vocabulary")

    # Experiment identifier
    code_snapshot = {p.as_posix(): p.read_text() for p in sorted(Path("src").rglob("*.py"))}
    hash_payload = {
        "config": config,
        "size": size,
        "seed": seed,
        "indices_sha256": sample["indices_sha256"],
        "code_sha256": digest(code_snapshot),
        "vocab_sha256": digest(vocab.token2id),
    }
    experiment_id = f"bilstm_crf_{size//1000:03d}k_seed{seed}_" + digest(hash_payload)[:12]
    model_dir = Path("models/bilstm_crf") / experiment_id
    report_dir = Path("reports/bilstm_crf") / experiment_id

    # Check for existing completed experiment with prefix match (handling legacy seed 42)
    if (report_dir / "complete.json").exists():
        print(f"Experiment already completed: {experiment_id}", flush=True)
        return read_json(report_dir / "complete.json")
    
    # Check legacy seed 42 folder
    if size == 50000 and seed == 42:
        legacy_report = Path("reports/bilstm_crf/bilstm_crf_050k_seed42_1c3a3bd180d5/complete.json")
        if legacy_report.exists():
            print(f"Legacy seed 42 experiment verified at {legacy_report.parent}", flush=True)
            return read_json(legacy_report)

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary and sample manifest
    vocab_meta = vocab.save(model_dir / "vocabulary.json")
    write_json(model_dir / "sample_manifest.json", sample)
    write_json(report_dir / "source_snapshot.json", {"source_sha256": digest(code_snapshot), "files": code_snapshot})

    # Prepare model
    model = BiLSTM_CRF(
        vocab_size=len(vocab),
        num_tags=7,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_lstm_layers=config["model"]["num_lstm_layers"],
        bidirectional=config["model"]["bidirectional"],
        dropout=config["model"]["dropout"],
        padding_idx=0,
        use_crf=config["model"]["use_crf"],
    ).to(device)

    # Dataloaders
    batch_size = config["training"]["batch_size"]
    train_loader = get_dataloader(train_records, vocab, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Training state
    max_epochs = config["training"]["maximum_epochs"]
    patience = config["training"]["early_stopping_patience"]
    clip_norm = config["training"]["gradient_clip_norm"]
    resource_tracker = NeuralResourceTracker()

    training_history = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    early_stopping_reason = "maximum_epochs_completed"
    total_train_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            tag_ids = batch["tag_ids"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids, tags=tag_ids, mask=mask, reduction="mean")
            loss.backward()

            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        epoch_duration = time.perf_counter() - epoch_start
        avg_train_loss = running_loss / max(1, num_batches)
        mem_sample = resource_tracker.sample()

        # Validation evaluation
        val_metrics = evaluate_model(
            model,
            val_records,
            vocab,
            device,
            batch_size=batch_size,
            split_name="validation_clean",
        )
        val_f1 = val_metrics["strict_entity_micro_f1"]
        val_macro = val_metrics["strict_entity_macro_f1"]
        val_loss = val_metrics.get("loss", 0.0)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "validation_loss": val_loss,
            "validation_strict_precision": val_metrics["strict_entity_precision"],
            "validation_strict_recall": val_metrics["strict_entity_recall"],
            "validation_strict_micro_f1": val_f1,
            "validation_strict_macro_f1": val_macro,
            "validation_per_class": val_metrics["per_class"],
            "validation_token_accuracy": val_metrics["token_accuracy"],
            "epoch_duration_seconds": epoch_duration,
            "memory_peak_rss_gb": mem_sample["peak_rss_gb"],
        }
        training_history.append(epoch_record)
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Micro F1: {val_f1:.6f} | Val Macro F1: {val_macro:.6f} | {epoch_duration:.1f}s",
            flush=True,
        )

        # Save last checkpoint
        torch.save(model.state_dict(), model_dir / "last_model.pt")

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopping_reason = f"early_stopping_patience_reached_at_epoch_{epoch}"
                print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with F1: {best_val_f1:.6f})", flush=True)
                break

    total_train_seconds = time.perf_counter() - total_train_start

    # Reload best checkpoint and verify final validation metrics
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    final_val_metrics = evaluate_model(
        model,
        val_records,
        vocab,
        device,
        batch_size=batch_size,
        split_name="validation_clean",
    )

    model_size_bytes = (model_dir / "best_model.pt").stat().st_size

    # Save metrics and config
    config_resolved = dict(config)
    config_resolved.update({
        "experiment_id": experiment_id,
        "sample_size": size,
        "seed": seed,
        "device_used": str(device),
        "seed_info": seed_info,
        "best_epoch": best_epoch,
        "early_stopping_reason": early_stopping_reason,
        "total_train_seconds": total_train_seconds,
        "vocab_size": len(vocab),
        "val_unk_rate": unk_stats["unknown_token_rate"],
        "model_size_bytes": model_size_bytes,
        "peak_rss_gb": resource_tracker.sample()["peak_rss_gb"],
    })
    (model_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config_resolved, sort_keys=False, allow_unicode=True))
    write_json(model_dir / "environment.json", system_diagnostics())
    write_json(model_dir / "training_history.json", training_history)
    write_json(model_dir / "validation_metrics.json", final_val_metrics)

    model_card = f"""# IndicNewsNER BiLSTM-CRF Model Card: {experiment_id}

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** {size:,} clean training records from `naamapadam_hi_crf_v1` (Seed: {seed}).
- **Vocabulary Size:** {len(vocab):,} tokens (built strictly from training sample).
- **Validation UNK Rate:** {unk_stats['unknown_token_rate']:.4%}.
- **Device Used:** {device}.
- **Best Epoch:** {best_epoch} (Early stopping patience: {patience}, Reason: {early_stopping_reason}).
- **Total Training Time:** {total_train_seconds:.2f} seconds.
- **Model Size:** {model_size_bytes / (1024 ** 2):.2f} MiB ({model_size_bytes:,} bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** {final_val_metrics['strict_entity_micro_f1']:.8f}
- **Strict Macro F1:** {final_val_metrics['strict_entity_macro_f1']:.8f}
- **Precision:** {final_val_metrics['strict_entity_precision']:.8f}
- **Recall:** {final_val_metrics['strict_entity_recall']:.8f}
- **PER F1:** {final_val_metrics['per_class']['PER']['f1']:.8f}
- **ORG F1:** {final_val_metrics['per_class']['ORG']['f1']:.8f}
- **LOC F1:** {final_val_metrics['per_class']['LOC']['f1']:.8f}
- **Token Accuracy:** {final_val_metrics['token_accuracy']:.8f}
- **Median Latency:** {final_val_metrics['median_latency_ms']:.3f} ms / sentence
- **P95 Latency:** {final_val_metrics['p95_latency_ms']:.3f} ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
"""
    (model_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    # Artifacts checksums (excluding checksums.json itself)
    artifact_files = {p.relative_to(model_dir).as_posix(): file_hash(p)
                      for p in sorted(model_dir.rglob("*")) if p.is_file() and p.name != "checksums.json"}
    artifacts_checksum = {
        "method": "SHA256 of UTF-8 canonical JSON mapping relative POSIX file paths to SHA256; sorted keys, no spaces, ensure_ascii=False; manifest stored outside root",
        "files": artifact_files,
        "combined_sha256": digest(artifact_files),
    }
    write_json(model_dir / "checksums.json", artifacts_checksum)

    summary = {
        "experiment_id": experiment_id,
        "sample_size": size,
        "seed": seed,
        "best_epoch": best_epoch,
        "early_stopping_reason": early_stopping_reason,
        "total_train_seconds": total_train_seconds,
        "validation_metrics": final_val_metrics,
        "training_history": training_history,
        "vocab_size": len(vocab),
        "val_unk_stats": unk_stats,
        "model_artifacts": artifacts_checksum,
        "model_size_bytes": model_size_bytes,
        "peak_rss_gb": resource_tracker.sample()["peak_rss_gb"],
    }
    write_json(report_dir / "complete.json", summary)
    print(f"[{size} records | Seed {seed}] Experiment {experiment_id} complete! Strict Val Micro F1: {final_val_metrics['strict_entity_micro_f1']:.6f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hindi BiLSTM-CRF on controlled sample sizes.")
    parser.add_argument("--size", type=int, choices=[1000, 10000, 50000, 100000], default=1000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train_bilstm_crf(args.size, seed=args.seed)
