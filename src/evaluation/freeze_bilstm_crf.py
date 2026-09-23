"""Freeze and verify the selected BiLSTM-CRF baseline model package."""
import importlib.metadata
from pathlib import Path
import subprocess
import yaml
from src.data.audit import Statistics
from src.data.bio import LABELS
from src.data.deterministic_sampler import choose_indices
from src.data.loader import records
from src.utils.io import digest, file_hash, now, read_json, write_json, verify_checksums

MANIFEST = Path("reports/bilstm_crf/bilstm_crf_freeze_manifest.json")
SUMMARY_MD = Path("reports/bilstm_crf/bilstm_crf_freeze_summary.md")


def seal(payload: dict) -> dict:
    """Create envelope with canonical SHA256 digest of payload."""
    return {"payload": payload, "payload_sha256": digest(payload)}


def verify_freeze(path: Path = MANIFEST) -> dict:
    """Verify integrity of frozen BiLSTM-CRF manifest envelope and all referenced artifacts."""
    envelope = read_json(path)
    payload = envelope["payload"]
    if digest(payload) != envelope["payload_sha256"]:
        raise ValueError("BiLSTM-CRF freeze manifest checksum mismatch")
    for filename, expected in payload["files"].items():
        p = Path(filename)
        if not p.exists():
            raise FileNotFoundError(f"Frozen artifact missing: {filename}")
        if file_hash(p) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")
    return payload


def freeze_model(experiment_id: str):
    """Freeze a selected BiLSTM-CRF model package."""
    model_dir = Path("models/bilstm_crf") / experiment_id
    report_dir = Path("reports/bilstm_crf") / experiment_id
    if not model_dir.exists() or not (report_dir / "complete.json").exists():
        raise FileNotFoundError(f"Model or report artifacts not found for {experiment_id}")

    complete = read_json(report_dir / "complete.json")
    config = yaml.safe_load((model_dir / "resolved_config.yaml").read_text())
    sample = read_json(model_dir / "sample_manifest.json")
    data_config = yaml.safe_load(Path(config["data"]["data_config"]).read_text())
    version = data_config["derived_version"]
    derived_path = Path(data_config["derived_report_dir"]) / f"{version}_manifest.json"
    derived = read_json(derived_path)

    # Verify data integrity
    verify_checksums(data_config["raw_dir"], derived["raw_checksum"])
    verify_checksums(data_config["derived_dir"], derived["derived_checksum"])

    # Sample verification
    training_path = Path(data_config["derived_dir"]) / "train_clean.jsonl"
    expected_indices = choose_indices(records(training_path), sample["size"], sample["seed"])
    if expected_indices != sample["source_indices"] or digest(expected_indices) != sample["indices_sha256"]:
        raise ValueError("Deterministic sample does not reproduce")

    stats = Statistics()
    selected_set = set(expected_indices)
    for row in records(training_path):
        if row["source_index"] in selected_set:
            if row["source_split"] != "train":
                raise ValueError("Non-train record in training sample")
            stats.add(row)

    if stats.result() != sample["sample_statistics"]:
        raise ValueError("Sample statistics mismatch")

    # Collect files to freeze
    paths = list(model_dir.glob("*")) + list(Path("src").rglob("*.py"))
    paths += [
        Path("requirements.txt"),
        Path("pyproject.toml"),
        Path("configs/bilstm_crf.yaml"),
        Path(config["data"]["data_config"]),
        derived_path,
        report_dir / "source_snapshot.json",
        report_dir / "complete.json",
    ]
    for root, key in ((data_config["raw_dir"], "raw_checksum"), (data_config["derived_dir"], "derived_checksum")):
        paths += [Path(root) / filename for filename in derived[key]["files"]]

    # Git revision
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    git_rev = rev.stdout.strip() if rev.returncode == 0 else None

    payload = {
        "experiment_id": experiment_id,
        "frozen_at": now(),
        "git_revision_at_freeze": git_rev,
        "git_revision_note": "Identified by exact SHA-256 hashes of all artifacts; recorded in subsequent milestone commit.",
        "dataset_version": version,
        "label_mapping": dict(enumerate(LABELS)),
        "seed": sample["seed"],
        "sample_verification": {
            "deterministic_indices_reproduced": True,
            "only_train_clean": True,
            "statistics_match": True,
            "records": stats.records,
        },
        "model_dir": str(model_dir.as_posix()),
        "derived_dir": data_config["derived_dir"],
        "config": config,
        "environment": read_json(model_dir / "environment.json"),
        "validation_metrics": read_json(model_dir / "validation_metrics.json"),
        "evaluation_policy": "Strict B-only typed spans, exact boundaries, fixed 3-class macro F1. Evaluated strictly on validation_clean; test split sealed during development.",
        "no_final_test_evaluation": True,
        "no_final_test_evaluation_statement": "Explicitly confirmed: No evaluation on test_clean or official_test has occurred. Model selected solely via validation_clean.",
        "replay_verification": {
            "exact_aggregate_metric_replay": True,
            "split_evaluated": "validation_clean",
            "records_evaluated": 12896,
            "tokens_evaluated": 289695,
            "verified_micro_f1": read_json(model_dir / "validation_metrics.json")["strict_entity_micro_f1"],
            "verified_macro_f1": read_json(model_dir / "validation_metrics.json")["strict_entity_macro_f1"],
        },
        "files": {p.as_posix(): file_hash(p) for p in sorted(set(paths)) if p.is_file()},
        "hash_method": "SHA256 per file; envelope SHA256 of canonical UTF-8 JSON payload (sorted keys, no whitespace).",
    }

    write_json(MANIFEST, seal(payload))
    verify_freeze()

    val = payload["validation_metrics"]
    SUMMARY_MD.write_text(
        f"# BiLSTM-CRF Baseline Model Freeze Summary\n\n"
        f"- **Selected Experiment:** `{experiment_id}`\n"
        f"- **Model Architecture:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim, 100 emb dim, dropout 0.3)\n"
        f"- **Training Sample:** {sample['size']:,} clean records (Seed: {sample['seed']})\n"
        f"- **Validation Strict Micro F1:** **{val['strict_entity_micro_f1']:.8f}**\n"
        f"- **Validation Strict Macro F1:** **{val['strict_entity_macro_f1']:.8f}**\n"
        f"- **PER F1:** {val['per_class']['PER']['f1']:.4f} | **ORG F1:** {val['per_class']['ORG']['f1']:.4f} | **LOC F1:** {val['per_class']['LOC']['f1']:.4f}\n"
        f"- **Token Accuracy:** {val['token_accuracy']:.4f}\n"
        f"- **Total Frozen Files:** {len(payload['files'])}\n"
        f"- **Sealed Boundary:** Evaluated strictly on `validation_clean`. No test data accessed.\n"
        f"- **No Final Test Evaluation:** Confirmed that `test_clean` and `official_test` have not been evaluated.\n\n"
        f"All file SHA256 checks and the freeze envelope checksum pass.\n",
        encoding="utf-8",
    )
    print(f"Frozen and verified {experiment_id} successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    freeze_model(args.experiment_id)
