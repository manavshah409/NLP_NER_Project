"""Verify and freeze the approved classical baseline before any test predictions."""
import subprocess
import importlib.metadata
from pathlib import Path
import yaml
from src.data.audit import Statistics
from src.data.bio import LABELS
from src.data.deterministic_sampler import choose_indices
from src.data.loader import records
from src.utils.io import digest, file_hash, now, read_json, write_json, verify_checksums

EXPERIMENT = "crf_100k_seed42_24dd05e84a26"
MANIFEST = Path("reports/crf/100k_baseline_freeze_manifest.json")
FINAL_DIR = Path("reports/crf/100k_final")


def seal(payload):
    return {"payload": payload, "payload_sha256": digest(payload)}


def verify_freeze(path=MANIFEST):
    envelope = read_json(path)
    payload = envelope["payload"]
    if digest(payload) != envelope["payload_sha256"]:
        raise ValueError("Freeze manifest checksum mismatch")
    for package, version in payload.get("environment", {}).get("packages", {}).items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Frozen environment version changed: {package}")
    for filename, expected in payload["files"].items():
        if file_hash(filename) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")
    return payload


def freeze():
    if MANIFEST.exists():
        verify_freeze()
        print("Existing baseline freeze verified")
        return
    model_dir = Path("models/crf") / EXPERIMENT
    run_dir = Path("reports/crf") / EXPERIMENT
    complete = read_json(run_dir / "complete.json")
    verify_checksums(model_dir, complete["model_artifacts"])
    config = yaml.safe_load((model_dir / "resolved_config.yaml").read_text())
    sample = read_json(model_dir / "sample_manifest.json")
    data_config = yaml.safe_load(Path(config["data"]["data_config"]).read_text())
    version = data_config["derived_version"]
    derived_path = Path(data_config["derived_report_dir"]) / f"{version}_manifest.json"
    derived = read_json(derived_path)
    verify_checksums(data_config["raw_dir"], derived["raw_checksum"])
    verify_checksums(data_config["derived_dir"], derived["derived_checksum"])
    if config["data"]["train_split"] != "train_clean" or config["data"]["validation_split"] != "validation_clean":
        raise ValueError("Unexpected development split")
    if sample["size"] != 100000 or sample["seed"] != config["experiment"]["seed"] or sample["source_split"] != "train_clean":
        raise ValueError("Unexpected sample metadata")
    if sample["derived_checksum"] != derived["derived_checksum"]["combined_sha256"]:
        raise ValueError("Sample data version mismatch")
    training_path = Path(data_config["derived_dir"]) / "train_clean.jsonl"
    expected = choose_indices(records(training_path), 100000, sample["seed"])
    if expected != sample["source_indices"] or digest(expected) != sample["indices_sha256"]:
        raise ValueError("Deterministic sample does not reproduce")
    stats, selected = Statistics(), set(expected)
    for row in records(training_path):
        if row["source_index"] in selected:
            if row["source_split"] != "train":
                raise ValueError("Test/validation record in training sample")
            stats.add(row)
    if stats.result() != sample["sample_statistics"] or stats.result() != complete["training"]["training_statistics"]:
        raise ValueError("Sample statistics mismatch")
    provenance = read_json(Path(data_config["raw_dir"]) / "provenance.json")
    if provenance["labels"] != list(LABELS):
        raise ValueError("Label mapping mismatch")
    paths = list(model_dir.glob("*")) + list(Path("src").rglob("*.py"))
    paths += [Path("requirements.txt"), Path("pyproject.toml"), Path("configs/crf_baseline.yaml"),
              Path(config["data"]["data_config"]), derived_path, run_dir / "source_snapshot.json", run_dir / "complete.json"]
    for root, key in ((data_config["raw_dir"], "raw_checksum"), (data_config["derived_dir"], "derived_checksum")):
        paths += [Path(root) / filename for filename in derived[key]["files"]]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    payload = {"experiment_id": EXPERIMENT, "frozen_at": now(), "git_revision_at_freeze": revision.stdout.strip() if revision.returncode == 0 else None,
               "git_revision_note": "No existing commit at freeze; exact files are identified by SHA256. The subsequent milestone commit records this manifest.",
               "dataset_version": version, "label_mapping": dict(enumerate(LABELS)), "seed": sample["seed"],
               "sample_verification": {"deterministic_indices_reproduced": True, "only_train_clean": True, "statistics_match": True, "records": stats.records},
               "model_dir": str(model_dir), "derived_dir": data_config["derived_dir"],
               "config": config, "environment": read_json(model_dir / "environment.json"),
               "evaluation_policy": "Strict B-only typed spans, exact boundaries, fixed 3-class macro F1. Preserve raw official BIO; include empty official records with empty predictions. No test-time repair.",
               "files": {str(p): file_hash(p) for p in sorted(set(paths)) if p.is_file()},
               "hash_method": "SHA256 per file; envelope SHA256 of canonical UTF-8 JSON payload (sorted keys, no whitespace)."}
    write_json(MANIFEST, seal(payload))
    verify_freeze()
    Path("reports/crf/100k_baseline_freeze_summary.md").write_text(
        f"# 100k classical baseline freeze\n\nExperiment: `{EXPERIMENT}`.\n\n"
        f"Frozen {len(payload['files'])} files, including the model binary, sample, configurations, labels, environment, source/evaluation implementation and dataset files.\n\n"
        "The 100,000 seed-42 source indices were independently reproduced from train_clean and their full statistics matched the saved training artifacts. No validation or test records were sampled.\n\n"
        "All file SHA256 checks and the freeze envelope checksum pass. No Git commit existed at freeze; source/file hashes provide exact identity. The later milestone commit records the manifest without changing it.\n\n"
        + payload["evaluation_policy"] + "\n", encoding="utf-8")
    print(f"Frozen and verified {EXPERIMENT}")


if __name__ == "__main__":
    freeze()
