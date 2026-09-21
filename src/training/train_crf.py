"""Verified CRF entry point; controlled sizes only; worker isolated for monitoring."""
import argparse
import gc
import sys
import time
from pathlib import Path
import psutil
import yaml
from src.data.audit import Statistics
from src.data.crf_features import sentence_features
from src.data.deterministic_sampler import create_manifest
from src.data.loader import records
from src.evaluation.evaluate_crf import evaluate
from src.models.crf_model import CRFModel
from src.utils.environment import environment
from src.utils.io import checksums, digest, read_json, write_json, verify_checksums
from src.utils.resource_monitor import limits, monitor


def worker(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    model_dir = Path(config_path).parent
    sample = read_json(model_dir / "sample_manifest.json")
    selected = set(sample["source_indices"])
    if len(selected) != config["data"]["sample_size"] or sample["source_split"] != "train_clean":
        raise ValueError("Invalid sample manifest")
    data_config = yaml.safe_load(Path(config["data"]["data_config"]).read_text())
    trainer = CRFModel(config["model"]).trainer()
    process, stats = psutil.Process(), Statistics()
    rss_before = process.memory_info().rss
    peak_rss = rss_before
    feature_count, feature_time, vocabulary = 0, 0.0, set()
    for row in records(Path(data_config["derived_dir"]) / "train_clean.jsonl"):
        if row["source_index"] not in selected:
            continue
        if row["source_split"] != "train" or len(row["tokens"]) != len(row["labels"]) or not row["tokens"]:
            raise ValueError("Invalid training record")
        begin = time.perf_counter()
        features = sentence_features(row["tokens"], config["features"])
        feature_time += time.perf_counter() - begin
        feature_count += sum(len(f) for f in features)
        trainer.append(features, row["labels"])
        stats.add(row)
        vocabulary.update(row["tokens"])
        peak_rss = max(peak_rss, process.memory_info().rss)
    if stats.result() != sample["sample_statistics"]:
        raise ValueError("Training sample statistics differ")
    print(f"Prepared {stats.records} records, {stats.tokens} tokens", flush=True)
    rss_after = process.memory_info().rss
    started = time.perf_counter()
    trainer.train(str(model_dir / "model.crfsuite"))
    elapsed = time.perf_counter() - started
    del trainer
    gc.collect()
    metrics = {"training_statistics": stats.result(), "feature_extraction_seconds": feature_time,
               "training_seconds": elapsed, "feature_key_value_occurrences": feature_count,
               "feature_proxy_definition": "Sum of Python feature dictionary key/value pairs across tokens including false boolean entries; not unique CRFsuite attributes",
               "observed_preparation_rss_delta_bytes": rss_after - rss_before,
               "feature_memory_note": "RSS delta includes native trainer storage, vocabulary, and allocator effects; isolated feature memory not measured",
               "peak_preparation_rss_bytes_record_sampled": peak_rss,
               "model_size_bytes": (model_dir / "model.crfsuite").stat().st_size}
    write_json(model_dir / "training_metrics.json", metrics)
    validation = evaluate(model_dir / "model.crfsuite", Path(data_config["derived_dir"]) / "validation_clean.jsonl",
                          config, vocabulary, config["report_dir"])
    write_json(model_dir / "validation_metrics.json", validation)
    (model_dir / "model_card.md").write_text(f"# Hindi CRF {config['experiment_id']}\n\nTrained on {stats.records} sampled clean training records. Seed 42.\n\nNative CRFsuite file. See configuration and environment for exact settings.\n\nValidation strict entity micro F1: {validation['strict_entity_micro_f1']:.8f}.\n\nNo test evaluation. Hindi only; news domain generalisation and English remain unverified.\n")


def train(size):
    if size not in (1000, 10000, 50000, 100000):
        raise ValueError("Only 1k, 10k, 50k and the approved 100k training is authorised")
    config = yaml.safe_load(Path("configs/crf_baseline.yaml").read_text())
    data_config = yaml.safe_load(Path(config["data"]["data_config"]).read_text())
    report = Path(data_config["derived_report_dir"])
    version = data_config["derived_version"]
    manifest = read_json(report / f"{version}_manifest.json")
    integrity = read_json(report / f"{version}_integrity.json")
    reproduction = read_json(report / f"{version}_reproducibility.json")
    if not integrity or not all(integrity.values()) or not all(s["identical"] for s in reproduction.values()):
        raise ValueError("Data integrity gate failed")
    verify_checksums(data_config["raw_dir"], manifest["raw_checksum"])
    verify_checksums(data_config["derived_dir"], manifest["derived_checksum"])
    sample_path = f"experiments/crf/manifests/train_{size//1000:03d}k_seed42.json"
    sample = create_manifest(sample_path, Path(data_config["derived_dir"]) / "train_clean.jsonl", size, 42,
                             manifest["derived_checksum"]["combined_sha256"], manifest["clean_statistics"]["train_clean"])
    config["data"].update(sample_size=size, sample_manifest=sample_path)
    code = {p.as_posix(): p.read_text() for p in sorted(Path("src").rglob("*.py"))}
    identifier = f"crf_{size//1000:03d}k_seed42_" + digest({"config": config, "indices": sample["indices_sha256"], "data": sample["derived_checksum"], "code": code, "requirements": Path("requirements.txt").read_text()})[:12]
    model_dir = Path("models/crf") / identifier
    report_dir = Path("reports/crf") / identifier
    if (report_dir / "complete.json").exists():
        verify_checksums(model_dir, read_json(report_dir / "complete.json")["model_artifacts"])
        print(f"Existing completed experiment verified: {identifier}", flush=True)
        return
    if model_dir.exists():
        raise ValueError(f"Incomplete experiment preserved at {model_dir}; review before retry")
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "source_snapshot.json", {"source_sha256": digest(code), "files": code})
    policy = limits()
    if policy["maximum_memory_bytes"] < 1024 ** 3:
        raise RuntimeError("Less than 1 GiB conservative memory budget; training deferred")
    config["resources"]["maximum_memory_gb"] = policy["maximum_memory_bytes"] / 1e9
    config["resources"]["resolved_policy"] = policy
    config.update(experiment_id=identifier, report_dir=str(report_dir))
    config_path = model_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    write_json(model_dir / "sample_manifest.json", sample)
    write_json(model_dir / "environment.json", environment())
    print(f"Starting {identifier}; memory budget {policy['maximum_memory_bytes']/1024**3:.2f} GiB", flush=True)
    result = monitor([sys.executable, "-m", "src.training.train_crf", "--worker", str(config_path)],
                     report_dir / "training.log", report_dir / "resources.json", policy)
    summary = {"experiment_id": identifier, "size": size, "training": read_json(model_dir / "training_metrics.json"),
               "validation": read_json(model_dir / "validation_metrics.json"), "resources": result,
               "model_artifacts": checksums(model_dir)}
    write_json(report_dir / "complete.json", summary)
    print(f"Completed {identifier}: strict micro F1={summary['validation']['strict_entity_micro_f1']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=[1000, 10000, 50000, 100000], default=1000)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        train(args.size)
