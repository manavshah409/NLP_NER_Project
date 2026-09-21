"""Verify completed model files, provenance IDs and sample nesting after training."""
from pathlib import Path
import yaml
from src.data.crf_features import sentence_features
from src.data.loader import records
from src.models.crf_model import CRFModel
from src.utils.io import checksums, digest, read_json, write_json, verify_checksums


def main():
    source = read_json("reports/crf/source_snapshot.json")
    if digest(source["files"]) != source["source_sha256"]:
        raise ValueError("Source snapshot hash mismatch")
    outcomes = {}
    for complete_path in sorted(Path("reports/crf").glob("*/complete.json")):
        complete = read_json(complete_path)
        run_source_path = complete_path.parent / "source_snapshot.json"
        run_source = read_json(run_source_path) if run_source_path.exists() else source
        if digest(run_source["files"]) != run_source["source_sha256"]:
            raise ValueError("Per-run source snapshot mismatch")
        model_dir = Path("models/crf") / complete["experiment_id"]
        verify_checksums(model_dir, complete["model_artifacts"])
        config = yaml.safe_load((model_dir / "resolved_config.yaml").read_text())
        sample = read_json(model_dir / "sample_manifest.json")
        original_config = yaml.safe_load(Path("configs/crf_baseline.yaml").read_text())
        original_config["data"].update(sample_size=sample["size"], sample_manifest=config["data"]["sample_manifest"])
        identifier = f"crf_{sample['size']//1000:03d}k_seed42_" + digest({"config": original_config,
                        "indices": sample["indices_sha256"], "data": sample["derived_checksum"],
                        "code": run_source["files"], "requirements": Path("requirements.txt").read_text()})[:12]
        if identifier != complete["experiment_id"]:
            raise ValueError("Experiment ID provenance mismatch")
        # Training-only smoke input: no extra validation or test inspection.
        selected = set(sample["source_indices"])
        row = next(r for r in records("data/derived/naamapadam_hi_crf_v1/train_clean.jsonl") if r["source_index"] in selected)
        features = sentence_features(row["tokens"], config["features"])
        first = CRFModel(config["model"]).load(model_dir / "model.crfsuite").predict(features)
        second = CRFModel(config["model"]).load(model_dir / "model.crfsuite").predict(features)
        if first != second or len(first) != len(row["tokens"]):
            raise ValueError("Model reload mismatch")
        outcomes[identifier] = {"artifacts_checksums_valid": True, "experiment_id_reproduced": True, "native_model_reload_consistent": True}
    if len(outcomes) != 4:
        raise ValueError("Four completed experiments required")
    write_json("reports/crf/verification.json", outcomes)
    print(outcomes)


if __name__ == "__main__":
    main()
