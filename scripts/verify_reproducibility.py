"""Rebuild clean streams in memory one row at a time and compare output hashes.

Run after derive: .venv/bin/python -m scripts.verify_reproducibility
No derived or raw files are modified.
"""
import hashlib
from pathlib import Path
import yaml
from src.data.derive import clean_split
from src.data.loader import raw_records, normalised_text
from src.utils.io import canonical, file_hash, read_json, write_json


def main():
    config = yaml.safe_load(Path("configs/data/naamapadam_hi.yaml").read_text())
    earlier, results = {}, {}
    report = Path(config["derived_report_dir"])
    manifest = read_json(report / f"{config['derived_version']}_manifest.json")
    for split in ("train", "validation", "test"):
        hasher, texts = hashlib.sha256(), set()
        for row in clean_split(raw_records(config["raw_dir"], split), split, earlier, lambda r: None, lambda k, r: None):
            hasher.update((canonical(row) + "\n").encode("utf-8"))
            texts.add(normalised_text(row["tokens"]))
        expected = file_hash(Path(config["derived_dir"]) / f"{split}_clean.jsonl")
        if hasher.hexdigest() != expected:
            raise ValueError(f"Non-deterministic derived stream: {split}")
        results[split] = {"sha256": expected, "identical": True}
        earlier[split] = texts
        print(f"Reconstruction identical: {split}", flush=True)
    write_json(report / f"{config['derived_version']}_reproducibility.json", results)


if __name__ == "__main__":
    main()
