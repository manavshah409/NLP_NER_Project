"""Download the pinned official Hindi archive without executing remote dataset code.

The official builder is parsed as Python syntax solely to discover ClassLabel names.
Its original source and dataset card are retained for provenance, not executed.
"""
import argparse
import ast
import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path
import yaml
import certifi
from src.utils.io import checksums, now, write_json, read_json, verify_checksums


def fetch(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "IndicNewsNER/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=120, context=context) as response, open(path, "wb") as output:
        shutil.copyfileobj(response, output)


def discover_labels(source):
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ClassLabel":
            for kw in node.keywords:
                if kw.arg == "names":
                    candidates.append(ast.literal_eval(kw.value))
    if len(candidates) != 1 or set(candidates[0]) != {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}:
        raise ValueError("Unexpected official label schema")
    return candidates[0]


def acquire(config):
    root = Path(config["raw_dir"])
    manifest_path = Path(config["audit_dir"]) / "raw_checksum_manifest.json"
    if root.exists():
        verify_checksums(root, read_json(manifest_path))
        print("Existing immutable raw source verified", flush=True)
        return
    staging = root.with_name(root.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=True)
    api_path = staging / "hub_metadata.json"
    fetch(f'https://huggingface.co/api/datasets/{config["dataset_id"]}/revision/{config["revision"]}', api_path)
    revision = read_json(api_path)["sha"]
    base = f'https://huggingface.co/datasets/{config["dataset_id"]}/resolve/{revision}'
    for filename in ("naamapadam.py", "README.md"):
        fetch(f"{base}/{filename}", staging / filename)
    labels = discover_labels((staging / "naamapadam.py").read_text())
    archive = staging / "hi_IndicNER_v1.0.zip"
    url = f"{base}/data/{archive.name}"
    print(f"Downloading {url}", flush=True)
    fetch(url, archive)
    expected = {"hi_train.json", "hi_val.json", "hi_test.json"}
    with zipfile.ZipFile(archive) as z:
        members = {Path(n).name: n for n in z.namelist() if Path(n).name in expected}
        if set(members) != expected:
            raise ValueError(f"Archive split files differ: {z.namelist()}")
        for name, member in members.items():
            with z.open(member) as inp, (staging / name).open("wb") as out:
                shutil.copyfileobj(inp, out)
    write_json(staging / "provenance.json", {
        "dataset_id": config["dataset_id"], "configuration": "hi", "revision": revision,
        "retrieved_at": now(), "archive_url": url, "labels": labels,
        "schema": {"source_tokens_field": "words", "source_labels_field": "ner", "record_format": "JSON Lines"},
        "loader": "Local parser of official JSONL; official builder retained but not executed",
        "license_note": "Hub card and builder may disagree; see retained originals before redistribution"})
    staging.rename(root)
    write_json(manifest_path, checksums(root))
    print(f"Saved and checksummed {root}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/naamapadam_hi.yaml")
    args = p.parse_args()
    acquire(yaml.safe_load(Path(args.config).read_text()))
