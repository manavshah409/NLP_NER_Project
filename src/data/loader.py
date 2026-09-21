"""Stream original JSONL while keeping official zero-based source indices."""
import json
import unicodedata
from pathlib import Path
from src.utils.io import read_json

SPLITS = {"train": "hi_train.json", "validation": "hi_val.json", "test": "hi_test.json"}


def raw_records(root, split):
    labels = read_json(Path(root) / "provenance.json")["labels"]
    with (Path(root) / SPLITS[split]).open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            row = json.loads(line)
            tags = row.get("ner")
            if isinstance(tags, list):
                tags = [labels[tag] if isinstance(tag, int) and not isinstance(tag, bool) and 0 <= tag < len(labels) else tag for tag in tags]
            yield {"tokens": row.get("words"), "labels": tags, "source_split": split, "source_index": i}


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def normalised_text(tokens):
    return " ".join(" ".join(unicodedata.normalize("NFC", token) for token in tokens).split())
