"""Nested seed-hash samples from clean training only, with entity coverage anchors."""
import heapq
from pathlib import Path
from src.data.audit import Statistics
from src.data.bio import TYPES, strict_spans
from src.data.loader import records
from src.utils.io import digest, read_json, write_json


def rank(seed, source_index):
    return digest([seed, source_index])


def choose_indices(rows, size, seed):
    if not isinstance(size, int) or size < len(TYPES):
        raise ValueError("Sample must contain at least three records")
    candidates, anchors, total = [], {}, 0
    for row in rows:
        if row["source_split"] != "train":
            raise ValueError("Only train_clean may be sampled")
        index = row["source_index"]
        key = int(rank(seed, index), 16)
        total += 1
        item = (-key, -index)
        if len(candidates) < size:
            heapq.heappush(candidates, item)
        elif item > candidates[0]:
            heapq.heapreplace(candidates, item)
        for kind in {s[2] for s in strict_spans(row["labels"])}:
            if kind not in anchors or (key, index) < anchors[kind]:
                anchors[kind] = (key, index)
    if size < len(TYPES) or size > total:
        raise ValueError("Sample size outside allowed range")
    if set(anchors) != set(TYPES):
        raise ValueError("Training data lack one or more entity types")
    ordered = list(dict.fromkeys(anchors[k][1] for k in TYPES))
    for _, index in sorted((-key, -i) for key, i in candidates):
        if index not in ordered[:3]:
            ordered.append(index)
        if len(ordered) >= size:
            break
    return sorted(ordered[:size])


def create_manifest(path, train_path, size, seed, derived_checksum, full_statistics):
    path = Path(path)
    if path.exists():
        manifest = read_json(path)
        if any(manifest[k] != v for k, v in {"size": size, "seed": seed, "derived_checksum": derived_checksum}.items()):
            raise ValueError("Existing sample manifest differs")
        if manifest["indices_sha256"] != digest(manifest["source_indices"]):
            raise ValueError("Sample index checksum mismatch")
        return manifest
    indices = choose_indices(records(train_path), size, seed)
    chosen, summary = set(indices), Statistics()
    for row in records(train_path):
        if row["source_index"] in chosen:
            summary.add(row)
    if summary.records != size:
        raise ValueError("Sample membership mismatch")
    manifest = {"source_split": "train_clean", "size": size, "seed": seed, "source_indices": indices,
                "indices_sha256": digest(indices), "derived_checksum": derived_checksum,
                "strategy": "Lowest SHA256(seed, source index), reserving lowest-hash record per entity type; nested across sizes",
                "full_training_statistics": full_statistics, "sample_statistics": summary.result()}
    write_json(path, manifest)
    return manifest
