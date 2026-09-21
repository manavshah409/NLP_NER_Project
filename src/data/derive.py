"""Versioned derived construction with explicit order and source traceability."""
import argparse
import json
from collections import Counter
from pathlib import Path
import yaml
from src.data.audit import Statistics
from src.data.bio import normalise, validate
from src.data.loader import raw_records, records, normalised_text
from src.utils.io import canonical, digest, now, read_json, write_json, checksums, verify_checksums


def clean_split(rows, split, earlier_texts, on_repair, on_remove):
    """Normalise all first; test empties quarantined; leakage precedes dedup."""
    exact = {}
    for source in rows:
        tokens, labels, repairs = normalise(source["tokens"], source["labels"])
        row = dict(source, tokens=tokens, labels=labels)
        for repair in repairs:
            on_repair(dict(repair, source_split=split, source_record_index=row["source_index"]))
        if not tokens:
            if split != "test":
                raise ValueError("Unexpected empty training/validation record; policy review required")
            on_remove("quarantine", dict(row, reason="empty_record"))
            continue
        text = normalised_text(tokens)
        matches = [name for name, text_set in earlier_texts.items() if text in text_set]
        if matches:
            on_remove("leakage", {"source_split": split, "source_index": row["source_index"], "overlaps": matches})
            continue
        key = digest([tokens, labels])
        if key in exact:
            on_remove("duplicate", {"source_split": split, "removed_index": row["source_index"], "retained_index": exact[key]})
            continue
        exact[key] = row["source_index"]
        yield row


def verify_derived(config, manifest):
    verify_checksums(config["raw_dir"], manifest["raw_checksum"])
    verify_checksums(config["derived_dir"], manifest["derived_checksum"])
    text_sets, stats = {}, {}
    for split in ("train", "validation", "test"):
        texts, exact, indices = set(), set(), set()
        summary = Statistics()
        for row in records(Path(config["derived_dir"]) / f"{split}_clean.jsonl"):
            if row["source_split"] != split or row["source_index"] in indices or validate(row["tokens"], row["labels"]):
                raise ValueError("Derived structure/BIO/source failure")
            indices.add(row["source_index"])
            text = normalised_text(row["tokens"])
            if any(text in previous for previous in text_sets.values()):
                raise ValueError("Cross-split leakage remains")
            key = digest([row["tokens"], row["labels"]])
            if key in exact:
                raise ValueError("Exact duplicate remains")
            exact.add(key)
            texts.add(text)
            summary.add(row)
        text_sets[split] = texts
        stats[split + "_clean"] = summary.result()
    if stats != manifest["clean_statistics"]:
        raise ValueError("Derived statistics mismatch")
    # Exact source reconstruction verifies contents, indices and official-test preservation.
    for split in ("train", "validation", "test"):
        kept = iter(records(Path(config["derived_dir"]) / f"{split}_clean.jsonl"))
        current = next(kept, None)
        for raw in raw_records(config["raw_dir"], split):
            if current is not None and current["source_index"] == raw["source_index"]:
                tokens, labels, _ = normalise(raw["tokens"], raw["labels"])
                if current != dict(raw, tokens=tokens, labels=labels):
                    raise ValueError("Source content mismatch")
                current = next(kept, None)
        if current is not None:
            raise ValueError("Untraceable source index")
    from itertools import zip_longest
    for raw, official in zip_longest(raw_records(config["raw_dir"], "test"), records(Path(config["derived_dir"]) / "official_test.jsonl")):
        if raw != official:
            raise ValueError("Official test changed")
    return {"raw_checksum_preserved": True, "derived_checksum_verified": True, "zero_invalid_BIO": True,
            "zero_cross_split_overlap": True, "no_clean_empty_records": True, "source_content_verified": True,
            "official_test_preserved": True, "zero_exact_duplicates": True}


def derive(config):
    root = Path(config["derived_dir"])
    report = Path(config["derived_report_dir"])
    version = config["derived_version"]
    report.mkdir(parents=True, exist_ok=True)
    manifest_path = report / f"{version}_manifest.json"
    if root.exists():
        result = verify_derived(config, read_json(manifest_path))
        print(result, flush=True)
        return
    raw_checksum = read_json(Path(config["audit_dir"]) / "raw_checksum_manifest.json")
    verify_checksums(config["raw_dir"], raw_checksum)
    audit = read_json(Path(config["audit_dir"]) / "dataset_summary.json")
    if any(set(s["errors"]) - {"empty_record", "orphan_I", "type_mismatch_I"} for s in audit["splits"].values()):
        raise ValueError("Unresolved raw audit failures")
    staging = root.with_name(root.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    repairs, removed, text_sets, stats = [], {"duplicate": [], "leakage": [], "quarantine": []}, {}, {}
    timestamp = now()
    code_revision = digest({p.as_posix(): p.read_text() for p in sorted(Path("src").rglob("*.py"))})

    def repair(item):
        repairs.append(dict(item, dataset_id=config["dataset_id"], dataset_configuration="hi",
                            derived_dataset_version=version, timestamp=timestamp, code_revision=code_revision))

    for split in ("train", "validation", "test"):
        texts, summary = set(), Statistics()
        with (staging / f"{split}_clean.jsonl").open("w", encoding="utf-8") as out:
            for row in clean_split(raw_records(config["raw_dir"], split), split, text_sets, repair,
                                   lambda kind, item: removed[kind].append(item)):
                out.write(canonical(row) + "\n")
                texts.add(normalised_text(row["tokens"]))
                summary.add(row)
        text_sets[split] = texts
        stats[split + "_clean"] = summary.result()
        print(f"Derived {split}: {summary.records} records", flush=True)
    for filename, rows in (("official_test", raw_records(config["raw_dir"], "test")), ("quarantined_records", removed["quarantine"])):
        with (staging / f"{filename}.jsonl").open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(canonical(row) + "\n")
    verify_checksums(config["raw_dir"], raw_checksum)
    staging.rename(root)
    manifest = {"version": version, "timestamp": timestamp, "code_revision": code_revision, "config": config,
                "raw_checksum": raw_checksum, "derived_checksum": checksums(root), "clean_statistics": stats,
                "repair_stage": "All raw records before any removal, independently in each split",
                "repair_counts": dict(Counter(r["error_category"] for r in repairs)),
                "removed_counts": {kind: len(items) for kind, items in removed.items()},
                "test_overlap_breakdown": dict(Counter("+".join(r["overlaps"]) for r in removed["leakage"] if r["source_split"] == "test")),
                "official_test_records": audit["splits"]["test"]["records"],
                "order": "BIO normalise; quarantine test empties; remove overlap with earlier clean splits; exact dedup retaining lowest source index"}
    write_json(manifest_path, manifest)
    for suffix, values in (("bio_repairs", repairs), ("removed_duplicates", removed["duplicate"]), ("removed_leakage", removed["leakage"])):
        write_json(report / f"{version}_{suffix}.json", values)
    assertions = verify_derived(config, manifest)
    # Account for every original index exactly once, including removed records.
    for split in ("train", "validation", "test"):
        indices = [r["source_index"] for r in records(root / f"{split}_clean.jsonl")]
        indices += [r["removed_index"] for r in removed["duplicate"] if r["source_split"] == split]
        indices += [r["source_index"] for kind in ("leakage", "quarantine") for r in removed[kind] if r["source_split"] == split]
        if sorted(indices) != list(range(audit["splits"][split]["records"])):
            raise ValueError("Source index accounting failed")
    assertions["complete_source_index_accounting"] = True
    write_json(report / f"{version}_integrity.json", assertions)
    lines = ["# Derived Hindi dataset checkpoint", "", "| Split | Records | Tokens |", "|---|---:|---:|"]
    lines += [f"| {s} | {v['records']} | {v['tokens']} |" for s, v in stats.items()]
    lines += ["", "Order: " + manifest["order"], "", "Repairs: " + str(manifest["repair_counts"]),
              "", "Repairs are counted before deduplication, leakage removal and quarantine.", "",
              "Removals: " + str(manifest["removed_counts"]), "", "Test overlap with clean predecessors: " + str(manifest["test_overlap_breakdown"]),
              "", "A 'both' overlap with clean train and clean validation is impossible after validation decontamination.", "",
              "Checksum method: " + raw_checksum["method"], "", "Integrity: " + str(assertions)]
    (report / f"{version}_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/naamapadam_hi.yaml")
    derive(yaml.safe_load(Path(p.parse_args().config).read_text()))
