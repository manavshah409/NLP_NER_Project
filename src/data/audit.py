"""Exact streaming audit; no model predictions or test-based feature selection."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import yaml
from src.data.bio import LABELS, TYPES, validate, strict_spans
from src.data.loader import raw_records, normalised_text
from src.utils.io import digest, read_json, write_json, verify_checksums


class Statistics:
    def __init__(self):
        self.records = 0
        self.tokens = 0
        self.labels = Counter({label: 0 for label in LABELS})
        self.entities = Counter({kind: 0 for kind in TYPES})
        self.lengths = Counter()
        self.token_lengths = Counter()
        self.errors = Counter()

    def add(self, row):
        self.records += 1
        tokens, labels = row["tokens"], row["labels"]
        errors = validate(tokens, labels)
        self.errors.update(e["category"] for e in errors)
        if isinstance(tokens, list):
            self.tokens += len(tokens)
            self.lengths[len(tokens)] += 1
            self.token_lengths.update(len(t) for t in tokens if isinstance(t, str))
        if isinstance(labels, list):
            self.labels.update(str(tag) for tag in labels)
            if all(tag in LABELS for tag in labels):
                self.entities.update(span[2] for span in strict_spans(labels))
        return errors

    def result(self):
        return {"records": self.records, "tokens": self.tokens, "label_tokens": dict(self.labels),
                "entity_spans": dict(self.entities), "sequence_length_histogram": {str(k): v for k, v in sorted(self.lengths.items())},
                "token_character_length_histogram": {str(k): v for k, v in sorted(self.token_lengths.items())}, "errors": dict(self.errors)}


def audit(config):
    report_dir = Path(config["audit_dir"])
    verify_checksums(config["raw_dir"], read_json(report_dir / "raw_checksum_manifest.json"))
    summaries, duplicate_report, text_sets, bio_errors, overlaps = {}, {}, {}, [], {}
    for split in ("train", "validation", "test"):
        stats = Statistics()
        exact, texts = {}, {}
        duplicates = []
        for row in raw_records(config["raw_dir"], split):
            errors = stats.add(row)
            bio_errors.extend(dict(e, source_split=split, source_index=row["source_index"]) for e in errors)
            if any(e["category"] not in {"empty_record", "orphan_I", "type_mismatch_I"} for e in errors):
                continue
            key = digest([row["tokens"], row["labels"]])
            if key in exact:
                duplicates.append({"removed_index": row["source_index"], "retained_index": exact[key]})
            else:
                exact[key] = row["source_index"]
            text = normalised_text(row["tokens"])
            texts[text] = texts.get(text, 0) + 1
            if row["source_index"] % 200000 == 0:
                print(f"Audit {split}: {row['source_index']} records", flush=True)
        summaries[split] = stats.result()
        duplicate_report[split] = {"exact_duplicate_copies": len(duplicates), "exact_duplicate_records": duplicates,
                                   "normalised_text_duplicate_copies": sum(n - 1 for n in texts.values())}
        for other, other_text in text_sets.items():
            intersection = texts.keys() & other_text.keys()
            overlaps[f"{other}--{split}"] = {"unique_texts": len(intersection),
                                            "earlier_split_records": sum(other_text[t] for t in intersection),
                                            "later_split_records": sum(texts[t] for t in intersection)}
        text_sets[split] = texts
    summary = {"splits": summaries, "total_records": sum(s["records"] for s in summaries.values()),
               "total_tokens": sum(s["tokens"] for s in summaries.values()),
               "label_mapping": dict(enumerate(read_json(Path(config["raw_dir"]) / "provenance.json")["labels"])),
               "normalised_text_policy": config["normalised_text_policy"],
               "span_policy": "Strict BIO; only B starts entities", "status": "audited"}
    write_json(report_dir / "dataset_summary.json", summary)
    write_json(report_dir / "bio_errors.json", bio_errors)
    write_json(report_dir / "duplicate_analysis.json", duplicate_report)
    write_json(report_dir / "cross_split_overlap.json", overlaps)
    for filename, key in (("label_distribution.csv", "label_tokens"), ("entity_distribution.csv", "entity_spans")):
        with (report_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "label", "count"])
            for split, stats in summaries.items():
                writer.writerows((split, label, count) for label, count in stats[key].items())
    lines = ["# Official Hindi Naamapadam audit", "", "Counts are computed from the official archive. No model evaluation was performed.", "",
             "| Split | Records | Tokens | Error categories |", "|---|---:|---:|---|"]
    lines += [f"| {s} | {v['records']} | {v['tokens']} | {v['errors']} |" for s, v in summaries.items()]
    lines += ["", "Normalised text: " + config["normalised_text_policy"], "", "Entity counts use strict raw BIO. See JSON reports for exact distributions and overlaps."]
    (report_dir / "dataset_summary.md").write_text("\n".join(lines) + "\n")
    if any(e["category"] not in {"empty_record", "orphan_I", "type_mismatch_I"} for e in bio_errors):
        raise ValueError("Structural audit failures; inspect reports before constructing derived data")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data/naamapadam_hi.yaml")
    audit(yaml.safe_load(Path(p.parse_args().config).read_text()))
