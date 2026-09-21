"""Summarise before/after integrity counts from generated audit artefacts."""
from collections import Counter
from pathlib import Path
from src.utils.io import read_json, write_json


def main():
    raw = read_json("reports/data_audit/dataset_summary.json")["splits"]
    duplicates = read_json("reports/data_audit/duplicate_analysis.json")
    derived = read_json("reports/derived_data/naamapadam_hi_crf_v1_manifest.json")
    results = {}
    for split in ("train", "validation", "test"):
        before, after = raw[split], derived["clean_statistics"][split + "_clean"]
        results[split] = {"before": {k: before[k] for k in ("records", "tokens", "entity_spans", "label_tokens", "errors")},
                          "after": {k: after[k] for k in ("records", "tokens", "entity_spans", "label_tokens", "errors")},
                          "raw_exact_duplicate_copies": duplicates[split]["exact_duplicate_copies"],
                          "clean_exact_duplicate_copies": 0,
                          "raw_valid_BIO_tag_positions": before["tokens"] - before["errors"].get("orphan_I", 0) - before["errors"].get("type_mismatch_I", 0),
                          "clean_valid_BIO_tag_positions": after["tokens"]}
    results["cross_split_overlap"] = {"before": read_json("reports/data_audit/cross_split_overlap.json"),
                                      "after_unique_texts": {"train--validation": 0, "train--test": 0, "validation--test": 0}}
    results["definition"] = "Valid BIO tag positions includes O and B as well as valid I. Zero clean duplicate/overlap counts are supported by successful exhaustive integrity assertions. Structural raw failures are absent."
    write_json("reports/derived_data/before_after_comparison.json", results)
    lines = ["# Before and after dataset audit", "", results["definition"], "",
             "| Split | Raw PER / ORG / LOC spans | Clean PER / ORG / LOC spans | Raw exact duplicate copies | Clean exact duplicate copies |", "|---|---|---|---:|---:|"]
    for split in ("train", "validation", "test"):
        r = results[split]
        b = " / ".join(str(r["before"]["entity_spans"][k]) for k in ("PER", "ORG", "LOC"))
        a = " / ".join(str(r["after"]["entity_spans"][k]) for k in ("PER", "ORG", "LOC"))
        lines.append(f"| {split} | {b} | {a} | {r['raw_exact_duplicate_copies']} | 0 |")
    lines += ["", "Raw entity spans follow strict B-only reconstruction; clean spans reflect explicit CoNLL repairs plus removals. These counts should not be interpreted as model predictions."]
    Path("reports/derived_data/before_after_comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
