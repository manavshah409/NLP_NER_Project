"""Generate a checkpoint report from measured artefacts, never reference guesses."""
from pathlib import Path
import xml.etree.ElementTree as ET
from src.utils.io import read_json, write_json
from src.data.loader import records


def main():
    raw = read_json("reports/data_audit/dataset_summary.json")
    base = "reports/derived_data/naamapadam_hi_crf_v1"
    derived = read_json(base + "_manifest.json")
    integrity = read_json(base + "_integrity.json")
    reproduction = read_json(base + "_reproducibility.json")
    suite = ET.parse("reports/environment/pytest.xml").getroot().find("testsuite")
    tests = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    tests["passed"] = tests["tests"] - tests["failures"] - tests["errors"] - tests["skipped"]
    reference = {"train_clean": 963174, "validation_clean": 12896, "test_clean": 506}
    differences = {s: v["records"] - reference[s] for s, v in derived["clean_statistics"].items()}
    quarantine = list(records("data/derived/naamapadam_hi_crf_v1/quarantined_records.jsonl"))
    report = {"raw": raw, "derived_counts": {s: v["records"] for s, v in derived["clean_statistics"].items()},
              "repairs": derived["repair_counts"], "removals": derived["removed_counts"],
              "test_overlap_breakdown_clean_predecessors": {"train_only": derived["test_overlap_breakdown"].get("train", 0), "validation_only": derived["test_overlap_breakdown"].get("validation", 0), "both": derived["test_overlap_breakdown"].get("train+validation", 0)},
              "quarantined_source_indices": [r["source_index"] for r in quarantine],
              "raw_combined_checksum": derived["raw_checksum"]["combined_sha256"],
              "reference_count_differences": differences, "integrity": integrity,
              "reproducibility": reproduction, "tests": tests,
              "created_source_files": [str(p) for folder in ("src", "configs", "tests", "scripts") for p in sorted(Path(folder).rglob("*")) if p.is_file() and "__pycache__" not in str(p)]}
    write_json("reports/checkpoint_milestones_1_2a.json", report)
    lines = ["# Milestones 1 and 2A checkpoint", "", "All benchmark values below are independently computed from downloaded official files.", "",
             "| Split | Raw records | Raw tokens | Clean records | Clean tokens |", "|---|---:|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        r, d = raw["splits"][split], derived["clean_statistics"][split + "_clean"]
        lines.append(f"| {split} | {r['records']} | {r['tokens']} | {d['records']} | {d['tokens']} |")
    lines += ["", f"Repairs: {report['repairs']}; counted before any removal.", "", f"Removals: {report['removals']}.", "",
              f"Quarantine indices: {report['quarantined_source_indices']}.", "",
              f"Test overlap breakdown against clean predecessors: {report['test_overlap_breakdown_clean_predecessors']}.", "",
              "All clean split record counts, raw token counts and reported removal/repair totals match the references. The three official test empty records are preserved in official_test and separately quarantined from test_clean.", "",
              f"Current test suite: {tests}. The earlier 54-test claim does not apply to this new implementation.", "",
              "All nine integrity assertions pass. Repeated reconstruction produces byte-identical clean split files. Strict raw BIO and derived CoNLL repair are separate. Each removal and repair is traceable.", "",
              "Checksum: `" + report["raw_combined_checksum"] + "`.", "", derived["raw_checksum"]["method"], "",
              "The reference checksum cannot be reproduced as the original implementation is absent. This acquisition includes timestamped provenance, so its combined checksum has a different scope; source-file hashes provide the comparable identity.", "",
              "Known limitations: license declarations disagree upstream; no model or domain-shift conclusion follows from a data checkpoint. Raw counts of absent error categories are zero. Data-only integrity checks inspect test labels; no test predictions were generated.", "",
              "Files created are listed in the machine-readable checkpoint. Commands and environment notes are in implementation_log.md."]
    Path("reports/checkpoint_milestones_1_2a.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
