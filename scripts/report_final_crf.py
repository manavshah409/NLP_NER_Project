"""Aggregate final results without loading test examples or rerunning inference."""
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from src.evaluation.freeze_baseline import verify_freeze, FINAL_DIR
from src.utils.io import read_json, write_json


def main():
    frozen = verify_freeze()
    model_dir = Path(frozen["model_dir"])
    validation = read_json(model_dir / "validation_metrics.json")
    clean = read_json(FINAL_DIR / "test_clean_metrics.json")
    official = read_json(FINAL_DIR / "official_test_metrics.json")
    results = {"validation_clean": validation, "test_clean": clean, "official_test": official}
    counts = {s: read_json(FINAL_DIR / f"{s}_error_counts.json")["counts"] for s in ("test_clean", "official_test")}
    suite = ET.parse("reports/environment/pytest_after_2c.xml").getroot().find("testsuite")
    tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
    tests["passed"] = tests["tests"] - sum(tests[k] for k in ("failures", "errors", "skipped"))
    csv_path = Path("reports/crf/crf_scaling_comparison.csv")
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    additions = [f"{s}_{k}" for s in ("test_clean", "official_test") for k in ("strict_entity_precision", "strict_entity_recall", "strict_entity_micro_f1", "strict_entity_macro_f1")]
    fields += [k for k in additions if k not in fields]
    for row in rows:
        for split in ("test_clean", "official_test"):
            for key in ("strict_entity_precision", "strict_entity_recall", "strict_entity_micro_f1", "strict_entity_macro_f1"):
                row[f"{split}_{key}"] = results[split][key] if int(row["training_records"]) == 100000 else "not evaluated"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Evaluation split | Records | Tokens | Precision | Recall | Strict micro F1 | Strict macro F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, r in results.items():
        table.append(f"| {split} | {r.get('records', r.get('validation_records'))} | {r['tokens']} | {r['strict_entity_precision']:.9f} | {r['strict_entity_recall']:.9f} | {r['strict_entity_micro_f1']:.9f} | {r['strict_entity_macro_f1']:.9f} |")
    scaling_path = Path("reports/crf/crf_scaling_comparison.md")
    original = scaling_path.read_text().split("\n## Frozen final evaluation")[0]
    scaling_path.write_text(original + "\n## Frozen final evaluation\n\nOnly the selected 100k model was evaluated on test data. These results were obtained after selection and are not scaling-selection criteria.\n\n" + "\n".join(table) + "\n\nMain conclusion: test_clean. Official test is a separate overlapping/raw-annotation benchmark comparison.\n")
    config = frozen["config"]
    lines = ["# Final classical CRF baseline report", "", f"Frozen experiment: `{frozen['experiment_id']}`.", "",
             "## Objective and data", "", "Extract complete Hindi PER, ORG and LOC spans using a reproducible classical baseline before neural-model comparison. Naamapadam is multi-domain; this is not evidence of current Indian-news domain performance.", "",
             f"Dataset: ai4bharat/naamapadam, hi; derived version `{frozen['dataset_version']}`. Clean training has 963,174 records; the model used a 100,000-record sample. All raw and derived file checksums were verified at freeze and around final inference.", "",
             "## Features and model configuration", "", "Original Unicode token, length, prefixes/suffixes 1–3, numeric/punctuation/alphabetic indicators, Latin/Devanagari/mixed-script indicators, Unicode categories and word shape. Latin-only lowercase derivatives; context window 1 with BOS/EOS. No gazetteers, transliteration or test-derived feature lists.", "",
             "```json", __import__('json').dumps(config["model"], indent=2), "```", "",
             "## Sampling and scaling", "", "Seed 42. Lowest seed/index SHA256 order with an entity-coverage anchor per type; source indices are retained. Samples are nested. Freeze independently reconstructed all selected indices and matched the saved training statistics. Only train_clean supplied training records.", "",
             "| Records | Validation micro F1 | Validation macro F1 | Training seconds | Peak RSS GiB |", "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['training_records']} | {float(row['strict_entity_micro_f1']):.9f} | {float(row['strict_entity_macro_f1']):.9f} | {float(row['training_seconds']):.6f} | {int(row['peak_rss_bytes_sampled'])/1024**3:.6f} |")
    lines += ["", "The 100k model was selected and approved from validation results before final test evaluation. Retention reflects its validation improvement and manageable observed memory; the test results did not influence selection. No larger model was trained.", "", "## Validation and final evaluation", ""] + table
    lines += ["", "test_clean is the main Naamapadam test conclusion. official_test is reported separately: it includes development overlap, 3 empty records and original invalid BIO transitions. Its gold labels are not repaired. The two test sets overlap each other and are not independent samples.", "",
              "Only B starts a span; correctness requires exact boundaries and type. Macro F1 averages PER/ORG/LOC, including zeros if unsupported. The official score is under this documented policy; it should not be called directly comparable to papers using different BIO handling.", "",
              "## Per-class results", "", "| Split | Type | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"]
    for split, r in results.items():
        for kind, v in r["per_class"].items():
            lines.append(f"| {split} | {kind} | {v['precision']:.9f} | {v['recall']:.9f} | {v['f1']:.9f} |")
    lines += ["", "## Inference resources", "", "| Split | Inference seconds | Median ms | P95 ms | Sampled RSS MiB | Secondary token accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for split, r in results.items():
        lines.append(f"| {split} | {r['inference_seconds']:.9f} | {r['median_sentence_latency_ms']:.9f} | {r['p95_sentence_latency_ms']:.9f} | {r['peak_inference_rss_bytes_sentence_sampled']/1024**2:.6f} | {r['token_accuracy']:.9f} |")
    lines += ["", f"Model binary: {clean['model_size_bytes']} bytes. " + clean["latency_definition"], "", clean["memory_definition"], "",
              "## Error analysis", "", "Counts below may overlap; they are not mutually exclusive totals. Confusion counts are overlapping span pairs; short-sequence counts are records; other slices are entity counts. Abbreviation/headline-like slices are explicit heuristics, not manually validated categories.", "",
              "| Category | Clean test count | Official test count |", "|---|---:|---:|"]
    lines += [f"| {key} | {counts['test_clean'][key]} | {counts['official_test'][key]} |" for key in counts["test_clean"]]
    lines += ["", "Illustrations are bounded to at most 10 per category and stored separately under excluded restricted_examples/. Test predictions and illustrations must not be opened during later model selection. This report contains aggregate counts only.", "",
              "## Reproducibility and verification", "", "The SHA256 freeze manifest covers the binary, configuration, sample, environment, label schema, data and source/evaluation code. The manifest payload has its own canonical JSON checksum. No Git commit existed at freeze; the later milestone commit records this exact manifest.", "",
              "Final inference ran once per split, clean first then official. Exclusive start markers reject repeat inference. Saved predictions include stable source IDs, gold/predicted labels and strict reconstructed spans; replay exactly reproduced aggregate metrics without calling the model again.", "",
              f"Post-evaluation tests: {tests}. Before the milestone 53 tests passed; before final evaluation the expanded 66-test suite passed. Synthetic tests never consume benchmark predictions.", "",
              "## Limitations and next milestone", "", "Single seed and no confidence interval; small clean test; benchmark annotation ambiguity; raw official BIO and overlap limit benchmark comparability; latency reflects this machine/load; RSS is sampled; no manually verified news-domain evaluation or English implementation. Upstream dataset license declarations conflict, as documented in README.", "",
              "Recommended next milestone: plan and implement BiLSTM-CRF after explicit approval, using only train_clean/validation_clean for development and keeping these final test examples sealed. Do not start it automatically. No transformer or English work is part of Milestone 2C."]
    Path("reports/crf/CRF_BASELINE_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(FINAL_DIR / "milestone_2c_summary.json", {"experiment": frozen["experiment_id"], "results": results, "error_counts": counts, "tests": tests, "freeze_verified": True, "inference_passes_per_split": 1})
    progress = Path("docs/progress")
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "milestone_2c.md").write_text("# Milestone 2C completed\n\n" + "\n".join(table) + f"\n\nTests: {tests}. Freeze checks and prediction replay pass. Each test split was evaluated once. No training or tuning occurred.\n\nSee reports/crf/CRF_BASELINE_FINAL_REPORT.md for methodology, class metrics, errors and limitations. Git delivery status is recorded separately.\n")


if __name__ == "__main__":
    main()
