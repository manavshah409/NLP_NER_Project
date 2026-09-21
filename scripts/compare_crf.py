"""Compare completed, identically configured controlled runs only."""
import csv
from pathlib import Path
import yaml
from src.utils.io import read_json, write_json


def main():
    root = Path("reports/crf")
    completed = [read_json(p) for p in root.glob("*/complete.json")]
    runs = sorted(completed, key=lambda r: r["size"])
    if [r["size"] for r in runs] != [1000, 10000, 50000, 100000]:
        raise ValueError("Exactly one completed experiment per approved size is required")
    common = None
    rows = []
    previous_indices = set()
    for run in runs:
        model_dir = Path("models/crf") / run["experiment_id"]
        config = yaml.safe_load((model_dir / "resolved_config.yaml").read_text())
        parameters = {k: config[k] for k in ("features", "model")}
        if common is not None and parameters != common:
            raise ValueError("Feature/model settings differ")
        common = parameters
        indices = set(read_json(model_dir / "sample_manifest.json")["source_indices"])
        if not previous_indices <= indices:
            raise ValueError("Samples are not nested")
        previous_indices = indices
        training, validation, resources = run["training"], run["validation"], run["resources"]
        stats = training["training_statistics"]
        row = {"experiment_id": run["experiment_id"], "training_records": run["size"], "training_tokens": stats["tokens"],
               **{f"{k}_training_spans": v for k, v in stats["entity_spans"].items()},
               "feature_extraction_seconds": training["feature_extraction_seconds"], "training_seconds": training["training_seconds"],
               "peak_rss_bytes_sampled": resources["peak_rss_bytes_sampled"], "model_size_bytes": training["model_size_bytes"],
               **{k: validation[k] for k in ("strict_entity_precision", "strict_entity_recall", "strict_entity_micro_f1", "strict_entity_macro_f1", "token_accuracy", "median_sentence_latency_ms", "p95_sentence_latency_ms")},
               **{k + "_f1": validation["per_class"][k]["f1"] for k in ("PER", "ORG", "LOC")}}
        rows.append(row)
    with (root / "crf_scaling_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Controlled CRF scaling comparison", "", "All metrics below are observed validation results. Samples are nested; features and model parameters are identical.", "",
             "| Records | Train tokens | Train seconds | Peak RSS GiB | Strict micro F1 | Strict macro F1 |", "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['training_records']} | {r['training_tokens']} | {r['training_seconds']:.2f} | {r['peak_rss_bytes_sampled']/1024**3:.3f} | {r['strict_entity_micro_f1']:.6f} | {r['strict_entity_macro_f1']:.6f} |" for r in rows]
    lines += ["", "| Records | PER F1 | ORG F1 | LOC F1 | Precision | Recall |", "|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['training_records']} | {r['PER_f1']:.6f} | {r['ORG_f1']:.6f} | {r['LOC_f1']:.6f} | {r['strict_entity_precision']:.6f} | {r['strict_entity_recall']:.6f} |" for r in rows]
    gain = rows[-1]["strict_entity_micro_f1"] - rows[-2]["strict_entity_micro_f1"]
    memory_fraction = rows[-1]["peak_rss_bytes_sampled"] / runs[-1]["resources"]["maximum_memory_bytes"]
    recommendation = "250,000 records, subject to a fresh memory assessment" if gain > 0.002 and memory_fraction < 0.4 else "Stop at the current scale"
    lines += ["", f"Recommendation: **{recommendation}**. The observed 50k→100k micro F1 change is {gain:.6f}; 100k peak RSS was {memory_fraction:.1%} of its configured budget.", "",
              "Decision rule: recommend considering 250k only if the latest micro F1 gain exceeds 0.002 and observed peak RSS stays below 40% of the current budget; otherwise stop scaling for now. This is a conservative project rule, not a statistical significance claim. A 250k run would require approval and a fresh resource assessment; linear memory or runtime scaling is not assumed. No larger run has been started.", "",
              "See the CSV for all class distributions, class F1 scores, precision/recall, timing and size fields. Peak RSS is sampled every 0.25 seconds over the worker, including evaluation. Latency includes feature extraction and tagging; no warmup is excluded. Token accuracy is secondary."]
    (root / "crf_scaling_comparison.md").write_text("\n".join(lines) + "\n")
    write_json(root / "recommendation.json", {"recommendation": recommendation, "micro_f1_gain_50k_to_100k": gain, "peak_rss_fraction_of_budget_100k": memory_fraction, "larger_run_started": False})


if __name__ == "__main__":
    main()
