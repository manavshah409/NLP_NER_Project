"""Generate BiLSTM-CRF scaling comparison table and comparison against frozen CRF baseline."""
import csv
from pathlib import Path
from src.utils.io import read_json

FROZEN_CRF_VAL_MICRO_F1 = 0.7137697275946124
FROZEN_CRF_VAL_MACRO_F1 = 0.7093171349890666


def collect_experiments() -> list:
    reports_root = Path("reports/bilstm_crf")
    if not reports_root.exists():
        return []
    
    experiments = []
    for comp_path in sorted(reports_root.glob("*/complete.json")):
        data = read_json(comp_path)
        val = data["validation_metrics"]
        unk_stats = data.get("val_unk_stats", {})
        
        row = {
            "sample_size": data["sample_size"],
            "experiment_id": data["experiment_id"],
            "vocab_size": data["vocab_size"],
            "val_unk_rate": unk_stats.get("unknown_token_rate", 0.0),
            "best_epoch": data["best_epoch"],
            "train_seconds": data["total_train_seconds"],
            "peak_rss_gb": data.get("peak_rss_gb", 0.0),
            "model_size_mib": data.get("model_size_bytes", 0) / (1024 ** 2),
            "val_precision": val["strict_entity_precision"],
            "val_recall": val["strict_entity_recall"],
            "val_micro_f1": val["strict_entity_micro_f1"],
            "val_macro_f1": val["strict_entity_macro_f1"],
            "per_f1": val["per_class"]["PER"]["f1"],
            "org_f1": val["per_class"]["ORG"]["f1"],
            "loc_f1": val["per_class"]["LOC"]["f1"],
            "token_accuracy": val["token_accuracy"],
            "delta_vs_crf_val_micro_f1": val["strict_entity_micro_f1"] - FROZEN_CRF_VAL_MICRO_F1,
        }
        experiments.append(row)
    
    experiments.sort(key=lambda x: x["sample_size"])
    return experiments


def generate_comparison_reports():
    experiments = collect_experiments()
    reports_root = Path("reports/bilstm_crf")
    reports_root.mkdir(parents=True, exist_ok=True)

    csv_path = reports_root / "bilstm_crf_scaling_comparison.csv"
    md_path = reports_root / "bilstm_crf_scaling_comparison.md"

    if not experiments:
        print("No completed BiLSTM-CRF experiments found.")
        return

    # Write CSV
    fieldnames = [
        "sample_size", "experiment_id", "vocab_size", "val_unk_rate", "best_epoch",
        "train_seconds", "peak_rss_gb", "model_size_mib", "val_precision", "val_recall",
        "val_micro_f1", "val_macro_f1", "per_f1", "org_f1", "loc_f1", "token_accuracy",
        "delta_vs_crf_val_micro_f1"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiments)

    # Write Markdown
    md_lines = [
        "# IndicNewsNER BiLSTM-CRF Scaling Comparison",
        "",
        "Evaluation strictly on `validation_clean` (12,896 records, 289,695 tokens).",
        f"Frozen Classical CRF 100k Baseline Validation Micro F1: **{FROZEN_CRF_VAL_MICRO_F1:.6f}** (Macro F1: **{FROZEN_CRF_VAL_MACRO_F1:.6f}**).",
        "",
        "| Sample Size | Vocab Size | Val UNK Rate | Best Epoch | Train Time (s) | Peak RSS (GiB) | Val Strict Micro F1 | Val Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Delta vs CRF Val F1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for exp in experiments:
        delta_str = f"{exp['delta_vs_crf_val_micro_f1']:+.6f}"
        md_lines.append(
            f"| {exp['sample_size']:,} | {exp['vocab_size']:,} | {exp['val_unk_rate']:.2%} | {exp['best_epoch']} | "
            f"{exp['train_seconds']:.2f} | {exp['peak_rss_gb']:.3f} | **{exp['val_micro_f1']:.6f}** | "
            f"{exp['val_macro_f1']:.6f} | {exp['per_f1']:.4f} | {exp['org_f1']:.4f} | {exp['loc_f1']:.4f} | {delta_str} |"
        )

    md_lines.extend([
        "",
        "## Baseline Comparison & Analysis",
        f"- **Frozen CRF 100k Baseline Validation Micro F1:** {FROZEN_CRF_VAL_MICRO_F1:.6f}",
        "- **Research Boundary:** Evaluated strictly on `validation_clean`. No test data accessed.",
    ])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Generated comparison reports at {csv_path} and {md_path}")


if __name__ == "__main__":
    generate_comparison_reports()
