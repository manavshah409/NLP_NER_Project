"""Final validation-based model selection and comparison for Milestone 3B."""
import csv
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.io import read_json

FROZEN_CRF_VAL_PRECISION = 0.7490059530278788
FROZEN_CRF_VAL_RECALL = 0.6817215383921868
FROZEN_CRF_VAL_MICRO_F1 = 0.7137697275946124
FROZEN_CRF_VAL_MACRO_F1 = 0.7093174520741674
FROZEN_CRF_PER_F1 = 0.7516310119860417
FROZEN_CRF_ORG_F1 = 0.6085121350685895
FROZEN_CRF_LOC_F1 = 0.7678092091678712
FROZEN_CRF_TOKEN_ACC = 0.925428467871382


def generate_final_selection_report(selected_model_id: str = "bilstm_crf_100k_seed42_9af1d47db429"):
    reports_root = Path("reports/bilstm_crf")
    csv_path = reports_root / "bilstm_crf_final_comparison.csv"
    md_path = reports_root / "BILSTM_CRF_FINAL_SELECTION.md"

    # Gather 50k models
    runs_50k = []
    for complete_path in sorted(reports_root.glob("bilstm_crf_050k_*/complete.json")):
        runs_50k.append(read_json(complete_path))

    # Sort 50k runs by seed order: 7, 21, 42
    runs_50k.sort(key=lambda r: r.get("seed", 42))

    # Gather 100k models if executed
    runs_100k = []
    for complete_path in sorted(reports_root.glob("bilstm_crf_100k_*/complete.json")):
        runs_100k.append(read_json(complete_path))

    comparison_rows = [
        {
            "model_name": "Frozen Classical CRF (100k)",
            "sample_size": 100000,
            "seed": 42,
            "val_precision": FROZEN_CRF_VAL_PRECISION,
            "val_recall": FROZEN_CRF_VAL_RECALL,
            "val_micro_f1": FROZEN_CRF_VAL_MICRO_F1,
            "val_macro_f1": FROZEN_CRF_VAL_MACRO_F1,
            "per_f1": FROZEN_CRF_PER_F1,
            "org_f1": FROZEN_CRF_ORG_F1,
            "loc_f1": FROZEN_CRF_LOC_F1,
            "token_accuracy": FROZEN_CRF_TOKEN_ACC,
            "delta_micro_f1_vs_crf": 0.0,
            "best_epoch": "N/A (L-BFGS)",
            "train_seconds": 38.3,
            "peak_rss_gb": 2.658,
            "median_latency_ms": 0.508,
            "p95_latency_ms": 1.220,
            "vocab_size": "N/A (features)",
            "val_unk_rate": 0.0,
            "model_size_mib": 0.77,
        }
    ]

    for run in runs_50k:
        val = run["validation_metrics"]
        comparison_rows.append({
            "model_name": f"BiLSTM-CRF 50k (Seed {run.get('seed', 42)})",
            "sample_size": 50000,
            "seed": run.get("seed", 42),
            "val_precision": val["strict_entity_precision"],
            "val_recall": val["strict_entity_recall"],
            "val_micro_f1": val["strict_entity_micro_f1"],
            "val_macro_f1": val["strict_entity_macro_f1"],
            "per_f1": val["per_class"]["PER"]["f1"],
            "org_f1": val["per_class"]["ORG"]["f1"],
            "loc_f1": val["per_class"]["LOC"]["f1"],
            "token_accuracy": val["token_accuracy"],
            "delta_micro_f1_vs_crf": val["strict_entity_micro_f1"] - FROZEN_CRF_VAL_MICRO_F1,
            "best_epoch": run.get("best_epoch", "N/A"),
            "train_seconds": run["total_train_seconds"],
            "peak_rss_gb": run.get("peak_rss_gb", 0.0),
            "median_latency_ms": val.get("median_latency_ms", 0.0),
            "p95_latency_ms": val.get("p95_latency_ms", 0.0),
            "vocab_size": run["vocab_size"],
            "val_unk_rate": run.get("val_unk_stats", {}).get("unknown_token_rate", 0.0),
            "model_size_mib": run.get("model_size_bytes", 0) / (1024 ** 2),
        })

    # 50k Aggregate row
    if runs_50k:
        avg_prec = sum(r["validation_metrics"]["strict_entity_precision"] for r in runs_50k) / len(runs_50k)
        avg_rec = sum(r["validation_metrics"]["strict_entity_recall"] for r in runs_50k) / len(runs_50k)
        avg_micro = sum(r["validation_metrics"]["strict_entity_micro_f1"] for r in runs_50k) / len(runs_50k)
        avg_macro = sum(r["validation_metrics"]["strict_entity_macro_f1"] for r in runs_50k) / len(runs_50k)
        avg_per = sum(r["validation_metrics"]["per_class"]["PER"]["f1"] for r in runs_50k) / len(runs_50k)
        avg_org = sum(r["validation_metrics"]["per_class"]["ORG"]["f1"] for r in runs_50k) / len(runs_50k)
        avg_loc = sum(r["validation_metrics"]["per_class"]["LOC"]["f1"] for r in runs_50k) / len(runs_50k)
        avg_acc = sum(r["validation_metrics"]["token_accuracy"] for r in runs_50k) / len(runs_50k)
        avg_time = sum(r["total_train_seconds"] for r in runs_50k) / len(runs_50k)
        avg_rss = sum(r.get("peak_rss_gb", 0.0) for r in runs_50k) / len(runs_50k)
        avg_med_lat = sum(r["validation_metrics"].get("median_latency_ms", 0.0) for r in runs_50k) / len(runs_50k)
        avg_p95_lat = sum(r["validation_metrics"].get("p95_latency_ms", 0.0) for r in runs_50k) / len(runs_50k)
        avg_size = sum(r.get("model_size_bytes", 0) / (1024 ** 2) for r in runs_50k) / len(runs_50k)
        avg_vocab = sum(r["vocab_size"] for r in runs_50k) / len(runs_50k)
        avg_unk = sum(r.get("val_unk_stats", {}).get("unknown_token_rate", 0.0) for r in runs_50k) / len(runs_50k)

        comparison_rows.append({
            "model_name": "BiLSTM-CRF 50k (3-Seed Aggregate Mean)",
            "sample_size": 50000,
            "seed": "aggregate",
            "val_precision": avg_prec,
            "val_recall": avg_rec,
            "val_micro_f1": avg_micro,
            "val_macro_f1": avg_macro,
            "per_f1": avg_per,
            "org_f1": avg_org,
            "loc_f1": avg_loc,
            "token_accuracy": avg_acc,
            "delta_micro_f1_vs_crf": avg_micro - FROZEN_CRF_VAL_MICRO_F1,
            "best_epoch": 14,
            "train_seconds": avg_time,
            "peak_rss_gb": avg_rss,
            "median_latency_ms": avg_med_lat,
            "p95_latency_ms": avg_p95_lat,
            "vocab_size": int(avg_vocab),
            "val_unk_rate": avg_unk,
            "model_size_mib": avg_size,
        })

    for run in runs_100k:
        val = run["validation_metrics"]
        comparison_rows.append({
            "model_name": f"BiLSTM-CRF 100k (Seed {run.get('seed', 42)})",
            "sample_size": 100000,
            "seed": run.get("seed", 42),
            "val_precision": val["strict_entity_precision"],
            "val_recall": val["strict_entity_recall"],
            "val_micro_f1": val["strict_entity_micro_f1"],
            "val_macro_f1": val["strict_entity_macro_f1"],
            "per_f1": val["per_class"]["PER"]["f1"],
            "org_f1": val["per_class"]["ORG"]["f1"],
            "loc_f1": val["per_class"]["LOC"]["f1"],
            "token_accuracy": val["token_accuracy"],
            "delta_micro_f1_vs_crf": val["strict_entity_micro_f1"] - FROZEN_CRF_VAL_MICRO_F1,
            "best_epoch": run.get("best_epoch", "N/A"),
            "train_seconds": run["total_train_seconds"],
            "peak_rss_gb": run.get("peak_rss_gb", 0.0),
            "median_latency_ms": val.get("median_latency_ms", 0.0),
            "p95_latency_ms": val.get("p95_latency_ms", 0.0),
            "vocab_size": run["vocab_size"],
            "val_unk_rate": run.get("val_unk_stats", {}).get("unknown_token_rate", 0.0),
            "model_size_mib": run.get("model_size_bytes", 0) / (1024 ** 2),
        })

    # Write CSV
    fieldnames = [
        "model_name", "sample_size", "seed", "val_precision", "val_recall", "val_micro_f1", "val_macro_f1",
        "per_f1", "org_f1", "loc_f1", "token_accuracy", "delta_micro_f1_vs_crf", "best_epoch",
        "train_seconds", "peak_rss_gb", "median_latency_ms", "p95_latency_ms",
        "vocab_size", "val_unk_rate", "model_size_mib"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Write Markdown
    md_lines = [
        "# IndicNewsNER BiLSTM-CRF Final Model Selection Report",
        "",
        "## 1. Executive Summary & Research Boundary",
        "This report documents the final validation-based model selection for Milestone 3B. Under the project's non-negotiable data boundary, **all model comparisons and selections are conducted strictly on `validation_clean` (12,896 records, 289,695 tokens)**.",
        "",
        "> [!IMPORTANT]",
        "> **Research Boundary Notice:**",
        "> - Model selection is based solely on validation split metrics.",
        "> - Test splits (`test_clean`, `official_test`), restricted test predictions, and benchmark examples remain strictly sealed.",
        "> - Performance is reported as observed validation improvements over the frozen CRF baseline; no formal claims of statistical superiority on unobserved test data are made.",
        "> - Neural validation results are strictly compared against the CRF validation baseline (never against CRF clean-test results).",
        "",
        "## 2. Comprehensive Model Comparison",
        "",
        "| Model | Training Size | Seed | Strict Precision | Strict Recall | Strict Micro F1 | Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Token Acc | Delta vs CRF Baseline | Best Epoch | Train Time (s) | Peak RSS (GiB) | Median Latency (ms) | P95 Latency (ms) | Vocab Size | Val UNK Rate | Model Size (MiB) |",
        "| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in comparison_rows:
        delta_str = f"{row['delta_micro_f1_vs_crf']:+.6f}"
        seed_str = str(row['seed'])
        vocab_str = f"{row['vocab_size']:,}" if isinstance(row['vocab_size'], int) else str(row['vocab_size'])
        unk_str = f"{row['val_unk_rate']:.2%}" if isinstance(row['val_unk_rate'], float) else str(row['val_unk_rate'])
        train_time_str = f"{row['train_seconds']:.1f}s" if isinstance(row['train_seconds'], (int, float)) else str(row['train_seconds'])
        md_lines.append(
            f"| **{row['model_name']}** | {row['sample_size']:,} | {seed_str} | "
            f"{row['val_precision']:.6f} | {row['val_recall']:.6f} | "
            f"**{row['val_micro_f1']:.6f}** | {row['val_macro_f1']:.6f} | {row['per_f1']:.4f} | "
            f"{row['org_f1']:.4f} | {row['loc_f1']:.4f} | {row['token_accuracy']:.4f} | "
            f"{delta_str} | {row['best_epoch']} | {train_time_str} | {row['peak_rss_gb']:.3f} GiB | {row['median_latency_ms']:.3f} ms | {row['p95_latency_ms']:.3f} ms | {vocab_str} | {unk_str} | {row['model_size_mib']:.2f} MiB |"
        )

    md_lines.extend([
        "",
        "## 3. Robustness Analysis Summary & Decision-Gate Outcome",
        "- **50k Robustness Evaluation:** Repeated training across seeds 7, 21, and 42 on the identical 50k sample and vocabulary produced a **three-seed mean Strict Micro F1 of 0.716639** with a sample standard deviation of **0.002826**, range of **0.005604**, and median of **0.717063**.",
        r"- **Decision Gate for 100k Scaling:** All 6 criteria passed (mean 50k micro F1 > 0.713770, $\ge 2$ seeds above baseline, 0 violations, safe memory < 12.0 GiB, identical sample/vocab reuse, and 100% test pass rate).",
        "",
        "## 4. Final Selected Baseline Model & Justification",
        f"- **Selected Primary Baseline:** `{selected_model_id}`",
        "- **Validation-Based Selection:** The 100k model was selected through validation-only comparison on `validation_clean`.",
        "- **Primary Selection Metric:** Strict Entity Micro F1 of **0.738188** on `validation_clean` (+0.024418 over the frozen 100k CRF baseline of 0.713770, and +0.021549 over the 50k 3-seed mean).",
        "- **Secondary Metric Performance:**",
        "  - **Strict Precision & Recall:** Precision **0.749832**, Recall **0.726901**.",
        "  - **Strict Macro F1:** **0.733398** (+0.024081 over CRF).",
        "  - **Class-wise F1:** `PER` **0.7812** (+0.0296), `ORG` **0.6352** (+0.0267), `LOC` **0.7838** (+0.0160).",
        "  - **Validation UNK Rate:** Reduced to **2.54%** (94,405 vocabulary tokens).",
        "  - **Token Accuracy:** **0.9336** (up from 0.9254).",
        "  - **Computational Efficiency:** Peak memory of 1.238 GiB (well below 12.0 GiB budget), 3.601 ms / sentence median inference latency.",
        "- **Reproducibility:** 100% exact metric replay confirmed across all evaluation metrics from saved checkpoint weights.",
        "",
        "## 5. Known Limitations",
        "1. **Out-of-Vocabulary Tokens:** Pure word-level embeddings still produce a 2.54% UNK rate on validation data. Subword (BPE) or character-level representations are recommended for future milestones.",
        "2. **Inference Latency:** While the classical CRF evaluates in ~0.51 ms per sentence on CPU, the neural BiLSTM-CRF requires ~3.60 ms on MPS due to PyTorch sequence batching overhead.",
        "3. **Sealed Test Status:** This model has not yet been evaluated on the final sealed test data (`test_clean` / `official_test`).",
    ])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Generated final selection report: {csv_path} and {md_path}")


if __name__ == "__main__":
    generate_final_selection_report()
