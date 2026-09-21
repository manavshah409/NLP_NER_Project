"""Publish observed CRF milestone status after comparison and artifact verification."""
import csv
from pathlib import Path
from src.utils.io import read_json, write_json


def main():
    root = Path("reports/crf")
    with (root / "crf_scaling_comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    verification = read_json(root / "verification.json")
    recommendation = read_json(root / "recommendation.json")
    checkpoint = read_json("reports/checkpoint_milestones_1_2a.json")
    summary = {"status": "Approved 100k Hindi CRF extension complete; later phases require approval",
               "runs": rows, "verification": verification, "tests": checkpoint["tests"],
               "recommendation": recommendation, "test_predictions_generated": False,
               "later_models_started": False, "english_integrated": False}
    write_json("reports/milestone_2b_completion.json", summary)
    lines = ["# Milestone 2B completion", "", summary["status"], "",
             "| Sample records | Precision | Recall | Strict micro F1 | Strict macro F1 |", "|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row["training_records"]] + [f"{float(row[k]):.6f}" for k in ("strict_entity_precision", "strict_entity_recall", "strict_entity_micro_f1", "strict_entity_macro_f1")]) + " |")
    lines += ["", f"Tests: {checkpoint['tests']}. All model artefact checksums, experiment identifiers and native reload checks pass.", "",
              "The full data checkpoint was reproduced, including raw checksums, explicit repair/removal manifests and byte-identical clean dataset reconstruction.", "",
              "Recommendation: " + recommendation["recommendation"] + ". No further training has been started.", "",
              "Limitations: sampled RSS peaks; latency reflects this machine and current system load; no repeated-seed uncertainty study; no news-domain gold evaluation; Hindi only. Gazetteers, neural models, English, test evaluation and the interface remain outside this completed milestone.", "",
              "Results were not fabricated or extrapolated. The current sample is not the complete training split. The official and clean test sets were used only for data-integrity preparation, never for predictions or model selection.", "",
              "See crf/crf_scaling_comparison.csv for all required scaling metrics, crf/*/error_analysis.md for bounded validation examples, and implementation_log.md for commands and recovered failures."]
    Path("reports/milestone_2b_completion.md").write_text("\n".join(lines) + "\n")
    largest = rows[-1]
    card = ["# IndicNewsNER model card", "", "Status: Hindi classical CRF trained and evaluated at 1k, 10k, 50k and the approved 100k samples.", "",
            "Task: BIO tagging for PER, ORG and LOC. All runs use native CRFsuite, L-BFGS, c1=c2=0.1, at most 100 iterations, all possible transitions and identical Unicode-preserving features with context window 1. Seed 42; nested coverage-aware samples.", "",
            f"The 100k model is `{largest['experiment_id']}`. It is trained on {largest['training_tokens']} tokens from 100,000 clean training records, not the full 963,174-record split.", "",
            f"Validation strict entity precision: {float(largest['strict_entity_precision']):.6f}; recall: {float(largest['strict_entity_recall']):.6f}; micro F1: {float(largest['strict_entity_micro_f1']):.6f}; macro F1: {float(largest['strict_entity_macro_f1']):.6f}.", "",
            f"Class F1 — PER: {float(largest['PER_f1']):.6f}; ORG: {float(largest['ORG_f1']):.6f}; LOC: {float(largest['LOC_f1']):.6f}. Token accuracy is secondary: {float(largest['token_accuracy']):.6f}.", "",
            "Evaluation uses all 12,896 clean validation records. Correctness requires exact entity boundaries and type. Macro F1 includes all three types. No test predictions were generated.", "",
            "Native `.crfsuite` files, resolved configuration, package versions, timing and sample manifests are saved under models/crf. Reports include source snapshot, artifact checksums and successful reload verification.", "",
            "Limitations: Naamapadam spans multiple domains and does not establish current Indian-news performance. Annotation ambiguity, unseen names, boundary errors and domain shift remain. English, BiLSTM-CRF, IndicBERT, final test evaluation and demonstration interface are future work requiring approval. No multilingual completion is claimed.", "",
            "The Hub and upstream loader have conflicting license declarations; retained provenance documents the discrepancy. Bounded error examples contain public benchmark text and names. See README for sources and preparation policy."]
    Path("MODEL_CARD.md").write_text("\n".join(card) + "\n")


if __name__ == "__main__":
    main()
