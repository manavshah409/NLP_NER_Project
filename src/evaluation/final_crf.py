"""One-shot, frozen final inference; saved predictions support metric replay only."""
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import psutil
from src.data.bio import strict_spans
from src.data.crf_features import sentence_features
from src.data.loader import records
from src.evaluation.evaluate_crf import analyse, CATEGORIES
from src.evaluation.freeze_baseline import FINAL_DIR, MANIFEST, verify_freeze
from src.evaluation.metrics import StrictMetrics
from src.models.crf_model import CRFModel
from src.utils.io import canonical, file_hash, now, read_json, write_json

SPLITS = ("test_clean", "official_test")


def select_split(root, split):
    if split not in SPLITS:
        raise ValueError("Final evaluator accepts only test_clean or official_test")
    return Path(root) / f"{split}.jsonl"


def prediction_record(row, predicted, split, experiment):
    if row["source_split"] != "test" or len(row["tokens"]) != len(row["labels"]) or len(predicted) != len(row["labels"]):
        raise ValueError("Invalid final prediction source/shape")
    return {"dataset_id": "ai4bharat/naamapadam", "configuration": "hi", "evaluation_split": split,
            "source_split": "test", "source_index": row["source_index"],
            "source_id": f"naamapadam:hi:test:{row['source_index']}", "experiment_id": experiment,
            "tokens": row["tokens"], "gold_labels": row["labels"], "predicted_labels": predicted,
            "gold_spans": strict_spans(row["labels"]), "predicted_spans": strict_spans(predicted)}


def reserve_once(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical({"started_at": now(), "rule": "Never rerun inference if this marker exists, including after interruption"}) + "\n")


def replay(predictions):
    """Verify aggregate metrics without calling the model a second time."""
    metrics = StrictMetrics()
    count = 0
    for row in records(predictions):
        if [list(s) for s in strict_spans(row["gold_labels"])] != row["gold_spans"] or [list(s) for s in strict_spans(row["predicted_labels"])] != row["predicted_spans"]:
            raise ValueError("Prediction span schema mismatch")
        metrics.add(row["gold_labels"], row["predicted_labels"])
        count += 1
    return metrics.result(), count


def final_evaluate(split):
    frozen = verify_freeze()
    path = select_split(frozen["derived_dir"], split)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if split == "official_test":
        previous = read_json(FINAL_DIR / "test_clean_completion.json")
        for filename, checksum in previous["files"].items():
            if file_hash(filename) != checksum:
                raise ValueError("Clean-test completion checksum mismatch")
    model_path = Path(frozen["model_dir"]) / "model.crfsuite"
    config = frozen["config"]
    sample = read_json(Path(frozen["model_dir"]) / "sample_manifest.json")
    selected, vocabulary = set(sample["source_indices"]), set()
    for row in records(Path(frozen["derived_dir"]) / "train_clean.jsonl"):
        if row["source_index"] in selected:
            vocabulary.update(row["tokens"])
    model = CRFModel(config["model"]).load(model_path)
    predictions_path = FINAL_DIR / f"{split}_predictions.jsonl"
    if predictions_path.exists():
        raise FileExistsError("Predictions already exist; inference rerun forbidden")
    reserve_once(FINAL_DIR / f"{split}_started.json")
    stats, latencies, counts, examples = StrictMetrics(), [], Counter({k: 0 for k in CATEGORIES}), defaultdict(list)
    process, peak, empty, seen = psutil.Process(), 0, 0, set()
    started = time.perf_counter()
    with predictions_path.open("x", encoding="utf-8") as output:
        for row in records(path):
            if row["source_index"] in seen:
                raise ValueError("Duplicate source identifier")
            seen.add(row["source_index"])
            begin = time.perf_counter()
            predicted = model.predict(sentence_features(row["tokens"], config["features"]))
            latencies.append(time.perf_counter() - begin)
            item = prediction_record(row, predicted, split, frozen["experiment_id"])
            output.write(canonical(item) + "\n")
            stats.add(row["labels"], predicted)
            analyse(row, predicted, vocabulary, counts, examples, 10)
            peak = max(peak, process.memory_info().rss)
            empty += not row["tokens"]
    result = dict(stats.result(), split=split, records=len(latencies), empty_records=empty,
                  inference_seconds=sum(latencies), evaluation_wall_seconds=time.perf_counter() - started,
                  median_sentence_latency_ms=float(np.median(latencies) * 1000),
                  p95_sentence_latency_ms=float(np.percentile(latencies, 95) * 1000),
                  model_size_bytes=model_path.stat().st_size, peak_inference_rss_bytes_sentence_sampled=peak,
                  latency_definition="Feature extraction plus tagging; includes empty records, excludes model load, file IO, scoring and error analysis; no warmup excluded",
                  memory_definition="Whole process RSS sampled after each record, including model, training vocabulary, metrics and bounded examples; not exact high-water memory",
                  freeze_payload_sha256=read_json(MANIFEST)["payload_sha256"], evaluation_policy=frozen["evaluation_policy"])
    replayed, count = replay(predictions_path)
    if replayed != stats.result() or count != result["records"]:
        raise ValueError("Saved prediction metrics do not reproduce")
    write_json(FINAL_DIR / f"{split}_metrics.json", result)
    write_json(FINAL_DIR / f"{split}_error_counts.json", {"counts": counts,
               "definitions": "Counts overlap; correct/boundary/missed/FP count spans, confusion counts overlapping span pairs, slices count gold entities except <=10-token short sequences count records. Abbreviations and headline-like sequences are heuristic categories. No test-driven tuning permitted."})
    restricted = FINAL_DIR / "restricted_examples"
    write_json(restricted / f"{split}_error_examples.json", {"purpose": "Final academic reporting only; never model selection", "examples": {k: examples[k] for k in CATEGORIES}, "maximum_examples_per_category": 10})
    example_lines = [f"# {split}: restricted illustrative examples", "", "Final reporting only. Do not read during later model selection. Examples are bounded and not representative counts."]
    for category in CATEGORIES:
        example_lines += ["", f"## {category}", ""]
        for e in examples[category]:
            example_lines += [" ".join(e["tokens"]), "", f"Source index {e['source_index']}; gold spans {e['gold']}; predicted spans {e['predicted']}.", ""]
    (restricted / f"{split}_error_examples.md").write_text("\n".join(example_lines) + "\n", encoding="utf-8")
    lines = [f"# Final {split} evaluation", "", "Main clean Naamapadam conclusion." if split == "test_clean" else "Separate official benchmark comparison; raw BIO retained. Contains overlap with development data and is not an independent clean estimate.", "",
             f"Records: {result['records']}; tokens: {result['tokens']}; empty records: {empty}.", "",
             "| Precision | Recall | Strict micro F1 | Strict macro F1 |", "|---:|---:|---:|---:|",
             "| " + " | ".join(f"{result[k]:.9f}" for k in ("strict_entity_precision", "strict_entity_recall", "strict_entity_micro_f1", "strict_entity_macro_f1")) + " |", "",
             "| Type | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {k} | {v['precision']:.9f} | {v['recall']:.9f} | {v['f1']:.9f} | {v['support']} |" for k, v in result["per_class"].items()]
    lines += ["", f"Secondary token accuracy: {result['token_accuracy']:.9f}.", "",
              f"Inference: {result['inference_seconds']:.9f} seconds; median {result['median_sentence_latency_ms']:.9f} ms; P95 {result['p95_sentence_latency_ms']:.9f} ms. Model: {result['model_size_bytes']} bytes; sampled RSS: {peak} bytes.", "", result["latency_definition"], "", result["memory_definition"], "", frozen["evaluation_policy"], "",
              "Predictions were generated in one pass. Metrics were independently replayed from saved predictions without a second model call. Test examples remain in the excluded restricted_examples directory.", "", "| Error category | Count |", "|---|---:|"]
    lines += [f"| {k} | {counts[k]} |" for k in CATEGORIES]
    (FINAL_DIR / f"{split}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verify_freeze()
    artifacts = list(FINAL_DIR.glob(f"{split}_*")) + list(restricted.glob(f"{split}_*"))
    write_json(FINAL_DIR / f"{split}_completion.json", {"finished_at": now(), "inference_passes": 1,
               "saved_predictions_replayed": True, "freeze_verified_before_after": True,
               "files": {str(p): file_hash(p) for p in artifacts if p.is_file()}})
    print(f"Completed {split}; metrics saved. Freeze verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=SPLITS)
    final_evaluate(parser.parse_args().split)
