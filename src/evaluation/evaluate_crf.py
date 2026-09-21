"""Validation-only inference, strict metrics, latency and bounded error analysis."""
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import psutil
from src.data.bio import strict_spans
from src.data.crf_features import sentence_features
from src.data.loader import records
from src.evaluation.metrics import StrictMetrics
from src.models.crf_model import CRFModel
from src.utils.io import write_json

CATEGORIES = ("correct_entities", "boundary_errors", "missed_entities", "false_positive_entities",
              "PER/ORG_confusion", "PER/LOC_confusion", "ORG/LOC_confusion", "unseen_entity_tokens",
              "abbreviations_acronyms", "mixed_script_tokens", "numeric_punctuation_adjacent", "headline_like_short_sequences")


def analyse(row, predicted, vocabulary, counts, examples, limit):
    tokens = row["tokens"]
    gold, pred = set(strict_spans(row["labels"])), set(strict_spans(predicted))
    snapshot = {"source_index": row["source_index"], "tokens": tokens, "gold": sorted(gold), "predicted": sorted(pred)}

    def add(category, detail=None):
        counts[category] += 1
        if len(examples[category]) < limit:
            examples[category].append(dict(snapshot, detail=detail))

    for span in sorted(gold & pred):
        add("correct_entities", span)
    for span in sorted(gold - pred):
        overlaps = [p for p in pred if p[0] < span[1] and span[0] < p[1]]
        if not overlaps:
            add("missed_entities", span)
        if any(p[2] == span[2] and p[:2] != span[:2] for p in overlaps):
            add("boundary_errors", span)
        for p in overlaps:
            if p[2] != span[2]:
                pair = frozenset((p[2], span[2]))
                category = {frozenset(("PER", "ORG")): "PER/ORG_confusion", frozenset(("PER", "LOC")): "PER/LOC_confusion", frozenset(("ORG", "LOC")): "ORG/LOC_confusion"}[pair]
                add(category, {"gold": span, "predicted": p})
    for span in sorted(pred - gold):
        if not any(g[0] < span[1] and span[0] < g[1] for g in gold):
            add("false_positive_entities", span)
    for start, end, kind in sorted(gold):
        entity_tokens = tokens[start:end]
        if any(t not in vocabulary for t in entity_tokens):
            add("unseen_entity_tokens", [start, end, kind])
        if any(t.isupper() and any("LATIN" in unicodedata.name(c, "") for c in t) or "." in t or "॰" in t for t in entity_tokens):
            add("abbreviations_acronyms", [start, end, kind])
        if any(any("LATIN" in unicodedata.name(c, "") for c in t) and any("DEVANAGARI" in unicodedata.name(c, "") for c in t) for t in entity_tokens):
            add("mixed_script_tokens", [start, end, kind])
        adjacent = tokens[max(0, start-1):min(len(tokens), end+1)]
        if any(c.isdigit() or unicodedata.category(c).startswith("P") for t in adjacent for c in t):
            add("numeric_punctuation_adjacent", [start, end, kind])
    if len(tokens) <= 10:
        add("headline_like_short_sequences")


def evaluate(model_path, validation_path, config, vocabulary, report_dir):
    if Path(validation_path).name != "validation_clean.jsonl":
        raise ValueError("Only validation_clean is authorised for development evaluation")
    model = CRFModel(config["model"]).load(model_path)
    metrics, latencies, counts, examples = StrictMetrics(), [], Counter({k: 0 for k in CATEGORIES}), defaultdict(list)
    rss_peak, process = 0, psutil.Process()
    started = time.perf_counter()
    for row in records(validation_path):
        if row["source_split"] != "validation":
            raise ValueError("Unexpected evaluation source split")
        begin = time.perf_counter()
        predicted = model.predict(sentence_features(row["tokens"], config["features"]))
        latencies.append(time.perf_counter() - begin)
        metrics.add(row["labels"], predicted)
        analyse(row, predicted, vocabulary, counts, examples, config["evaluation"]["max_examples_per_category"])
        rss_peak = max(rss_peak, process.memory_info().rss)
    wall = time.perf_counter() - started
    result = metrics.result()
    result.update(validation_records=len(latencies), inference_seconds=sum(latencies),
                  evaluation_wall_seconds=wall, median_sentence_latency_ms=float(np.median(latencies) * 1000),
                  p95_sentence_latency_ms=float(np.percentile(latencies, 95) * 1000),
                  peak_inference_rss_bytes_sentence_sampled=rss_peak,
                  latency_definition="Feature extraction plus tag call per sentence; excludes disk read, metric and error analysis; no warmup excluded")
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "validation_metrics.json", result)
    analysis = {"counts": counts, "examples": {k: examples[k] for k in CATEGORIES},
                "definitions": "Counts may overlap. Correct/boundary/missed/FP count spans; confusion counts overlapping span pairs; slices count gold entities except short sequences (<=10 tokens) count records. Acronyms use Latin uppercase or dot marks as a heuristic; short length is a headline-like proxy, not a verified headline. Unseen means absent from sampled training token vocabulary."}
    write_json(report_dir / "error_analysis.json", analysis)
    lines = ["# Bounded validation error analysis", "", analysis["definitions"], "", "| Category | Count |", "|---|---:|"]
    lines += [f"| {k} | {counts[k]} |" for k in CATEGORIES]
    for category in CATEGORIES:
        if examples[category]:
            e = examples[category][0]
            lines += ["", f"## {category}", "", " ".join(e["tokens"]), "", f"Gold spans: {e['gold']}; predicted spans: {e['predicted']}."]
    (report_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
