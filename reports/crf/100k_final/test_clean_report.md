# Final test_clean evaluation

Main clean Naamapadam conclusion.

Records: 506; tokens: 12388; empty records: 0.

| Precision | Recall | Strict micro F1 | Strict macro F1 |
|---:|---:|---:|---:|
| 0.762803235 | 0.728130360 | 0.745063624 | 0.732890036 |

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PER | 0.803468208 | 0.762340037 | 0.782363977 | 547 |
| ORG | 0.658273381 | 0.637630662 | 0.647787611 | 287 |
| LOC | 0.787974684 | 0.750000000 | 0.768518519 | 332 |

Secondary token accuracy: 0.938569583.

Inference: 0.323672844 seconds; median 0.572354504 ms; P95 1.302667253 ms. Model: 22707636 bytes; sampled RSS: 93716480 bytes.

Feature extraction plus tagging; includes empty records, excludes model load, file IO, scoring and error analysis; no warmup excluded

Whole process RSS sampled after each record, including model, training vocabulary, metrics and bounded examples; not exact high-water memory

Strict B-only typed spans, exact boundaries, fixed 3-class macro F1. Preserve raw official BIO; include empty official records with empty predictions. No test-time repair.

Predictions were generated in one pass. Metrics were independently replayed from saved predictions without a second model call. Test examples remain in the excluded restricted_examples directory.

| Error category | Count |
|---|---:|
| correct_entities | 849 |
| boundary_errors | 114 |
| missed_entities | 141 |
| false_positive_entities | 100 |
| PER/ORG_confusion | 21 |
| PER/LOC_confusion | 17 |
| ORG/LOC_confusion | 27 |
| unseen_entity_tokens | 162 |
| abbreviations_acronyms | 13 |
| mixed_script_tokens | 0 |
| numeric_punctuation_adjacent | 264 |
| headline_like_short_sequences | 66 |
