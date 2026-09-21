# Final official_test evaluation

Separate official benchmark comparison; raw BIO retained. Contains overlap with development data and is not an independent clean estimate.

Records: 867; tokens: 19893; empty records: 3.

| Precision | Recall | Strict micro F1 | Strict macro F1 |
|---:|---:|---:|---:|
| 0.787712562 | 0.747138398 | 0.766889186 | 0.759028464 |

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PER | 0.819261214 | 0.788071066 | 0.803363519 | 788 |
| ORG | 0.698380567 | 0.662188100 | 0.679802956 | 521 |
| LOC | 0.823117338 | 0.766721044 | 0.793918919 | 613 |

Secondary token accuracy: 0.942542603.

Inference: 0.532870390 seconds; median 0.550292025 ms; P95 1.293271099 ms. Model: 22707636 bytes; sampled RSS: 93929472 bytes.

Feature extraction plus tagging; includes empty records, excludes model load, file IO, scoring and error analysis; no warmup excluded

Whole process RSS sampled after each record, including model, training vocabulary, metrics and bounded examples; not exact high-water memory

Strict B-only typed spans, exact boundaries, fixed 3-class macro F1. Preserve raw official BIO; include empty official records with empty predictions. No test-time repair.

Predictions were generated in one pass. Metrics were independently replayed from saved predictions without a second model call. Test examples remain in the excluded restricted_examples directory.

| Error category | Count |
|---|---:|
| correct_entities | 1436 |
| boundary_errors | 176 |
| missed_entities | 224 |
| false_positive_entities | 146 |
| PER/ORG_confusion | 26 |
| PER/LOC_confusion | 26 |
| ORG/LOC_confusion | 38 |
| unseen_entity_tokens | 248 |
| abbreviations_acronyms | 19 |
| mixed_script_tokens | 1 |
| numeric_punctuation_adjacent | 541 |
| headline_like_short_sequences | 153 |
