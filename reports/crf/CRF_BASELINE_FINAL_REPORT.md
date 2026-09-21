# Final classical CRF baseline report

Frozen experiment: `crf_100k_seed42_24dd05e84a26`.

## Objective and data

Extract complete Hindi PER, ORG and LOC spans using a reproducible classical baseline before neural-model comparison. Naamapadam is multi-domain; this is not evidence of current Indian-news domain performance.

Dataset: ai4bharat/naamapadam, hi; derived version `naamapadam_hi_crf_v1`. Clean training has 963,174 records; the model used a 100,000-record sample. All raw and derived file checksums were verified at freeze and around final inference.

## Features and model configuration

Original Unicode token, length, prefixes/suffixes 1–3, numeric/punctuation/alphabetic indicators, Latin/Devanagari/mixed-script indicators, Unicode categories and word shape. Latin-only lowercase derivatives; context window 1 with BOS/EOS. No gazetteers, transliteration or test-derived feature lists.

```json
{
  "algorithm": "lbfgs",
  "c1": 0.1,
  "c2": 0.1,
  "max_iterations": 100,
  "all_possible_transitions": true
}
```

## Sampling and scaling

Seed 42. Lowest seed/index SHA256 order with an entity-coverage anchor per type; source indices are retained. Samples are nested. Freeze independently reconstructed all selected indices and matched the saved training statistics. Only train_clean supplied training records.

| Records | Validation micro F1 | Validation macro F1 | Training seconds | Peak RSS GiB |
|---:|---:|---:|---:|---:|
| 1000 | 0.518880587 | 0.515804384 | 3.825170 | 0.108734 |
| 10000 | 0.638998658 | 0.635748688 | 39.683235 | 0.419052 |
| 50000 | 0.695145561 | 0.690853430 | 199.757245 | 1.488129 |
| 100000 | 0.713769728 | 0.709317452 | 368.139129 | 2.657623 |

The 100k model was selected and approved from validation results before final test evaluation. Retention reflects its validation improvement and manageable observed memory; the test results did not influence selection. No larger model was trained.

## Validation and final evaluation

| Evaluation split | Records | Tokens | Precision | Recall | Strict micro F1 | Strict macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| validation_clean | 12896 | 289695 | 0.740147601 | 0.689207298 | 0.713769728 | 0.709317452 |
| test_clean | 506 | 12388 | 0.762803235 | 0.728130360 | 0.745063624 | 0.732890036 |
| official_test | 867 | 19893 | 0.787712562 | 0.747138398 | 0.766889186 | 0.759028464 |

test_clean is the main Naamapadam test conclusion. official_test is reported separately: it includes development overlap, 3 empty records and original invalid BIO transitions. Its gold labels are not repaired. The two test sets overlap each other and are not independent samples.

Only B starts a span; correctness requires exact boundaries and type. Macro F1 averages PER/ORG/LOC, including zeros if unsupported. The official score is under this documented policy; it should not be called directly comparable to papers using different BIO handling.

## Per-class results

| Split | Type | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| validation_clean | PER | 0.767982637 | 0.735961177 | 0.751631012 |
| validation_clean | ORG | 0.671496960 | 0.556329725 | 0.608512135 |
| validation_clean | LOC | 0.767096441 | 0.768523303 | 0.767809209 |
| test_clean | PER | 0.803468208 | 0.762340037 | 0.782363977 |
| test_clean | ORG | 0.658273381 | 0.637630662 | 0.647787611 |
| test_clean | LOC | 0.787974684 | 0.750000000 | 0.768518519 |
| official_test | PER | 0.819261214 | 0.788071066 | 0.803363519 |
| official_test | ORG | 0.698380567 | 0.662188100 | 0.679802956 |
| official_test | LOC | 0.823117338 | 0.766721044 | 0.793918919 |

## Inference resources

| Split | Inference seconds | Median ms | P95 ms | Sampled RSS MiB | Secondary token accuracy |
|---|---:|---:|---:|---:|---:|
| validation_clean | 7.398991892 | 0.507916498 | 1.220395738 | 2672.000000 | 0.925428468 |
| test_clean | 0.323672844 | 0.572354504 | 1.302667253 | 89.375000 | 0.938569583 |
| official_test | 0.532870390 | 0.550292025 | 1.293271099 | 89.578125 | 0.942542603 |

Model binary: 22707636 bytes. Feature extraction plus tagging; includes empty records, excludes model load, file IO, scoring and error analysis; no warmup excluded

Whole process RSS sampled after each record, including model, training vocabulary, metrics and bounded examples; not exact high-water memory

## Error analysis

Counts below may overlap; they are not mutually exclusive totals. Confusion counts are overlapping span pairs; short-sequence counts are records; other slices are entity counts. Abbreviation/headline-like slices are explicit heuristics, not manually validated categories.

| Category | Clean test count | Official test count |
|---|---:|---:|
| correct_entities | 849 | 1436 |
| boundary_errors | 114 | 176 |
| missed_entities | 141 | 224 |
| false_positive_entities | 100 | 146 |
| PER/ORG_confusion | 21 | 26 |
| PER/LOC_confusion | 17 | 26 |
| ORG/LOC_confusion | 27 | 38 |
| unseen_entity_tokens | 162 | 248 |
| abbreviations_acronyms | 13 | 19 |
| mixed_script_tokens | 0 | 1 |
| numeric_punctuation_adjacent | 264 | 541 |
| headline_like_short_sequences | 66 | 153 |

Illustrations are bounded to at most 10 per category and stored separately under excluded restricted_examples/. Test predictions and illustrations must not be opened during later model selection. This report contains aggregate counts only.

## Reproducibility and verification

The SHA256 freeze manifest covers the binary, configuration, sample, environment, label schema, data and source/evaluation code. The manifest payload has its own canonical JSON checksum. No Git commit existed at freeze; the later milestone commit records this exact manifest.

Final inference ran once per split, clean first then official. Exclusive start markers reject repeat inference. Saved predictions include stable source IDs, gold/predicted labels and strict reconstructed spans; replay exactly reproduced aggregate metrics without calling the model again.

Post-evaluation tests: {'tests': 66, 'failures': 0, 'errors': 0, 'skipped': 0, 'passed': 66}. Before the milestone 53 tests passed; before final evaluation the expanded 66-test suite passed. Synthetic tests never consume benchmark predictions.

## Limitations and next milestone

Single seed and no confidence interval; small clean test; benchmark annotation ambiguity; raw official BIO and overlap limit benchmark comparability; latency reflects this machine/load; RSS is sampled; no manually verified news-domain evaluation or English implementation. Upstream dataset license declarations conflict, as documented in README.

Recommended next milestone: plan and implement BiLSTM-CRF after explicit approval, using only train_clean/validation_clean for development and keeping these final test examples sealed. Do not start it automatically. No transformer or English work is part of Milestone 2C.
