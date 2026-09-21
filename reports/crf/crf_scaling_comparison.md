# Controlled CRF scaling comparison

All metrics below are observed validation results. Samples are nested; features and model parameters are identical.

| Records | Train tokens | Train seconds | Peak RSS GiB | Strict micro F1 | Strict macro F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 22675 | 3.83 | 0.109 | 0.518881 | 0.515804 |
| 10000 | 225176 | 39.68 | 0.419 | 0.638999 | 0.635749 |
| 50000 | 1120452 | 199.76 | 1.488 | 0.695146 | 0.690853 |
| 100000 | 2234979 | 368.14 | 2.658 | 0.713770 | 0.709317 |

| Records | PER F1 | ORG F1 | LOC F1 | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.532839 | 0.413295 | 0.601279 | 0.622166 | 0.445006 |
| 10000 | 0.665341 | 0.534637 | 0.707268 | 0.686966 | 0.597292 |
| 50000 | 0.728689 | 0.588035 | 0.755836 | 0.726697 | 0.666220 |
| 100000 | 0.751631 | 0.608512 | 0.767809 | 0.740148 | 0.689207 |

Recommendation: **Stop at the current scale**. The observed 50k→100k micro F1 change is 0.018624; 100k peak RSS was 51.8% of its configured budget.

Decision rule: recommend considering 250k only if the latest micro F1 gain exceeds 0.002 and observed peak RSS stays below 40% of the current budget; otherwise stop scaling for now. This is a conservative project rule, not a statistical significance claim. A 250k run would require approval and a fresh resource assessment; linear memory or runtime scaling is not assumed. No larger run has been started.

See the CSV for all class distributions, class F1 scores, precision/recall, timing and size fields. Peak RSS is sampled every 0.25 seconds over the worker, including evaluation. Latency includes feature extraction and tagging; no warmup is excluded. Token accuracy is secondary.

## Frozen final evaluation

Only the selected 100k model was evaluated on test data. These results were obtained after selection and are not scaling-selection criteria.

| Evaluation split | Records | Tokens | Precision | Recall | Strict micro F1 | Strict macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| validation_clean | 12896 | 289695 | 0.740147601 | 0.689207298 | 0.713769728 | 0.709317452 |
| test_clean | 506 | 12388 | 0.762803235 | 0.728130360 | 0.745063624 | 0.732890036 |
| official_test | 867 | 19893 | 0.787712562 | 0.747138398 | 0.766889186 | 0.759028464 |

Main conclusion: test_clean. Official test is a separate overlapping/raw-annotation benchmark comparison.
