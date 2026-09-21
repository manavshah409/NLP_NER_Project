# Controlled CRF scaling comparison

All metrics below are observed validation results. Samples are nested; features and model parameters are identical.

| Records | Train tokens | Train seconds | Peak RSS GiB | Strict micro F1 | Strict macro F1 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 22675 | 3.83 | 0.109 | 0.518881 | 0.515804 |
| 10000 | 225176 | 39.68 | 0.419 | 0.638999 | 0.635749 |
| 50000 | 1120452 | 199.76 | 1.488 | 0.695146 | 0.690853 |

Recommendation: **100,000 records**. The observed 10k→50k micro F1 change is 0.056147; 50k peak RSS was 29.2% of its configured budget.

This is a practical project decision, not a statistical significance claim. A 100k run would require a fresh resource assessment; linear memory or runtime scaling is not assumed. No larger run has been started.

See the CSV for all class distributions, class F1 scores, precision/recall, timing and size fields. Peak RSS is sampled every 0.25 seconds over the worker, including evaluation. Latency includes feature extraction and tagging; no warmup is excluded. Token accuracy is secondary.
