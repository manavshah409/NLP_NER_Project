# Milestone 2B completion

Approved 100k Hindi CRF extension complete; later phases require approval

| Sample records | Precision | Recall | Strict micro F1 | Strict macro F1 |
|---:|---:|---:|---:|---:|
| 1000 | 0.622166 | 0.445006 | 0.518881 | 0.515804 |
| 10000 | 0.686966 | 0.597292 | 0.638999 | 0.635749 |
| 50000 | 0.726697 | 0.666220 | 0.695146 | 0.690853 |
| 100000 | 0.740148 | 0.689207 | 0.713770 | 0.709317 |

Tests: {'tests': 53, 'failures': 0, 'errors': 0, 'skipped': 0, 'passed': 53}. All model artefact checksums, experiment identifiers and native reload checks pass.

The full data checkpoint was reproduced, including raw checksums, explicit repair/removal manifests and byte-identical clean dataset reconstruction.

Recommendation: Stop at the current scale. No further training has been started.

Limitations: sampled RSS peaks; latency reflects this machine and current system load; no repeated-seed uncertainty study; no news-domain gold evaluation; Hindi only. Gazetteers, neural models, English, test evaluation and the interface remain outside this completed milestone.

Results were not fabricated or extrapolated. The current sample is not the complete training split. The official and clean test sets were used only for data-integrity preparation, never for predictions or model selection.

See crf/crf_scaling_comparison.csv for all required scaling metrics, crf/*/error_analysis.md for bounded validation examples, and implementation_log.md for commands and recovered failures.
