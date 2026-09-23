# Changelog

## Milestone 3B

- Evaluated 50k BiLSTM-CRF multi-seed robustness across seeds 7, 21, and 42 on identical 50k sample and vocabulary.
- Measured 50k 3-seed mean validation micro F1 of **0.716639** ($s_{N-1} = 0.002826$, range = $0.005604$), confirming consistent superiority over classical CRF (0.713770).
- Passed all 6 conditions of the pre-specified 100k scaling decision gate.
- Executed controlled 100k BiLSTM-CRF training on Apple Silicon MPS with seed 42, achieving **0.738188 validation strict micro F1** (+2.44 pp over classical CRF baseline).
- Selected and froze `bilstm_crf_100k_seed42_9af1d47db429` as primary validation baseline in `reports/bilstm_crf/bilstm_crf_freeze_manifest.json` with SHA-256 envelope verification.
- Verified 100% bitwise & numerical metric replay across all model checkpoints on `validation_clean`.
- Expanded test suite to 129 tests (all passing). Sealed test splits remain strictly preserved.

## Milestone 3A

- Implemented PyTorch BiLSTM-CRF sequence tagger with custom Linear-Chain CRF module.
- Built training-only vocabulary builder, dataset collator with boolean sequence masks.
- Added guarded development support and strict validation-only evaluator.
- Conducted controlled pilot scaling: 1k (F1: 0.3680), 10k (F1: 0.6189), and 50k (F1: 0.7192).
- Verified 50k BiLSTM-CRF surpasses frozen 100k CRF baseline (0.7138) by +0.55 pp.
- Added 12 new unit and integration tests; all 122 tests pass.


- Added the local Streamlit demo for the frozen 100k CRF.
- Added Unicode tokenisation, exclusive offsets, safe highlights, counts and JSON/CSV exports.
- Exposed explicitly defined mean CRFsuite tag marginals.
- Retained all frozen artifacts/package versions; demo verification needs no test data.
- Verified five original examples and captured five real screenshots.
- Added 19 tests; all 85 pass. Included faculty guide and progress report.

## Milestone 2C

- Froze the approved 100k classical CRF, dataset files, sample, label schema, configuration, code and environment with SHA256 verification.
- Independently reproduced the training indices and statistics.
- Evaluated test_clean once: strict micro F1 0.7450636243966652.
- Evaluated official_test once: strict micro F1 0.7668891855807745.
- Added bounded, excluded test illustrations and stable prediction identifiers.
- Expanded the passing suite from 53 to 66 tests.
- Updated scaling results, final report, model card and progress documentation. No new training or tuning.

## v0.2.0-crf-demo

- Complete local Hindi CRF demo and phase report with captured interface evidence.
- Add guarded development entry point and synthetic access-boundary tests.
- Document sealed final evaluation, exact model restoration and licensing limits.
- Preserve the 100k model and all 51 frozen artifacts without retraining.
