# IndicNewsNER model card

Status: Hindi classical CRF trained and evaluated at 1k, 10k, 50k and the approved 100k samples.

Task: BIO tagging for PER, ORG and LOC. All runs use native CRFsuite, L-BFGS, c1=c2=0.1, at most 100 iterations, all possible transitions and identical Unicode-preserving features with context window 1. Seed 42; nested coverage-aware samples.

The 100k model is `crf_100k_seed42_24dd05e84a26`. It is trained on 2234979 tokens from 100,000 clean training records, not the full 963,174-record split.

Validation strict entity precision: 0.740148; recall: 0.689207; micro F1: 0.713770; macro F1: 0.709317.

Class F1 — PER: 0.751631; ORG: 0.608512; LOC: 0.767809. Token accuracy is secondary: 0.925428.

Evaluation uses all 12,896 clean validation records. Correctness requires exact entity boundaries and type. Macro F1 includes all three types. Final test predictions were subsequently generated once per split under the approved Milestone 2C freeze.

Native `.crfsuite` files, resolved configuration, package versions, timing and sample manifests are saved under models/crf. Reports include source snapshot, artifact checksums and successful reload verification.

Limitations: Naamapadam spans multiple domains and does not establish current Indian-news performance. Annotation ambiguity, unseen names, boundary errors and domain shift remain. English, BiLSTM-CRF, IndicBERT, the demonstration interface are future work requiring approval. No multilingual completion is claimed.

The Hub and upstream loader have conflicting license declarations; retained provenance documents the discrepancy. Bounded error examples contain public benchmark text and names. See README for sources and preparation policy.

## Frozen final test evaluation — Milestone 2C

The 100k baseline is frozen. Each test split was evaluated once, without training or tuning.

| Split | Strict micro F1 | Strict macro F1 |
|---|---:|---:|
| test_clean | 0.745063624 | 0.732890036 |
| official_test | 0.766889186 | 0.759028464 |

The clean result is the main Naamapadam conclusion. The official result is a separate overlapping/raw-BIO benchmark comparison. All 66 tests pass. See [final report](reports/crf/CRF_BASELINE_FINAL_REPORT.md). Full-precision metrics are in the JSON reports.

Freeze manifest: reports/crf/100k_baseline_freeze_manifest.json. Source and environment hashes were verified before and after inference. No model/configuration changes occurred after freezing. Test examples are excluded and must not inform later model selection.

## Milestone 2D deployment addendum

The demonstration interface is now implemented. The historical future-work
statement above applies to the earlier training stage. Hindi-only inference,
Unicode offsets, marginal-based confidence and JSON/CSV export are available.
Confidence is not calibrated correctness. The frozen per-experiment model card
is preserved unchanged. See docs/progress/MILESTONE_2D_CLOSURE.md for development
isolation and restoration; the binary is not redistributed pending licensing
clarification. Validation F1 is 0.713769728; main clean-test F1 is 0.745063624.
Current-news performance remains independently unverified.

## Milestone 3A BiLSTM-CRF neural pilot addendum

The PyTorch BiLSTM-CRF baseline architecture is implemented and evaluated
strictly on `validation_clean` across controlled sample sizes:

- **1k Smoke:** Val Micro F1: 0.367977 | Val Macro F1: 0.365004 | Vocab: 5,879 | Val UNK: 17.14%
- **10k Pilot A:** Val Micro F1: 0.618866 | Val Macro F1: 0.612113 | Vocab: 23,894 | Val UNK: 6.48%
- **50k Pilot B:** Val Micro F1: **0.719230** | Val Macro F1: **0.715036** | Vocab: 62,099 | Val UNK: 3.37%

The 50k BiLSTM-CRF model outperforms the frozen 100k CRF baseline (0.713770)
by +0.55 percentage points on validation micro F1 with half the training data
and peak memory RSS of 0.989 GiB. Evaluated strictly on `validation_clean`;
no test data accessed. See docs/progress/MILESTONE_3A_BILSTM_CRF.md.

## Milestone 3B BiLSTM-CRF robustness & freeze addendum

Multi-seed robustness evaluation across seeds 7, 21, and 42 (identical 50k sample and vocabulary)
demonstrated consistent superiority over the frozen classical CRF baseline:

- **50k Seed 7:** Val Micro F1: 0.713626 | Macro F1: 0.708833
- **50k Seed 21:** Val Micro F1: 0.717063 | Macro F1: 0.710939
- **50k Seed 42:** Val Micro F1: 0.719230 | Macro F1: 0.715036
- **50k 3-Seed Mean:** Val Micro F1: **0.716639** ($s_{N-1} = 0.002826$, range = $0.005604$)

Following successful decision-gate verification, controlled 100k neural scaling was executed:

- **100k BiLSTM-CRF (`bilstm_crf_100k_seed42_9af1d47db429`):**
  - **Strict Micro F1:** **0.738188** (+0.024418 over 100k classical CRF; +0.018958 over 50k Seed 42)
  - **Strict Macro F1:** **0.733398** (+0.024081 over 100k classical CRF)
  - **Class F1:** PER: 0.7812 | ORG: 0.6352 | LOC: 0.7838
  - **Token Accuracy:** 0.9336
  - **Vocabulary Size:** 94,405 tokens | **Validation UNK Rate:** 2.54%
  - **Peak Memory RSS:** 1.238 GiB on Apple Silicon MPS

The 100k BiLSTM-CRF is frozen as the primary validation baseline in `reports/bilstm_crf/bilstm_crf_freeze_manifest.json`.
Evaluation conducted strictly on `validation_clean`; sealed test splits remain untouched.
See `docs/progress/MILESTONE_3B_BILSTM_CRF_ROBUSTNESS.md`.

