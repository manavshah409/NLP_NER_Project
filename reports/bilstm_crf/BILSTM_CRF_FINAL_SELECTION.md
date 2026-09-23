# IndicNewsNER BiLSTM-CRF Final Model Selection Report

## 1. Executive Summary & Research Boundary
This report documents the final validation-based model selection for Milestone 3B. Under the project's non-negotiable data boundary, **all model comparisons and selections are conducted strictly on `validation_clean` (12,896 records, 289,695 tokens)**.

> [!IMPORTANT]
> **Research Boundary Notice:**
> - Model selection is based solely on validation split metrics.
> - Test splits (`test_clean`, `official_test`), restricted test predictions, and benchmark examples remain strictly sealed.
> - Performance is reported as observed validation improvements over the frozen CRF baseline; no formal claims of statistical superiority on unobserved test data are made.
> - Neural validation results are strictly compared against the CRF validation baseline (never against CRF clean-test results).

## 2. Comprehensive Model Comparison

| Model | Training Size | Seed | Strict Precision | Strict Recall | Strict Micro F1 | Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Token Acc | Delta vs CRF Baseline | Best Epoch | Train Time (s) | Peak RSS (GiB) | Median Latency (ms) | P95 Latency (ms) | Vocab Size | Val UNK Rate | Model Size (MiB) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frozen Classical CRF (100k)** | 100,000 | 42 | 0.749006 | 0.681722 | **0.713770** | 0.709317 | 0.7516 | 0.6085 | 0.7678 | 0.9254 | +0.000000 | N/A (L-BFGS) | 38.3s | 2.658 GiB | 0.508 ms | 1.220 ms | N/A (features) | 0.00% | 0.77 MiB |
| **BiLSTM-CRF 50k (Seed 7)** | 50,000 | 7 | 0.741788 | 0.687524 | **0.713626** | 0.708833 | 0.7532 | 0.6050 | 0.7683 | 0.9272 | -0.000144 | 14 | 2962.3s | 0.871 GiB | 3.422 ms | 4.300 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (Seed 21)** | 50,000 | 21 | 0.721857 | 0.712332 | **0.717063** | 0.710939 | 0.7521 | 0.6061 | 0.7747 | 0.9279 | +0.003293 | 14 | 3015.0s | 1.028 GiB | 3.527 ms | 4.522 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (Seed 42)** | 50,000 | 42 | 0.740164 | 0.699447 | **0.719230** | 0.715036 | 0.7587 | 0.6123 | 0.7741 | 0.9284 | +0.005460 | 14 | 2461.7s | 0.989 GiB | 4.660 ms | 5.646 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (3-Seed Aggregate Mean)** | 50,000 | aggregate | 0.734603 | 0.699767 | **0.716639** | 0.711603 | 0.7547 | 0.6078 | 0.7723 | 0.9278 | +0.002870 | 14 | 2813.0s | 0.963 GiB | 3.870 ms | 4.823 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 100k (Seed 42)** | 100,000 | 42 | 0.749832 | 0.726901 | **0.738188** | 0.733398 | 0.7812 | 0.6352 | 0.7838 | 0.9336 | +0.024419 | 9 | 3949.7s | 1.238 GiB | 3.601 ms | 4.505 ms | 94,405 | 2.54% | 36.92 MiB |

## 3. Robustness Analysis Summary & Decision-Gate Outcome
- **50k Robustness Evaluation:** Repeated training across seeds 7, 21, and 42 on the identical 50k sample and vocabulary produced a **three-seed mean Strict Micro F1 of 0.716639** with a sample standard deviation of **0.002826**, range of **0.005604**, and median of **0.717063**.
- **Decision Gate for 100k Scaling:** All 6 criteria passed (mean 50k micro F1 > 0.713770, $\ge 2$ seeds above baseline, 0 violations, safe memory < 12.0 GiB, identical sample/vocab reuse, and 100% test pass rate).

## 4. Final Selected Baseline Model & Justification
- **Selected Primary Baseline:** `bilstm_crf_100k_seed42_9af1d47db429`
- **Validation-Based Selection:** The 100k model was selected through validation-only comparison on `validation_clean`.
- **Primary Selection Metric:** Strict Entity Micro F1 of **0.738188** on `validation_clean` (+0.024418 over the frozen 100k CRF baseline of 0.713770, and +0.021549 over the 50k 3-seed mean).
- **Secondary Metric Performance:**
  - **Strict Precision & Recall:** Precision **0.749832**, Recall **0.726901**.
  - **Strict Macro F1:** **0.733398** (+0.024081 over CRF).
  - **Class-wise F1:** `PER` **0.7812** (+0.0296), `ORG` **0.6352** (+0.0267), `LOC` **0.7838** (+0.0160).
  - **Validation UNK Rate:** Reduced to **2.54%** (94,405 vocabulary tokens).
  - **Token Accuracy:** **0.9336** (up from 0.9254).
  - **Computational Efficiency:** Peak memory of 1.238 GiB (well below 12.0 GiB budget), 3.601 ms / sentence median inference latency.
- **Reproducibility:** 100% exact metric replay confirmed across all evaluation metrics from saved checkpoint weights.

## 5. Known Limitations
1. **Out-of-Vocabulary Tokens:** Pure word-level embeddings still produce a 2.54% UNK rate on validation data. Subword (BPE) or character-level representations are recommended for future milestones.
2. **Inference Latency:** While the classical CRF evaluates in ~0.51 ms per sentence on CPU, the neural BiLSTM-CRF requires ~3.60 ms on MPS due to PyTorch sequence batching overhead.
3. **Sealed Test Status:** This model has not yet been evaluated on the final sealed test data (`test_clean` / `official_test`).
