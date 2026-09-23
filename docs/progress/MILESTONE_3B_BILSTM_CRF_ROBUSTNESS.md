# Milestone 3B: BiLSTM-CRF Robustness Validation & Final Model Selection

Fourth-year undergraduate NLP project | 23 September 2026

## 1. Objective

Milestone 3B evaluates whether the observed 50k BiLSTM-CRF validation improvement over the frozen classical CRF baseline (+0.005460 strict micro F1) is reproducible across random neural seeds, executes a rigorous 6-condition decision gate for full 100k neural scaling, conducts validation-based model selection, freezes the final selected neural baseline, and enforces all project sealed-evaluation safeguards.

Under the project's strict data boundaries, **all evaluations and model selections were conducted solely on `validation_clean` (12,896 records, 289,695 tokens)**. No test splits (`test_clean`, `official_test`), restricted test predictions, or benchmark text examples were accessed.

---

## 2. Computational Environment & Resource Thresholds

* **Target Hardware:** MacBook Pro Apple Silicon (`arm64`, 24.0 GiB physical RAM)
* **OS:** macOS 26.6.2 (Darwin 25.3.0)
* **Python:** 3.14.7
* **PyTorch Version:** 2.14.0
* **Acceleration Device:** `mps` (Metal Performance Shaders) verified and active across all training runs
* **Peak Resident Memory during Training:** 1.238 GiB (well within the conservative 12.0 GiB budget)
* **Storage Requirement:** Model binary 36.92 MiB, training history and artifacts < 50 MiB

---

## 3. Data Isolation & Sealed Evaluation Boundaries

* **Dataset Version:** `naamapadam_hi_crf_v1`
* **Permitted Development Splits:**
  * Training: `data/derived/naamapadam_hi_crf_v1/train_clean.jsonl` (963,174 clean records)
  * Validation: `data/derived/naamapadam_hi_crf_v1/validation_clean.jsonl` (12,896 clean records)
* **Sealed Data Safeguards:** `test_clean.jsonl`, `official_test.jsonl`, and `reports/crf/100k_final/` are strictly blocked via `src/development_guard.py` and evaluation entry-point assertions.
* **Label Schema (7 BIO tags):** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`.

---

## 4. Multi-Seed Methodology & Non-Negotiable Experimental Controls

To isolate neural initialization and batch-shuffling stochasticity from dataset-sampling variation:
* **Identical Training Records:** Repeated 50k runs (seeds 7, 21, and 42) used the exact same 50,000 source records from `experiments/bilstm_crf/manifests/train_050k_seed42.json`.
* **Identical Vocabulary:** Reused the exact same 62,099-token training-only vocabulary (`vocabulary.json`, SHA-256: `a5187854...`).
* **Fixed Hyperparameters:** AdamW ($lr=0.001$, weight decay=$0.01$), batch size 32, max epochs 15, early stopping patience 3, gradient clipping norm 5.0, dropout 0.3, hidden dimension 128 per direction.
* **Deterministic Seed Propagation:** Random seed propagated across Python (`random`, `PYTHONHASHSEED`), NumPy, PyTorch CPU, PyTorch MPS, and DataLoader collation.

---

## 5. 50k Multi-Seed Experimental Results

Evaluations conducted strictly on `validation_clean` (12,896 records, 289,695 tokens):

| Experiment / Seed | Best Epoch | Val Strict Micro F1 | Val Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Val Token Acc | Train Time (s) | Peak RSS (GiB) | Delta vs Frozen CRF Baseline |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Seed 7** (`c99f52e0e1fb`) | 14 | 0.713626 | 0.708833 | 0.7532 | 0.6050 | 0.7683 | 0.9272 | 2962.34s | -0.000144 |
| **Seed 21** (`2e1de81e3085`) | 14 | 0.717063 | 0.710939 | 0.7521 | 0.6061 | 0.7747 | 0.9279 | 3014.95s | +0.003293 |
| **Seed 42** (`1c3a3bd180d5`) | 14 | 0.719230 | 0.715036 | 0.7587 | 0.6123 | 0.7741 | 0.9284 | 2461.72s | +0.005460 |

---

## 6. Three-Seed Statistical Robustness Aggregation

Multi-seed statistical aggregates computed across the three 50k runs ($N=3$):

| Metric | Mean ($\mu$) | Sample Std ($s_{N-1}$) | Min | Max | Range (Max-Min) | Median | Frozen CRF Baseline | Mean Delta vs CRF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Strict Micro F1** | **0.716639** | 0.002826 | 0.713626 | 0.719230 | 0.005604 | 0.717063 | 0.713770 | **+0.002870** |
| **Strict Macro F1** | **0.711603** | 0.003154 | 0.708833 | 0.715036 | 0.006203 | 0.710939 | 0.709317 | **+0.002285** |
| **PER F1** | **0.7547** | 0.0036 | 0.7521 | 0.7587 | 0.0067 | 0.7532 | 0.7516 | +0.0030 |
| **ORG F1** | **0.6078** | 0.0039 | 0.6050 | 0.6123 | 0.0073 | 0.6061 | 0.6085 | -0.0007 |
| **LOC F1** | **0.7723** | 0.0035 | 0.7683 | 0.7747 | 0.0064 | 0.7741 | 0.7678 | +0.0045 |
| **Training Time (s)** | **2813.0s** | 305.4s | 2461.7s | 3015.0s | 553.2s | 2962.3s | N/A | N/A |
| **Peak RSS (GiB)** | **0.963 GiB** | 0.082 GiB | 0.871 GiB | 1.028 GiB | 0.157 GiB | 0.989 GiB | 2.658 GiB | -1.695 GiB |
| **Median Latency (ms)** | **3.870 ms** | 0.687 ms | 3.422 ms | 4.660 ms | 1.238 ms | 3.527 ms | 0.508 ms | +3.362 ms |
| **P95 Latency (ms)** | **4.823 ms** | 0.722 ms | 4.300 ms | 5.646 ms | 1.346 ms | 4.522 ms | 1.220 ms | +3.603 ms |

> [!NOTE]
> Due to the small sample size ($N=3$), formal asymptotic confidence intervals are omitted in favor of exact sample standard deviation ($s_{N-1}$ with Bessel's correction) and empirical range bounds.

---

## 7. Decision-Gate Outcome for 100k Neural Scaling

The 6 pre-specified decision-gate conditions were evaluated:

1. **Mean 50k Strict Micro F1 > 0.713769728:** **PASSED** ($\mu = 0.716639$, $+0.002870$).
2. **At least 2 of 3 seeds exceed 0.713769728:** **PASSED** (Seed 21: 0.717063, Seed 42: 0.719230).
3. **Zero pipeline, checksum, checkpoint, or sealed-data violations:** **PASSED** (All runs clean and verified).
4. **Sampled memory within safe limit (< 12.0 GiB):** **PASSED** (Peak RSS $\approx 1.03\text{ GiB}$).
5. **Repeated 50k runs reuse identical 50k sample and vocabulary:** **PASSED** (Checksums verified).
6. **Complete test suite passes:** **PASSED** (128 passed before launch).

**Outcome: DECISION GATE PASSED — Controlled 100k scaling authorized.**

---

## 8. Controlled 100k BiLSTM-CRF Result (Seed 42)

* **Sample Manifest:** `experiments/bilstm_crf/manifests/train_100k_seed42.json` (100,000 clean records, strictly nesting all 50,000 records from the 50k sample).
* **Vocabulary:** 94,405 unique tokens (built strictly from the 100k training records).
* **Validation UNK Rate:** **2.5372%** (down from 3.3660% at 50k).
* **Experiment ID:** `bilstm_crf_100k_seed42_9af1d47db429`
* **Best Epoch:** 9 (early stopping triggered at epoch 12).
* **Training Time:** 3,114.71s (~51.9 minutes).
* **Validation Metrics:**
  * **Strict Micro F1:** **0.738188** (+0.024418 vs 100k Frozen CRF baseline; +0.018958 vs 50k Seed 42)
  * **Strict Macro F1:** **0.733398** (+0.024081 vs 100k Frozen CRF baseline)
  * **Strict Precision:** 0.749832
  * **Strict Recall:** 0.726901
  * **PER F1:** **0.7812** (+0.0296 vs CRF)
  * **ORG F1:** **0.6352** (+0.0267 vs CRF)
  * **LOC F1:** **0.7838** (+0.0160 vs CRF)
  * **Token Accuracy:** **0.9336** (up from 0.9254)

---

## 9. Classical CRF vs BiLSTM-CRF Validation Comparison

| Model Architecture | Sample Size | Seed | Strict Micro F1 | Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Token Acc | Delta vs CRF Baseline | Peak RSS | Median Latency | Vocab Size | Val UNK Rate | Model Size |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frozen Classical CRF** | 100,000 | 42 | 0.713770 | 0.709317 | 0.7516 | 0.6085 | 0.7678 | 0.9254 | baseline | 2.658 GiB | 0.508 ms | N/A (features) | 0.00% | 0.77 MiB |
| **BiLSTM-CRF 50k (Seed 7)** | 50,000 | 7 | 0.713626 | 0.708833 | 0.7532 | 0.6050 | 0.7683 | 0.9272 | -0.000144 | 0.871 GiB | 3.422 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (Seed 21)** | 50,000 | 21 | 0.717063 | 0.710939 | 0.7521 | 0.6061 | 0.7747 | 0.9279 | +0.003293 | 1.028 GiB | 3.527 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (Seed 42)** | 50,000 | 42 | 0.719230 | 0.715036 | 0.7587 | 0.6123 | 0.7741 | 0.9284 | +0.005460 | 0.989 GiB | 4.660 ms | 62,099 | 3.37% | 24.60 MiB |
| **BiLSTM-CRF 50k (3-Seed Mean)** | 50,000 | $\mu$ | **0.716639** | **0.711603** | **0.7547** | **0.6078** | **0.7723** | **0.9278** | **+0.002870** | **0.963 GiB** | **3.870 ms** | **62,099** | **3.37%** | **24.60 MiB** |
| **BiLSTM-CRF 100k (Selected)** | **100,000** | **42** | **0.738188** | **0.733398** | **0.7812** | **0.6352** | **0.7838** | **0.9336** | **+0.024418** | **1.238 GiB** | **3.601 ms** | **94,405** | **2.54%** | **36.92 MiB** |

---

## 10. Model Selection Justification

`bilstm_crf_100k_seed42_9af1d47db429` is selected as the primary BiLSTM-CRF validation baseline:
1. **Validation Micro F1:** Achieved **0.738188**, outperforming the 100k classical CRF baseline by **+0.024418** (+2.44 pp) and the 50k neural mean by **+0.021549** (+2.15 pp).
2. **Entity Class Balance:** Substantial gains across all 3 entity classes:
   - `PER`: **0.7812** vs 0.7516 (+2.96 pp)
   - `ORG`: **0.6352** vs 0.6085 (+2.67 pp)
   - `LOC`: **0.7838** vs 0.7678 (+1.60 pp)
3. **Vocabulary & Out-of-Vocabulary Coverage:** Vocabulary expanded to 94,405 words, decreasing validation UNK rate to 2.54%.
4. **Computational Feasibility:** Peak RSS of 1.238 GiB is less than half the classical CRF's peak preparation memory (2.658 GiB).

---

## 11. Replay & Checkpoint Verification

Every trained model checkpoint was independently reloaded and re-evaluated on `validation_clean`:
* **50k Seed 42 Replay:** Saved F1: 0.71922975 | Replayed F1: 0.71922975 ($\Delta = 0.00000000$)
* **50k Seed 7 Replay:** Saved F1: 0.71362591 | Replayed F1: 0.71362591 ($\Delta = 0.00000000$)
* **50k Seed 21 Replay:** Saved F1: 0.71706271 | Replayed F1: 0.71706271 ($\Delta = 0.00000000$)
* **100k Seed 42 Replay:** Saved F1: 0.73818829 | Replayed F1: 0.73818829 ($\Delta = 0.00000000$)

All metric replays matched with 100% bitwise and numerical precision.

---

## 12. Freeze Manifest Verification

The baseline package was sealed using `src/evaluation/freeze_bilstm_crf.py`:
* **Manifest Path:** `reports/bilstm_crf/bilstm_crf_freeze_manifest.json`
* **Selected Model ID:** `bilstm_crf_100k_seed42_9af1d47db429`
* **Frozen Files Count:** 68 files
* **Payload SHA-256:** `758ed03e196c85f31116c072c41256a97e3b0086c7983c28a892aabf860c5abc`
* **Verification Status:** `verify_freeze()` passed with 0 checksum mismatches.

---

## 13. Resource Usage Analysis

| Resource | Classical CRF 100k | BiLSTM-CRF 50k Mean | BiLSTM-CRF 100k |
| :--- | :--- | :--- | :--- |
| **Peak Resident Memory (RSS)** | 2.658 GiB | 0.963 GiB | 1.238 GiB |
| **Training Time** | 38.3s | 2,813.0s (~46.9 min) | 3,114.7s (~51.9 min) |
| **Model Checkpoint Size** | 0.77 MiB | 24.60 MiB | 36.92 MiB |
| **Median Sentence Latency** | 0.508 ms | 3.870 ms | 3.601 ms |
| **P95 Sentence Latency** | 1.220 ms | 4.823 ms | 4.502 ms |

---

## 14. Known Limitations

1. **Word-Level Vocabulary:** Pure word-level embeddings still produce a 2.54% UNK rate on validation data. Subword tokenization (BPE/WordPiece) or character embeddings remain necessary for highly inflected Hindi words.
2. **Inference Latency:** Classical CRF remains ~7x faster in per-sentence inference on CPU (0.51 ms vs 3.60 ms on MPS).
3. **Domain Transfer:** Evaluated exclusively on news crawl data from Naamapadam.
4. **Sealed Test Status:** These models have not been evaluated on `test_clean` or `official_test`.

---

## 15. Next Recommended Milestone

* **Milestone 4: Pretrained Multilingual Transformers (IndicBERT / XLM-RoBERTa)**
  1. Fine-tune `ai4bharat/indic-bert` and `xlm-roberta-base` using the clean training splits.
  2. Leverage subword tokenizers to resolve out-of-vocabulary limitations.
  3. Benchmark contextual embeddings against the frozen 100k BiLSTM-CRF baseline.
