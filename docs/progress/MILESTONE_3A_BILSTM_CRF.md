# Milestone 3A: BiLSTM-CRF Architecture & Controlled Pilot Training

Fourth-year undergraduate NLP project | 21 September 2026

## 1. Objective

Milestone 3A implements a reproducible PyTorch-based **BiLSTM-CRF** neural sequence tagger for Hindi Named Entity Recognition and evaluates it strictly on `validation_clean` to determine whether a neural sequence model improves upon the frozen classical CRF baseline validation micro F1 of **0.713769728** while operating safely within Apple Silicon computational limits.

Under the project's sealed-evaluation policy, **no test split (`test_clean`, `official_test`), restricted test predictions, or benchmark examples were accessed or evaluated**.

---

## 2. Computational Environment

* **Target Hardware:** MacBook Pro Apple Silicon (`arm64`, 24.0 GiB physical RAM)
* **OS:** macOS 26.6.2 (Darwin 25.3.0)
* **Python:** 3.14.7
* **PyTorch Version:** 2.14.0
* **Acceleration Device:** `mps` (Metal Performance Shaders) verified and active
* **Peak Resident Memory during Training:** 0.989 GiB (well within the conservative 12.0 GiB budget)

---

## 3. Data Isolation & Sampling

* **Dataset Version:** `naamapadam_hi_crf_v1`
* **Permitted Development Splits:**
  * Training: `data/derived/naamapadam_hi_crf_v1/train_clean.jsonl` (963,174 clean records)
  * Validation: `data/derived/naamapadam_hi_crf_v1/validation_clean.jsonl` (12,896 clean records)
* **Sealed Data Boundary:** `test_clean.jsonl`, `official_test.jsonl`, and `reports/crf/100k_final/` are strictly blocked via `src/development_guard.py`.
* **BIO Label Schema (7 tags):** `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`.
* **Deterministic Nested Sampling:** Seed 42 with entity-coverage anchors for `PER`, `ORG`, and `LOC`:
  * **1k Smoke:** `experiments/bilstm_crf/manifests/train_001k_seed42.json` (1,000 records)
  * **10k Pilot A:** `experiments/bilstm_crf/manifests/train_010k_seed42.json` (10,000 records)
  * **50k Pilot B:** `experiments/bilstm_crf/manifests/train_050k_seed42.json` (50,000 records)

---

## 4. Architecture & Implementation

### 4.1 PyTorch BiLSTM-CRF Architecture
* **Word Embeddings:** Dimension 100, initialized randomly, padding index 0.
* **Dropout:** 0.3 applied to embeddings and LSTM outputs.
* **Encoder:** 1-layer Bidirectional LSTM (`hidden_dim = 128` per direction, producing 256-dimensional hidden state).
* **Emission Projection:** Linear layer `(256 -> 7)`.
* **Decoder:** Trainable `LinearChainCRF` layer supporting start/end transitions, sequence masking, forward log-partition $\log Z(\mathbf{x})$, gold sequence scoring, and Viterbi decoding.

### 4.2 Training-Only Vocabulary
* Built **strictly from the sampled training records**.
* Special tokens: `<PAD>` (ID 0), `<UNK>` (ID 1).
* Preserves raw Hindi Unicode (no lowercasing or transliteration).

### 4.3 Training Parameters
* **Optimizer:** AdamW ($lr = 0.001$, weight decay = $0.01$)
* **Batch Size:** 32 (variable-length batching with boolean mask)
* **Gradient Clipping:** Max norm $5.0$
* **Max Epochs:** 15 (early stopping patience 3 on validation strict micro F1)

---

## 5. Verification & Test Suite

All **122 tests passed** (100% pass rate, 0 failures, 0 errors, 0 skips) in 3.01s:
* 110 baseline tests preserved intact (CRF models, data integrity, demo app, development guard).
* 12 new unit and integration tests covering:
  * Training-only vocabulary construction and UNK rate calculation.
  * Rejection of non-training splits for vocabulary.
  * Padded sequence collation and mask invariance.
  * CRF log-partition and Viterbi decoding verified mathematically against brute-force path enumeration.
  * Model forward loss and backward gradient flow across all parameters.
  * Checkpoint reload and deterministic inference.
  * Rejection of test splits by the neural evaluator.

---

## 6. Experimental Results & Validation Scaling

All evaluations conducted strictly on `validation_clean` (12,896 records, 289,695 tokens):

| Experiment | Sample Size | Vocab Size | Val UNK Rate | Best Epoch | Train Time (s) | Peak RSS (GiB) | Val Strict Micro F1 | Val Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Token Accuracy | Delta vs CRF Val Baseline |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bilstm_crf_001k` | 1,000 | 5,879 | 17.14% | 14 | 924.99 | 0.684 | 0.367977 | 0.365004 | 0.3871 | 0.2501 | 0.4579 | 0.8542 | -0.345793 |
| `bilstm_crf_010k` | 10,000 | 23,894 | 6.48% | 13 | 1187.02 | 0.787 | 0.618866 | 0.612113 | 0.6748 | 0.4797 | 0.6819 | 0.9069 | -0.094904 |
| `bilstm_crf_050k` | 50,000 | 62,099 | 3.37% | 14 | 2461.72 | 0.989 | **0.719230** | **0.715036** | **0.7587** | **0.6123** | **0.7741** | **0.9284** | **+0.005460** |

---

## 7. Comparison with Frozen Classical CRF Baseline

| Model Architecture | Training Size | Val Strict Micro F1 | Val Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Val Token Acc | Peak Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical CRF (1k)** | 1,000 | 0.518881 | 0.515804 | 0.5369 | 0.3957 | 0.6148 | 0.8770 | 0.109 GiB |
| **BiLSTM-CRF (1k)** | 1,000 | 0.367977 | 0.365004 | 0.3871 | 0.2501 | 0.4579 | 0.8542 | 0.684 GiB |
| **Classical CRF (10k)** | 10,000 | 0.638999 | 0.635749 | 0.6698 | 0.5312 | 0.7063 | 0.9099 | 0.419 GiB |
| **BiLSTM-CRF (10k)** | 10,000 | 0.618866 | 0.612113 | 0.6748 | 0.4797 | 0.6819 | 0.9069 | 0.787 GiB |
| **Classical CRF (50k)** | 50,000 | 0.695146 | 0.690853 | 0.7303 | 0.5900 | 0.7523 | 0.9213 | 1.488 GiB |
| **Classical CRF (100k Frozen)** | 100,000 | 0.713770 | 0.709317 | 0.7516 | 0.6085 | 0.7678 | 0.9254 | 2.658 GiB |
| **BiLSTM-CRF (50k Pilot B)** | **50,000** | **0.719230** | **0.715036** | **0.7587** | **0.6123** | **0.7741** | **0.9284** | **0.989 GiB** |

### Key Analysis:
1. **Low-Resource Regime (1k–10k):** The hand-engineered features of the classical CRF (prefix, suffix, shape) outperformed randomly initialized embeddings when training data was sparse (CRF 0.6390 vs BiLSTM-CRF 0.6189 at 10k).
2. **Medium-Resource Scaling (50k):** With sufficient sequence data (50k records, 1.12M tokens), the BiLSTM-CRF learned rich contextual representations, achieving **0.719230 validation micro F1**, outperforming:
   - The 50k CRF baseline (0.695146) by **+2.41 percentage points**.
   - The 100k Frozen CRF baseline (0.713770) by **+0.55 percentage points**, using half the training records and less peak memory (0.989 GiB vs 2.658 GiB).
3. **Class-Wise Gains:** The 50k BiLSTM-CRF improved across all three entity types compared to the 100k classical baseline:
   - `PER`: **0.7587** vs 0.7516 (+0.71 pp)
   - `ORG`: **0.6123** vs 0.6085 (+0.38 pp)
   - `LOC`: **0.7741** vs 0.7678 (+0.63 pp)

---

## 8. Limitations

1. **Unseen Words & Morphology:** Word-level embeddings suffer when tokens are out-of-vocabulary (UNK rate was 3.37% at 50k). Subword/character-level representations are needed to better model rich Hindi inflections.
2. **Single Seed:** Experiments were conducted with seed 42 without multi-run variance bounds.
3. **Domain Coverage:** Naamapadam crawl text does not guarantee domain adaptation to breaking real-time news.
4. **Research Boundary:** No test splits evaluated. Final test evaluation remains sealed for future milestones.

---

## 9. Recommendations for Next Steps

* **Milestone 3B (Approved Scaling):**
  1. Train the 100k BiLSTM-CRF configuration to establish the full neural baseline scaling curve.
  2. Implement a Character CNN/LSTM representation layer to address out-of-vocabulary words in morphologically rich Hindi.
* **Milestone 4:** Benchmark pretrained multilingual transformer models (`ai4bharat/indic-bert` / `xlm-roberta-base`).
