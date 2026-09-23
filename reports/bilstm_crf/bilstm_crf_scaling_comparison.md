# IndicNewsNER BiLSTM-CRF Scaling Comparison

Evaluation strictly on `validation_clean` (12,896 records, 289,695 tokens).
Frozen Classical CRF 100k Baseline Validation Micro F1: **0.713770** (Macro F1: **0.709317**).

| Sample Size | Vocab Size | Val UNK Rate | Best Epoch | Train Time (s) | Peak RSS (GiB) | Val Strict Micro F1 | Val Strict Macro F1 | PER F1 | ORG F1 | LOC F1 | Delta vs CRF Val F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1,000 | 5,879 | 17.14% | 14 | 924.99 | 0.684 | **0.367977** | 0.365004 | 0.3871 | 0.2501 | 0.4579 | -0.345793 |
| 10,000 | 23,894 | 6.48% | 13 | 1187.02 | 0.787 | **0.618866** | 0.612113 | 0.6748 | 0.4797 | 0.6819 | -0.094904 |
| 50,000 | 62,099 | 3.37% | 14 | 3014.95 | 1.028 | **0.717063** | 0.710939 | 0.7521 | 0.6061 | 0.7747 | +0.003293 |
| 50,000 | 62,099 | 3.37% | 14 | 2461.72 | 0.989 | **0.719230** | 0.715036 | 0.7587 | 0.6123 | 0.7741 | +0.005460 |
| 50,000 | 62,099 | 3.37% | 14 | 2962.34 | 0.871 | **0.713626** | 0.708833 | 0.7532 | 0.6050 | 0.7683 | -0.000144 |
| 100,000 | 94,405 | 2.54% | 9 | 3949.68 | 1.238 | **0.738188** | 0.733398 | 0.7812 | 0.6352 | 0.7838 | +0.024419 |

## Baseline Comparison & Analysis
- **Frozen CRF 100k Baseline Validation Micro F1:** 0.713770
- **Research Boundary:** Evaluated strictly on `validation_clean`. No test data accessed.
