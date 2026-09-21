# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_001k_seed42_14ca16a6ddf0

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 1,000 clean training records from `naamapadam_hi_crf_v1`.
- **Vocabulary Size:** 5,879 tokens (built strictly from training sample).
- **Validation UNK Rate:** 17.1439%.
- **Device Used:** mps.
- **Best Epoch:** 14 (Early stopping patience: 3).
- **Total Training Time:** 924.99 seconds.
- **Model Size:** 3.15 MiB (3,306,293 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.36797699
- **Strict Macro F1:** 0.36500379
- **Precision:** 0.47839235
- **Recall:** 0.29897261
- **PER F1:** 0.38708381
- **ORG F1:** 0.25006505
- **LOC F1:** 0.45786252
- **Token Accuracy:** 0.85416731
- **Median Latency:** 4.603 ms / sentence
- **P95 Latency:** 5.601 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
