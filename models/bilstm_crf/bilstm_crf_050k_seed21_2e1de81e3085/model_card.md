# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_050k_seed21_2e1de81e3085

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 50,000 clean training records from `naamapadam_hi_crf_v1` (Seed: 21).
- **Vocabulary Size:** 62,099 tokens (built strictly from training sample).
- **Validation UNK Rate:** 3.3660%.
- **Device Used:** mps.
- **Best Epoch:** 14 (Early stopping patience: 3, Reason: maximum_epochs_completed).
- **Total Training Time:** 3014.95 seconds.
- **Model Size:** 24.60 MiB (25,794,293 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.71706271
- **Strict Macro F1:** 0.71093855
- **Precision:** 0.72185661
- **Recall:** 0.71233206
- **PER F1:** 0.75206888
- **ORG F1:** 0.60607490
- **LOC F1:** 0.77467188
- **Token Accuracy:** 0.92791039
- **Median Latency:** 3.527 ms / sentence
- **P95 Latency:** 4.522 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
