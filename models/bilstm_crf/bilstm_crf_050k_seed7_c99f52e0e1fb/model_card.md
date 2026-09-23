# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_050k_seed7_c99f52e0e1fb

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 50,000 clean training records from `naamapadam_hi_crf_v1` (Seed: 7).
- **Vocabulary Size:** 62,099 tokens (built strictly from training sample).
- **Validation UNK Rate:** 3.3660%.
- **Device Used:** mps.
- **Best Epoch:** 14 (Early stopping patience: 3, Reason: maximum_epochs_completed).
- **Total Training Time:** 2962.34 seconds.
- **Model Size:** 24.60 MiB (25,794,293 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.71362591
- **Strict Macro F1:** 0.70883323
- **Precision:** 0.74178839
- **Recall:** 0.68752362
- **PER F1:** 0.75323023
- **ORG F1:** 0.60499822
- **LOC F1:** 0.76827123
- **Token Accuracy:** 0.92721656
- **Median Latency:** 3.422 ms / sentence
- **P95 Latency:** 4.300 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
