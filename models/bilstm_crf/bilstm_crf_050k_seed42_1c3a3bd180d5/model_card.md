# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_050k_seed42_1c3a3bd180d5

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 50,000 clean training records from `naamapadam_hi_crf_v1`.
- **Vocabulary Size:** 62,099 tokens (built strictly from training sample).
- **Validation UNK Rate:** 3.3660%.
- **Device Used:** mps.
- **Best Epoch:** 14 (Early stopping patience: 3).
- **Total Training Time:** 2461.72 seconds.
- **Model Size:** 24.60 MiB (25,794,293 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.71922975
- **Strict Macro F1:** 0.71503602
- **Precision:** 0.74016435
- **Recall:** 0.69944679
- **PER F1:** 0.75873845
- **ORG F1:** 0.61229916
- **LOC F1:** 0.77407045
- **Token Accuracy:** 0.92841437
- **Median Latency:** 4.660 ms / sentence
- **P95 Latency:** 5.646 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
