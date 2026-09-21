# IndicNewsNER BiLSTM-CRF Model Card: bilstm_crf_010k_seed42_6ee37e2676b5

## Summary
- **Model Type:** PyTorch BiLSTM-CRF (1 layer, 128 hidden dim per direction, 100 word embedding dim).
- **Training Sample:** 10,000 clean training records from `naamapadam_hi_crf_v1`.
- **Vocabulary Size:** 23,894 tokens (built strictly from training sample).
- **Validation UNK Rate:** 6.4834%.
- **Device Used:** mps.
- **Best Epoch:** 13 (Early stopping patience: 3).
- **Total Training Time:** 1187.02 seconds.
- **Model Size:** 10.03 MiB (10,512,309 bytes).

## Validation Strict Metrics (`validation_clean`, 12,896 records)
- **Strict Micro F1:** 0.61886618
- **Strict Macro F1:** 0.61211255
- **Precision:** 0.70331977
- **Recall:** 0.55252036
- **PER F1:** 0.67476852
- **ORG F1:** 0.47968056
- **LOC F1:** 0.68188859
- **Token Accuracy:** 0.90687102
- **Median Latency:** 4.617 ms / sentence
- **P95 Latency:** 5.508 ms / sentence

## Research Policy
Evaluated strictly on `validation_clean`. No test split was accessed.
