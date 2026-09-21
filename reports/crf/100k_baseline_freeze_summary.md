# 100k classical baseline freeze

Experiment: `crf_100k_seed42_24dd05e84a26`.

Frozen 51 files, including the model binary, sample, configurations, labels, environment, source/evaluation implementation and dataset files.

The 100,000 seed-42 source indices were independently reproduced from train_clean and their full statistics matched the saved training artifacts. No validation or test records were sampled.

All file SHA256 checks and the freeze envelope checksum pass. No Git commit existed at freeze; source/file hashes provide exact identity. The later milestone commit records the manifest without changing it.

Strict B-only typed spans, exact boundaries, fixed 3-class macro F1. Preserve raw official BIO; include empty official records with empty predictions. No test-time repair.
