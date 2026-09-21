# Derived Hindi dataset checkpoint

| Split | Records | Tokens |
|---|---:|---:|
| train_clean | 963174 | 21483205 |
| validation_clean | 12896 | 289695 |
| test_clean | 506 | 12388 |

Order: BIO normalise; quarantine test empties; remove overlap with earlier clean splits; exact dedup retaining lowest source index

Repairs: {'type_mismatch_I': 5589, 'orphan_I': 5}

Repairs are counted before deduplication, leakage removal and quarantine.

Removals: {'duplicate': 22619, 'leakage': 916, 'quarantine': 3}

Test overlap with clean predecessors: {'validation': 341, 'train': 16}

A 'both' overlap with clean train and clean validation is impossible after validation decontamination.

Checksum method: SHA256 of UTF-8 canonical JSON mapping relative POSIX file paths to SHA256; sorted keys, no spaces, ensure_ascii=False; manifest stored outside root

Integrity: {'raw_checksum_preserved': True, 'derived_checksum_verified': True, 'zero_invalid_BIO': True, 'zero_cross_split_overlap': True, 'no_clean_empty_records': True, 'source_content_verified': True, 'official_test_preserved': True, 'zero_exact_duplicates': True, 'complete_source_index_accounting': True}
