# Milestones 1 and 2A checkpoint

All benchmark values below are independently computed from downloaded official files.

| Split | Raw records | Raw tokens | Clean records | Clean tokens |
|---|---:|---:|---:|---:|
| train | 985787 | 22029408 | 963174 | 21483205 |
| validation | 13460 | 304282 | 12896 | 289695 |
| test | 867 | 19893 | 506 | 12388 |

Repairs: {'type_mismatch_I': 5589, 'orphan_I': 5}; counted before any removal.

Removals: {'duplicate': 22619, 'leakage': 916, 'quarantine': 3}.

Quarantine indices: [156, 272, 331].

Test overlap breakdown against clean predecessors: {'train_only': 16, 'validation_only': 341, 'both': 0}.

All clean split record counts, raw token counts and reported removal/repair totals match the references. The three official test empty records are preserved in official_test and separately quarantined from test_clean.

Current test suite: {'tests': 53, 'failures': 0, 'errors': 0, 'skipped': 0, 'passed': 53}. The earlier 54-test claim does not apply to this new implementation.

All nine integrity assertions pass. Repeated reconstruction produces byte-identical clean split files. Strict raw BIO and derived CoNLL repair are separate. Each removal and repair is traceable.

Checksum: `ff9854a573973de7247660851f614b530395981580aaebf28dde9485e1c1f581`.

SHA256 of UTF-8 canonical JSON mapping relative POSIX file paths to SHA256; sorted keys, no spaces, ensure_ascii=False; manifest stored outside root

The reference checksum cannot be reproduced as the original implementation is absent. This acquisition includes timestamped provenance, so its combined checksum has a different scope; source-file hashes provide the comparable identity.

Known limitations: license declarations disagree upstream; no model or domain-shift conclusion follows from a data checkpoint. Raw counts of absent error categories are zero. Data-only integrity checks inspect test labels; no test predictions were generated.

Files created are listed in the machine-readable checkpoint. Commands and environment notes are in implementation_log.md.
