# Official Hindi Naamapadam audit

Counts are computed from the official archive. No model evaluation was performed.

| Split | Records | Tokens | Error categories |
|---|---:|---:|---|
| train | 985787 | 22029408 | {'type_mismatch_I': 5513} |
| validation | 13460 | 304282 | {'type_mismatch_I': 74} |
| test | 867 | 19893 | {'orphan_I': 5, 'empty_record': 3, 'type_mismatch_I': 2} |

Normalised text: NFC each token, join with one space, collapse whitespace; case preserved

Entity counts use strict raw BIO. See JSON reports for exact distributions and overlaps.
