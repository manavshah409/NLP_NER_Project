# Before and after dataset audit

Valid BIO tag positions includes O and B as well as valid I. Zero clean duplicate/overlap counts are supported by successful exhaustive integrity assertions. Structural raw failures are absent.

| Split | Raw PER / ORG / LOC spans | Clean PER / ORG / LOC spans | Raw exact duplicate copies | Clean exact duplicate copies |
|---|---|---|---:|---:|
| train | 767003 / 686388 / 731183 | 751157 / 672911 / 711267 | 22613 | 0 |
| validation | 10549 / 9735 / 10209 | 10097 / 9329 / 9677 | 5 | 0 |
| test | 788 / 521 / 613 | 547 / 287 / 332 | 3 | 0 |

Raw entity spans follow strict B-only reconstruction; clean spans reflect explicit CoNLL repairs plus removals. These counts should not be interpreted as model predictions.
