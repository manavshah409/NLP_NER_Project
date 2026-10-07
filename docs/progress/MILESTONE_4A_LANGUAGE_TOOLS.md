# Milestone 4A — initial English and translation integration

Authorized and implemented locally on 2026-10-07 after Milestone 3C verification.
Scope: external pretrained English NER plus Hindi → English only. No training,
tuning, IndicBERT, reverse translation or final benchmark evaluation was performed.

The implementation adds a separate Streamlit page, a bounded subprocess bridge,
an offline guarded worker, exact model-file checksums, an isolated Python 3.12
dependency lock and a pinned translation restoration script. Frozen Hindi files
and environment remain unchanged. The confirmed remote remains
https://github.com/manavshah409/NLP_NER_Project on branch main; this work is prepared for a reviewed publication checkpoint on main.
No new release tag is included.

English smoke inference found Meera Sharma/PER, London/LOC and Microsoft/ORG in
an authored sentence and preserved exact offsets. Translation smoke inference
produced non-empty English but corrupted a name and location. This failure is
explicitly shown in LANGUAGE_TOOLS.md and the UI warns prominently. The initial
integration is functional, not quality-certified. No dataset or restricted
evaluation examples were opened. Final checks are in MILESTONE_3C_VERIFICATION.md.

Next: create an independently authored English/news and translation development
set with clear entity and name-preservation criteria; human-review translations;
compare candidate translation models within a separately versioned environment.
Do not use sealed Hindi test artifacts for this work. A new final evaluation needs
explicit approval. Avoid presenting the current translator as reliable news translation.
