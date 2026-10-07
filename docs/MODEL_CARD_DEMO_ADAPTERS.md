# Demonstration model card addendum

This addendum versions inference behavior without editing frozen model cards.
Hindi baseline identities, metrics and checksum boundaries are in
`progress/MILESTONE_3C_DUAL_MODEL_DEMO.md`. Both Hindi models retain PER/ORG/LOC,
their original parameters and training environments. No new final evaluation exists.

The optional English model is spaCy `en_core_web_sm` 3.8.0, running on CPU in
`.venv-language`. It is an external pretrained NER pipeline, not a project-trained
baseline. Its source label mapping and exclusions are documented in
`demo/LANGUAGE_TOOLS.md`; English tokens are independent of Hindi tokens.

The optional translator is `Helsinki-NLP/opus-mt-hi-en`, pinned at
`a7d96a16729f812578bd55e7366147beda625d86`. It runs deterministic beam search
(four beams, no sampling) on CPU, with no gradients and bounded lengths. It is
not fine-tuned. It has demonstrated name/place corruption and must be treated as
experimental. No alignment or factual fidelity guarantee is made. Both new tools
have no project validation/test quality metric. Runtime versions and file hashes
are in `configs/language_tools_manifest.json` and the separate dependency lock.

Intended use: undergraduate local demonstrations and controlled future development.
Not intended for unattended publication, factual verification, or decisions about
people. Current-news performance remains independently unverified for every model.
