# Optional English NER and Hindi → English translation

The Streamlit sidebar includes **English and Translation**. English text goes
directly to an English NER model. Hindi translation is a separate action and does
not translate entity offsets or feed translated text automatically into another
model. Only Hindi → English is supported, as requested.

## Isolated installation

Use Python 3.12 in a separate environment; do not install these requirements into
the frozen Hindi `.venv`:

```sh
python3.12 -m venv .venv-language
.venv-language/bin/python -m pip install -r requirements-language.lock.txt
.venv-language/bin/python scripts/setup_language_models.py
.venv-language/bin/python -m pip check
```

The setup command downloads only eight allowlisted runtime files from revision
`a7d96a16729f812578bd55e7366147beda625d86` of
`Helsinki-NLP/opus-mt-hi-en`, checks the committed manifest's SHA-256 values and
refuses altered files. The spaCy 3.8.0 model comes from its official release wheel.
No project datasets are included. Runtime never downloads missing artifacts.
Missing files or failed checks require restoring the pinned files/environment.

Start the main app using its existing `.venv`, then select the sidebar page.
Each request starts a separate offline Python worker; startup overhead is included
in displayed time. This trades latency for isolation and bounded lifetime. Workers
receive input over stdin, return JSON over stdout, and are killed after 120 seconds.
They install the project's development guard before model reads. User input is not
saved to files or sent to an external API. Offline flags disable Hugging Face
network use; this is not a general OS network sandbox. Explicit downloads save results.

## Limits and interpretation

- Both tools accept 1–5,000 Unicode code points of non-empty input.
- Translation accepts at most 512 model tokens; oversize input is rejected,
  never silently truncated. Output-limit exhaustion is also rejected.
- English uses spaCy-native tokens with exact original character offsets.
  PERSON→PER, ORG→ORG, GPE/LOC→LOC; all other labels are excluded. Original labels
  remain in JSON for traceability. Confidence is not fabricated.
- Translation has no entity alignment. Source and translation are separate strings.
  It may change names, places, facts and meaning. In the original demonstration,
  “मीरा शर्मा ने जयपुर में पुस्तक प्रदर्शनी देखी।” became
  “Eva shame saw the book exhibition in Mumbai.” This is an observed failure,
  not a correct reference translation. Review every output against its source.
- No project English F1 or translation-quality score has been measured. These
  pretrained tools are not comparable to the controlled Hindi validation experiment.

The English model is MIT-licensed according to its official metadata. The pinned
translation model card declares Apache-2.0, while the upstream OPUS project describes
other licensing; provenance is recorded without resolving that discrepancy.
Downloaded weights remain local and ignored; no model redistribution is included.

Sources: [spaCy English models](https://spacy.io/models/en),
[pinned translation model card](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/blob/a7d96a16729f812578bd55e7366147beda625d86/README.md),
[upstream OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT).
