# Five-minute demonstration — Hindi model comparison

1. **0:00–0:45 — Problem.** Explain Hindi person, organisation and location
   extraction. Introduce the classical CRF and BiLSTM-CRF as separate frozen
   100k baselines. State that current-news performance is unverified.
2. **0:45–1:45 — Input.** Choose Compare Both and the original Person and place
   example. Extract entities. Point out preserved Hindi text, highlights and the
   CPU/MPS device labels. Neither model is retrained during this demonstration.
3. **1:45–2:45 — Inspect.** Show each table and exclusive character offsets.
   Explain repeated mentions and UNK diagnostics. Agreement is not accuracy.
   Neural confidence is intentionally absent; CRF marginals are not correctness.
4. **2:45–3:30 — Export.** Download a model's JSON and CSV, then the comparison
   JSON. Explain that exports retain all entities even when the table is filtered.
5. **3:30–4:15 — Evidence.** Open model details. Compare validation micro F1:
   71.38% versus 73.82%. Do not compare neural validation with classical clean-test
   results. No neural final-test result exists.
6. **4:15–5:00 — Limits and next work.** Explain original-text limits and model
   checksum failure behavior. Show the optional English and Translation sidebar
   page if time permits, explicitly identifying its external pretrained models
   and experimental translation quality. No quality score is claimed for it.

Start with `.venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false`.
Both original Hindi artifacts are required; a fresh clone does not contain weights.
See LANGUAGE_TOOLS.md for the separate optional installation.
