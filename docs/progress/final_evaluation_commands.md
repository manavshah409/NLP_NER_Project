# Milestone 2C execution instructions

Executed in this order from the project root:

```sh
.venv/bin/python -m pytest --junitxml=reports/environment/pytest_before_2c.xml
.venv/bin/python -m pytest --junitxml=reports/environment/pytest_pre_final_evaluation.xml
.venv/bin/python -m src.evaluation.freeze_baseline
.venv/bin/python -m src.evaluation.final_crf --split test_clean
.venv/bin/python -m src.evaluation.final_crf --split official_test
.venv/bin/python -m pytest --junitxml=reports/environment/pytest_after_2c.xml
.venv/bin/python -m scripts.report_final_crf
```

The first suite had 53 tests; the expanded pre/post suites each passed all 66.
The freeze command independently reconstructed all 100k indices and hashed the
artifacts. Each final evaluator ran once. Do not repeat final inference or remove
its start markers. An interrupted attempt requires review, not an automatic retry.
Report generation may be repeated from saved metrics without further inference.

For read-only freeze verification:

```sh
.venv/bin/python -c 'from src.evaluation.freeze_baseline import verify_freeze; verify_freeze(); print("Freeze verified")'
```

Test predictions and restricted examples are excluded from Git. AGENTS.md forbids
using those examples in later model selection. No new model training is authorised
by these commands. The development evaluator still rejects test splits.

The commit is prepared on codex/crf-baseline-2c. No remote was configured during
initial inspection, so pushing requires the user's GitHub repository URL.
