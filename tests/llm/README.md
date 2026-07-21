# Live LLM Evaluations

These tests call the configured provider and are excluded from normal test and
CI runs. Configure the `MELLEA_LRC_LLM_*` values in `.env`, then run:

```bash
uv run pytest tests/llm --no-cov --run-llm-evaluations
```

Their assertions intentionally surface model-behavior drift.
