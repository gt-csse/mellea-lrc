# Live LLM Evaluations

These evaluations call the configured provider and can vary because LLM output
is inherently non-deterministic. A failed evaluation does not by itself prove
that the implementation or model is wrong, but it is a useful signal to review
the result and determine whether behavior has meaningfully changed.

They are deliberately excluded from normal test and CI runs. The explicit
option accounts for provider availability, cost, and output randomness.
Configure the `MELLEA_LRC_LLM_*` values in `.env`, then run:

```bash
uv run pytest tests/llm --no-cov --run-llm-evaluations
```

The assertions provide concrete examples for periodic review, not a deterministic
CI gate.
