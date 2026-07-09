# Remote and LLM Sanity Tests

These tests are opt-in checks for deployed services. They are skipped by default
during normal `uv run pytest`.

Pass `--run-remote-smoke` to acknowledge that the tests call external services.

Run deployed Modal service checks with explicit service URLs:

```bash
uv run pytest tests/smoke --no-cov --run-remote-smoke \
  --courtlistener-url https://your-courtlistener-service.modal.run
```

Run the Label Studio upload/extraction smoke test with credentials from `.env`:

```bash
uv run --group label-studio pytest tests/smoke/test_label_studio_upload_remote.py --no-cov --run-remote-smoke
```

## LLM remote sanity

The `llm_remote_sanity` suite covers live checks against the configured LLM
endpoint and LLM-backed workflows. These tests are intentionally non-CI sanity
checks: they can validate prompt shape, structured-output behavior, retry
observability, and endpoint compatibility, but they depend on the configured
remote model and may vary with model behavior.

Run the full LLM remote sanity suite with credentials from `.env`:

```bash
uv run --group pipeline pytest tests/smoke -m llm_remote_sanity --no-cov --run-llm-remote-sanity
```

Run only the OpenAI-compatible LLM endpoint sanity test:

```bash
uv run --group llm pytest tests/smoke/test_llm_api_remote.py --no-cov --run-llm-remote-sanity
```

Run only the live case-name preparation sanity checks:

```bash
uv run --group pipeline pytest tests/smoke/test_case_name_prepare_remote.py --no-cov --run-llm-remote-sanity
```

Use `--remote-timeout <seconds>` to adjust HTTP timeouts for cold starts.
