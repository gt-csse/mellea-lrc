# Remote Smoke Tests

These tests are opt-in checks for deployed services. They are skipped by default
during normal `uv run pytest`.

Pass `--run-remote-smoke` to acknowledge that the tests call external services.

Run deployed Modal service checks with explicit service URLs:

```bash
uv run pytest tests/smoke --no-cov --run-remote-smoke \
  --courtlistener-url https://your-courtlistener-service.modal.run \
  --label-studio-url https://your-label-studio-service.modal.run
```

Run the Label Studio upload/extraction smoke test with credentials from `.env`:

```bash
uv run --group label-studio pytest tests/smoke/test_label_studio_upload_remote.py --no-cov --run-remote-smoke
```

Run the OpenAI-compatible LLM smoke test with credentials from `.env` (requires `uv sync --group llm`):

```bash
uv run --group llm pytest tests/smoke/test_llm_provider_remote.py --no-cov --run-remote-smoke
```

Use `--remote-timeout <seconds>` to adjust HTTP timeouts for cold starts.
