# Remote Smoke Tests

These tests are opt-in checks for external services. They are skipped during
normal `uv run pytest`.

Run the Label Studio upload smoke test with credentials from `.env`:

```bash
uv run --group label-studio pytest tests/smoke --no-cov --run-remote-smoke
```
