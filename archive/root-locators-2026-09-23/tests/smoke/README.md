# Remote Smoke Tests

These tests are opt-in checks for external services. They are skipped during
normal `uv run pytest`.

Run them with credentials from `.env`:

```bash
uv run pytest tests/smoke --no-cov --run-remote-smoke
```
