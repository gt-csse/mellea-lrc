# Mellea + LiteLLM MWE

This standalone example asks one question through Mellea using an
OpenAI-compatible LiteLLM endpoint. It demonstrates both Mellea execution paths:

- `MelleaSession.instruct(...)` with a synchronous `httpx.Client`.
- `MelleaSession.ainstruct(...)` with an asynchronous `httpx.AsyncClient`.

The small backend subclass is needed because the OpenAI SDK requires different
HTTP client types for its synchronous and asynchronous clients. Both clients use
the same certificate-verification setting.

## Run

Copy `.env.example` to `.env` and provide the endpoint values. Then run:

```bash
uv run --env-file .env python mwe.py --mode both
```

Run only one path when needed:

```bash
uv run --env-file .env python mwe.py --mode sync
uv run --env-file .env python mwe.py --mode async
```

Certificate verification defaults to `true`. Set
`MELLEA_VERIFY_CERTIFICATES=false` only when the endpoint requires that TLS
workaround.
