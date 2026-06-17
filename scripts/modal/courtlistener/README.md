# CourtListener Modal Backend

This Modal app exposes the reusable CourtListener access API from
`src/mellea_lrc/courtlistener`. The Modal layer only defines deployment details;
routes, cache behavior, rate limiting, and CourtListener response normalization
live in the reusable package.

Modal app name: `courtlistener-access`.

## Secrets

Runtime configuration comes from named Modal secrets:

```env
# Modal secret: courtlistener
COURTLISTENER_API_TOKEN_1=<courtlistener-api-token>

# Modal secret: courtlistener-r2-cache
R2_BUCKET=<bucket-name>
R2_ENDPOINT_URL=<r2-endpoint-url>
AWS_ACCESS_KEY_ID=<r2-access-key>
AWS_SECRET_ACCESS_KEY=<r2-secret-key>
```

Optional environment values supported by the reusable client include
`COURTLISTENER_API_TOKEN_2`, `COURTLISTENER_BASE_URL`, and `R2_PREFIX`.

## Runtime Assumptions

- The service is a general-purpose CourtListener backend, not only citation
  validation.
- Citation validation uses `POST /citation-lookup` with form fields `volume`,
  `reporter`, and `page`.
- The reusable client handles CourtListener token rotation, rate limiting, and
  cache envelopes.

## Deploy

```bash
uv run --group modal modal deploy scripts/modal/courtlistener/server.py
```

## Local Serve

```bash
uv run --group modal modal serve scripts/modal/courtlistener/server.py
```

Useful endpoints:

- `GET /health`
- `GET /dockets/resolve`
- `GET /dockets/{cl_docket_id}`
- `GET /docket-entries/search`
- `GET /recap-documents/search`
- `GET /recap-documents/{recap_document_id}`
- `GET /recap-documents/{recap_document_id}/download-url`
- `GET /courts`
- `GET /courts/{court_id}`
- `GET /search`
- `POST /citation-lookup`
