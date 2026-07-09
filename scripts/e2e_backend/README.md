# E2E Backend

Command index: [../README.md](../README.md).

This backend exposes the assembled mellea-lrc preprocessing, extraction,
retrieval, and assessment pipeline for the custom frontend review UI. It can
run locally as a standalone FastAPI app or be wrapped by Modal for deployment.

Modal app name: `mellea-lrc-prototype`.

## Secrets

Runtime configuration comes from named Modal secrets:

```env
# Modal secret: cl-access-modal
CL_ACCESS_MODAL_URL=<courtlistener-access-service-url>

# Modal secret: mellea-assessment
MELLEA_LRC_LLM_TEMPERATURE=0
MELLEA_LRC_LLM_MODEL=<model-id>
MELLEA_LRC_LLM_API_BASE=<openai-compatible-api-base>
MELLEA_LRC_LLM_API_KEY=<api-key>
MELLEA_LRC_LLM_RESPONSE_FORMAT=json_schema
MELLEA_LRC_LLM_CERT_REQUIRED=true

# Use json_object when the endpoint does not accept JSON Schema response formats.
# Disable certificate verification only for an endpoint that requires this workaround.
```

Retrieval uses the CourtListener Modal backend through `CL_ACCESS_MODAL_URL`.
Assessment uses the OpenAI-compatible API bound by the `mellea-assessment` secret.
Label Studio is no longer integrated as a prediction-retrieval backend: upload
scripts live under `scripts/label_studio/` and are independent of this service.

## Runtime Assumptions

- The assembled backend API is defined by `pipeline.E2EBackend`.
- The custom frontend uses staged review endpoints: extract first, then retrieve
  the existing review payload, then assess the retrieved payload.
- Docling is initialized lazily on the first PDF request.

## Local Test Corpus Preprocessing

CourtListener plain-text exports are no longer the Layer 2 source for local tests.
Regenerate text from the PDFs under `local/test_data/pdfs/` with:

```bash
uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs
```

See [Preprocessing Development](../../docs/development/Preprocessing%20Development.md) for
rationale, defaults (Docling + Tesseract CLI OCR), and the evaluation drill.

## Deploy

```bash
uv run --group modal modal deploy scripts/modal/e2e_backend/server.py
```

## Local Serve

```bash
uv run --group modal modal serve scripts/modal/e2e_backend/server.py
```

For local frontend E2E testing without Modal:

```bash
uv run --group modal fastapi run \
  scripts/e2e_backend/local_server.py \
  --host 127.0.0.1 \
  --port 8011

cd frontend
MELLEA_E2E_BACKEND_URL=http://127.0.0.1:8011 npm run dev -- \
  --hostname 127.0.0.1 \
  --port 3000
```

Useful endpoints:

- `GET /health`: service probe
- `POST /api/extract-text`: frontend text extraction stage
- `POST /api/extract-document`: frontend document extraction stage
- `POST /api/review-snapshot`: frontend dev loader for serialized interface artifacts
- `POST /api/retrieve-review`: frontend retrieval stage for an existing review payload
- `POST /api/assess-review`: frontend Mellea assessment stage for an existing review payload

## Snapshot Loading

The frontend can load a JSON snapshot produced by the neutral serializers for:
`PreprocessedDocument`, `ExtractedDocument`, `RetrievedDocument`, or
`AssessedDocument`. Use the `Load snapshot` button in the input panel and choose
the artifact JSON file. The backend deserializes the artifact and returns the same
review payload shape used by the normal staged workflow.

## Snapshot Regeneration

Regenerate the same strict stage snapshots as the development notebook without
running cells manually. Full runs also write `citation_nodes.json`, a
citation-level projection between extraction and jurisdiction inference.

```bash
# One configured fixture
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file 3 --phase assessment

# The configured bookmarked text
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file bookmarked --phase retrieval

# Stop after writing citation_nodes.json
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file bookmarked --phase citation_nodes

# The inclusive numeric text-fixture range configured in snapshot_corpus.py
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus
```

The CLI intentionally accepts only `--file` and `--phase`. Supported phases are
`preprocessed`, `extraction`, `citation_nodes`, `inferred`, `retrieval`, and
`assessment`. Configure the env, test-data, bookmark, and snapshot paths,
numeric batch range, and Mellea concurrency through `SnapshotConfig`/`CONFIG` in
`snapshot_corpus.py`.
