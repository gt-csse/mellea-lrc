# E2E Backend

This backend exposes the assembled mellea-lrc preprocessing, extraction,
validation, assessment, and prediction pipeline. It can run locally as a
standalone FastAPI app or be wrapped by Modal for deployment.

Modal app name: `mellea-lrc-prototype`.

## Secrets

Runtime configuration comes from named Modal secrets:

```env
# Modal secret: label-studio
LS_URL=https://your-label-studio-instance.com
LS_ACCOUNT_AUTH=<label-studio-refresh-token>

# Modal secret: cl-access-modal
CL_ACCESS_MODAL_URL=<courtlistener-access-service-url>

# Modal secret for /api/assess-review, if served remotely.
# Set MELLEA_LRC_LLM_PROVIDER to deepseek, openrouter, or digitalocean.
MELLEA_LRC_LLM_PROVIDER=deepseek
MELLEA_LRC_LLM_TEMPERATURE=0

# Active OpenAI-compatible endpoint.
MELLEA_LRC_LLM_MODEL=deepseek-v4-pro
MELLEA_LRC_LLM_API_BASE=https://api.deepseek.com
MELLEA_LRC_LLM_API_KEY=<deepseek-api-key>

# OpenRouter example:
# MELLEA_LRC_LLM_MODEL=openai/gpt-4.1-mini
# MELLEA_LRC_LLM_API_BASE=https://openrouter.ai/api/v1
# MELLEA_LRC_LLM_API_KEY=<openrouter-api-key>

# DigitalOcean Gradient example:
# MELLEA_LRC_LLM_MODEL=openai-gpt-oss-20b
# MELLEA_LRC_LLM_API_BASE=https://inference.do-ai.run/v1
# MELLEA_LRC_LLM_API_KEY=<digitalocean-inference-key>

# OpenRouter-only: require a provider that supports all requested params,
# including structured JSON schema output.
# MELLEA_LRC_LLM_OPENROUTER_REQUIRE_PARAMETERS=1
```

`LS_ACCOUNT_AUTH` is only used by the Label Studio bridge to fetch uploaded PDF
assets and patch extracted text back onto the same task. Validation uses the
CourtListener Modal backend through `CL_ACCESS_MODAL_URL`.

## Runtime Assumptions

- The assembled backend API is defined by `pipeline.E2EBackend`.
- Label Studio-specific task bridging is isolated in `label_studio_bridge.py`.
- Label Studio calls this service with `/setup` and `/predict`.
- A PDF task may store the upload path in `data.pdf`, or directly in `data.text`
  when imported into a text-based Label Studio project.
- The app fetches the PDF from `LS_URL`, extracts plain text with Docling, patches
  the same task to keep the source path in `data.pdf` and put extracted text in
  `data.text`, then returns the prediction for Label Studio to attach.
- Citation spans are relative to the extracted `data.text`.
- Validation is optional for `/predict_text` via `{"validate": false}`.
- The custom frontend uses staged review endpoints: extract first, then validate
  the existing review payload, then assess the validated payload.
- Docling is initialized lazily on the first PDF request.

## Local Test Corpus Preprocessing

CourtListener plain-text exports are no longer the Layer 2 source for local tests.
Regenerate text from the PDFs under `local/test_data/pdfs/` with:

```bash
uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs
```

See [Preprocessing Development](../../docs/Preprocessing%20Development.md) for
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
- `POST /setup`: Label Studio ML backend setup acknowledgement
- `POST /api/extract-text`: frontend text extraction stage
- `POST /api/extract-document`: frontend document extraction stage
- `POST /api/review-snapshot`: frontend dev loader for serialized interface artifacts
- `POST /api/validate-review`: frontend validation stage for an existing review payload
- `POST /api/assess-review`: frontend Mellea assessment stage for an existing review payload
- `POST /predict_text`: direct text prediction and optional validation
- `POST /predict`: Label Studio ML backend contract for uploaded PDFs

## Snapshot Loading

The frontend can load a JSON snapshot produced by the neutral serializers for:
`PreprocessedDocument`, `DocumentExtraction`, `DocumentValidation`, or
`DocumentAssessment`. Use the `Load snapshot` button in the input panel and choose
the artifact JSON file. The backend deserializes the artifact and returns the same
review payload shape used by the normal staged workflow.
