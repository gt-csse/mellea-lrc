# E2E Modal Backend

This Modal app exposes the assembled mellea-lrc preprocessing, extraction,
validation, and prediction pipeline. The backend has a direct text API and a
Label Studio bridge for the ML-backend `/setup` and `/predict` contract.

Modal app name: `mellea-lrc-prototype`.

## Secrets

Runtime configuration comes from named Modal secrets:

```env
# Modal secret: label-studio
LS_URL=https://your-label-studio-instance.com
LS_ACCOUNT_AUTH=<label-studio-refresh-token>

# Modal secret: cl-access-modal
CL_ACCESS_MODAL_URL=<courtlistener-access-service-url>
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
- Docling is initialized lazily on the first PDF request.

## Deploy

```bash
uv run --group modal modal deploy scripts/modal/e2e_backend/server.py
```

## Local Serve

```bash
uv run --group modal modal serve scripts/modal/e2e_backend/server.py
```

Useful endpoints:

- `GET /health`: service probe
- `POST /setup`: Label Studio ML backend setup acknowledgement
- `POST /predict_text`: direct text prediction and optional validation
- `POST /predict`: Label Studio ML backend contract for uploaded PDFs
