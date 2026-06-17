# Label Studio Modal Backend

This Modal app exposes the mellea-lrc extraction and validation pipeline as a
Label Studio ML backend.

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

`LS_ACCOUNT_AUTH` is the Label Studio refresh token used to fetch uploaded PDF
assets and patch extracted text back onto the same task. Validation uses the
CourtListener Modal backend through `CL_ACCESS_MODAL_URL`.

## Runtime Assumptions

- Label Studio calls this service with `/setup` and `/predict`.
- A PDF task stores an uploaded PDF path in `data.pdf`, or in another string
  field that looks like a Label Studio upload path.
- The app fetches the PDF from `LS_URL`, extracts plain text with Docling, patches
  that text back onto the same Label Studio task as `data.text`, then returns the
  prediction for Label Studio to attach.
- Citation spans are relative to the extracted `data.text`.
- Validation is optional for `/predict_text` via `{"validate": false}`.
- Docling is initialized lazily on the first PDF request.

## Deploy

```bash
uv run --group modal modal deploy scripts/modal/label_studio/server.py
```

## Local Serve

```bash
uv run --group modal modal serve scripts/modal/label_studio/server.py
```

Useful endpoints:

- `GET /health`: service probe
- `POST /setup`: Label Studio ML backend setup acknowledgement
- `POST /predict_text`: direct text prediction and optional validation
- `POST /predict`: Label Studio ML backend contract for uploaded PDFs
