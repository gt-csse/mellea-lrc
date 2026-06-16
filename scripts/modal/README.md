# Modal Pipeline

This directory packages the reusable `src/mellea_lrc` preprocessing, extraction,
and validation pipeline as a Modal-hosted service. It is the replacement shape
for the old standalone `ml_backend/modal_app.py`: Modal and Label Studio stay in
`scripts`, while the reusable pipeline remains in `src`.

Pipeline Modal app name: `mellea-lrc-prototype`.
CourtListener Modal app name: `courtlistener-access`.

## Secrets

The Modal app does not read project-root `.env` during deployment or runtime.
Runtime configuration comes from named Modal secrets:

```env
# Modal secret: label-studio
LS_URL=https://your-label-studio-instance.com
LS_ACCOUNT_AUTH=<label-studio-refresh-token>

# Modal secret: cl-access-modal
CL_ACCESS_MODAL_URL=<cl-access-service-url>
```

`LS_ACCOUNT_AUTH` is the Label Studio refresh token used to fetch uploaded PDF
assets and patch extracted text back onto the same task. The local upload scripts
can still use `LS_EMAIL` / `LS_PASSWORD`.

## Runtime Assumptions

- Label Studio calls this service as an ML backend using `/setup` and `/predict`.
- A PDF task stores an uploaded PDF path in `data.pdf`, or in another string
  field that looks like a Label Studio upload path.
- The app fetches the PDF from `LS_URL`, extracts plain text with Docling, patches
  that text back onto the same Label Studio task as `data.text`, then returns the
  prediction for Label Studio to attach.
- Citation spans are always relative to the extracted `data.text`, not the source
  PDF bytes or an upload metadata wrapper.
- Validation is optional for `/predict_text` via `{"validate": false}`. When
  enabled, `CL_ACCESS_MODAL_URL` must point at the deployed cl-access service.
- The cl-access service is expected to expose `POST /citation-lookup` with
  form fields `volume`, `reporter`, and `page`.
- Docling is initialized lazily on the first PDF request. Text-only requests do
  not initialize the PDF converter.
- Modal CPU memory snapshot is enabled, but the snapshot captures lightweight app
  state; it does not pre-warm Docling's PDF model.

## Deploy

```bash
uv run --group modal modal deploy scripts/modal/app.py
```

The CourtListener access service uses reusable code from
`src/mellea_lrc/courtlistener` and keeps Modal-specific deployment glue in
`scripts/modal/courtlistener_server.py`:

```bash
uv run --group modal modal deploy scripts/modal/courtlistener_server.py
```

## Local Serve

```bash
uv run --group modal modal serve scripts/modal/app.py
```

Useful endpoints:

- `GET /health`: service probe
- `POST /setup`: Label Studio ML backend setup acknowledgement
- `POST /predict_text`: direct text prediction and validation
- `POST /predict`: Label Studio ML backend contract for uploaded PDFs
