# Scripts

Workflow entry points for local pipeline runs, snapshot regeneration, frontend
E2E testing, Modal deployment, and Label Studio uploads. Reusable library code
lives under `src/mellea_lrc/`; these scripts wire that code into runnable
commands.

Run commands from the repository root unless noted otherwise.

## Layout

```text
scripts/
├── README.md                          # this file — command index
├── reextract_case_name.py             # standalone case-name re-extraction
├── e2e_backend/                       # pipeline snapshots + local/Modal API
│   ├── README.md
│   ├── preprocess_test_pdfs.py
│   ├── snapshot_corpus.py             # notebook-equivalent batch snapshots
│   ├── run_artifact_pipeline.py       # single-input extract → retrieve → assess
│   ├── local_server.py                # local FastAPI for frontend E2E
│   ├── api.py                         # shared FastAPI app factory
│   └── pipeline.py                    # assembled E2EBackend
├── label_studio/                      # Label Studio upload workflow
│   └── README.md
└── modal/                             # Modal deploy wrappers
    ├── e2e_backend/server.py          # Modal app: mellea-lrc-prototype
    └── courtlistener/server.py        # Modal app: courtlistener-access
        └── README.md
```

## Dependency groups

Install only what a workflow needs with `uv sync --group <name>`.

| Group | Use for |
| --- | --- |
| `preprocessing` | PDF → plain-text test corpus |
| `pipeline` | Full local pipeline and snapshot regeneration |
| `llm` | Mellea LLM calls (assessment, re-extraction) |
| `label-studio` | Label Studio schema and task uploads |
| `modal` | Modal deploy/serve and local FastAPI E2E server |
| `scripts` | Shared script utilities (`python-dotenv`, `requests`) |
| `notebook` | Jupyter (`notebooks/explore.example.ipynb`) |

Most commands below use `uv run --group …` so dependencies resolve on demand.

## Quick reference

| Task | Command |
| --- | --- |
| Regenerate test PDF text | `uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs` |
| Regenerate pipeline snapshots (notebook equivalent) | `uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus` |
| Run one input through extract → retrieve | `uv run --group pipeline python -m scripts.e2e_backend.run_artifact_pipeline` |
| Serve E2E backend locally for frontend | `uv run --group modal fastapi run scripts/e2e_backend/local_server.py --host 127.0.0.1 --port 8011` |
| Deploy E2E backend to Modal | `uv run --group modal modal deploy scripts/modal/e2e_backend/server.py` |
| Deploy CourtListener access to Modal | `uv run --group modal modal deploy scripts/modal/courtlistener/server.py` |
| Upload Label Studio schema | `uv run --group label-studio python -m scripts.label_studio.cli upload-schema` |
| Upload Label Studio tasks | `uv run --group label-studio python -m scripts.label_studio.cli upload-tasks path/to/doc.txt` |
| Re-extract a case name (LLM) | `uv run --group llm python scripts/reextract_case_name.py --courtlistener-case-name "…"` |

---

## Local pipeline and snapshots

### Preprocess test PDFs

Convert PDFs under `local/test_data/pdfs/` into Layer 2 plain-text fixtures under
`local/test_data/`. See [Preprocessing Development](../docs/development/Preprocessing%20Development.md).

```bash
# All PDFs
uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs

# Specific stems
uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs --pdf 1
uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs --pdf 1 3 5
```

| Option | Default | Description |
| --- | --- | --- |
| `--pdf-dir` | `local/test_data/pdfs` | Source PDF directory |
| `--output-dir` | `local/test_data` | Output `.txt` directory |
| `--pdf` | all PDFs in `--pdf-dir` | One or more PDF stems or filenames |

### Regenerate pipeline snapshots

Script equivalent of `notebooks/explore.example.ipynb`. Runs preprocessing →
extraction → jurisdiction inference → retrieval → assessment, writing strict
stage JSON under `local/snapshots/<doc>/`.

```bash
# Configured inclusive numeric range (see CONFIG in snapshot_corpus.py)
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus

# One fixture through a chosen phase
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file 3 --phase assessment

# Bookmarked fixture through retrieval only
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file bookmarked --phase retrieval
```

| Option | Default | Description |
| --- | --- | --- |
| `--file` | numeric batch range | `bookmarked` or a numeric text-fixture stem |
| `--phase` | `assessment` | Stop after `preprocessed`, `extraction`, `inferred`, `retrieval`, or `assessment` |

Paths, batch range (`batch_start`/`batch_end`), bookmark path, and Mellea
concurrency are configured in `SnapshotConfig` / `CONFIG` inside
`snapshot_corpus.py` — not CLI flags.

Requires a populated `.env` (CourtListener URL, Mellea LLM settings).

### Run artifact pipeline (single input)

Lighter-weight runner for one document: preprocess or plain-text load → extract
→ retrieve, with optional Mellea assessment. Writes `retrieval.json` and
optionally `assessment.json` under `local/snapshots/<doc>/`.

```bash
uv run --group pipeline python -m scripts.e2e_backend.run_artifact_pipeline

uv run --group pipeline python -m scripts.e2e_backend.run_artifact_pipeline \
  --input local/test_data/1.txt \
  --refresh-retrieval \
  --assess-mellea
```

| Option | Default | Description |
| --- | --- | --- |
| `--input` | `local/test_data/pdfs/1.pdf` | PDF or `.txt` input |
| `--snapshot-dir` | `local/snapshots/<stem>` | Output directory |
| `--refresh-retrieval` | off | Re-run retrieval even if `retrieval.json` exists |
| `--assess-mellea` | off | Run assessment and write `assessment.json` |

---

## E2E backend (frontend review UI)

Exposes the assembled pipeline for the custom frontend. Deeper runtime notes,
secrets, and API endpoints: [e2e_backend/README.md](e2e_backend/README.md).

Modal app name: `mellea-lrc-prototype`.

### Local serve (no Modal)

```bash
uv run --group modal fastapi run \
  scripts/e2e_backend/local_server.py \
  --host 127.0.0.1 \
  --port 8011
```

Then point the frontend at the backend:

```bash
cd frontend
MELLEA_E2E_BACKEND_URL=http://127.0.0.1:8011 npm run dev -- \
  --hostname 127.0.0.1 \
  --port 3000
```

### Modal serve / deploy

```bash
# Ephemeral dev deployment
uv run --group modal modal serve scripts/modal/e2e_backend/server.py

# Production deploy
uv run --group modal modal deploy scripts/modal/e2e_backend/server.py
```

Key endpoints: `GET /health`, `POST /api/extract-text`, `POST /api/extract-document`,
`POST /api/review-snapshot`, `POST /api/retrieve-review`, `POST /api/assess-review`.

---

## CourtListener Modal backend

Reusable CourtListener access API. Details and endpoint list:
[modal/courtlistener/README.md](modal/courtlistener/README.md).

Modal app name: `courtlistener-access`.

```bash
uv run --group modal modal serve scripts/modal/courtlistener/server.py
uv run --group modal modal deploy scripts/modal/courtlistener/server.py
```

The E2E pipeline and snapshot scripts reach this service through
`CL_ACCESS_MODAL_URL` in `.env` or Modal secrets.

---

## Label Studio workflow

Pre-annotate documents with source extraction and upload to Label Studio for
human review. Setup, env vars, and task data shape:
[label_studio/README.md](label_studio/README.md).

```bash
uv sync --group label-studio

uv run --group label-studio python -m scripts.label_studio.cli upload-schema

uv run --group label-studio python -m scripts.label_studio.cli upload-tasks path/to/document.txt
uv run --group label-studio python -m scripts.label_studio.cli upload-tasks docs-text/*.txt
```

Required `.env` values: `LS_URL`, `LS_EMAIL`, `LS_PASSWORD`, `LS_PROJECT_ID`.

---

## Assessment utilities

### Re-extract case name

Standalone CLI for the case-name re-extraction Mellea workflow. Reads context
from a file or stdin; prints JSON to stdout.

```bash
uv run --group llm python scripts/reextract_case_name.py \
  --context-file fixtures/bookmarked/bookmarked.txt \
  --extracted-case-name "<NO_EXTRACTED_CASE_NAME>" \
  --courtlistener-case-name "Brown v. Board"
```

| Option | Required | Description |
| --- | --- | --- |
| `--context-file` | no | Local citation context; reads stdin when omitted |
| `--extracted-case-name` | no | Current extracted name; omit when none |
| `--courtlistener-case-name` | yes | CourtListener name to compare against |

---

## Notebook

Interactive walkthrough of the same staged snapshot workflow:

`notebooks/explore.example.ipynb`

Prefer `snapshot_corpus.py` for batch regeneration without running cells manually.
