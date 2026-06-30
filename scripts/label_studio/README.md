# Label Studio Workflow

Pre-annotates legal documents with the source extraction pipeline and uploads them to Label Studio for human review.

Label Studio is an annotation UI adapter. Reusable preprocessing, extraction, and neutral JSON serialization live under `src/mellea_lrc`.

## Setup

Install the Label Studio script dependencies:

```bash
uv sync --group label-studio
```

Copy the project-root example environment file and fill in your Label Studio credentials:

```bash
cp .env.example .env
```

Required values:

```env
LS_URL=https://your-label-studio-instance.com
LS_EMAIL=your@email.com
LS_PASSWORD=your_password_here
LS_PROJECT_ID=123
```

## Workflow

1. Create a new project in Label Studio and set `LS_PROJECT_ID` from the `/projects/<id>/` URL.
2. Push the labeling schema:

```bash
uv run --group label-studio python -m scripts.label_studio.cli upload-schema
```

3. Upload documents with pre-annotations:

```bash
uv run --group label-studio python -m scripts.label_studio.cli upload-tasks path/to/document.txt
uv run --group label-studio python -m scripts.label_studio.cli upload-tasks docs-text/*.txt
```

## Task Data Shape

The current labeling schema renders only `data.text`:

```xml
<Text name="text" value="$text" />
```

So Label Studio annotations and predictions are text-span annotations over:

```json
{"data": {"text": "document text"}}
```

If a PDF is imported directly into this text-based project, Label Studio may put
the upload path in `data.text`:

```json
{"data": {"text": "/data/upload/.../document.pdf"}}
```

That path is then rendered as plain text because the XML consumes `$text`.
Downstream PDF workflows should extract text first and write tasks shaped like:

```json
{"data": {"pdf": "/data/upload/.../document.pdf", "text": "extracted text"}}
```

Predictions should always use spans relative to `data.text`.

## Files

| File | Purpose |
|---|---|
| `pre_annotate.py` | Runs source extraction and adapts the result to a Label Studio prediction dict |
| `label_studio.py` | Label Studio-specific prediction serializer |
| `upload_tasks.py` | Uploads `.txt` files as tasks with pre-annotations |
| `upload_schema.py` | Pushes `label_studio_config.xml` to the active project |
| `label_studio_config.xml` | Label Studio annotation schema |
