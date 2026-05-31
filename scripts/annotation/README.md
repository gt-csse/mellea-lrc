# Annotation Pipeline

Pre-annotates legal documents with eyecite citation extraction and uploads them to Label Studio for human review.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` at the repo root and fill in your credentials:

```
LS_URL=https://your-label-studio-instance.com
LS_EMAIL=your@email.com
LS_PASSWORD=your_password_here
LS_PROJECT_ID=<project_id>
```

## Workflow

**1. Create a new project in Label Studio**

Go to your Label Studio instance and create a new project. Note the project ID from the URL (`/projects/<id>/`). Set `LS_PROJECT_ID` in `.env`.

**2. Push the labeling schema**

```bash
python -m scripts.upload_schema
```

Do this once per project, or whenever `label_studio_config.xml` changes.

**3. Upload documents with pre-annotations**

```bash
python -m scripts.upload_tasks path/to/document.txt
# or in bulk:
python -m scripts.upload_tasks docs-text/*.txt
```

Each document is uploaded as a task with eyecite pre-annotations attached. Annotators review and correct in Label Studio.

## Scripts

| Script | Purpose |
|---|---|
| `pre_annotate.py` | Core eyecite extraction logic — returns a Label Studio prediction dict |
| `upload_tasks.py` | Upload one or more `.txt` files as tasks with pre-annotations |
| `upload_schema.py` | Push `label_studio_config.xml` to the active project |
| `label_studio_config.xml` | Label Studio annotation schema |
