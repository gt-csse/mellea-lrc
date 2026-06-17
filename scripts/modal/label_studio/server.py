"""Modal server for the Label Studio extraction and validation pipeline."""

from __future__ import annotations

import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import modal

from mellea_lrc.preprocessing.types import (
    PreprocessedDocument,
    PreprocessedDocumentMetadata,
    PreprocessingBackend,
    SourceFormat,
)
from scripts.modal.label_studio.pipeline import predict_preprocessed

APP_NAME = "mellea-lrc-prototype"
REMOTE_REPO_ROOT = Path("/root")
LOCAL_REPO_PARENT_INDEX = 3


def _source_repo_root() -> Path:
    path = Path(__file__).resolve()
    if len(path.parents) > LOCAL_REPO_PARENT_INDEX:
        return path.parents[LOCAL_REPO_PARENT_INDEX]
    return REMOTE_REPO_ROOT


LOCAL_REPO_ROOT = _source_repo_root()

logger = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO)


class DoclingDocument(Protocol):
    """Protocol for the Docling document export surface used here."""

    def export_to_text(self) -> str:
        """Export extracted document content as plain text."""


class DoclingResult(Protocol):
    """Protocol for a Docling conversion result."""

    document: DoclingDocument


class DoclingConverter(Protocol):
    """Protocol for the Docling converter surface used here."""

    def convert(self, source: object) -> DoclingResult:
        """Convert a document source."""


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "docling>=2.0",
        "eyecite>=2.6",
        "fastapi[standard]>=0.115",
        "requests>=2.32",
    )
    .add_local_dir(LOCAL_REPO_ROOT / "src", REMOTE_REPO_ROOT / "src", copy=True)
    .add_local_dir(LOCAL_REPO_ROOT / "scripts", REMOTE_REPO_ROOT / "scripts", copy=True)
    .env({"PYTHONPATH": "/root/src:/root"})
)


def _build_converter() -> DoclingConverter:
    from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
    from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _pdf_to_preprocessed(
    converter: DoclingConverter,
    content: bytes,
    filename: str,
) -> PreprocessedDocument:
    from docling.datamodel.base_models import DocumentStream  # noqa: PLC0415

    source = DocumentStream(name=filename, stream=BytesIO(content))
    result = converter.convert(source)
    return PreprocessedDocument(
        text=result.document.export_to_text(),
        metadata=PreprocessedDocumentMetadata(
            source_path=filename,
            source_format=SourceFormat.PDF,
            backend=PreprocessingBackend.DOCLING,
        ),
    )


def _text_to_preprocessed(text: str) -> PreprocessedDocument:
    from mellea_lrc.preprocessing import preprocess_plain_text_from_string  # noqa: PLC0415

    return preprocess_plain_text_from_string(text)


def _get_access_token(ls_url: str, refresh_token: str) -> str:
    import requests  # noqa: PLC0415

    response = requests.post(
        f"{ls_url}/api/token/refresh",
        json={"refresh": refresh_token},
        timeout=30,
    )
    response.raise_for_status()
    access = response.json().get("access")
    if not access:
        msg = "Label Studio token refresh returned no access token"
        raise RuntimeError(msg)
    return str(access)


def _fetch_ls_asset(ls_url: str, asset_path: str, access_token: str) -> bytes:
    import requests  # noqa: PLC0415

    split = urlsplit(asset_path)
    path = split.path
    if split.query:
        path += f"?{split.query}"
    if not path.startswith("/"):
        path = f"/{path}"

    response = requests.get(
        f"{ls_url}{path}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=120,
    )
    response.raise_for_status()
    return response.content


def _extract_pdf_path(task: dict[str, object]) -> str:
    data = task.get("data")
    if not isinstance(data, dict):
        msg = "Task data is missing"
        raise KeyError(msg)

    pdf = data.get("pdf")
    if isinstance(pdf, str) and pdf:
        return pdf

    for value in data.values():
        if isinstance(value, str) and (value.lower().endswith(".pdf") or "/data/upload/" in value):
            return value

    msg = f"No PDF path found in task data (keys: {list(data.keys())})"
    raise KeyError(msg)


def _filename_from_path(asset_path: str) -> str:
    return urlsplit(asset_path).path.rsplit("/", 1)[-1] or asset_path


def _set_task_text(
    ls_url: str,
    task_id: object,
    existing_data: dict[str, object],
    pdf_path: str,
    text: str,
    access_token: str,
) -> None:
    import requests  # noqa: PLC0415

    response = requests.patch(
        f"{ls_url}/api/tasks/{task_id}",
        json={"data": {**existing_data, "pdf": pdf_path, "text": text}},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=60,
    )
    response.raise_for_status()


_converter: DoclingConverter | None = None


def _get_converter() -> DoclingConverter:
    global _converter  # noqa: PLW0603
    if _converter is None:
        _converter = _build_converter()
    return _converter


def _predict_text(text: str, *, validate: bool = True) -> dict[str, object]:
    return predict_preprocessed(_text_to_preprocessed(text), validate=validate)


def _predict_pdf_bytes(
    content: bytes,
    filename: str,
    *,
    validate: bool = True,
) -> dict[str, object]:
    if content[:4] != b"%PDF":
        msg = f"{filename} is not a PDF"
        raise ValueError(msg)
    return predict_preprocessed(
        _pdf_to_preprocessed(_get_converter(), content, filename),
        validate=validate,
    )


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("label-studio"),
        modal.Secret.from_name("cl-access-modal"),
    ],
    gpu="L4",
    memory=4096,
    timeout=600,
    enable_memory_snapshot=True,
)
@modal.asgi_app()
def web() -> object:
    """Expose the Modal pipeline through FastAPI."""
    from fastapi import FastAPI  # noqa: PLC0415

    web_app = FastAPI(title="Mellea LRC Pipeline", version="0.1.0")

    @web_app.get("/")
    @web_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "UP", "model_class": APP_NAME}

    @web_app.post("/setup")
    async def setup(_payload: dict[str, object]) -> dict[str, str]:
        return {"model_version": "eyecite-pre-annotation"}

    @web_app.post("/predict_text")
    async def predict_text(payload: dict[str, object]) -> dict[str, object]:
        text = payload.get("text") or ""
        validate = bool(payload.get("validate", True))
        return _predict_text(str(text), validate=validate)

    @web_app.post("/predict")
    async def predict(payload: dict[str, object]) -> dict[str, list[object]]:
        tasks = payload.get("tasks", []) or []
        if not isinstance(tasks, list):
            tasks = []

        ls_url = os.environ["LS_URL"].rstrip("/")
        refresh_token = os.environ["LS_ACCOUNT_AUTH"]
        access_token = _get_access_token(ls_url, refresh_token)

        results = []
        for task in tasks:
            if not isinstance(task, dict):
                results.append({"result": []})
                continue
            try:
                asset_path = _extract_pdf_path(task)
                content = _fetch_ls_asset(ls_url, asset_path, access_token)
                filename = _filename_from_path(asset_path)
                output = _predict_pdf_bytes(content, filename)
                task_id = task.get("id")
                data = task.get("data") if isinstance(task.get("data"), dict) else {}
                if task_id is not None:
                    _set_task_text(
                        ls_url,
                        task_id,
                        data,
                        asset_path,
                        str(output["text"]),
                        access_token,
                    )
                results.append(output["prediction"])
            except Exception:
                logger.exception("Failed to process Label Studio task %s", task.get("id"))
                results.append({"result": []})
        return {"results": results}

    return web_app
