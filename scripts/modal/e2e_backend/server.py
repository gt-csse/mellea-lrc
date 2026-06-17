"""Modal server for the mellea-lrc end-to-end backend."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import modal

from scripts.modal.e2e_backend.label_studio_bridge import LabelStudioBridge, LabelStudioConfig
from scripts.modal.e2e_backend.pipeline import E2EBackend

APP_NAME = "mellea-lrc-prototype"

logger = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO)

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
    .add_local_python_source("mellea_lrc", "scripts")
)

backend = E2EBackend()


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
    """Expose the E2E backend through FastAPI."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: PLC0415
    from fastapi.middleware.cors import CORSMiddleware  # noqa: PLC0415

    web_app = FastAPI(title="Mellea LRC E2E Backend", version="0.1.0")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=_frontend_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

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
        return backend.predict_text(str(text), validate=validate)

    @web_app.post("/api/review-text")
    async def review_text(payload: dict[str, object]) -> dict[str, object]:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        validate = bool(payload.get("validate", True))
        return backend.review_text(text, validate=validate)

    @web_app.post("/api/review-document")
    async def review_document(
        *,
        file: Annotated[UploadFile, File()],
        validate: Annotated[bool, Form()] = True,
    ) -> dict[str, object]:
        filename = file.filename or "document.pdf"
        try:
            return backend.review_document_bytes(await file.read(), filename, validate=validate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/predict")
    async def predict(payload: dict[str, object]) -> dict[str, list[object]]:
        bridge = LabelStudioBridge(
            backend,
            LabelStudioConfig.from_env(),
            logger=logger,
        )
        return bridge.predict_tasks(payload.get("tasks", []))

    return web_app


def _frontend_origins() -> list[str]:
    raw_origins = os.environ.get("MELLEA_LRC_FRONTEND_ORIGINS")
    if raw_origins:
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]
