"""Reusable FastAPI app for the E2E backend."""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from mellea_lrc.serialization import (
    deserialize_document_assessment,
    deserialize_document_extraction,
    deserialize_document_validation,
    deserialize_preprocessed_document,
)
from scripts.e2e_backend.pipeline import E2EBackend

APP_NAME = "mellea-lrc-prototype"


def create_app(backend: E2EBackend | None = None) -> FastAPI:  # noqa: C901, PLR0915
    """Create the E2E backend app for Modal or local serving."""
    pipeline = backend or E2EBackend()
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
        return pipeline.predict_text(str(text), validate=validate)

    @web_app.post("/api/extract-text")
    async def extract_text(payload: dict[str, object]) -> dict[str, Any]:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        return pipeline.extract_text(text)

    @web_app.post("/api/extract-document")
    async def extract_document(
        *,
        file: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        filename = file.filename or "document.pdf"
        try:
            return pipeline.extract_document_bytes(await file.read(), filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/validate-review")
    async def validate_review(payload: dict[str, object]) -> dict[str, Any]:
        try:
            return pipeline.validate_review_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/validate-review-citation")
    async def validate_review_citation(payload: dict[str, object]) -> dict[str, Any]:
        try:
            return pipeline.validate_review_citation_payload(payload)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/assess-review")
    async def assess_review(payload: dict[str, object]) -> dict[str, Any]:
        try:
            return await pipeline.assess_review_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @web_app.post("/api/review-snapshot")
    async def review_snapshot(
        *,
        file: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        try:
            payload = json.loads((await file.read()).decode("utf-8"))
            return _review_snapshot_payload(payload, pipeline)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Snapshot must be a JSON artifact.") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @web_app.post("/api/review-text")
    async def review_text(payload: dict[str, object]) -> dict[str, object]:
        text = str(payload.get("text") or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        validate = bool(payload.get("validate", True))
        return pipeline.review_text(text, validate=validate)

    @web_app.post("/api/review-document")
    async def review_document(
        *,
        file: Annotated[UploadFile, File()],
        validate: Annotated[bool, Form()] = True,
    ) -> dict[str, object]:
        filename = file.filename or "document.pdf"
        try:
            return pipeline.review_document_bytes(await file.read(), filename, validate=validate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return web_app


def _frontend_origins() -> list[str]:
    raw_origins = os.environ.get("MELLEA_LRC_FRONTEND_ORIGINS")
    if raw_origins:
        return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


def _review_snapshot_payload(payload: object, pipeline: E2EBackend) -> dict[str, Any]:
    if not isinstance(payload, dict):
        msg = "Snapshot must be a JSON object."
        raise TypeError(msg)
    if "assessments" in payload:
        return {
            "stage": "assessed",
            "result": pipeline.review_document_assessment(deserialize_document_assessment(payload)),
        }
    if "validations" in payload:
        return {
            "stage": "validated",
            "result": pipeline.review_document_validation(deserialize_document_validation(payload)),
        }
    if "citations" in payload:
        return {
            "stage": "extracted",
            "result": pipeline.review_document_extraction(deserialize_document_extraction(payload)),
        }
    if "metadata" in payload and "text" in payload:
        return {
            "stage": "preprocessed",
            "result": pipeline.review_preprocessed_document(deserialize_preprocessed_document(payload)),
        }
    msg = (
        "Snapshot does not look like a serialized PreprocessedDocument, "
        "DocumentExtraction, DocumentValidation, or DocumentAssessment."
    )
    raise ValueError(msg)
