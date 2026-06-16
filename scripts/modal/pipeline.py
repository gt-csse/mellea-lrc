"""End-to-end pipeline helpers used by the Modal app."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from mellea_lrc.extraction import run_extraction
from mellea_lrc.validation import validate_extraction
from mellea_lrc.validation.types import ValidationStatus
from scripts.label_studio.label_studio import to_label_studio_prediction

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import DocumentExtraction
    from mellea_lrc.preprocessing.types import PreprocessedDocument
    from mellea_lrc.validation import CourtListenerAccessClient, DocumentValidation


def predict_preprocessed(
    preprocessed: PreprocessedDocument,
    *,
    validate: bool = True,
    client: CourtListenerAccessClient | None = None,
) -> dict[str, Any]:
    """Run extraction, optional validation, and Label Studio prediction serialization."""
    extraction = run_extraction(preprocessed)
    validation = (
        validate_extraction(
            extraction,
            client_mode="custom" if client is not None else "deployed",
            client=client,
        )
        if validate
        else None
    )
    prediction = to_label_studio_prediction(extraction)
    if validation is not None:
        prediction = add_validation_notes(prediction, validation)

    return {
        "text": extraction.text,
        "prediction": prediction,
        "validation": _validation_payload(validation),
        "stats": _stats(extraction, validation),
    }


def add_validation_notes(
    prediction: dict[str, Any],
    validation: DocumentValidation,
) -> dict[str, Any]:
    """Add CourtListener validation messages as Label Studio per-region notes."""
    result = list(prediction.get("result", []))
    for item in validation.validations:
        if item.status == ValidationStatus.SKIPPED:
            continue
        result.append(
            {
                "id": item.citation_id,
                "from_name": "notes",
                "to_name": "text",
                "type": "textarea",
                "value": {"text": [item.message]},
            }
        )
    return {**prediction, "result": result}


def _validation_payload(validation: DocumentValidation | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {
        "validations": [{**asdict(item), "status": item.status.value} for item in validation.validations],
        "counts": {
            "total": len(validation.validations),
            "found": len(validation.found),
        },
    }


def _stats(
    extraction: DocumentExtraction,
    validation: DocumentValidation | None,
) -> dict[str, int]:
    label_count = len(extraction.citations)
    stats = {
        "chars": len(extraction.text),
        "citation_spans": label_count,
        "full_citations": len(extraction.full_citations),
    }
    if validation is not None:
        stats["validated"] = sum(
            1 for item in validation.validations if item.status != ValidationStatus.SKIPPED
        )
        stats["found"] = len(validation.found)
    return stats
