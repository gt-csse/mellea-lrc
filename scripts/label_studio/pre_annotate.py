"""Run source extraction and adapt the result for Label Studio pre-annotation."""

from mellea_lrc.extraction import run_extraction_from_text

from .label_studio import to_label_studio_prediction


def citations_to_prediction(text: str) -> dict[str, object]:
    """Extract citations and serialize them for Label Studio pre-annotation."""
    extraction = run_extraction_from_text(text)
    return to_label_studio_prediction(extraction)


def build_task_payload(text: str, *, source_path: str | None = None) -> dict[str, object]:
    """Build a Label Studio task whose text matches extraction span offsets."""
    extraction = run_extraction_from_text(text, source_path=source_path)
    data: dict[str, object] = {"text": extraction.text}
    if extraction.metadata.header is not None:
        data["source_header"] = extraction.metadata.header
    if source_path is not None:
        data["source_path"] = source_path
    return {
        "data": data,
        "predictions": [to_label_studio_prediction(extraction)],
    }
