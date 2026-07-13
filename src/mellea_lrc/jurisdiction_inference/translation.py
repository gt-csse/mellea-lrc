"""Translation Layer between MLZ Jurisdictions and CourtListener Taxonomies."""

from __future__ import annotations

from mellea_lrc.jurisdiction_inference.types import (
    ReporterInference,
    CourtInference,
    CourtInferenceStatus,
    ReporterInferenceStatus,
    TranslationLayerResult,
    TranslationLayerStatus,
)
from mellea_lrc.jurisdiction_inference.registry import MLZ_TO_CL_MAP

def triangulate_court_id(
    reporter_inference: ReporterInference, court_inference: CourtInference
) -> TranslationLayerResult:
    """Triangulate the exact CourtListener Court ID based on MLZ and Extracted leads."""

    # Heuristic 1: If the CourtInference is resolved, it takes precedence.
    if court_inference.status is CourtInferenceStatus.RESOLVED and court_inference.courts_db_classification:
        return TranslationLayerResult(
            status=TranslationLayerStatus.RESOLVED,
            translated_court_id=court_inference.courts_db_classification.court_id,
        )

    # Heuristic 2: Use the ReporterInference MLZ mapping.
    if reporter_inference.status is ReporterInferenceStatus.RECOGNIZED:
        possible_courts: set[str] = set()
        for mlz in reporter_inference.mlz_jurisdictions:
            if courts := MLZ_TO_CL_MAP.get(mlz):
                possible_courts.update(courts)

        if len(possible_courts) == 1:
            return TranslationLayerResult(
                status=TranslationLayerStatus.RESOLVED,
                translated_court_id=possible_courts.pop(),
            )
        if len(possible_courts) > 1:
            return TranslationLayerResult(
                status=TranslationLayerStatus.AMBIGUOUS,
                translated_court_id=None,
            )

    return TranslationLayerResult(
        status=TranslationLayerStatus.UNAVAILABLE,
        translated_court_id=None,
    )
