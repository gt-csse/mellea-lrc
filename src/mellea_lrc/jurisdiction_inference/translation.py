"""Translation Layer between MLZ Jurisdictions and CourtListener Taxonomies."""

from __future__ import annotations

from mellea_lrc.jurisdiction_inference.types import (
    ReporterLead,
    CourtLead,
    CourtLeadStatus,
    ReporterLeadStatus,
    TranslationLayerResult,
    TranslationLayerStatus,
)
from mellea_lrc.jurisdiction_inference.registry import MLZ_TO_CL_MAP

def triangulate_court_id(
    reporter_lead: ReporterLead, court_lead: CourtLead
) -> TranslationLayerResult:
    """Triangulate the exact CourtListener Court ID based on MLZ and Extracted leads."""
    
    # Heuristic 1: If the CourtLead is resolved, it takes precedence.
    if court_lead.status is CourtLeadStatus.RESOLVED and court_lead.cl_court_taxonomy:
        return TranslationLayerResult(
            status=TranslationLayerStatus.RESOLVED,
            translated_court_id=court_lead.cl_court_taxonomy.court_id,
        )

    # Heuristic 2: Use the ReporterLead MLZ mapping.
    if reporter_lead.status is ReporterLeadStatus.RECOGNIZED:
        possible_courts: set[str] = set()
        for mlz in reporter_lead.mlz_jurisdictions:
            if courts := MLZ_TO_CL_MAP.get(mlz):
                possible_courts.update(courts)
        
        if len(possible_courts) == 1:
            return TranslationLayerResult(
                status=TranslationLayerStatus.RESOLVED,
                translated_court_id=possible_courts.pop(),
            )
        elif len(possible_courts) > 1:
            return TranslationLayerResult(
                status=TranslationLayerStatus.AMBIGUOUS,
                translated_court_id=None,
            )

    return TranslationLayerResult(
        status=TranslationLayerStatus.UNAVAILABLE,
        translated_court_id=None,
    )
