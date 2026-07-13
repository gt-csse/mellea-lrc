"""Jurisdiction inference pipeline stage."""

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction.types import ExtractedDocument
from mellea_lrc.jurisdiction_inference.leads import evaluate_court_inference, evaluate_reporter_inference
from mellea_lrc.jurisdiction_inference.types import (
    CourtInference,
    CourtInferenceStatus,
    InferredDocument,
    Jurisdiction,
    ReporterInference,
    ReporterInferenceStatus,
)


def infer_jurisdiction(doc: ExtractedDocument) -> InferredDocument:
    """Run jurisdiction inference on every citation in an extracted document.

    Only ``FullCaseCitation`` citations are evaluated. Law, journal, and other
    citation kinds are explicitly marked as unsupported.
    """
    jurisdictions: list[Jurisdiction] = []
    for citation in doc.citations:
        if isinstance(citation.citation, FullCaseCitation):
            reporter_inference = evaluate_reporter_inference(citation.citation.reporter)
            court_inference = evaluate_court_inference(citation.citation.court)
        else:
            reporter_inference = ReporterInference(
                reporter=None,
                status=ReporterInferenceStatus.UNSUPPORTED,
                mlz_jurisdictions=(),
            )
            court_inference = CourtInference(
                extracted_court=None,
                status=CourtInferenceStatus.UNSUPPORTED,
                courts_db_classification=None,
            )

        jurisdictions.append(Jurisdiction(
            reporter_inference=reporter_inference,
            court_inference=court_inference,
        ))

    return InferredDocument(
        source_metadata=doc.source_metadata,
        text=doc.text,
        preprocessing_metadata=doc.preprocessing_metadata,
        citations=doc.citations,
        extraction_metadata=doc.extraction_metadata,
        jurisdictions=tuple(jurisdictions),
    )
