"""Read case names using document-aware rules or saved ML spans."""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.extraction.resolve_case_names.reader import read_case_name
from mellea_lrc.model.citation_windows import before
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.read_root_fields.resolve_case_names"


def resolve_case_names(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read a name before the complete locator group, then normalize its quote."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    config = rules or stable()
    citations = document.full_locators
    if fields.case_name is FieldSource.ML:
        spans = [
            fields.validate_prediction(citation.get_ml_field_prediction()).case_name for citation in citations
        ]
    else:
        spans = []
        for citation in citations:
            prefix, start = before(document, citation, config.case_name_window)
            quote = citation.locator[-1].quote if isinstance(citation, FullReporterCitation) else None
            spans.append(read_case_name(prefix, start, quote))
    for citation, span in zip(citations, spans, strict=True):
        updated = citation.record(SUBSTAGE)
        if span is not None:
            updated = updated.with_case_name(document.text, span)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
