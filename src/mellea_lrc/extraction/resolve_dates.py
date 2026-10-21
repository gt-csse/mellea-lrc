"""Read dates using document-aware rules or saved ML spans."""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.full_citations import dated_parenthetical, require_structure
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.dates import FULL_DATE_RE, YEAR_RE
from mellea_lrc.parsing.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_dates"


def resolve_dates(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read a day or year after the complete locator group, then normalize it."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    config = rules or stable()
    citations = document.full_locators
    if fields.date is FieldSource.ML:
        spans = [
            fields.validate_prediction(citation.get_ml_field_prediction()).date for citation in citations
        ]
    else:
        spans = []
        for citation in citations:
            found = dated_parenthetical(document, citation, config.post_locator_window)
            span = None
            if found:
                parenthetical, start = found
                match = FULL_DATE_RE.search(parenthetical.group("body")) or YEAR_RE.search(
                    parenthetical.group("body")
                )
                if match is not None:
                    offset = start + parenthetical.start("body")
                    span = Span(offset + match.start(), offset + match.end())
            spans.append(span)
    for citation, span in zip(citations, spans, strict=True):
        updated = citation.record(SUBSTAGE)
        if span is not None:
            updated = updated.with_date(document.text, span)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
