"""Read case-name fields before each full locator site."""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction.case_names.reader import read_case_name
from mellea_lrc.extraction.context.full_citations import require_structure
from mellea_lrc.model.citation_windows import before as bounded_before
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.field_reading.case_names"


def resolve_case_names(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read a name before each citation site, never through another locator."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        before, start = bounded_before(document, citation, config.case_name_window)
        reporter_quote = citation.locator[-1].quote if isinstance(citation, FullReporterCitation) else None
        if span := read_case_name(before, start, reporter_quote):
            document = document.replace_citation(
                citation.record(SUBSTAGE).with_case_name(document.text, span)
            )
    return document.complete_substage(SUBSTAGE)
