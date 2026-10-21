"""Read written courts or retain reporter-based court inference."""

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.full_citations import dated_parenthetical, require_structure
from mellea_lrc.extraction.field_readers.court_inference import infer_court_from_reporter
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.dates import FULL_DATE_RE, YEAR_RE
from mellea_lrc.parsing.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_courts"

_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)


def resolve_courts(
    document: Document,
    rules: ExtractionRules | None = None,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read a court after the locator group; infer one only when no court is written."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    require_structure(document)
    fields = field_extraction or FieldExtractionConfig.from_env()
    config = rules or stable()
    citations = document.full_locators
    if fields.court is FieldSource.ML:
        spans = [
            fields.validate_prediction(citation.get_ml_field_prediction()).court for citation in citations
        ]
    else:
        spans = []
        for citation in citations:
            found = dated_parenthetical(document, citation, config.post_locator_window)
            span = None
            if found:
                parenthetical, start = found
                body = parenthetical.group("body")
                date = FULL_DATE_RE.search(body) or YEAR_RE.search(body)
                if date is not None:
                    region = body[: date.start()]
                    if event := _DATE_EVENT.search(region):
                        region = region[: event.start()]
                    written = region.strip(" ,;")
                    if written:
                        offset = start + parenthetical.start("body") + len(region) - len(region.lstrip(" ,;"))
                        span = Span(offset, offset + len(written))
            spans.append(span)
    for citation, span in zip(citations, spans, strict=True):
        updated = citation.record(SUBSTAGE)
        if span is not None:
            updated = updated.with_court(document.text, span)
        elif (court := infer_court_from_reporter(citation)) is not None:
            updated = updated.with_inferred_court(court)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
