"""Create full reporter occurrences from the shared source reader."""

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import full_reporter_readings

STAGE = "1_full_reporter_locators"


def find_full_reporter_locators(document: Document) -> Document:
    """Create one typed occurrence for each shared-reader reporter span.

    The field re-reads its exact quote with the same reader. That keeps its
    normalized identity recoverable from a serialized citation even if eyecite
    used surrounding document context while finding the span.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "5_colocations" in document.stage_runs:
        raise ValueError("Discover all locators before resolving colocations")
    for reading in sorted(full_reporter_readings(document.text), key=lambda item: item.span):
        span = Span(*reading.span)
        if any(span.overlaps(item.site_span) for item in document.citations):
            continue
        identifier = f"reporter:{span.start}:{span.end}"
        document = document.add_citation(
            FullReporterCitation.from_locator(
                citation_id=identifier,
                stage=STAGE,
                source=document.text,
                span=span,
            )
        )
    return document.complete(STAGE)
