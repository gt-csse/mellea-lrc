"""Apply source-grounded docket re-read results as durable field operations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.extraction.reading.courts import resolve_court
from mellea_lrc.llm import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations import CitationDate, CitationField, DocketCitation
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.model.operations import observe_citation, update_fields
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.spans import Span
from mellea_lrc.validation.field_checks.source_case_name import ground_source_case_name
from mellea_lrc.validation.root_identity.context import masked_root_context
from mellea_lrc.validation.types import (
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord, Node


_LITERAL_SOURCE = FuzzinessOption.whitespace_relaxation()
_SOURCE_DATE = re.compile(
    r"(?:(?P<month>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?)"
    r"\s+(?P<day>\d{1,2}),?\s+)?(?P<year>\d{4})",
    re.IGNORECASE,
)


def apply_docket_source_review(
    record: CitationRecord,
    document: Document,
    review: MelleaDocketCitationReextractionNode,
    trace: Node,
) -> bool:
    """Write supported re-read fields; return whether any field changed.

    The review is a single document reading. Its model strings can change a
    citation only after being matched to the target-only source view and then
    verified against the original document bytes. Retrieved records are not
    inputs to this operation.
    """
    if review.status is not ValidationNodeStatus.SUCCEEDED:
        observe_citation(record, trace)
        return False
    citation = record.fields
    if not isinstance(citation, DocketCitation):
        raise TypeError("Docket source review requires a docket citation")

    context = masked_root_context(document, record, before=320, after=320)
    before_locator = context.text[: record.locator_span.start - context.start]
    after_locator = context.text[record.locator_span.end - context.start :]
    changes: dict[CitationField, object] = {}

    if (
        review.outcome is MelleaDocketCitationReextractionOutcome.CORRECTED
        and review.grounded_docket_number is not None
        and review.grounded_docket_number != citation.docket_number
    ):
        changes[CitationField.DOCKET_NUMBER] = review.grounded_docket_number

    name = review.grounded_case_name or ground_source_case_name(
        review.reparsed_case_name,
        before_locator=before_locator,
        source_start=context.start,
    )
    if (
        name is not None
        and name.span.end <= record.locator_span.start
        and document.text[name.span.start : name.span.end] == name.text
    ):
        if name != citation.case_name:
            changes[CitationField.CASE_NAME] = name
        if name.plaintiff is not None or name.defendant is not None:
            if name.plaintiff != citation.plaintiff:
                changes[CitationField.PLAINTIFF] = name.plaintiff
            if name.defendant != citation.defendant:
                changes[CitationField.DEFENDANT] = name.defendant

    court = _ground_after_locator(
        review.reparsed_court, after_locator, source_start=record.locator_span.end, document=document
    )
    if court is not None:
        court_text = court[0]
        court_id = resolve_court(court_text)
        if court_id is not None:
            if court_id != citation.court:
                changes[CitationField.COURT] = court_id
                if citation.court_name is not None:
                    changes[CitationField.COURT_NAME] = None
            if court_text != citation.court_text:
                changes[CitationField.COURT_TEXT] = court_text

    stated_date = _ground_after_locator(
        review.reparsed_date, after_locator, source_start=record.locator_span.end, document=document
    )
    if stated_date is not None:
        parsed_date = _parse_source_date(stated_date[0])
        prior = citation.date
        loses_precision = (
            prior is not None
            and prior.is_exact
            and parsed_date is not None
            and not parsed_date.is_exact
            and parsed_date.year == prior.year
        )
        if parsed_date is not None and parsed_date != prior and not loses_precision:
            changes[CitationField.DATE] = parsed_date

    # A pin cite belongs beside its own locator. Keep the window short so a
    # later page, docket entry, or date cannot become this citation's pin.
    pin = _ground_after_locator(
        review.reparsed_pin_cite,
        after_locator[:80],
        source_start=record.locator_span.end,
        document=document,
    )
    if pin is not None:
        pin_text, pin_span = pin
        parsed_pin = PinCite.read(pin_text, pin_span)
        if parsed_pin != citation.pin_cite:
            changes[CitationField.PIN_CITE] = parsed_pin

    if changes:
        update_fields(
            record,
            trace,
            changes,
            reason=review.reason or "Model re-read the target docket citation in the filing.",
        )
        return True
    observe_citation(record, trace)
    return False


def _ground_after_locator(
    quote: str | None,
    source: str,
    *,
    source_start: int,
    document: Document,
) -> tuple[str, Span] | None:
    if quote is None or not quote.strip():
        return None
    found = GroundingEvidence((EvidenceCandidate(source, None),)).find_fragment(quote, _LITERAL_SOURCE)
    if found is None:
        return None
    start = source_start + found.start
    end = source_start + found.end
    text = document.text[start:end]
    if text != found.text:
        return None
    return text, Span(start, end)


def _parse_source_date(value: str) -> CitationDate | None:
    matched = _SOURCE_DATE.fullmatch(value.strip())
    if matched is None:
        return None
    return CitationDate(
        year=matched.group("year"),
        month=matched.group("month"),
        day=matched.group("day"),
    )
