"""Adapt short-reporter colocations to the shared raw field prediction shape."""

from mellea_lrc.extraction.context.leaves import preceding_name
from mellea_lrc.field_extraction.models import FieldPrediction
from mellea_lrc.model.citations import ShortReporterCitation, latest
from mellea_lrc.model.citations.fields.base import current_reading
from mellea_lrc.model.citations.rule_field_reading import RuleFieldReading
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span

CASE_NAME_WINDOW = 160
POST_LOCATOR_WINDOW = 120


def with_short_rule_prediction(citation: ShortReporterCitation, document: Document) -> ShortReporterCitation:
    """Cache a group's written name and this occurrence's locator-derived pin.

    Short reporters do not read courts or dates. The name is shared through
    the first colocated member, while every member retains its own pinpoint.
    """
    if citation.rule_field_reading is not None:
        saved = citation.rule_field_reading
        if (saved.case_name_window, saved.post_locator_window) != (CASE_NAME_WINDOW, POST_LOCATOR_WINDOW):
            raise ValueError("Rule prediction windows changed; rewind before reading fields")
        return citation
    group_id = latest(citation.colocation_id)
    if group_id is None:
        first = citation
    else:
        group = next(group for group in document.short_reporter_colocations if group.id == group_id)
        first = next(item for item in document.short_reporters if item.id == group.citation_ids[0])
    name = preceding_name(document, first.short_locator_span)
    pin = current_reading(citation.pin_cite)
    prediction = FieldPrediction(
        context_span=Span(
            max(0, first.short_locator_span.start - CASE_NAME_WINDOW), citation.short_locator_span.end
        ),
        case_name=name,
        pin_cite=pin.span if pin is not None else None,
    )
    prediction.validate_source(document.text, citation.short_locator_span)
    return citation.with_rule_field_reading(
        RuleFieldReading(
            node_id=citation.nodes[-1].id,
            prediction=prediction,
            case_name_window=CASE_NAME_WINDOW,
            post_locator_window=POST_LOCATOR_WINDOW,
        )
    )
