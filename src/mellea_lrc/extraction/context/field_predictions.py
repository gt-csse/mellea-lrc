"""Adapt document structure to the standalone span readers."""

from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.field_extraction.models import FieldPrediction
from mellea_lrc.field_extraction.rule import read_case_name, read_court_and_date, read_pin_cite
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.rule_field_reading import RuleFieldReading
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span


def with_root_rule_prediction(citation, document: Document, rules: ExtractionRules):
    """Cache all four rule spans once, using the document's colocation boundaries.

    Standalone readers receive source text and a locator. The document adapter
    additionally knows which adjacent locators share one citation, so it uses
    the same primitive rules on the group's bounded prefix and suffix.
    """
    if citation.rule_field_reading is not None:
        saved = citation.rule_field_reading
        if (saved.case_name_window, saved.post_locator_window) != (
            rules.case_name_window,
            rules.post_locator_window,
        ):
            raise ValueError("Rule prediction windows changed; rewind before reading fields")
        return citation
    prefix, start = before(document, citation, rules.case_name_window)
    suffix, suffix_start = after(document, citation, rules.post_locator_window)
    court, date = read_court_and_date(suffix, suffix_start)
    site = citation.locator_span
    next_start = min(
        (item.locator_span.start for item in document.full_locators if item.locator_span.start >= site.end),
        default=len(document.text),
    )
    stop = min(next_start, site.end + rules.post_locator_window)
    prediction = FieldPrediction(
        context_span=Span(min(start, site.start), max(suffix_start + len(suffix), stop, site.end)),
        case_name=read_case_name(
            prefix, start, citation.locator[-1].quote if isinstance(citation, FullReporterCitation) else None
        ),
        court=court,
        date=date,
        pin_cite=read_pin_cite(document.text, site.end, stop),
    )
    prediction.validate_source(document.text, site)
    return citation.with_rule_field_reading(
        RuleFieldReading(
            node_id=citation.nodes[-1].id,
            prediction=prediction,
            case_name_window=rules.case_name_window,
            post_locator_window=rules.post_locator_window,
        )
    )
