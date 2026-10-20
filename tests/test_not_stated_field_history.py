"""Explicit absence is a reversible append-only field reading."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.model.citations import (
    CaseNameKind,
    FullReporterCitation,
    MatchResult,
    NotStatedField,
    current_reading,
    latest,
)
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span
from mellea_lrc.providers.courtlistener.models import CourtListenerCitationLookup

SOURCE = "Smith v. Jones, 347 U.S. 483, 495 (9th Cir. 1954)."
FIELDS = ("case_name", "court", "date", "pin_cite")
QUOTES = {"case_name": "Smith v. Jones", "court": "9th Cir.", "date": "1954", "pin_cite": "495"}


def _span(quote: str) -> Span:
    start = SOURCE.index(quote)
    return Span(start, start + len(quote))


def _created() -> Document:
    citation = FullReporterCitation.from_locator(
        citation_id="reporter", substage="create", source=SOURCE, span=_span("347 U.S. 483")
    )
    return Document.from_source(SOURCE).add_citation(citation).complete_substage("create")


def _parsed() -> Document:
    document = _created()
    citation = document.citations[0].record("read_fields")
    for field in FIELDS:
        citation = getattr(citation, f"with_{field}")(SOURCE, _span(QUOTES[field]))
    return document.replace_citation(citation).complete_substage("read_fields")


@pytest.mark.parametrize("field", FIELDS)
def test_clearing_appends_absence_without_falling_back_to_old_reading(field: str) -> None:
    original = _parsed()
    previous = original.citations[0]
    citation = getattr(previous.record("review"), f"with_{field}_not_stated")()
    history = getattr(citation, field)

    assert history[:-1] == getattr(previous, field)
    assert isinstance(history[-1], NotStatedField)
    assert history[-1].node_id == citation.nodes[-1].id
    assert history[-1].span is None
    assert history[-1].quote is None
    assert history[-1].normalizable is False
    assert current_reading(history) is None
    assert latest(history) is None
    assert current_reading(history[:-1]) is history[-2]
    if field == "case_name":
        assert citation.get_case_name().kind is CaseNameKind.NOT_STATED
    if field == "pin_cite":
        assert citation.get_pin_cite() is None
    with pytest.raises(ValueError, match="not stated"):
        history[-1].get_normalized()

    updated = original.replace_citation(citation).complete_substage("review")
    restored = Document.model_validate_json(updated.model_dump_json())
    assert restored == updated
    assert restored.get_substage("read_fields") == original
    assert restored.get_substage("create") == _created()
    assert restored.citations[0].locator == previous.locator


@pytest.mark.parametrize("field", FIELDS)
def test_not_stated_can_be_first_reading_and_later_replaced(field: str) -> None:
    document = _created()
    citation = getattr(document.citations[0].record("review"), f"with_{field}_not_stated")()
    cleared = document.replace_citation(citation).complete_substage("review")
    reading = getattr(citation.record("reread"), f"with_{field}")(SOURCE, _span(QUOTES[field]))
    restored = cleared.replace_citation(reading).complete_substage("reread")

    assert current_reading(getattr(restored.citations[0], field)).quote == QUOTES[field]
    assert isinstance(getattr(restored.citations[0], field)[0], NotStatedField)
    assert restored.get_substage("review") == cleared
    assert Document.model_validate_json(restored.model_dump_json()) == restored


def test_all_fields_can_be_cleared_at_one_review_node() -> None:
    document = _parsed()
    citation = document.citations[0].record("review")
    for field in FIELDS:
        citation = getattr(citation, f"with_{field}_not_stated")()
    updated = document.replace_citation(citation).complete_substage("review")
    saved = json.loads(updated.model_dump_json())["citations"][0]

    for field in FIELDS:
        assert saved[field][-1] == {"node_id": citation.nodes[-1].id, "kind": "not_stated"}
    updated.citations[0].validate_source(SOURCE)
    assert Document.model_validate_json(updated.model_dump_json()) == updated


@pytest.mark.parametrize("field", FIELDS)
def test_not_stated_requires_a_decision_node_and_rejects_duplicate_field_updates(field: str) -> None:
    created = _created().citations[0]
    with pytest.raises(ValueError, match="decision node"):
        getattr(created, f"with_{field}_not_stated")()
    cleared = getattr(created.record("review"), f"with_{field}_not_stated")()
    with pytest.raises(ValueError, match=f"{field} updates are out of order"):
        getattr(cleared, f"with_{field}_not_stated")()
    with pytest.raises(ValueError, match=f"{field} updates are out of order"):
        getattr(cleared, f"with_{field}")(SOURCE, _span(QUOTES[field]))
    parsed = getattr(created.record("review"), f"with_{field}")(SOURCE, _span(QUOTES[field]))
    with pytest.raises(ValueError, match=f"{field} updates are out of order"):
        getattr(parsed, f"with_{field}_not_stated")()


@pytest.mark.parametrize("field", FIELDS)
def test_not_stated_rejects_missing_or_out_of_order_nodes(field: str) -> None:
    citation = _parsed().citations[0]
    data = citation.model_dump(mode="python")
    data[field] = (*getattr(citation, field), NotStatedField(node_id="missing"))
    with pytest.raises(ValidationError, match="missing node"):
        FullReporterCitation.model_validate(data)
    data[field] = (*getattr(citation, field), NotStatedField(node_id=citation.nodes[0].id))
    with pytest.raises(ValidationError, match="out of order"):
        FullReporterCitation.model_validate(data)


@pytest.mark.parametrize("field", FIELDS)
def test_clearing_cannot_delete_or_rewrite_history(field: str) -> None:
    original = _parsed()
    citation = getattr(original.citations[0].record("review"), f"with_{field}_not_stated")()
    erased = FullReporterCitation.model_validate(
        {**citation.model_dump(mode="python"), field: (getattr(citation, field)[-1],)}
    )
    with pytest.raises(ValueError, match="append-only"):
        original.replace_citation(erased)


def test_required_locator_cannot_be_cleared_with_absence_marker() -> None:
    citation = _created().citations[0]
    with pytest.raises(ValidationError, match="locator"):
        FullReporterCitation.model_validate(
            {
                **citation.model_dump(mode="python"),
                "locator": (NotStatedField(node_id=citation.nodes[0].id),),
            }
        )


def test_marker_cannot_hold_quote_or_normalized_values() -> None:
    for extra in ({"quote": "Smith"}, {"normalized": "Smith"}, {"span": Span(0, 5)}):
        with pytest.raises(ValidationError, match="Extra inputs"):
            NotStatedField.model_validate({"node_id": "node", **extra})


def test_current_reading_does_not_hide_normalization_failure() -> None:
    citation = _created().citations[0].record("read_fields").with_pin_cite(SOURCE, _span("Jones"))
    reading = current_reading(citation.pin_cite)
    assert reading is citation.pin_cite[-1]
    assert reading.normalizable is False
    with pytest.raises(ValueError, match="not normalizable"):
        latest(citation.pin_cite)
    assert current_reading(()) is None
    assert current_reading(None) is None


@pytest.mark.parametrize("field", ("case_name", "court", "date"))
def test_judgments_can_reference_absence_but_cannot_treat_it_as_a_value(field: str) -> None:
    citation = _parsed().citations[0].record("lookup")
    lookup = ReporterExactLookup(
        node_id=citation.nodes[-1].id,
        outcome=ReporterExactLookupOutcome.UNIQUE,
        query=ReporterExactLookupQuery(volume=347, edition="U.S.", page="483"),
        response=CourtListenerCitationLookup.model_validate(
            {"citation": "347 U.S. 483", "status": 200, "clusters": [{"id": 1}]}
        ),
    )
    citation = citation.with_reporter_exact_lookup(lookup).record("review")
    citation = getattr(citation, f"with_{field}_not_stated")()
    index = len(getattr(citation, field)) - 1
    judged = getattr(citation, f"with_{field}_judgment")(index, 0, MatchResult.UNAVAILABLE)
    assert getattr(judged, f"{field}_judgments")[-1].reading_index == index
    assert FullReporterCitation.model_validate_json(judged.model_dump_json()) == judged
    for verdict in (MatchResult.MATCH, MatchResult.MISMATCH):
        with pytest.raises(ValueError, match="not-stated field reading"):
            getattr(citation, f"with_{field}_judgment")(index, 0, verdict)
