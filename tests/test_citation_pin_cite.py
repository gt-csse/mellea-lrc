"""Shared nullable pin-cite histories preserve source readings and checkpoints."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.extraction.supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.model.citations import (
    Citation,
    FullCitation,
    FullDocketCitation,
    FullReporterCitation,
    IdCitation,
    LeafCitation,
    Node,
    PinCiteField,
    PinCiteKind,
    PinCiteTarget,
    ReferenceCitation,
    ShortReporterCitation,
    SupraCitation,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

CONCRETE_TYPES = (
    FullReporterCitation,
    FullDocketCitation,
    ShortReporterCitation,
    IdCitation,
    SupraCitation,
    ReferenceCitation,
)
ALL_TYPES = (Citation, *CONCRETE_TYPES)
FIRST_PIN = (PinCiteTarget(first=495, last=497, kind=PinCiteKind.PAGE),)
SECOND_PIN = (PinCiteTarget(first=501, last=501, kind=PinCiteKind.PAGE, footnote="3"),)


def _span(source: str, quote: str) -> Span:
    start = source.index(quote)
    return Span(start, start + len(quote))


def _citation(citation_type: type[Citation]) -> tuple[str, Citation]:
    sites = {
        Citation: "Citation",
        FullReporterCitation: "347 U.S. 483",
        FullDocketCitation: "No. 1:24-cv-00001",
        ShortReporterCitation: "347 U.S. at 495",
        IdCitation: "Id.",
        SupraCitation: "Smith, supra",
        ReferenceCitation: "Smith",
    }
    site = sites[citation_type]
    source = f"{site}; 495-97; 501 n.3; 495a."
    span = _span(source, site)
    if citation_type is Citation:
        citation = Citation(id="citation", kind="test", nodes=(Node(id="citation:node:0", stage="create"),))
    elif citation_type is FullReporterCitation:
        citation = FullReporterCitation.from_locator(
            citation_id="reporter", stage="create", source=source, span=span
        )
    elif citation_type is FullDocketCitation:
        citation = FullDocketCitation.from_locator(
            citation_id="docket",
            stage="create",
            source=source,
            span=span,
            number_span=_span(source, "1:24-cv-00001"),
        )
    elif citation_type is ShortReporterCitation:
        citation = ShortReporterCitation.from_short_locator(
            citation_id="short_reporter", stage="create", source=source, span=span
        )
    else:
        citation = citation_type.from_source(source=source, span=span, stage="create")
    return source, citation


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
def test_unread_pin_cite_is_null_and_roundtrips(citation_type: type[Citation]) -> None:
    _, citation = _citation(citation_type)

    assert citation.pin_cite is None
    assert citation.get_pin_cite() is None
    saved = json.loads(citation.model_dump_json())
    assert saved["pin_cite"] is None
    assert citation_type.model_validate_json(citation.model_dump_json()) == citation
    del saved["pin_cite"]
    assert citation_type.model_validate(saved) == citation

    alternatives = citation_type.model_json_schema()["properties"]["pin_cite"]["anyOf"]
    assert any(item.get("type") == "null" for item in alternatives)
    assert any(item.get("type") == "array" and item.get("minItems") == 1 for item in alternatives)


def test_pin_cite_api_is_owned_by_the_shared_citation_base() -> None:
    for citation_type in (FullCitation, LeafCitation, *CONCRETE_TYPES):
        assert citation_type.with_pin_cite is Citation.with_pin_cite
        assert citation_type.get_pin_cite is Citation.get_pin_cite
    assert "pin_cite" not in FullCitation.__annotations__
    assert "pin_cite" not in LeafCitation.__annotations__


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
def test_pin_cite_reading_is_grounded_normalized_and_roundtrips(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    span = _span(source, "495-97")
    updated = citation.record("pin_parse").with_pin_cite(source, span)

    assert type(updated) is citation_type
    assert updated.get_pin_cite() == FIRST_PIN
    assert updated.pin_cite is not None
    reading = updated.pin_cite[0]
    assert reading.quote == source[span.start : span.end] == "495-97"
    assert reading.span == span
    assert reading.node_id == updated.nodes[-1].id
    assert reading.normalizable is True
    assert reading.normalization_error is None
    assert citation.pin_cite is None
    assert len(citation.nodes) == 1
    saved = json.loads(updated.model_dump_json())
    assert len(saved["pin_cite"]) == 1
    assert citation_type.model_validate_json(updated.model_dump_json()) == updated
    updated.validate_source(source)


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("empty", [(), []], ids=["tuple", "list"])
def test_empty_pin_cite_log_is_invalid(citation_type: type[Citation], empty: tuple | list) -> None:
    _, citation = _citation(citation_type)

    with pytest.raises(ValidationError, match="pin_cite"):
        citation_type.model_validate({**citation.model_dump(mode="python"), "pin_cite": empty})
    with pytest.raises(ValidationError, match="pin_cite"):
        citation_type.model_validate_json(json.dumps({**citation.model_dump(mode="json"), "pin_cite": empty}))


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
def test_failed_pin_reading_retains_quote_and_error(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    span = _span(source, "495a")
    failed = citation.record("pin_parse").with_pin_cite(source, span)
    restored = citation_type.model_validate_json(failed.model_dump_json())

    assert restored == failed
    assert restored.pin_cite is not None
    reading = restored.pin_cite[0]
    assert reading.quote == "495a"
    assert reading.span == span
    assert reading.normalizable is False
    assert reading.normalization_error
    assert reading.model_dump(mode="json")["normalized"] is None
    with pytest.raises(ValueError, match="not normalizable"):
        restored.get_pin_cite()

    corrected = restored.record("pin_review").with_pin_cite(source, _span(source, "501 n.3"))
    assert corrected.pin_cite[0] == reading
    assert corrected.get_pin_cite() == SECOND_PIN
    assert restored.pin_cite == (reading,)


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
def test_pin_updates_append_immutable_history(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    first = citation.record("pin_parse").with_pin_cite(source, _span(source, "495-97"))
    second = first.record("pin_review").with_pin_cite(source, _span(source, "501 n.3"))

    assert second.nodes[:-1] == first.nodes
    assert second.pin_cite[:-1] == first.pin_cite
    assert len(second.pin_cite) == 2
    assert first.get_pin_cite() == FIRST_PIN
    assert second.get_pin_cite() == SECOND_PIN
    assert citation_type.model_validate_json(second.model_dump_json()) == second
    with pytest.raises(ValidationError, match="frozen"):
        second.pin_cite = None
    with pytest.raises(ValidationError, match="frozen"):
        second.pin_cite[0].quote = "changed"


@pytest.mark.parametrize("citation_type", ALL_TYPES, ids=lambda cls: cls.__name__)
def test_pin_updates_require_a_decision_and_reject_same_node_duplicates(
    citation_type: type[Citation],
) -> None:
    source, citation = _citation(citation_type)
    with pytest.raises(ValueError, match="decision node"):
        citation.with_pin_cite(source, _span(source, "495-97"))
    first = citation.record("pin_parse").with_pin_cite(source, _span(source, "495-97"))
    with pytest.raises(ValueError, match="pin_cite updates are out of order"):
        first.with_pin_cite(source, _span(source, "501 n.3"))


@pytest.mark.parametrize("citation_type", CONCRETE_TYPES, ids=lambda cls: cls.__name__)
def test_document_accepts_first_pin_log_with_multiple_new_nodes(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    original = Document.from_source(source).add_citation(citation).complete("create")
    first = citation.record("pin_parse").with_pin_cite(source, _span(source, "495-97"))
    second = first.record("pin_parse").with_pin_cite(source, _span(source, "501 n.3"))

    updated = original.replace_citation(second).complete("pin_parse")
    assert updated.citations[0].pin_cite == second.pin_cite
    assert len(updated.citations[0].pin_cite) == 2
    assert updated.citations[0].get_pin_cite() == SECOND_PIN
    assert updated.get_stage("create") == original
    assert original.citations[0].pin_cite is None
    assert Document.model_validate_json(updated.model_dump_json()) == updated


@pytest.mark.parametrize("citation_type", CONCRETE_TYPES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("change", ["erase", "truncate", "mutate"])
def test_document_rejects_changes_to_existing_pin_history(citation_type: type[Citation], change: str) -> None:
    source, citation = _citation(citation_type)
    original = Document.from_source(source).add_citation(citation).complete("create")
    citation = citation.record("pin_parse").with_pin_cite(source, _span(source, "495-97"))
    citation = citation.record("pin_parse").with_pin_cite(source, _span(source, "501 n.3"))
    original = original.replace_citation(citation).complete("pin_parse")
    candidate = citation.record("pin_review")
    if change == "erase":
        pin_cite = None
    elif change == "truncate":
        pin_cite = citation.pin_cite[:1]
    else:
        replacement = PinCiteField.from_source(
            source, _span(source, "501 n.3"), node_id=citation.pin_cite[0].node_id
        )
        pin_cite = (replacement, citation.pin_cite[1])
    candidate = citation_type.model_validate({**candidate.model_dump(mode="python"), "pin_cite": pin_cite})

    with pytest.raises(ValueError, match="pin_cite must be append-only"):
        original.replace_citation(candidate)


@pytest.mark.parametrize("citation_type", CONCRETE_TYPES, ids=lambda cls: cls.__name__)
def test_document_rejects_pin_reading_without_new_node(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    original = Document.from_source(source).add_citation(citation).complete("create")
    citation = citation.record("pin_parse")
    original = original.replace_citation(citation)
    candidate = citation.with_pin_cite(source, _span(source, "495-97"))

    with pytest.raises(ValueError, match="new decision node"):
        original.replace_citation(candidate)


@pytest.mark.parametrize("citation_type", CONCRETE_TYPES, ids=lambda cls: cls.__name__)
def test_document_checkpoints_recover_null_and_exact_pin_histories(citation_type: type[Citation]) -> None:
    source, citation = _citation(citation_type)
    creation = Document.from_source(source).add_citation(citation).complete("create")
    first = citation.record("pin_parse").with_pin_cite(source, _span(source, "495-97"))
    parsed = creation.replace_citation(first).complete("pin_parse")
    second = first.record("pin_review").with_pin_cite(source, _span(source, "501 n.3"))
    reviewed = parsed.replace_citation(second).complete("pin_review")
    restored = Document.model_validate_json(reviewed.model_dump_json())

    assert restored.get_stage("create") == creation
    assert restored.get_stage("create").citations[0].pin_cite is None
    assert restored.get_stage("pin_parse") == parsed
    assert restored.get_stage("pin_parse").citations[0].get_pin_cite() == FIRST_PIN
    assert restored.get_stage("pin_review") == reviewed
    assert restored.citations[0].get_pin_cite() == SECOND_PIN


@pytest.mark.parametrize(
    "source,expected", [("Smith, supra.", None), ("Smith, supra, at 495-97.", FIRST_PIN)]
)
def test_supra_pin_reading_uses_shared_nullable_history(source: str, expected: tuple | None) -> None:
    discovered = find_supra_citations(Document.from_source(source).complete("10_roots"))
    assert discovered.citations[0].pin_cite is None
    document = resolve_supra_pin_cites(discovered)

    assert len(document.citations) == 1
    citation = document.citations[0]
    assert isinstance(citation, SupraCitation)
    assert citation.get_pin_cite() == expected
    if expected is None:
        assert citation.pin_cite is None
    else:
        assert citation.pin_cite[-1].quote == "495-97"
        span = citation.pin_cite[-1].span
        assert source[span.start : span.end] == "495-97"
    restored = Document.model_validate_json(document.model_dump_json())
    assert restored == document
    assert restored.get_stage("34_supra_citations") == discovered
    assert restored.get_stage("34_supra_citations").citations[0].pin_cite is None
