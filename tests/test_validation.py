"""Tests for the post-extraction validation-node progression."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
    CourtListenerError,
)
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.validation import (
    CaseNameCheckNode,
    CaseNameCheckOutcome,
    CaseNameReextractionNode,
    CaseNameReextractionOutcome,
    CaseSearchNode,
    CaseSearchOutcome,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorLookupOutcome,
    RecheckedCaseNameNode,
    ValidationNodeStatus,
    YearCheckNode,
    initialize_validation,
    validate_document,
)
from mellea_lrc.validation.field_checks.mellea_case_name_check import mellea_case_names_match


class LookupClient:
    """Minimal deterministic exact-lookup client for pipeline tests."""

    def __init__(self, response: CourtListenerCitationLookup | CourtListenerError) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Record the locator and return or raise the configured outcome."""
        self.calls.append((volume, reporter, page))
        if isinstance(self.response, CourtListenerError):
            raise self.response
        return self.response


def _document(citation: FullCaseCitation | FullLawCitation) -> ExtractedDocument:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = preprocess_plain_text_from_string(text)
    locator = "347 U.S. 483"
    start = text.index(locator)
    extracted = ExtractedCitation(
        citation_id="cite-0001",
        span=Span(start, start + len(locator)),
        matched_text=locator,
        citation=citation,
    )
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(extracted,),
        extraction_metadata=ExtractionMetadata(),
    )


def test_initialize_validation_instances_one_progression_per_extracted_citation() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))

    validation = initialize_validation(extracted)

    assert validation.source is extracted
    assert validation.text == extracted.text
    assert validation.citations[0].citation is extracted.citations[0]
    assert validation.citations[0].nodes == ()


def test_exact_locator_found_fans_out_to_case_name_and_year_checks() -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            year="1954",
        )
    )
    record = CourtListenerCitationRecord(
        case_name="Brown v. Board",
        date_filed="1954-05-17",
        court_id="scotus",
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(record,),
        )
    )

    validation = validate_document(extracted, client=client)

    progression = validation.citation_by_id("cite-0001")
    assert client.calls == [("347", "U.S.", "483")]
    assert len(progression.nodes) == 3
    lookup, case_name, year = progression.nodes
    assert isinstance(lookup, ExactLocatorLookupNode)
    assert lookup.outcome is LocatorLookupOutcome.FOUND
    assert lookup.record is record
    assert isinstance(case_name, CaseNameCheckNode)
    assert case_name.outcome is CaseNameCheckOutcome.EXACT_MATCH
    assert case_name.depends_on == (lookup.node_id,)
    assert isinstance(year, YearCheckNode)
    assert year.outcome is FieldCheckOutcome.MATCH
    assert year.depends_on == (lookup.node_id,)


def _configure_mellea(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")


def _mock_reextraction(monkeypatch: pytest.MonkeyPatch) -> None:
    async def reextract(validation: object, **_kwargs: object) -> CaseNameReextractionNode:
        trigger = validation.nodes[-1]
        if not isinstance(trigger, (CaseNameCheckNode, ExactLocatorLookupNode)):
            trigger = next(
                node
                for node in reversed(validation.nodes)
                if isinstance(node, (CaseNameCheckNode, ExactLocatorLookupNode))
            )
        return CaseNameReextractionNode(
            node_id=f"{validation.citation_id}:case_name_reextract",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=CaseNameReextractionOutcome.ACCEPTED,
            plaintiff="Brown",
            defendant="Board of Education",
            depends_on=(trigger.node_id,),
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.execution.run_mellea_case_name_reextract",
        reextract,
    )


def test_found_case_name_uses_mellea_for_semantic_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
            year="1954",
        )
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(
                CourtListenerCitationRecord(
                    case_name="Brown v. Board of Education",
                    date_filed="1955-01-01",
                ),
            ),
        )
    )

    calls: list[dict[str, Any]] = []

    async def semantic_match(extracted: str, retrieved: str, **kwargs: Any) -> bool:
        calls.append({"extracted": extracted, "retrieved": retrieved, **kwargs})
        return True

    _configure_mellea(monkeypatch)
    monkeypatch.setattr(
        "mellea_lrc.validation.execution.mellea_case_names_match",
        semantic_match,
    )

    _, case_name, year = (
        validate_document(
            extracted,
            client=client,
            mellea_session=object(),
        )
        .citations[0]
        .nodes
    )

    assert case_name.status is ValidationNodeStatus.SUCCEEDED
    assert case_name.outcome is CaseNameCheckOutcome.SEMANTIC_MATCH
    assert calls[0]["extracted"] == "Brown v. Board"
    assert calls[0]["retrieved"] == "Brown v. Board of Education"
    assert year.status is ValidationNodeStatus.SUCCEEDED
    assert year.outcome is FieldCheckOutcome.MISMATCH


def test_case_name_semantic_match_uses_instruct_ivr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run case-name classification through the direct instruct IVR entrypoint."""
    calls: list[dict[str, object]] = []
    _configure_mellea(monkeypatch)

    async def run_instruct(_session: object, spec: object, **kwargs: object) -> SimpleNamespace:
        calls.append({"spec": spec, **kwargs})
        return SimpleNamespace(success=True, result=SimpleNamespace(value='{"verdict":"semantic_match"}'))

    monkeypatch.setattr(
        "mellea_lrc.validation.field_checks.mellea_case_name_check.run_instruct_ivr",
        run_instruct,
    )

    result = asyncio.run(
        mellea_case_names_match(
            extracted_case_name="Brown v. Board",
            retrieved_case_name="Brown v. Board of Education",
            session=object(),
        )
    )

    assert result is True
    spec = calls[0]["spec"]
    assert spec.grounding_context == {}
    assert spec.user_variables["extracted_case_name"] == "Brown v. Board"
    assert calls[0]["model_options"]["max_tokens"] == 128


def test_nonsemantic_case_name_is_available_for_future_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(
        FullCaseCitation(
            plaintiff="Brown",
            defendant="Board",
            volume="347",
            reporter="U.S.",
            page="483",
        )
    )
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(CourtListenerCitationRecord(case_name="Different v. Case"),),
        )
    )

    async def not_semantic(*_args: str, **_kwargs: Any) -> bool:
        return False

    reextraction_calls: list[object] = []

    async def run_reextract(_session: object, spec: object, **_kwargs: object) -> SimpleNamespace:
        reextraction_calls.append(spec)
        return SimpleNamespace(
            success=True,
            result=SimpleNamespace(
                value='{"classification":"complete_case_name","plaintiff":"Brown","defendant":"Board"}'
            ),
        )

    _configure_mellea(monkeypatch)
    monkeypatch.setattr(
        "mellea_lrc.validation.execution.mellea_case_names_match",
        not_semantic,
    )
    monkeypatch.setattr(
        "mellea_lrc.validation.field_checks.mellea_case_name_reextract.run_instruct_ivr",
        run_reextract,
    )

    lookup, case_name, year, reextraction, recheck = (
        validate_document(
            extracted,
            client=client,
            mellea_session=object(),
        )
        .citations[0]
        .nodes
    )

    assert case_name.status is ValidationNodeStatus.SUCCEEDED
    assert case_name.outcome is CaseNameCheckOutcome.NOT_SEMANTIC_MATCH
    assert case_name.error is None
    assert reextraction.depends_on == (case_name.node_id,)
    assert isinstance(recheck, RecheckedCaseNameNode)
    assert recheck.depends_on == (reextraction.node_id,)
    assert recheck.outcome is CaseNameCheckOutcome.NOT_SEMANTIC_MATCH
    assert lookup.node_id in year.depends_on
    spec = reextraction_calls[0]
    assert spec.user_variables == {"locator": "347 U.S. 483"}
    assert "retrieved_case_name" not in spec.user_variables
    assert "Brown v. Board" in spec.grounding_context["local_context"]


def test_found_field_checks_skip_unavailable_values() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(CourtListenerCitationRecord(),),
        )
    )

    _, case_name, year = validate_document(extracted, client=client).citations[0].nodes

    assert case_name.status is ValidationNodeStatus.SKIPPED
    assert case_name.outcome is CaseNameCheckOutcome.UNASSESSABLE
    assert year.status is ValidationNodeStatus.SKIPPED
    assert year.outcome is FieldCheckOutcome.UNAVAILABLE


def test_not_found_reextracts_then_routes_to_case_search_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="9999"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 9999",
            status=404,
            records=(),
        )
    )

    _mock_reextraction(monkeypatch)
    validation = validate_document(extracted, client=client)

    nodes = validation.citations[0].nodes
    assert len(nodes) == 3
    assert nodes[0].status is ValidationNodeStatus.SUCCEEDED
    assert nodes[0].outcome is LocatorLookupOutcome.NOT_FOUND
    assert nodes[0].record is None
    assert isinstance(nodes[1], CaseNameReextractionNode)
    assert nodes[1].depends_on == (nodes[0].node_id,)
    assert isinstance(nodes[2], CaseSearchNode)
    assert nodes[2].status is ValidationNodeStatus.SKIPPED
    assert nodes[2].outcome is CaseSearchOutcome.NOT_IMPLEMENTED
    assert nodes[2].depends_on == (nodes[0].node_id, nodes[1].node_id)


def test_ambiguous_lookup_stops_without_candidate_processing() -> None:
    extracted = _document(FullCaseCitation(volume="1", reporter="F.2d", page="2"))
    records = (
        CourtListenerCitationRecord(case_name="First"),
        CourtListenerCitationRecord(case_name="Second"),
    )
    client = LookupClient(CourtListenerCitationLookup(citation="1 F.2d 2", status=300, records=records))

    node = validate_document(extracted, client=client).citations[0].nodes[0]

    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is LocatorLookupOutcome.AMBIGUOUS
    assert node.candidate_count == 2
    assert node.record is None


def test_unsupported_citation_is_skipped_without_service_access() -> None:
    extracted = _document(FullLawCitation(volume="28", reporter="U.S.C.", page="636"))
    client = LookupClient(CourtListenerCitationLookup(citation="28 U.S.C. 636", status=200, records=()))

    node = validate_document(extracted, client=client).citations[0].nodes[0]

    assert client.calls == []
    assert node.status is ValidationNodeStatus.SKIPPED
    assert node.outcome is LocatorLookupOutcome.UNSUPPORTED_CITATION


def test_service_failure_is_a_terminal_validation_node() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(
        CourtListenerError(
            "service unavailable",
            failure_type="upstream_error",
            retryable=True,
        )
    )

    node = validate_document(extracted, client=client).citations[0].nodes[0]

    assert node.status is ValidationNodeStatus.FAILED
    assert node.outcome is LocatorLookupOutcome.FAILED
    assert node.error == "service unavailable"


def test_unexpected_lookup_response_raises() -> None:
    """Reject a response that violates the expected lookup contract."""
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(CourtListenerCitationLookup(citation="347 U.S. 483", status=200, records=()))

    with pytest.raises(AssertionError, match="Unexpected CourtListener lookup response"):
        validate_document(extracted, client=client)
