"""Tests for the post-extraction validation-node progression."""

import asyncio
from types import SimpleNamespace

import pytest
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel

from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerCitationRecord,
    CourtListenerError,
)
from mellea_lrc.extraction import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.preprocessing import preprocess_plain_text_from_string
from mellea_lrc.validation import (
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    ValidationNodeStatus,
    YearCheckNode,
    initialize_validation,
    validate_document,
)
from mellea_lrc.validation.field_checks.mellea_case_name_reextraction import (
    run_mellea_case_name_reextraction,
)


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


def test_instruct_ivr_forwards_the_pydantic_output_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward the schema through Mellea's structured-output interface."""

    class ExpectedOutput(BaseModel):
        value: str

    calls: list[dict[str, object]] = []

    def fake_instruct(*_args: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return '{"value":"structured"}'

    monkeypatch.setattr("mellea_lrc.llm.ivr.mfuncs.instruct", fake_instruct)
    result = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=object()),
            InstructIvrSpec(description="Return a value.", output_format=ExpectedOutput),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options={},
        )
    )

    assert result == '{"value":"structured"}'
    assert calls[0]["format"] is ExpectedOutput


def test_initialize_validation_instances_one_progression_per_extracted_citation() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))

    validation = initialize_validation(extracted)

    assert validation.source is extracted
    assert validation.text == extracted.text
    assert validation.citations[0].citation is extracted.citations[0]
    assert validation.citations[0].nodes == ()


def test_exact_locator_found_fans_out_to_exact_case_name_and_year_checks() -> None:
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
    exact_locator_lookup_node, exact_case_name_check_node, year_check_node = progression.nodes
    assert isinstance(exact_locator_lookup_node, ExactLocatorLookupNode)
    assert exact_locator_lookup_node.outcome is LocatorLookupOutcome.FOUND
    assert exact_locator_lookup_node.record is record
    assert isinstance(exact_case_name_check_node, ExactCaseNameCheckNode)
    assert exact_case_name_check_node.outcome is FieldCheckOutcome.MATCH
    assert exact_case_name_check_node.depends_on == (exact_locator_lookup_node.node_id,)
    assert isinstance(year_check_node, YearCheckNode)
    assert year_check_node.outcome is FieldCheckOutcome.MATCH
    assert year_check_node.depends_on == (exact_locator_lookup_node.node_id,)


def test_found_field_checks_record_mismatch_without_failing_execution() -> None:
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
                    case_name="Different v. Case",
                    date_filed="1955-01-01",
                ),
            ),
        )
    )

    _, exact_case_name_check_node, year_check_node = (
        validate_document(extracted, client=client).citations[0].nodes
    )

    assert exact_case_name_check_node.status is ValidationNodeStatus.SUCCEEDED
    assert exact_case_name_check_node.outcome is FieldCheckOutcome.MISMATCH
    assert year_check_node.status is ValidationNodeStatus.SUCCEEDED
    assert year_check_node.outcome is FieldCheckOutcome.MISMATCH


def test_found_field_checks_skip_unavailable_values() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 483",
            status=200,
            records=(CourtListenerCitationRecord(),),
        )
    )

    _, exact_case_name_check_node, year_check_node = (
        validate_document(extracted, client=client).citations[0].nodes
    )

    assert exact_case_name_check_node.status is ValidationNodeStatus.SKIPPED
    assert exact_case_name_check_node.outcome is FieldCheckOutcome.UNAVAILABLE
    assert year_check_node.status is ValidationNodeStatus.SKIPPED
    assert year_check_node.outcome is FieldCheckOutcome.UNAVAILABLE


def test_mellea_case_name_reextraction_uses_only_local_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ground Mellea re-extraction in local text without a retrieved case name."""
    document = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))
    validation = initialize_validation(document).citations[0]
    exact_locator_lookup_node = ExactLocatorLookupNode(
        node_id="cite-0001:exact_locator_lookup",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorLookupOutcome.FOUND,
        locator="347 U.S. 483",
        record=CourtListenerCitationRecord(case_name="Brown v. Board of Education"),
        candidate_count=1,
    )
    validation = validation.append(exact_locator_lookup_node)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test-model")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test-key")
    calls: list[object] = []

    async def fake_instruct(_session: object, spec: object, **_kwargs: object) -> SimpleNamespace:
        calls.append(spec)
        return SimpleNamespace(
            success=True,
            result=SimpleNamespace(
                value=(
                    '{"classification":"complete_case_name",'
                    '"plaintiff":"Brown","defendant":"Board"}'
                )
            ),
        )

    monkeypatch.setattr(
        "mellea_lrc.validation.field_checks.mellea_case_name_reextraction.run_instruct_ivr",
        fake_instruct,
    )

    node = asyncio.run(
        run_mellea_case_name_reextraction(
            validation,
            document_text=document.text,
            session=object(),
        )
    )

    assert isinstance(node, MelleaCaseNameReextractionNode)
    assert node.outcome is MelleaCaseNameReextractionOutcome.COMPLETE
    assert node.plaintiff == "Brown"
    assert node.defendant == "Board"
    assert node.depends_on == (exact_locator_lookup_node.node_id,)
    spec = calls[0]
    assert spec.user_variables == {"locator": "347 U.S. 483"}
    assert spec.grounding_context.keys() == {"local_context"}
    assert "Brown v. Board" in spec.grounding_context["local_context"]
    assert spec.output_format.__name__ == "_PartyProposal"


def test_not_found_stops_without_starting_a_fallback_branch() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="9999"))
    client = LookupClient(
        CourtListenerCitationLookup(
            citation="347 U.S. 9999",
            status=404,
            records=(),
        )
    )

    validation = validate_document(extracted, client=client)

    nodes = validation.citations[0].nodes
    assert len(nodes) == 1
    assert nodes[0].status is ValidationNodeStatus.SUCCEEDED
    assert nodes[0].outcome is LocatorLookupOutcome.NOT_FOUND
    assert nodes[0].record is None


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
