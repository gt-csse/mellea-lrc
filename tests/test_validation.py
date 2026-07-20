"""Tests for the post-extraction validation-node progression."""

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
    LocatorLookupOutcome,
    ValidationNodeStatus,
    initialize_validation,
    validate_document,
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


def test_initialize_validation_instances_one_progression_per_extracted_citation() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483"))

    validation = initialize_validation(extracted)

    assert validation.source is extracted
    assert validation.text == extracted.text
    assert validation.citations[0].citation is extracted.citations[0]
    assert validation.citations[0].nodes == ()


def test_exact_locator_found_appends_the_only_implemented_node() -> None:
    extracted = _document(FullCaseCitation(volume="347", reporter="U.S.", page="483", year="1954"))
    record = CourtListenerCitationRecord(
        case_name="Brown v. Board of Education",
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
    assert len(progression.nodes) == 1
    node = progression.nodes[0]
    assert node.node_id == "cite-0001:exact_locator_lookup"
    assert node.status is ValidationNodeStatus.SUCCEEDED
    assert node.outcome is LocatorLookupOutcome.FOUND
    assert node.locator == "347 U.S. 483"
    assert node.record is record
    assert node.candidate_count == 1


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
