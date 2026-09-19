"""Contract tests for document-native root identity."""

from __future__ import annotations

import asyncio

from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.core.citations import CitationDate, FullCaseCitation, placed
from mellea_lrc.core.record import CitationRecord, Question
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerOpinionCluster
from mellea_lrc.extraction import Document, ExtractionMetadata
from mellea_lrc.preprocessing import preprocess


class _LookupClient:
    """A unique exact-lookup response; this route does not need a model call."""

    def __init__(self, cluster: CourtListenerOpinionCluster) -> None:
        self.cluster = cluster
        self.calls: list[tuple[str, str, str]] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.calls.append((volume, reporter, page))
        return CourtListenerCitationLookup(
            citation=f"{volume} {reporter} {page}", status=200, clusters=(self.cluster,)
        )


def test_root_identity_writes_trace_and_state_to_the_original_document() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    locator = "347 U.S. 483"
    locator_start = text.index(locator)
    record = CitationRecord(
        citation_id="cite-0001",
        source=placed(
            FullCaseCitation(
                plaintiff="Brown",
                defendant="Board",
                volume="347",
                reporter="U.S.",
                page="483",
                date=CitationDate(year="1954"),
                court="scotus",
            ),
            span=Span(0, len(text)),
            locator_span=Span(locator_start, locator_start + len(locator)),
            matched_text=locator,
        ),
    )
    document = Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="brown",
            case_name="Brown v. Board",
            date_filed="1954-05-17",
            court_id="scotus",
        )
    )

    formed = form_roots(document)
    lookup = asyncio.run(lookup_full_reporter_locators_exact(formed, client=client))
    assert lookup.citations[0].judgement(Question.LOCATOR_LOOKUP).outcome == "found"
    assert lookup.citations[0].judgement(Question.IDENTITY).outcome == "unjudged"

    unique = asyncio.run(validate_unique_full_reporter_locator_identities(lookup, client=client))
    result = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=client))
    resumed = asyncio.run(lookup_full_reporter_locators_exact(result, client=client))
    restored = Document.from_serialized(result.serialize())
    resolved = restored.citations[0]

    assert client.calls == [("347", "U.S.", "483")]
    assert resolved.found is not None and resolved.found.cluster_id == "brown"
    assert resolved.authority_id == "brown"
    assert resolved.judgement(Question.IDENTITY).outcome == "resolved"
    assert resolved.judgement(Question.IDENTITY).node_id == "cite-0001:locator_identity_resolution"
    assert resolved.trace[-1].details["validation_node_type"] == "LocatorIdentityResolutionNode"
    assert result.passes[-4:] == (
        "root_formation",
        "full_reporter_locator_exact_lookup",
        "full_reporter_locator_unique_identity",
        "full_reporter_locator_ambiguity_resolution",
    )
    assert resumed is result
    assert client.calls == [("347", "U.S.", "483")]


def test_root_identity_requires_explicit_root_formation() -> None:
    document = Document.from_source("A filing without citations.")

    try:
        asyncio.run(lookup_full_reporter_locators_exact(document, client=object()))
    except ValueError as error:
        assert str(error) == "Exact full-reporter lookup requires root_formation before validation."
    else:
        raise AssertionError("identity accepted a document without root formation")


class _AmbiguousLookupClient:
    """An over-limit exact lookup exercises deferred ambiguity without a model call."""

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        clusters = tuple(CourtListenerOpinionCluster(cluster_id=f"candidate-{index}") for index in range(20))
        return CourtListenerCitationLookup(
            citation=f"{volume} {reporter} {page}", status=300, clusters=clusters
        )


def test_ambiguity_stage_resumes_a_serialized_lookup_and_defers_over_limit_candidates() -> None:
    text = "See 347 U.S. 483."
    preprocessed = preprocess(text)
    record = CitationRecord(
        citation_id="cite-0001",
        source=placed(
            FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            span=Span(4, 16),
            locator_span=Span(4, 16),
            matched_text="347 U.S. 483",
        ),
    )
    document = Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )
    client = _AmbiguousLookupClient()

    lookup = asyncio.run(lookup_full_reporter_locators_exact(form_roots(document), client=client))
    checkpoint = Document.from_serialized(lookup.serialize())
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(checkpoint, client=client))
    completed = asyncio.run(resolve_full_reporter_locator_ambiguities(unique, client=client))
    root = completed.citations[0]

    assert root.judgement(Question.LOCATOR_LOOKUP).outcome == "deferred_to_ambiguity"
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
    selection = next(
        node for node in root.trace if node.details.get("validation_node_type") == "CandidateSelectionNode"
    )
    assert selection.details["validation"]["total_candidate_count"] == 20
    assert selection.details["validation"]["selected_candidate_count"] == 0
