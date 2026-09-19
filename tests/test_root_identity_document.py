"""Contract tests for document-native root identity."""

from __future__ import annotations

import asyncio

from mellea_lrc.api import validate_roots_identity
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

    result = asyncio.run(validate_roots_identity(document, client=client))
    resumed = asyncio.run(validate_roots_identity(result, client=client))
    restored = Document.from_serialized(result.serialize())
    resolved = restored.citations[0]

    assert client.calls == [("347", "U.S.", "483")]
    assert resolved.found is not None and resolved.found.cluster_id == "brown"
    assert resolved.authority_id == "brown"
    assert resolved.judgement(Question.IDENTITY).outcome == "resolved"
    assert resolved.judgement(Question.IDENTITY).node_id == "cite-0001:locator_identity_resolution"
    assert resolved.trace[-1].details["validation_node_type"] == "LocatorIdentityResolutionNode"
    assert result.passes[-1] == "root_identity"
    assert resumed is result
    assert client.calls == [("347", "U.S.", "483")]
