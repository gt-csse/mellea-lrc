"""Contract tests for document-native root identity."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from mellea_lrc.api import (
    form_roots,
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    review_full_reporter_exact_dates,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerDocket,
    CourtListenerOpinionCluster,
    CourtListenerOpinionClusterCitation,
)
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationDate, CitationField, FullCaseCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.record import CitationRecord, Question
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.types import (
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReviewNode,
    ValidationNodeStatus,
)
from tests.record_fixtures import read_citation, revise_citation


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


class _DocketCourtClient(_LookupClient):
    def __init__(self, cluster: CourtListenerOpinionCluster) -> None:
        super().__init__(cluster)
        self.docket_calls = 0

    def get_docket(self, docket_id: str) -> CourtListenerDocket:
        self.docket_calls += 1
        return CourtListenerDocket(docket_id=docket_id, court_id="cand")


def _reporter_court_document(
    citation_date: CitationDate | None = None, *, missing_date: bool = False
) -> Document:
    text = "Smith v. Jones, 157 F.R.D. 477 (C.D. Cal. 1994)."
    preprocessed = preprocess(text)
    locator = "157 F.R.D. 477"
    start = text.index(locator)
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(
            read_citation(
                citation_id="cite-court",
                fields=placed(
                    FullCaseCitation(
                        plaintiff="Smith",
                        defendant="Jones",
                        volume="157",
                        reporter="F.R.D.",
                        page="477",
                        date=None if missing_date else citation_date or CitationDate(year="1994"),
                        court="cacd",
                    ),
                    span=Span(0, len(text)),
                    locator_span=Span(start, start + len(locator)),
                    matched_text=locator,
                ),
            ),
        ),
        extraction_metadata=ExtractionMetadata(),
    )


def test_exact_opinion_defers_conflicting_docket_only_court() -> None:
    client = _DocketCourtClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed="1994-06-01",
            court_id=None,
            docket_id="docket-1",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    looked_up = asyncio.run(
        lookup_full_reporter_locators_exact(form_roots(_reporter_court_document()), client=client)
    )
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))

    assert reviewed.citations[0].judgement(Question.IDENTITY).outcome == "deferred_to_search"
    assert client.docket_calls == 1


def test_exact_opinion_prefers_its_own_court_over_linked_docket() -> None:
    client = _DocketCourtClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed="1994-06-01",
            court_id="cacd",
            docket_id="docket-1",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    looked_up = asyncio.run(
        lookup_full_reporter_locators_exact(form_roots(_reporter_court_document()), client=client)
    )
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))

    assert reviewed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"
    assert client.docket_calls == 0


@pytest.mark.parametrize(
    ("caption", "plaintiff", "defendant"),
    [
        ("Smith v. Jones", "Smith", "Jones"),
        ("Matter of M4 Enters., Inc.", None, "M4 Enters., Inc."),
    ],
)
def test_reporter_combined_review_updates_case_name_through_document_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    caption: str,
    plaintiff: str | None,
    defendant: str,
) -> None:
    text = f"{caption}, 157 F.R.D. 477 (C.D. Cal. 1994)."
    locator = "157 F.R.D. 477"
    preprocessed = preprocess(text)
    initial_name = (
        CaseName(span=Span(0, len(plaintiff)), text=plaintiff, plaintiff=plaintiff)
        if plaintiff is not None
        else None
    )
    record = read_citation(
        citation_id="cite-recovered",
        fields=placed(
            FullCaseCitation(
                case_name=initial_name,
                volume="157",
                reporter="F.R.D.",
                page="477",
                date=CitationDate(year="1994"),
                court="cacd",
            ),
            span=Span(0, len(text)),
            locator_span=Span(text.index(locator), text.index(locator) + len(locator)),
            matched_text=locator,
        ),
    )
    document = form_roots(
        Document(
            source_metadata=preprocessed.source_metadata,
            text=text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(record,),
            extraction_metadata=ExtractionMetadata(),
        )
    )
    assert document.citations[0].case_name == initial_name
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="recovered",
            case_name=caption,
            date_filed="1994-06-01",
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )

    async def combined_review(_validation, *, trigger, **_kwargs):
        return MelleaCaseNameReviewNode(
            node_id=f"{trigger.node_id}:mellea_case_name_review",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.MATCH,
            retrieved_case_name=caption,
            plaintiff=plaintiff,
            defendant=defendant,
            case_name=CaseName(
                span=Span(0, len(caption)),
                text=caption,
                plaintiff=plaintiff,
                defendant=defendant,
            ),
            rationale="The reread source name matches the retrieved opinion.",
            depends_on=(trigger.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_case_name_review", combined_review)

    looked_up = asyncio.run(lookup_full_reporter_locators_exact(document, client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))
    restored = Document.model_validate(reviewed.model_dump(mode="json"))
    root = restored.citations[0]

    recovered = root.case_name
    assert recovered == CaseName(
        span=Span(0, len(caption)),
        text=caption,
        plaintiff=plaintiff,
        defendant=defendant,
    )
    assert root.get_field(CitationField.CASE_NAME) == recovered
    assert root.fields.case_name == recovered
    assert restored.text[recovered.span.start : recovered.span.end] == recovered.text
    updates = [update for update in root.field_updates if update.field is CitationField.CASE_NAME]
    assert len(updates) == (2 if initial_name is not None else 1)
    assert updates[-1].before == initial_name
    assert updates[-1].after == recovered
    assert updates[-1].node_id.endswith(":mellea_case_name_review")


def test_reporter_partial_reread_updates_party_without_replacing_complete_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _reporter_court_document()
    full_name = CaseName(
        span=Span(0, len("Smith v. Jones")),
        text="Smith v. Jones",
        plaintiff="Smith",
        defendant="Jones",
    )
    revise_citation(document.citations[0], case_name=full_name, plaintiff=None)
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="other-smith",
            case_name="Smith v. Allen",
            date_filed="1994-06-01",
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )

    async def partial_review(_validation, *, trigger, **_kwargs):
        return MelleaCaseNameReviewNode(
            node_id=f"{trigger.node_id}:mellea_case_name_review",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaCaseNameCheckOutcome.UNAVAILABLE,
            retrieved_case_name="Smith v. Allen",
            case_name=None,
            plaintiff="Smith",
            defendant=None,
            rationale="The filing supplies only one party for this review.",
            depends_on=(trigger.node_id,),
        )

    monkeypatch.setattr("mellea_lrc.validation.execution.run_mellea_case_name_review", partial_review)

    looked_up = asyncio.run(lookup_full_reporter_locators_exact(form_roots(document), client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))
    root = Document.model_validate(reviewed.model_dump(mode="json")).citations[0]

    assert root.case_name == full_name
    assert root.get_field(CitationField.PLAINTIFF) == "Smith"
    assert root.fields.defendant == "Jones"
    reread_updates = [
        update for update in root.field_updates if update.node_id.endswith(":mellea_case_name_review")
    ]
    assert [(update.field, update.before, update.after) for update in reread_updates] == [
        (CitationField.PLAINTIFF, None, "Smith")
    ]


def test_exact_reporter_lookup_compares_full_decision_date() -> None:
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed="1994-06-01",
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    source = _reporter_court_document(CitationDate(year="1994", month="June", day="1"))
    looked_up = asyncio.run(lookup_full_reporter_locators_exact(form_roots(source), client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))

    assert reviewed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"
    check = next(
        node
        for node in reviewed.citations[0].trace
        if node.details.get("validation_node_type") == "YearCheckNode"
    ).details["validation"]
    assert check["outcome"] == "match"
    assert check["extracted_date"] == check["retrieved_date"] == "1994-06-01"


def test_exact_reporter_lookup_rejects_different_full_decision_date_directly() -> None:
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed="1994-06-01",
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    source = _reporter_court_document(CitationDate(year="1994", month="June", day="2"))
    looked_up = asyncio.run(lookup_full_reporter_locators_exact(form_roots(source), client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))

    assert reviewed.citations[0].judgement(Question.IDENTITY).outcome == "no_match"
    check = next(
        node
        for node in reviewed.citations[0].trace
        if node.details.get("validation_node_type") == "YearCheckNode"
    ).details["validation"]
    assert check["outcome"] == "mismatch"
    assert check["extracted_date"] == "1994-06-02"
    assert check["retrieved_date"] == "1994-06-01"
    later = asyncio.run(review_full_reporter_exact_dates(reviewed, client=client))
    assert later.citations[0].judgement(Question.IDENTITY).outcome == "no_match"
    assert later.citations[0].trace == reviewed.citations[0].trace


@pytest.mark.parametrize(
    ("citation_date", "record_date", "expected_outcome", "expected_identity"),
    [
        (CitationDate(year="1994", month="June", day="1"), "1994", "match", "resolved"),
        (CitationDate(year="1994"), "1994-06-01", "match", "resolved"),
        (CitationDate(year="1994", month="June", day="1"), "1995", "mismatch", "no_match"),
    ],
)
def test_exact_reporter_lookup_compares_year_when_only_one_side_has_a_year(
    citation_date: CitationDate,
    record_date: str,
    expected_outcome: str,
    expected_identity: str,
) -> None:
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed=record_date,
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    source = _reporter_court_document(citation_date)
    looked_up = asyncio.run(lookup_full_reporter_locators_exact(form_roots(source), client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))
    restored = Document.model_validate(reviewed.model_dump(mode="json"))

    root = restored.citations[0]
    check = next(
        node for node in root.trace if node.details.get("validation_node_type") == "YearCheckNode"
    ).details["validation"]
    assert check["outcome"] == expected_outcome
    assert check["extracted_year"] == citation_date.year
    assert check["retrieved_year"] == record_date[:4]
    assert root.judgement(Question.IDENTITY).outcome == expected_identity


@pytest.mark.parametrize("missing_side", ["citing", "source"])
def test_exact_reporter_lookup_ignores_missing_date_in_identity(
    missing_side: str,
) -> None:
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed=None if missing_side == "source" else "1994-06-01",
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    source = _reporter_court_document(
        CitationDate(year="1994", month="June", day="1"),
        missing_date=missing_side == "citing",
    )
    looked_up = asyncio.run(lookup_full_reporter_locators_exact(form_roots(source), client=client))
    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(looked_up, client=client))
    restored = Document.model_validate(reviewed.model_dump(mode="json"))

    root = restored.citations[0]
    check = next(
        node for node in root.trace if node.details.get("validation_node_type") == "YearCheckNode"
    ).details["validation"]
    assert check["outcome"] is None
    assert root.judgement(Question.IDENTITY).outcome == "resolved"


def test_exact_opinion_other_dates_survive_document_checkpoint() -> None:
    other_dates = "Argued and Submitted April 18, 2013., Amended Feb. 5, 2014."
    client = _LookupClient(
        CourtListenerOpinionCluster(
            cluster_id="smith",
            case_name="Smith v. Jones",
            date_filed="1994-06-01",
            other_dates=other_dates,
            court_id="cacd",
            citations=(CourtListenerOpinionClusterCitation("157", "F.R.D.", "477"),),
        )
    )
    looked_up = asyncio.run(
        lookup_full_reporter_locators_exact(form_roots(_reporter_court_document()), client=client)
    )
    restored = Document.model_validate(looked_up.model_dump(mode="json"))
    lookup_node = next(
        node
        for node in restored.citations[0].trace
        if node.details.get("validation_node_type") == "ExactLocatorLookupNode"
    )
    assert lookup_node.details["validation"]["cluster"]["other_dates"] == other_dates

    reviewed = asyncio.run(validate_unique_full_reporter_locator_identities(restored, client=client))
    candidate_node = next(
        node
        for node in reviewed.citations[0].trace
        if node.details.get("validation_node_type") == "CandidateEvaluationNode"
    )
    assert candidate_node.details["validation"]["record"]["other_dates"] == other_dates


def test_root_identity_writes_trace_and_state_to_the_original_document() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    locator = "347 U.S. 483"
    locator_start = text.index(locator)
    record = read_citation(
        citation_id="cite-0001",
        fields=placed(
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
    restored = Document.model_validate(result.model_dump(mode="json"))
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
    assert resumed == result
    assert resumed is not result
    assert client.calls == [("347", "U.S.", "483")]
    with pytest.raises(ValueError, match="restart from the preceding checkpoint"):
        asyncio.run(
            resolve_full_reporter_locator_ambiguities(
                result, client=client, retrospective_date=date(1954, 12, 31)
            )
        )


def test_exact_lookup_cutoff_excludes_later_opinion_before_identity_review() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954)."
    preprocessed = preprocess(text)
    start = text.index("347 U.S. 483")
    record = read_citation(
        citation_id="cite-0001",
        fields=placed(
            FullCaseCitation(volume="347", reporter="U.S.", page="483"),
            span=Span(0, len(text)),
            locator_span=Span(start, start + len("347 U.S. 483")),
            matched_text="347 U.S. 483",
        ),
    )
    document = form_roots(
        Document(
            source_metadata=preprocessed.source_metadata,
            text=text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(record,),
            extraction_metadata=ExtractionMetadata(),
        )
    )
    client = _LookupClient(
        CourtListenerOpinionCluster(cluster_id="later", date_filed="1955-01-01", case_name="Brown v. Board")
    )
    lookup = asyncio.run(
        lookup_full_reporter_locators_exact(document, client=client, retrospective_date=date(1954, 12, 31))
    )
    restored = Document.model_validate(lookup.model_dump(mode="json"))
    saved = next(
        node
        for node in restored.citations[0].trace
        if node.details.get("validation_node_type") == "ExactLocatorLookupNode"
    )
    assert saved.details["validation"]["raw_candidate_count"] == 1
    assert saved.details["validation"]["excluded_candidate_count"] == 1
    assert saved.details["validation"]["candidate_count"] == 0
    assert restored.citations[0].judgement(Question.LOCATOR_LOOKUP).outcome == "deferred_to_search"
    unique = asyncio.run(validate_unique_full_reporter_locator_identities(restored, client=client))
    assert unique.citations[0].judgement(Question.IDENTITY).outcome == "unjudged"
    with pytest.raises(ValueError, match="restart from the preceding checkpoint"):
        asyncio.run(
            lookup_full_reporter_locators_exact(
                restored, client=client, retrospective_date=date(1955, 12, 31)
            )
        )


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
    record = read_citation(
        citation_id="cite-0001",
        fields=placed(
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

    lookup = asyncio.run(
        lookup_full_reporter_locators_exact(
            form_roots(document), client=client, retrospective_date=date(1954, 12, 31)
        )
    )
    checkpoint = Document.model_validate(lookup.model_dump(mode="json"))
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
