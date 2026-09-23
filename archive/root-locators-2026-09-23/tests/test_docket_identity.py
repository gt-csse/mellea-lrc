"""Contract tests for the unified docket-root query-and-resolution plan."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

import mellea_lrc.validation.root_identity.docket_resolution as docket_identity
from mellea_lrc.courtlistener import CourtListenerSearchResult
from mellea_lrc.extraction.root_stages import form_roots
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.model.case_names import CaseName
from mellea_lrc.model.citations import CitationDate, CitationField, DocketCitation, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.operations import observe_citation
from mellea_lrc.model.record import CitationRecord, Question
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.search.common import MetadataTermPlan
from mellea_lrc.validation.types import (
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNodeStatus,
)
from tests.record_fixtures import read_citation, revise_citation


class _CourtListener:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert search_type == "d"
        assert cursor is None
        self.calls.append(query)
        results = (
            [
                {
                    "docket_id": "18",
                    "docketNumber": "1:20-cv-06835",
                    "caseName": "Smith v. Jones",
                    "court_id": "nysd",
                    "dateFiled": "2020-02-01",
                    "party": ["Smith", "Jones"],
                }
            ]
            if query == 'caseName:("Smith") AND court_id:nysd'
            else []
        )
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=False,
            count=len(results),
            results=results,
            next_cursor=None,
            previous_cursor=None,
        )


class _GovInfo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        self.calls.append(query)
        return GovInfoSearchResult(query=query, count=0, results=(), next_offset_mark=None)


def _document() -> Document:
    text = "Smith v. Jones, No. 20 Civ. 6835 (S.D.N.Y. 2021)."
    locator = "20 Civ. 6835"
    preprocessed = preprocess(text)
    citation = DocketCitation(
        docket_number=locator,
        court="nysd",
        date=CitationDate(year="2021"),
        case_name=CaseName(text="Smith v. Jones", span=Span(0, 34)),
    )
    record = read_citation(
        citation_id="cite-0001",
        fields=placed(
            citation,
            span=Span(0, len(text)),
            locator_span=Span(text.index(locator), text.index(locator) + len(locator)),
            matched_text=locator,
        ),
    )
    return form_roots(
        Document(
            source_metadata=preprocessed.source_metadata,
            text=text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(record,),
            extraction_metadata=ExtractionMetadata(),
        )
    )


@pytest.fixture
def planned_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    async def plan(record, *, stage: str, made_by: str, source_case_name: str | None, session):
        return MetadataTermPlan(
            terms=("Smith",),
            node=docket_identity.Node(
                node_id=f"{record.citation_id}:{stage}:case_name_terms",
                reads=docket_identity.Reads.DOCUMENT,
                stage=stage,
                made_by=made_by,
                outcome="prepared",
                details={"terms": ["Smith"]},
            ),
        )

    monkeypatch.setattr(docket_identity, "prepare_case_name_terms", plan)


def test_direct_query_and_case_name_query_share_one_resolution_drill(
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later name fragment can resolve a docket written in another form."""
    courtlistener = _CourtListener()
    govinfo = _GovInfo()

    async def choose_candidate(validation, *, summary, shortlist, **kwargs):
        assert tuple(item.candidate_index for item in shortlist.candidates) == (1,)
        assert summary.candidates[0].docket_number == "1:20-cv-06835"
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1,),
            selected_candidate_index=1,
            rationale="The local and canonical docket forms identify the same matter.",
            depends_on=(summary.node_id,),
            docket_equivalent=True,
            case_name_equivalent=True,
        )

    monkeypatch.setattr(docket_identity, "run_mellea_docket_metadata_choice", choose_candidate)

    async def no_source_change(*args, **kwargs):
        return None

    monkeypatch.setattr(docket_identity, "_review_unresolved_docket_citation", no_source_change)
    document = _document()
    revise_citation(document.citations[0], date=None)
    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            document,
            client=courtlistener,
            govinfo_client=govinfo,
        )
    )
    root = result.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert courtlistener.calls == [
        "20 Civ. 6835",
        'caseName:("Smith") AND court_id:nysd',
        'caseName:("Smith")',
    ]
    assert len(govinfo.calls) == 3
    assert root.found is not None and root.found.docket_id == "18"
    assert result.passes[-1] == docket_identity.DOCKET_ROOT_IDENTITY_STAGE


@pytest.mark.parametrize("filed", ["2021-02-02", None, "not-a-date"])
def test_retrospective_cutoff_withholds_future_or_undated_docket_metadata(
    filed: str | None,
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DatedCourtListener(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            self.calls.append(query)
            results = (
                [
                    {
                        "docket_id": "18",
                        "docketNumber": "20 Civ. 6835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "dateFiled": filed,
                    }
                ]
                if query == "20 Civ. 6835"
                else []
            )
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=len(results),
                results=results,
                next_cursor=None,
                previous_cursor=None,
            )

    async def no_source_change(*args, **kwargs):
        return None

    async def must_not_choose(*args, **kwargs):
        raise AssertionError("Excluded metadata must not reach semantic review")

    monkeypatch.setattr(docket_identity, "_review_unresolved_docket_citation", no_source_change)
    monkeypatch.setattr(docket_identity, "run_mellea_docket_metadata_choice", must_not_choose)
    document = _document()
    revise_citation(document.citations[0], date=None)
    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            document,
            retrospective_date=date(2021, 2, 1),
            client=DatedCourtListener(),
            govinfo_client=_GovInfo(),
        )
    )

    root = result.citations[0]
    assert root.found is None
    searches = [
        node.details["validation"]
        for node in root.trace
        if node.details.get("validation_node_type") == "DocketRootSearchNode"
    ]
    direct = searches[0]
    assert direct["retrospective_date"] == "2021-02-01"
    assert direct["raw_candidate_count"] == 1
    assert direct["candidate_count"] == 0
    assert direct["excluded_candidate_count"] == 1
    assert direct["candidates"] == []
    assert direct["attempts"][0]["candidate_count"] == 1
    assert direct["attempts"][0]["candidates"] == []


def test_retrospective_cutoff_keeps_same_day_docket_metadata(planned_terms) -> None:
    record = _document().citations[0]

    class SameDayCourtListener(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=1,
                results=[
                    {
                        "docket_id": "18",
                        "docketNumber": "20 Civ. 6835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "dateFiled": "2021-02-01",
                    }
                ],
                next_cursor=None,
                previous_cursor=None,
            )

    search = docket_identity._run_query(
        record,
        query=docket_identity._DocketQuery("courtlistener", "direct_docket", "20 Civ. 6835", None),
        ordinal=1,
        courtlistener=SameDayCourtListener(),
        govinfo=_GovInfo(),
        retrospective_date=date(2021, 2, 1),
    )
    assert search.candidate_count == 1
    assert search.raw_candidate_count == 1
    assert search.excluded_candidate_count == 0
    assert search.outcome.value == "found"


def test_retrospective_cutoff_excludes_undated_govinfo_package() -> None:
    record = _document().citations[0]

    class PackageGovInfo(_GovInfo):
        def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
            return GovInfoSearchResult(
                query=query,
                count=1,
                results=(
                    {
                        "packageId": "USCOURTS-nysd-1_20-cv-06835",
                        "title": "Smith v. Jones",
                        "dateIssued": "2020-01-01",
                    },
                ),
                next_offset_mark=None,
            )

    search = docket_identity._run_query(
        record,
        query=docket_identity._DocketQuery("govinfo", "direct_docket", "20 Civ. 6835", None),
        ordinal=1,
        courtlistener=_CourtListener(),
        govinfo=PackageGovInfo(),
        retrospective_date=date(2021, 2, 1),
    )
    assert search.raw_candidate_count == 1
    assert search.candidate_count == 0
    assert search.excluded_candidate_count == 1


def test_retrospective_filter_preserves_complete_result_for_remaining_candidate() -> None:
    record = _document().citations[0]
    revise_citation(record, date=None)

    class MixedCourtListener(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=2,
                results=[
                    {
                        "docket_id": "18",
                        "docketNumber": "20 Civ. 6835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "dateFiled": "2020-02-01",
                    },
                    {
                        "docket_id": "19",
                        "docketNumber": "20 Civ. 6835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "dateFiled": "2022-02-01",
                    },
                ],
                next_cursor=None,
                previous_cursor=None,
            )

    search = docket_identity._run_query(
        record,
        query=docket_identity._DocketQuery("courtlistener", "direct_docket", "20 Civ. 6835", None),
        ordinal=1,
        courtlistener=MixedCourtListener(),
        govinfo=_GovInfo(),
        retrospective_date=date(2021, 2, 1),
    )
    assert search.raw_candidate_count == 2
    assert search.candidate_count == 1
    assert search.excluded_candidate_count == 1
    assert search.attempts[0].candidate_count == 2
    assert search.outcome.value == "found"

    resolved = docket_identity._resolve_query_programmatically(
        record, search=search, provider="courtlistener"
    )
    assert resolved.terminal is True
    assert record.found is not None and record.found.docket_id == "18"


@pytest.mark.parametrize("saved_cutoff", [None, date(2021, 1, 1)])
def test_completed_docket_identity_rejects_different_retrospective_cutoff(
    saved_cutoff: date | None,
) -> None:
    document = _document()
    search = docket_identity._run_query(
        document.citations[0],
        query=docket_identity._DocketQuery("courtlistener", "direct_docket", "20 Civ. 6835", None),
        ordinal=1,
        courtlistener=_CourtListener(),
        govinfo=_GovInfo(),
        retrospective_date=saved_cutoff,
    )
    observe_citation(document.citations[0], docket_identity._trace(search))
    completed = document.evolve(passes=(*document.passes, docket_identity.DOCKET_ROOT_IDENTITY_STAGE))

    with pytest.raises(ValueError, match="cannot reuse it"):
        asyncio.run(
            docket_identity.resolve_docket_root_identities(completed, retrospective_date=date(2021, 2, 1))
        )


def test_completed_docket_identity_reuses_matching_retrospective_cutoff() -> None:
    document = _document()
    cutoff = date(2021, 2, 1)
    search = docket_identity._run_query(
        document.citations[0],
        query=docket_identity._DocketQuery("courtlistener", "direct_docket", "20 Civ. 6835", None),
        ordinal=1,
        courtlistener=_CourtListener(),
        govinfo=_GovInfo(),
        retrospective_date=cutoff,
    )
    observe_citation(document.citations[0], docket_identity._trace(search))
    completed = document.evolve(passes=(*document.passes, docket_identity.DOCKET_ROOT_IDENTITY_STAGE))

    assert (
        asyncio.run(docket_identity.resolve_docket_root_identities(completed, retrospective_date=cutoff))
        is completed
    )


def test_court_and_filing_conflicts_reach_semantics_with_independent_fields(
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Docket similarity alone controls review eligibility, including conflicted fields."""

    class ConflictingCourtListener(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            self.calls.append(query)
            results = (
                [
                    {
                        "docket_id": "19",
                        "docketNumber": "1:20-cv-06835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nyed",
                        "dateFiled": "2022-02-01",
                        "party": ["Smith", "Jones"],
                    }
                ]
                if query == "20 Civ. 6835"
                else []
            )
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=len(results),
                results=results,
                next_cursor=None,
                previous_cursor=None,
            )

    reviewed = []

    async def choose_candidate(validation, *, summary, shortlist, **kwargs):
        reviewed.append(summary.candidates[0])
        assert tuple(item.candidate_index for item in shortlist.candidates) == (1,)
        candidate = summary.candidates[0]
        assert candidate.case_name_outcome.value == "match"
        assert candidate.year_outcome.value == "mismatch"
        assert candidate.court_outcome.value == "mismatch"
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1,),
            selected_candidate_index=1,
            rationale="The docket and caption agree, while court and chronology conflict.",
            depends_on=(summary.node_id,),
            docket_equivalent=True,
            case_name_equivalent=True,
        )

    async def no_source_change(*args, **kwargs):
        return None

    monkeypatch.setattr(docket_identity, "_review_unresolved_docket_citation", no_source_change)
    monkeypatch.setattr(docket_identity, "run_mellea_docket_metadata_choice", choose_candidate)
    client = ConflictingCourtListener()
    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            _document(),
            client=client,
            govinfo_client=_GovInfo(),
        )
    )

    assert reviewed
    assert client.calls[0] == "20 Civ. 6835"
    root = result.citations[0]
    assert root.found is None
    field_outcomes = {
        node.details.get("validation_node_type"): node.details["validation"]["outcome"]
        for node in root.trace
        if node.details.get("validation_node_type")
        in {"ExactCaseNameCheckNode", "YearCheckNode", "CourtCheckNode"}
    }
    assert field_outcomes == {
        "ExactCaseNameCheckNode": "match",
        "YearCheckNode": "mismatch",
        "CourtCheckNode": "mismatch",
    }


def test_semantic_docket_choice_reviews_a_compatible_filing_date(
    planned_terms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An earlier filing date is compatible and does not block semantic review."""

    async def no_source_change(*args, **kwargs):
        return None

    async def choose_candidate(validation, *, summary, shortlist, **kwargs):
        assert tuple(item.candidate_index for item in shortlist.candidates) == (1,)
        assert summary.candidates[0].case_name_outcome.value == "match"
        assert summary.candidates[0].year_outcome is None
        assert summary.candidates[0].court_outcome.value == "match"
        return MelleaLocatorCandidateChoiceNode(
            node_id=f"{summary.node_id}:mellea_docket_metadata_choice",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaLocatorCandidateChoiceOutcome.SELECTED,
            candidate_indices=(1,),
            selected_candidate_index=1,
            rationale="The docket and case name identify the candidate; filing predates the cited decision.",
            depends_on=(summary.node_id,),
            docket_equivalent=True,
            case_name_equivalent=True,
        )

    monkeypatch.setattr(docket_identity, "_review_unresolved_docket_citation", no_source_change)
    monkeypatch.setattr(docket_identity, "run_mellea_docket_metadata_choice", choose_candidate)

    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            _document(),
            client=_CourtListener(),
            govinfo_client=_GovInfo(),
        )
    )

    root = result.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.found is not None and root.found.docket_id == "18"


def test_case_name_query_plan_keeps_lenient_fragments_and_adds_one_bounded_pair() -> None:
    """Independent terms preserve recall; one pair controls broad result sets."""
    record = _document().citations[0]
    plan = docket_identity._query_plan(record, ("Smith", "Regents"))
    courtlistener_queries = tuple(query.query for query in plan if query.provider == "courtlistener")

    assert 'caseName:("Smith") AND court_id:nysd' in courtlistener_queries
    assert 'caseName:("Regents")' in courtlistener_queries
    assert 'caseName:("Smith" AND "Regents") AND court_id:nysd' in courtlistener_queries


def test_caption_pair_cannot_bypass_the_docket_similarity_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a matching caption cannot shortlist a dissimilar docket number."""

    class PairClient(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            self.calls.append(query)
            results = (
                [
                    {
                        "docket_id": "18",
                        "docketNumber": "1:16-cv-00318",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "party": ["Smith", "Jones"],
                    }
                ]
                if query == 'caseName:("Smith" AND "Jones") AND court_id:nysd'
                else []
            )
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=len(results),
                results=results,
                next_cursor=None,
                previous_cursor=None,
            )

    async def pair_terms(record, *, stage: str, made_by: str, source_case_name: str | None, session):
        return MetadataTermPlan(
            terms=("Smith", "Jones"),
            node=docket_identity.Node(
                node_id=f"{record.citation_id}:{stage}:case_name_terms",
                reads=docket_identity.Reads.DOCUMENT,
                stage=stage,
                made_by=made_by,
                outcome="prepared",
                details={"terms": ["Smith", "Jones"]},
            ),
        )

    async def must_not_choose(*args, **kwargs):
        raise AssertionError("A docket below 40% similarity must not reach semantic selection")

    async def no_source_change(*args, **kwargs):
        return None

    monkeypatch.setattr(docket_identity, "prepare_case_name_terms", pair_terms)
    monkeypatch.setattr(docket_identity, "run_mellea_docket_metadata_choice", must_not_choose)
    monkeypatch.setattr(docket_identity, "_review_unresolved_docket_citation", no_source_change)
    document = _document()
    revise_citation(document.citations[0], docket_number="CIV 16-0318 JB/SCY", date=None)

    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            document,
            client=PairClient(),
            govinfo_client=_GovInfo(),
        )
    )

    assert result.citations[0].judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
    assert result.citations[0].found is None


def test_matching_caption_does_not_shortlist_a_below_threshold_govinfo_docket() -> None:
    """The common metadata shortlist applies 40% docket similarity to GovInfo too."""
    record = _document().citations[0]
    shortlist = docket_identity.build_docket_metadata_shortlist(
        record,
        candidates=(
            {
                "docketNumber": "9:99-cr-99999",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
            },
        ),
        candidate_count=1,
        complete_result_set=True,
        search_node_id="test:govinfo",
        depends_on=("test:govinfo",),
        node_id="test:shortlist",
    )

    assert shortlist.outcome.value == "no_candidates"
    assert shortlist.complete_result_set is True
    assert shortlist.candidates == ()


def test_source_recovered_case_name_rechecks_direct_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source re-read changes comparison evidence even when its query stays identical."""

    class DirectCandidateClient(_CourtListener):
        def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
            self.calls.append(query)
            results = (
                [
                    {
                        "docket_id": "18",
                        "docketNumber": "20 Civ. 6835",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "party": ["Smith", "Jones"],
                    }
                ]
                if query == "20 Civ. 6835"
                else []
            )
            return CourtListenerSearchResult.from_payload(
                query=query,
                search_type=search_type,
                semantic=False,
                count=len(results),
                results=results,
                next_cursor=None,
                previous_cursor=None,
            )

    async def recovered_name(record, **kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation="Smith v. Jones, No. 20 Civ. 6835 (S.D.N.Y.).",
            source_locator="20 Civ. 6835",
            extracted_docket_number="20 Civ. 6835",
            reparsed_case_name="Smith v. Jones",
            grounded_case_name=CaseName(
                span=Span(0, len("Smith v. Jones")),
                text="Smith v. Jones",
                plaintiff="Smith",
                defendant="Jones",
            ),
            reparsed_docket_number="20 Civ. 6835",
            reparsed_court="S.D.N.Y.",
            reparsed_date=None,
            reparsed_pin_cite=None,
            grounded_docket_number="20 Civ. 6835",
            reason="The case name immediately precedes the docket locator.",
            depends_on=("cite-0001:docket_root_identity:query:2:govinfo",),
        )

    monkeypatch.setattr(docket_identity, "run_mellea_docket_citation_reextraction", recovered_name)
    document = _document()
    revise_citation(document.citations[0], case_name=None, date=None)
    assert document.citations[0].case_name is None
    courtlistener = DirectCandidateClient()

    result = asyncio.run(
        docket_identity.resolve_docket_root_identities(
            document,
            client=courtlistener,
            govinfo_client=_GovInfo(),
        )
    )

    restored = Document.model_validate(result.model_dump(mode="json"))
    root = restored.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    recovered = root.case_name
    assert recovered is not None
    assert root.get_field(CitationField.CASE_NAME) == recovered
    assert root.fields.case_name == recovered
    assert recovered.text == "Smith v. Jones"
    assert restored.text[recovered.span.start : recovered.span.end] == recovered.text
    name_updates = [
        update for update in root.field_updates if update.field is CitationField.CASE_NAME
    ]
    assert name_updates[-1].before is None
    assert name_updates[-1].after == recovered
    assert name_updates[-1].node_id == "cite-0001:mellea_docket_citation_reextraction"
    assert courtlistener.calls == [
        "20 Civ. 6835",
        "20 Civ. 6835",
    ]
    query_nodes = [
        node for node in result.citations[0].trace if node.stage == docket_identity.DOCKET_ROOT_IDENTITY_STAGE
    ]
    assert len({node.node_id for node in query_nodes}) == len(query_nodes)


def test_source_recovered_docket_pin_cite_survives_document_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "Smith v. Jones, No. 20 Civ. 6835, at 9 (S.D.N.Y. 2021)."
    locator = "20 Civ. 6835"
    preprocessed = preprocess(text)
    record = read_citation(
        citation_id="cite-pin",
        fields=placed(
            DocketCitation(docket_number=locator, court="nysd"),
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
    assert document.citations[0].fields.pin_cite is None

    async def recovered_pin(root, **_kwargs):
        return MelleaDocketCitationReextractionNode(
            node_id=f"{root.citation_id}:mellea_docket_citation_reextraction",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.UNCHANGED,
            source_citation=text,
            source_locator=locator,
            extracted_docket_number=locator,
            reparsed_case_name=None,
            reparsed_docket_number=locator,
            reparsed_court=None,
            reparsed_date=None,
            reparsed_pin_cite="9",
            grounded_docket_number=locator,
            reason="The source states page 9 immediately after the docket locator.",
            depends_on=("test:trigger",),
        )

    monkeypatch.setattr(docket_identity, "run_mellea_docket_citation_reextraction", recovered_pin)
    asyncio.run(
        docket_identity._review_unresolved_docket_citation(
            document.citations[0],
            document=document,
            trigger_node_id="test:trigger",
            session=None,
        )
    )
    restored = Document.model_validate(document.model_dump(mode="json"))
    root = restored.citations[0]
    pin = root.fields.pin_cite
    assert pin is not None
    assert pin.text == "9"
    assert pin.span == Span(text.index("at 9") + len("at "), text.index("at 9") + len("at 9"))
    assert pin.pages[0].first == 9
    assert restored.text[pin.span.start : pin.span.end] == pin.text
    updates = [update for update in root.field_updates if update.field is CitationField.PIN_CITE]
    assert len(updates) == 1
    assert updates[0].before is None
    assert updates[0].after == pin
    assert updates[0].node_id == "cite-pin:mellea_docket_citation_reextraction"
