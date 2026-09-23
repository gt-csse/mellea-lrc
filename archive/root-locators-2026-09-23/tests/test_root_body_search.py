"""Contract tests for the three-shape root body-corroboration route."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from mellea_lrc.api import form_roots
from mellea_lrc.courtlistener import CourtListenerError, CourtListenerSearchResult
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.extraction.stages import CASE_NAME_STAGE
from mellea_lrc.govinfo import GovInfoSearchResult
from mellea_lrc.llm import EvidenceCandidate, GroundingEvidence, IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.model.citations import (
    CitationDate,
    CitationField,
    DocketCitation,
    FullCaseCitation,
    Reporter,
    placed,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.model.operations import (
    assign_root,
    attribute_authority,
    create_citation,
    field_values,
    judge_citation,
    record_colocation,
    update_fields,
)
from mellea_lrc.model.record import CitationRecord, Node, Question, Reads
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.serialization.validated_document import deserialize_validation_node
from mellea_lrc.validation.root_identity.body import (
    ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
    ROOT_BODY_CORROBORATION_SEARCH_STAGE,
    SHARED_BODY_EVIDENCE_REVIEW_STAGE,
    _CitationFieldValues,
    _courtlistener_body_node,
    _DirectBodyCandidate,
    _govinfo_body_node,
    _ground_review_fields,
    _locator_context,
    _read_courtlistener_body,
    _resolve_direct_candidates,
    _same_source_document,
    _saved_body_nodes,
    _ThirdPartyEvidence,
    _validate_direct_candidate_grounding,
    _validate_third_party_grounding,
    _validate_third_party_source_locator,
    resolve_root_body_corroboration,
    review_shared_body_evidence,
    search_root_body_corroboration,
)
from mellea_lrc.validation.root_identity.docket_resolution import DOCKET_ROOT_IDENTITY_STAGE
from mellea_lrc.validation.root_identity.reporter import FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE
from mellea_lrc.validation.types import RootBodySearchSource


def _fixture_fields(record: CitationRecord, **changes: object) -> None:
    """Give a test's earlier reading its own replayable operation."""
    node = Node(
        f"{record.citation_id}:test_fixture:{len(record.trace)}",
        Reads.DOCUMENT,
        "test_fixture",
        __name__,
        "corrected",
    )
    update_fields(
        record, node, {CitationField(name): value for name, value in changes.items()}, reason="Test fixture"
    )


def _fixture_record(citation_id: str, fields: DocketCitation | FullCaseCitation) -> CitationRecord:
    """Admit an initial reading with the same CREATE history as extraction."""
    node = Node(f"{citation_id}:test_fixture:create", Reads.DOCUMENT, "test_fixture", __name__, "identified")
    record = create_citation(citation_id, fields.kind, node)
    update_fields(record, node, field_values(fields), reason="Initial test reading")
    return record


def test_reporter_body_context_preserves_digits_and_skips_earlier_near_match() -> None:
    wrong = "2014 U.S. Dist. LEXIS 147975"
    correct = "2021 U.S. Dist. LEXIS 147975"
    locator = correct

    assert _locator_context(locator, f"Cites {wrong} only.", require_reporter_digits=True) is None
    found = _locator_context(locator, f"Cites {wrong}; later cites {correct}.", require_reporter_digits=True)
    assert found is not None
    excerpt, offset = found
    assert correct in excerpt
    assert offset == 0
    assert _locator_context(locator, f"Cites {wrong} only.") is not None


def test_shared_body_quote_gets_a_separate_review_for_deferred_root(monkeypatch) -> None:
    document = _ready(_docket_document_with_parallel_reporter())
    target = document.citations[0]
    target = record_colocation(
        target,
        "group-1",
        Node("cite-1:test_fixture:colocation", Reads.RECORD, "test_fixture", __name__, "grouped"),
    )
    judge_citation(
        target,
        Node("cite-1:test_fixture:identity", Reads.RECORD, "test_fixture", __name__, "deferred"),
        Question.IDENTITY,
        "deferred_to_open_web_search",
        message="Needs more evidence.",
    )
    locator = "2024 WL 1234567"
    start = document.text.index(locator)
    sibling = _fixture_record(
        "reporter-1",
        placed(
            FullCaseCitation(
                plaintiff="Smith",
                defendant="Jones",
                volume="2024",
                reporter=Reporter(as_written="WL", short_name="WL"),
                page="1234567",
                court="nysd",
                date=CitationDate(year="2024"),
            ),
            span=Span(0, len(document.text)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )
    sibling = record_colocation(
        sibling,
        "group-1",
        Node("reporter-1:test_fixture:colocation", Reads.RECORD, "test_fixture", __name__, "grouped"),
    )
    sibling = assign_root(
        sibling,
        "reporter-1",
        Node("reporter-1:test_fixture:root", Reads.RECORD, "test_fixture", __name__, "root"),
    )
    attribute_authority(
        sibling,
        Node("reporter-1:test_fixture:authority", Reads.RECORD, "test_fixture", __name__, "attributed"),
        "third_party:shared-authority",
    )
    quote = "Smith v. Jones, No. 1:24-cv-08760, 2024 WL 1234567 (S.D.N.Y. 2024)."
    excerpt = f"Earlier text. {quote} Later text."
    judge_citation(
        sibling,
        Node(
            node_id="sibling-review",
            reads=Reads.RECORD,
            stage=ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
            made_by="test",
            outcome="resolved",
            details={
                "judgement_type": "third_party",
                "selected_evidence_index": 1,
                "citation_quote": quote,
                "citation_grounding": {"origin": "opinions:42:full_text", "matched_text": quote},
                "evidence": [
                    {
                        "index": 1,
                        "source": "courtlistener_cluster",
                        "citing_id": "42",
                        "citing_case_name": "Other Case",
                        "excerpt": excerpt,
                        "excerpt_start": 100,
                        "origin": "opinions:42:full_text",
                    }
                ],
            },
        ),
        Question.IDENTITY,
        "resolved",
        message="Grounded.",
    )
    document = document.evolve(
        citations=(target, sibling),
        passes=(*document.passes, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE),
    )
    calls = []

    async def review(_document, record, evidence, **kwargs):
        calls.append((record.citation_id, evidence[0].origin, kwargs["extra_depends_on"]))
        return Node(
            node_id="target-shared-review",
            reads=Reads.RECORD,
            stage=SHARED_BODY_EVIDENCE_REVIEW_STAGE,
            made_by="test",
            outcome="resolved",
            details={"cited_case_name": "Smith v. Jones", "selected_evidence_index": 1},
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body._review_third_party", review)
    completed = asyncio.run(review_shared_body_evidence(document))

    assert calls == [("cite-1", "opinions:42:full_text", ("sibling-review",))]
    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"
    assert completed.citations[0].authority_id == sibling.authority_id
    assert completed.passes[-1] == SHARED_BODY_EVIDENCE_REVIEW_STAGE
    assert asyncio.run(review_shared_body_evidence(completed)) is completed


class _BodyCourtListener:
    def __init__(
        self,
        *,
        opinion: list[dict[str, object]] = (),
        recap: list[dict[str, object]] = (),
        opinion_text: dict[str, str] | None = None,
        recap_text: dict[str, str] | None = None,
    ) -> None:
        self.opinion = opinion
        self.recap = recap
        self.opinion_text = opinion_text or {}
        self.recap_text = recap_text or {}
        self.calls: list[tuple[str, str]] = []

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert cursor is None and not semantic
        self.calls.append((query, search_type))
        results = self.opinion if search_type == "o" else self.recap
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=False,
            count=len(results),
            results=results,
            next_cursor=None,
            previous_cursor=None,
        )

    def get_recap_document(self, document_id: str):
        from mellea_lrc.courtlistener import CourtListenerRecapDocument

        return CourtListenerRecapDocument(document_id=document_id, plain_text=self.recap_text[document_id])

    def get_opinion(self, opinion_id: str):
        from mellea_lrc.courtlistener import CourtListenerOpinion

        return CourtListenerOpinion(
            opinion_id=opinion_id,
            cluster_id=None,
            opinion_type="lead-opinion",
            html_with_citations=self.opinion_text[opinion_id],
        )


class _PagedBodyCourtListener(_BodyCourtListener):
    def __init__(self, pages: dict[str, tuple[int, list[dict[str, object]], str | None]]) -> None:
        super().__init__()
        self.pages = pages

    def search(self, query: str, search_type: str, cursor: str | None = None, *, semantic: bool = False):
        assert cursor is None and not semantic
        self.calls.append((query, search_type))
        count, results, next_cursor = self.pages[query]
        return CourtListenerSearchResult.from_payload(
            query=query,
            search_type=search_type,
            semantic=False,
            count=count,
            results=results,
            next_cursor=next_cursor,
            previous_cursor=None,
        )


class _BodyGovInfo:
    def __init__(self, *, results: list[dict[str, object]] = (), texts: dict[str, str] | None = None) -> None:
        self.results = results
        self.texts = texts or {}
        self.calls: list[tuple[str, int]] = []
        self.text_calls: list[tuple[str, date | None]] = []

    def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
        self.calls.append((query, page_size))
        return GovInfoSearchResult(
            query=query,
            count=len(self.results),
            results=tuple(self.results),
            next_offset_mark=None,
        )

    def get_package_text(self, package_id: str, *, retrospective_date: date | None = None) -> str:
        self.text_calls.append((package_id, retrospective_date))
        return self.texts[package_id]


def _ready(document: Document) -> Document:
    """Put one formed root at the explicit pre-body checkpoint."""
    formed = form_roots(document)
    root = formed.citations[0]
    judge_citation(
        root,
        Node("cite-1:test_fixture:prior_identity", Reads.RECORD, "test_fixture", __name__, "deferred"),
        Question.IDENTITY,
        "deferred_to_future_implementation",
        message="Previous metadata routes did not admit an identity.",
    )
    return formed.evolve(
        passes=(
            ROOT_FORMATION_STAGE,
            CASE_NAME_STAGE,
            DOCKET_ROOT_IDENTITY_STAGE,
            FULL_REPORTER_SEARCH_CANDIDATE_RESOLUTION_STAGE,
        ),
    )


def _docket_document() -> Document:
    text = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)."
    locator = "1:24-cv-08760"
    start = text.index(locator)
    preprocessed = preprocess(text)
    record = _fixture_record(
        "cite-1",
        placed(
            DocketCitation(
                plaintiff="Smith",
                defendant="Jones",
                docket_number=locator,
                court="nysd",
                date=CitationDate(year="2024"),
            ),
            span=Span(0, len(text)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )


def _docket_document_with_parallel_reporter() -> Document:
    text = "Smith v. Jones, Case No. 1:24-cv-08760, 2024 WL 1234567 (S.D.N.Y. 2024)."
    locator = "1:24-cv-08760"
    start = text.index(locator)
    preprocessed = preprocess(text)
    record = _fixture_record(
        "cite-1",
        placed(
            DocketCitation(
                plaintiff="Smith",
                defendant="Jones",
                docket_number=locator,
                court="nysd",
                date=CitationDate(year="2024"),
            ),
            span=Span(0, len(text)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )


def _reporter_document() -> Document:
    text = "Smith v. Jones, 123 F.3d 456 (S.D.N.Y. 2024)."
    locator = "123 F.3d 456"
    start = text.index(locator)
    preprocessed = preprocess(text)
    record = _fixture_record(
        "cite-1",
        placed(
            FullCaseCitation(
                plaintiff="Smith",
                defendant="Jones",
                volume="123",
                reporter=Reporter(as_written="F.3d", short_name="F.3d"),
                page="456",
                court="nysd",
                date=CitationDate(year="2024"),
            ),
            span=Span(0, len(text)),
            locator_span=Span(start, start + len(locator)),
            matched_text=locator,
        ),
    )
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=(record,),
        extraction_metadata=ExtractionMetadata(),
    )


@pytest.mark.parametrize(
    ("courtlistener", "govinfo", "expected_identity"),
    [
        (
            _BodyCourtListener(
                opinion=[
                    {
                        "cluster_id": "77",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "decisionDate": "2024-02-01",
                    }
                ]
            ),
            _BodyGovInfo(),
            "deferred_to_open_web_search",
        ),
        (
            _BodyCourtListener(
                recap=[
                    {
                        "docket_id": "44",
                        "docketNumber": "1:24-cv-08760",
                        "caseName": "Smith v. Jones",
                        "court_id": "nysd",
                        "dateFiled": "2024-01-05",
                    }
                ]
            ),
            _BodyGovInfo(),
            "resolved",
        ),
        (
            _BodyCourtListener(),
            _BodyGovInfo(
                results=[
                    {
                        "packageId": "USCOURTS-nysd-1_24-cv-08760",
                        "title": "Smith v. Jones",
                        "dateIssued": "2024-02-01",
                    }
                ]
            ),
            "resolved",
        ),
    ],
)
def test_body_search_preserves_docket_hits_without_misattributing_citing_records(
    courtlistener: _BodyCourtListener,
    govinfo: _BodyGovInfo,
    expected_identity: str,
) -> None:
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=courtlistener, govinfo_client=govinfo
        )
    )
    completed = asyncio.run(resolve_root_body_corroboration(searched))
    restored = Document.model_validate(completed.model_dump(mode="json"))
    root = restored.citations[0]

    assert courtlistener.calls == [("1:24-cv-08760", "o"), ("1:24-cv-08760", "r")]
    assert root.judgement(Question.IDENTITY).outcome == expected_identity
    if expected_identity == "resolved":
        assert root.authority_id is not None
    else:
        assert root.authority_id is None
    assert completed.passes[-2:] == (
        ROOT_BODY_CORROBORATION_SEARCH_STAGE,
        ROOT_BODY_CORROBORATION_RESOLUTION_STAGE,
    )


def test_zero_hit_labeled_docket_retries_parsed_identifier_in_both_body_indexes() -> None:
    record = _docket_document().citations[0]
    labeled = "Case No. 1:24-cv-08760"
    _fixture_fields(record, matched_text=labeled)
    courtlistener = _PagedBodyCourtListener(
        {
            labeled: (0, [], None),
            "1:24-cv-08760": (
                1,
                [{"docket_id": "44", "docketNumber": "1:24-cv-08760"}],
                None,
            ),
        }
    )

    courtlistener_node = _courtlistener_body_node(
        record,
        locator=labeled,
        source=RootBodySearchSource.COURTLISTENER_DOCKET,
        search_type="r",
        client=courtlistener,
    )

    assert courtlistener.calls == [(labeled, "r"), ("1:24-cv-08760", "r")]
    assert courtlistener_node.outcome.value == "found"
    assert [attempt.kind for attempt in courtlistener_node.search_attempts] == [
        "raw_locator",
        "parsed_docket",
    ]
    assert [attempt.used_for_candidates for attempt in courtlistener_node.search_attempts] == [
        False,
        True,
    ]

    class QueryGovInfo:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def search_uscourts(self, query: str, *, page_size: int) -> GovInfoSearchResult:
            assert page_size > 0
            self.calls.append(query)
            results = () if "Case No." in query else ({"packageId": "USCOURTS-nysd-1_24-cv-08760"},)
            return GovInfoSearchResult(
                query=query, count=len(results), results=results, next_offset_mark=None
            )

    govinfo = QueryGovInfo()
    govinfo_node = _govinfo_body_node(record, locator=labeled, client=govinfo)

    assert len(govinfo.calls) == 2
    assert "Case No." in govinfo.calls[0]
    assert "Case No." not in govinfo.calls[1]
    assert govinfo_node.outcome.value == "found"
    assert [attempt.kind for attempt in govinfo_node.search_attempts] == [
        "raw_locator",
        "parsed_docket",
    ]


def test_body_search_preserves_a_reporter_hit_without_resolving_it_and_round_trips() -> None:
    courtlistener = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": "77",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "decisionDate": "2024-02-01",
            }
        ]
    )
    completed = asyncio.run(
        resolve_root_body_corroboration(
            asyncio.run(
                search_root_body_corroboration(
                    _ready(_reporter_document()),
                    client=courtlistener,
                    govinfo_client=_BodyGovInfo(),
                )
            )
        )
    )

    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.authority_id is None


def test_same_number_docket_in_another_court_does_not_reject_identity() -> None:
    courtlistener = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:24-cv-08760",
                "caseName": "Unrelated v. Matter",
                "court_id": "alacivapp",
                "dateFiled": "2002-01-05",
            }
        ]
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=courtlistener, govinfo_client=_BodyGovInfo()
        )
    )
    completed = asyncio.run(
        resolve_root_body_corroboration(searched, client=courtlistener, govinfo_client=_BodyGovInfo())
    )
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.authority_id is None
    direct_review = next(node for node in root.trace if node.node_id.endswith("root_body:identity_review"))
    assert direct_review.outcome == "deferred"


def test_conflicting_exact_reporter_hit_keeps_terminal_wrong_identity() -> None:
    courtlistener = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": "77",
                "caseName": "Unrelated v. Matter",
                "citation": ["123 F.3d 456"],
                "court_id": "ca9",
                "decisionDate": "2002-02-01",
            }
        ]
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_reporter_document()), client=courtlistener, govinfo_client=_BodyGovInfo()
        )
    )
    completed = asyncio.run(
        resolve_root_body_corroboration(searched, client=courtlistener, govinfo_client=_BodyGovInfo())
    )

    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "no_match"


def test_body_search_keeps_an_ambiguous_hit_set_for_open_web_instead_of_selecting_rank_one() -> None:
    courtlistener = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": "77",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "decisionDate": "2024-02-01",
            },
            {
                "cluster_id": "78",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "decisionDate": "2024-02-02",
            },
        ]
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_reporter_document()), client=courtlistener, govinfo_client=_BodyGovInfo()
        )
    )
    completed = asyncio.run(resolve_root_body_corroboration(searched))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.found is None


def _successful_run(output: str) -> IvrRun:
    return IvrRun(
        success=True,
        selected_attempt=0,
        attempts=(IvrAttempt(output=output, requirements=()),),
        backend="test",
        model="test",
        model_options={},
        instruction="",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )


def _direct_proposal(
    *, candidate_index: int | None, anchor_quote: str | None, decision: str = "select_candidate"
) -> str:
    return json.dumps(
        {
            "decision": decision,
            "candidate_index": candidate_index,
            "anchor_quote": anchor_quote,
            "rationale": "The candidate identifies the cited case.",
        }
    )


def _validation_context(output: str):
    return SimpleNamespace(last_output=lambda: SimpleNamespace(value=output))


def test_third_party_source_locator_requirement_rejects_a_parallel_read() -> None:
    document = _reporter_document()
    record = document.citations[0]
    kwargs = {
        "decision": "corroborates",
        "citation_quote": "Smith v. Jones, 123 F.3d 456 (S.D.N.Y. 2024)",
        "cited_name": "Smith v. Jones",
        "cited_locator": "123 F.3d 456",
    }
    wrong = _validate_third_party_source_locator(
        _validation_context(_third_party_proposal(**kwargs, source_locator="Smith v. Jones")),
        document,
        record,
    )
    corrected = _validate_third_party_source_locator(
        _validation_context(_third_party_proposal(**kwargs, source_locator="123 F.3d 456")),
        document,
        record,
    )
    assert not wrong.as_bool()
    assert "marked target_locator" in wrong.reason
    assert corrected.as_bool()


def test_third_party_grounding_requirement_returns_specific_repair_feedback() -> None:
    excerpt = "Smith v. Jones, 123 F.3d 456 (S.D.N.Y. 2024)"
    evidence = (
        _ThirdPartyEvidence(
            index=1,
            source=RootBodySearchSource.COURTLISTENER_CLUSTER,
            citing_id="1",
            citing_case_name=None,
            excerpt=excerpt,
            excerpt_start=0,
            origin="opinion:1",
        ),
    )
    source = excerpt
    valid = _third_party_proposal(
        decision="corroborates",
        citation_quote=excerpt,
        cited_name="Smith v. Jones",
        cited_locator="123 F.3d 456",
        source_locator="123 F.3d 456",
    )
    assert _validate_third_party_grounding(_validation_context(valid), source, evidence).as_bool()
    wrong_quote = _third_party_proposal(
        decision="corroborates",
        citation_quote="Other v. Case, 999 F.3d 999 (C.D. Cal. 2018)",
        cited_name="Other v. Case",
        cited_locator="999 F.3d 999",
        source_locator="123 F.3d 456",
    )
    result = _validate_third_party_grounding(_validation_context(wrong_quote), source, evidence)
    assert not result.as_bool()
    assert "citation_quote" in result.reason
    wrong_field = _third_party_proposal(
        decision="corroborates",
        citation_quote=excerpt,
        cited_name="Other v. Case",
        cited_locator="123 F.3d 456",
        source_locator="123 F.3d 456",
    )
    result = _validate_third_party_grounding(_validation_context(wrong_field), source, evidence)
    assert not result.as_bool()
    assert "cited_fields.case_name" in result.reason
    broad = _third_party_proposal(
        decision="corroborates",
        citation_quote="See " + excerpt,
        cited_name="Smith v. Jones",
        cited_locator="123 F.3d 456",
        source_locator="123 F.3d 456",
    )
    broad_evidence = replace(evidence[0], excerpt="See " + excerpt)
    result = _validate_third_party_grounding(_validation_context(broad), source, (broad_evidence,))
    assert not result.as_bool()
    assert "must begin" in result.reason


def test_cited_fields_use_the_grounded_quote_without_pdf_margin_numbers() -> None:
    excerpt = (
        "                  18   Prelude.\n\n"
        "                  19   See Sunsauce Foods\n\n"
        "                  20   Indus. Corp. v. Son Fish Sauce USA Corp., 2024 WL 778395,\n\n"
        "                  21   (N.D. Cal. Feb. 26, 2024).\n\n"
        "                  22   More text."
    )
    quote = (
        "Sunsauce Foods Indus. Corp. v. Son Fish Sauce USA Corp., 2024 WL 778395, (N.D. Cal. Feb. 26, 2024)"
    )
    policy = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
    grounded = GroundingEvidence((EvidenceCandidate(excerpt, excerpt),)).find_fragment(
        quote, policy, line_number_aware=True
    )
    assert grounded is not None
    assert "20   Indus." in grounded.text
    assert "20   Indus." not in grounded.normalized_text
    values, _, fully_grounded = _ground_review_fields(
        _CitationFieldValues(
            locator="2024 WL 778395",
            case_name="Sunsauce Foods Indus. Corp. v. Son Fish Sauce USA Corp.",
            court="N.D. Cal.",
            date="Feb. 26, 2024",
        ),
        grounded.normalized_text,
        0,
        policy,
    )
    assert fully_grounded
    assert values["case_name"] is not None
    assert "20" not in values["case_name"]


def test_direct_candidate_grounding_rejects_wrong_quote_and_allows_defer() -> None:
    selected = _DirectBodyCandidate(
        index=3,
        source=RootBodySearchSource.COURTLISTENER_DOCKET,
        record={"docketNumber": "25-11030"},
        anchor="25-11030",
    )
    candidates = (selected,)
    invalid = _validate_direct_candidate_grounding(
        _validation_context(_direct_proposal(candidate_index=3, anchor_quote="99-99999")), candidates
    )
    assert not invalid.as_bool()
    assert "candidate_index 3" in invalid.reason
    assert "25-11030" in invalid.reason
    assert "90%" in invalid.reason
    assert _validate_direct_candidate_grounding(
        _validation_context(_direct_proposal(candidate_index=3, anchor_quote="25-11030")), candidates
    ).as_bool()
    assert _validate_direct_candidate_grounding(
        _validation_context(_direct_proposal(candidate_index=None, anchor_quote=None, decision="defer")),
        candidates,
    ).as_bool()


def test_direct_candidate_grounding_repair_retains_both_ivr_attempts(monkeypatch) -> None:
    record = _docket_document().citations[0]
    candidates = (
        _DirectBodyCandidate(
            index=1,
            source=RootBodySearchSource.COURTLISTENER_DOCKET,
            record={
                "docketNumber": "1:24-cv-08760",
                "caseName": "Smith and Jones",
                "court_id": "nysd",
                "dateFiled": "2024-01-01",
                "docket_id": "one",
            },
            anchor="1:24-cv-08760",
        ),
        _DirectBodyCandidate(
            index=2,
            source=RootBodySearchSource.COURTLISTENER_DOCKET,
            record={
                "docketNumber": "1:24-cv-08760",
                "caseName": "Other Matter",
                "court_id": "nysd",
                "dateFiled": "2024-01-01",
                "docket_id": "two",
            },
            anchor="1:24-cv-08760",
        ),
    )

    async def _repair(_session, spec, *, strategy, model_options):
        del _session, model_options
        assert strategy.loop_budget == 2
        validate = spec.requirements[0].validation_fn
        bad = _direct_proposal(candidate_index=1, anchor_quote="unrelated identifier")
        good = _direct_proposal(candidate_index=1, anchor_quote="1:24-cv-08760")
        bad_result = validate(_validation_context(bad))
        good_result = validate(_validation_context(good))
        assert not bad_result.as_bool()
        assert good_result.as_bool()
        return IvrRun(
            success=True,
            selected_attempt=1,
            attempts=(
                IvrAttempt(
                    output=bad,
                    requirements=(
                        IvrRequirementAttempt(
                            description=spec.requirements[0].description,
                            passed=False,
                            reason=bad_result.reason,
                            score=None,
                        ),
                    ),
                ),
                IvrAttempt(
                    output=good,
                    requirements=(
                        IvrRequirementAttempt(
                            description=spec.requirements[0].description,
                            passed=True,
                            reason=good_result.reason,
                            score=None,
                        ),
                    ),
                ),
            ),
            backend="test",
            model="test",
            model_options={},
            instruction=spec.description,
            prefix=spec.prefix,
            grounding_context={},
            user_variables=spec.user_variables,
            output_schema=spec.output_format.model_json_schema(),
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _repair)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    node = asyncio.run(_resolve_direct_candidates(record, candidates, session=object()))

    assert node.outcome == "resolved"
    assert node.details["selected_candidate_index"] == 1
    assert len(node.details["ivr"]["attempts"]) == 2
    assert "does not reproduce" in node.details["ivr"]["attempts"][0]["requirements"][0]["reason"]


def test_direct_candidate_grounding_exhaustion_remains_deferred_with_feedback(monkeypatch) -> None:
    record = _docket_document().citations[0]
    candidates = (
        _DirectBodyCandidate(
            index=1,
            source=RootBodySearchSource.COURTLISTENER_DOCKET,
            record={
                "docketNumber": "1:24-cv-08760",
                "caseName": "Smith and Jones",
                "court_id": "nysd",
                "dateFiled": "2024-01-01",
            },
            anchor="1:24-cv-08760",
        ),
    )

    async def _exhaust(_session, spec, *, strategy, model_options):
        del _session, strategy, model_options
        output = _direct_proposal(candidate_index=1, anchor_quote="unrelated identifier")
        failed = spec.requirements[0].validation_fn(_validation_context(output))
        assert not failed.as_bool()
        attempt = IvrAttempt(
            output=output,
            requirements=(
                IvrRequirementAttempt(
                    description=spec.requirements[0].description,
                    passed=False,
                    reason=failed.reason,
                    score=None,
                ),
            ),
        )
        return IvrRun(
            success=False,
            selected_attempt=1,
            attempts=(attempt, attempt),
            backend="test",
            model="test",
            model_options={},
            instruction=spec.description,
            prefix=spec.prefix,
            grounding_context={},
            user_variables=spec.user_variables,
            output_schema=spec.output_format.model_json_schema(),
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _exhaust)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    node = asyncio.run(_resolve_direct_candidates(record, candidates, session=object()))

    assert node.outcome == "deferred"
    assert "does not reproduce" in node.details["error"]
    assert len(node.details["ivr"]["attempts"]) == 2


def _third_party_proposal(
    *,
    decision: str,
    citation_quote: str,
    cited_name: str | None,
    cited_locator: str | None,
    cited_court: str | None = "S.D.N.Y.",
    cited_date: str | None = "2024",
    source_locator: str = "1:24-cv-08760",
    source_name: str | None = "Smith v. Jones",
    source_court: str | None = "S.D.N.Y.",
    source_date: str | None = "2024",
    conflict: str | None = None,
    reason: str = "The cited fields were compared with the source citation.",
) -> str:
    """Return both model-extracted sides and its semantic field judgments."""
    source_values = {
        "locator": source_locator,
        "case_name": source_name,
        "court": source_court,
        "date": source_date,
    }
    cited_values = {
        "locator": cited_locator,
        "case_name": cited_name,
        "court": cited_court,
        "date": cited_date,
    }
    comparisons = {
        field: "unavailable" if value is None else "mismatch" if field == conflict else "match"
        for field, value in cited_values.items()
    }
    return json.dumps(
        {
            "decision": decision,
            "evidence_index": 1,
            "citation_quote": citation_quote,
            "source_fields": source_values,
            "cited_fields": cited_values,
            "comparisons": comparisons,
            "reason": reason,
        }
    )


def test_retrospective_date_fetches_only_eligible_recap_documents(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Later Filing",
                "dateFiled": "2022-01-01",
                "recap_documents": [{"id": "future", "entry_date_filed": "2024-06-02"}],
            },
            {
                "docket_id": "45",
                "docketNumber": "1:22-cv-00002",
                "caseName": "Undated Filing",
                "dateFiled": "2022-01-01",
                "recap_documents": [{"id": "undated"}],
            },
            {
                "docket_id": "46",
                "docketNumber": "1:22-cv-00003",
                "caseName": "Filing On Cutoff",
                "dateFiled": "2022-01-01",
                "recap_documents": [{"id": "eligible", "entry_date_filed": "2024-06-01"}],
            },
        ],
        recap_text={
            "future": f"This later filing follows {citation}.",
            "undated": f"This undated filing follows {citation}.",
            "eligible": f"This timely filing follows {citation}.",
        },
    )
    fetched: list[str] = []
    get_document = client.get_recap_document

    def _get_document(document_id: str):
        fetched.append(document_id)
        return get_document(document_id)

    monkeypatch.setattr(client, "get_recap_document", _get_document)
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()),
            client=client,
            govinfo_client=_BodyGovInfo(),
            retrospective_date=date(2024, 6, 1),
        )
    )
    review_count = 0

    async def _review(*args, **kwargs):
        nonlocal review_count
        del args, kwargs
        review_count += 1
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    restored = Document.model_validate(searched.model_dump(mode="json"))
    recap_node = next(
        node
        for node in _saved_body_nodes(restored.citations[0])
        if node.source is RootBodySearchSource.COURTLISTENER_DOCKET
    )
    assert recap_node.retrospective_date == "2024-06-01"
    assert recap_node.excluded_by_date == 2
    assert recap_node.excluded_body_items_by_date == 2
    assert [item["id"] for candidate in recap_node.candidates for item in candidate["recap_documents"]] == [
        "eligible"
    ]
    completed = asyncio.run(resolve_root_body_corroboration(restored, client=client, session=object()))

    assert fetched == ["eligible"]
    assert review_count == 1
    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "resolved"
    review = next(
        node for node in completed.citations[0].trace if node.node_id.endswith("third_party_review")
    )
    assert [item["citing_id"] for item in review.details["evidence"]] == ["46"]


def test_retrospective_date_with_no_eligible_body_never_reviews_model(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Later Filing",
                "dateFiled": "2022-01-01",
                "recap_documents": [{"id": "future", "entry_date_filed": "2024-06-02"}],
            },
            {
                "docket_id": "45",
                "docketNumber": "1:22-cv-00002",
                "caseName": "Badly Dated Filing",
                "dateFiled": "2022-01-01",
                "recap_documents": [{"id": "invalid", "entry_date_filed": "tomorrow"}],
            },
        ],
        recap_text={
            "future": f"This later filing follows {citation}.",
            "invalid": f"This unverified filing follows {citation}.",
        },
    )
    fetched: list[str] = []

    def _unexpected_fetch(document_id: str):
        fetched.append(document_id)
        raise AssertionError("An ineligible document must not be fetched")

    async def _unexpected_review(*args, **kwargs):
        raise AssertionError("An ineligible document must not reach model review")

    monkeypatch.setattr(client, "get_recap_document", _unexpected_fetch)
    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _unexpected_review)
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()),
            client=client,
            govinfo_client=_BodyGovInfo(),
            retrospective_date=date(2024, 6, 1),
        )
    )
    completed = asyncio.run(
        resolve_root_body_corroboration(
            Document.model_validate(searched.model_dump(mode="json")), client=client, session=object()
        )
    )

    assert fetched == []
    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert completed.citations[0].authority_id is None


def test_retrospective_date_excludes_future_direct_opinion_candidate(monkeypatch) -> None:
    client = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": "future",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "decisionDate": "2024-06-02",
                "citation": ["123 F.3d 456"],
            },
            {
                "cluster_id": "undated",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "citation": ["123 F.3d 456"],
            },
            {
                "cluster_id": "eligible",
                "caseName": "Smith v. Jones",
                "court_id": "nysd",
                "decisionDate": "2024-06-01",
                "citation": ["123 F.3d 456"],
            },
        ]
    )

    async def _unexpected_review(*args, **kwargs):
        raise AssertionError("The future opinion must not enter candidate selection")

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _unexpected_review)
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_reporter_document()),
            client=client,
            govinfo_client=_BodyGovInfo(),
            retrospective_date=date(2024, 6, 1),
        )
    )
    opinion_node = next(
        node
        for node in _saved_body_nodes(searched.citations[0])
        if node.source is RootBodySearchSource.COURTLISTENER_CLUSTER
    )
    assert [candidate["cluster_id"] for candidate in opinion_node.candidates] == ["eligible"]
    with pytest.raises(ValueError, match="retrospective date differs"):
        asyncio.run(
            resolve_root_body_corroboration(
                Document.model_validate(searched.model_dump(mode="json")),
                client=client,
                retrospective_date=date(2024, 6, 2),
            )
        )
    completed = asyncio.run(
        resolve_root_body_corroboration(
            Document.model_validate(searched.model_dump(mode="json")), client=client
        )
    )

    root = completed.citations[0]
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "courtlistener:cluster:eligible"
    with pytest.raises(ValueError, match="already run without this retrospective date"):
        asyncio.run(
            resolve_root_body_corroboration(completed, client=client, retrospective_date=date(2024, 6, 2))
        )


def test_retrospective_date_is_forwarded_to_govinfo_granule_fetch() -> None:
    package_id = "USCOURTS-nysd-1_22-cv-00001"
    govinfo = _BodyGovInfo(
        results=[{"packageId": package_id, "dateIssued": "2024-06-02", "title": "Other matter"}],
        texts={package_id: "An older opinion without the source locator."},
    )
    source = _ready(_reporter_document())
    searched = asyncio.run(
        search_root_body_corroboration(
            source,
            client=_BodyCourtListener(),
            govinfo_client=govinfo,
            retrospective_date=date(2024, 6, 1),
        )
    )
    asyncio.run(
        resolve_root_body_corroboration(
            Document.model_validate(searched.model_dump(mode="json")),
            client=_BodyCourtListener(),
            govinfo_client=govinfo,
        )
    )

    assert govinfo.text_calls == [(package_id, date(2024, 6, 1))]


def test_later_recap_citation_corroborates_root_as_third_party(monkeypatch) -> None:
    source_citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991", "snippet": "Brown v. State, order on motion."}],
            }
        ],
        recap_text={"991": f"The court follows {source_citation} on the disputed point."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )
    review_specs = []

    async def _review(*args, **kwargs):
        review_specs.append(args[1])
        del kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=source_citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                reason="The later citation agrees on all stated fields.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved", [
        (node.node_id, node.outcome, node.details.get("error")) for node in root.trace
    ]
    assert root.judgement(Question.IDENTITY).type == "third_party"
    assert root.authority_id is not None and root.authority_id.startswith("third_party:")
    assert root.found is not None and root.found.case_name == "Smith v. Jones"
    assert root.found.docket_id is None
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert trace.details["citation_quote"] == source_citation
    assert trace.details["evidence"][0]["citing_id"] == "44"
    assert trace.details["source_fields"]["case_name"] == "Smith v. Jones"
    assert trace.details["cited_fields"]["case_name"] == "Smith v. Jones"
    assert trace.details["field_comparisons"] == {
        "locator": {
            "original_stated_value": "1:24-cv-08760",
            "source_value": "1:24-cv-08760",
            "cited_value": "1:24-cv-08760",
            "outcome": "match",
        },
        "case_name": {
            "original_stated_value": "Smith v. Jones",
            "source_value": "Smith v. Jones",
            "cited_value": "Smith v. Jones",
            "outcome": "match",
        },
        "court": {
            "original_stated_value": "nysd",
            "source_value": "S.D.N.Y.",
            "cited_value": "S.D.N.Y.",
            "outcome": "match",
        },
        "date": {
            "original_stated_value": "2024",
            "source_value": "2024",
            "cited_value": "2024",
            "outcome": "match",
        },
    }
    grounding = trace.details["citation_grounding"]
    assert grounding["origin"] == "recap_documents:991:full_text"
    excerpt = trace.details["evidence"][0]["excerpt"]
    start, end = grounding["excerpt_span"]["start"], grounding["excerpt_span"]["end"]
    assert excerpt[start:end] == grounding["matched_text"] == source_citation
    body = client.recap_text["991"]
    start, end = grounding["document_span"]["start"], grounding["document_span"]["end"]
    assert body[start:end] == source_citation
    assert grounding["match_type"] == "perfect_match"
    prompt = review_specs[0].prefix.lower()
    assert "only that entire citation" in prompt
    assert "do not quote the surrounding sentence" in prompt


def test_third_party_review_corrects_source_court_and_keeps_both_sides_after_round_trip(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    document = _docket_document()
    record = document.citations[0]
    _fixture_fields(record, court="nyed")
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"This filing follows {citation} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(_ready(document), client=client, govinfo_client=_BodyGovInfo())
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                source_court="S.D.N.Y.",
                reason="Both source and independent citation identify the Southern District of New York.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.fields.court == "nysd"
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.extraction_reviewed_by_llm
    correction = next(
        correction
        for correction in root.field_updates
        if correction.node_id.endswith("source_reread") and correction.field == "court"
    )
    assert (correction.field, correction.after) == ("court", "nysd")
    correction_node = next(node for node in root.trace if node.node_id == correction.node_id)
    assert correction_node.reads is Reads.DOCUMENT
    review = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert review.reads is Reads.RECORD
    assert root.judgement(Question.IDENTITY).node_id == review.node_id
    assert review.details["field_comparisons"]["court"] == {
        "original_stated_value": "nyed",
        "source_value": "S.D.N.Y.",
        "cited_value": "S.D.N.Y.",
        "outcome": "match",
    }


def test_third_party_semantic_match_allows_different_case_name_spelling(monkeypatch) -> None:
    quoted = "Smith versus Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"This filing follows {quoted} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=quoted,
                cited_name="Smith versus Jones",
                cited_locator="1:24-cv-08760",
                reason="The variant case-name spelling refers to the same parties.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    comparison = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert comparison.details["field_comparisons"]["case_name"] == {
        "original_stated_value": "Smith v. Jones",
        "source_value": "Smith v. Jones",
        "cited_value": "Smith versus Jones",
        "outcome": "match",
    }


def test_source_reread_correction_applies_when_third_party_disagrees(monkeypatch) -> None:
    quoted = "Brown v. State, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    document = _docket_document()
    record = document.citations[0]
    _fixture_fields(record, court="nyed")
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"This filing cites {quoted} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(_ready(document), client=client, govinfo_client=_BodyGovInfo())
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="does_not_corroborate",
                citation_quote=quoted,
                cited_name="Brown v. State",
                cited_locator="1:24-cv-08760",
                source_court="S.D.N.Y.",
                conflict="case_name",
                reason="The source court is S.D.N.Y.; the cited case name conflicts.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "no_match"
    assert root.fields.court == "nysd"
    court_correction = next(
        correction
        for correction in root.field_updates
        if correction.node_id.endswith("source_reread") and correction.field == "court"
    )
    assert (
        next(node for node in root.trace if node.node_id == court_correction.node_id).reads is Reads.DOCUMENT
    )
    comparison = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert comparison.details["field_comparisons"]["case_name"]["outcome"] == "mismatch"


def test_source_reread_updates_typed_name_date_and_docket_fields(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    document = _docket_document()
    record = document.citations[0]
    _fixture_fields(
        record,
        docket_number="1:24-cv-08761",
        date=CitationDate(year="2023"),
    )
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"The later filing follows {citation} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(_ready(document), client=client, govinfo_client=_BodyGovInfo())
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                source_locator="1:24-cv-08760",
                reason="The source reread and independent citation agree.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.fields.docket_number == "1:24-cv-08760"
    assert root.fields.case_name is not None
    assert root.fields.case_name.text == citation[: len("Smith v. Jones")]
    assert root.fields.case_name.span == Span(0, len("Smith v. Jones"))
    assert root.fields.date == CitationDate(year="2024")
    for field in ("docket_number", "case_name", "date"):
        correction = next(
            item
            for item in root.field_updates
            if item.node_id.endswith("source_reread") and item.field == field
        )
        assert next(node for node in root.trace if node.node_id == correction.node_id).reads is Reads.DOCUMENT
    assert any(item.field == "docket_number" and item.after == "1:24-cv-08761" for item in root.field_updates)
    assert any(
        item.field == "date" and item.after == CitationDate(year="2023") for item in root.field_updates
    )


def test_ungrounded_source_reread_cannot_correct_or_resolve(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"The later filing follows {citation} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                source_name="Invented v. Party",
                reason="The two citations agree.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    original_fields = searched.citations[0].fields
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.authority_id is None
    assert not any(item.node_id.endswith("source_reread") for item in root.field_updates)
    assert not root.extraction_reviewed_by_llm
    assert root.fields == original_fields


@pytest.mark.parametrize("locator_kind", ["docket", "reporter"])
@pytest.mark.parametrize(
    ("conflict", "cited_name", "cited_court", "cited_year"),
    [
        ("name", "Brown v. State", "S.D.N.Y.", "2024"),
        ("court", "Smith v. Jones", "E.D.N.Y.", "2024"),
        ("date", "Smith v. Jones", "S.D.N.Y.", "2023"),
    ],
)
def test_independent_citation_with_same_locator_and_conflicting_identity_is_terminal_no_match(
    monkeypatch, locator_kind: str, conflict: str, cited_name: str, cited_court: str, cited_year: str
) -> None:
    locator = "1:24-cv-08760" if locator_kind == "docket" else "123 F.3d 456"
    document = _docket_document() if locator_kind == "docket" else _reporter_document()
    if locator_kind == "docket":
        citation_quote = f"{cited_name}, Case No. {locator} ({cited_court} {cited_year})"
        client = _BodyCourtListener(
            recap=[
                {
                    "docket_id": "44",
                    "docketNumber": "1:22-cv-00001",
                    "caseName": "Unrelated Later Filing",
                    "recap_documents": [{"id": "991"}],
                }
            ],
            recap_text={"991": f"This later filing cites {citation_quote} on the issue."},
        )
    else:
        citation_quote = f"{cited_name}, {locator} ({cited_court} {cited_year})"
        client = _BodyCourtListener(
            opinion=[
                {
                    "cluster_id": "77",
                    "caseName": "Unrelated Later Opinion",
                    "citation": ["555 F.3d 90"],
                    "opinions": [{"id": "88"}],
                }
            ],
            opinion_text={"88": f"<p>This later opinion cites {citation_quote} on the issue.</p>"},
        )
    searched = asyncio.run(
        search_root_body_corroboration(_ready(document), client=client, govinfo_client=_BodyGovInfo())
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="does_not_corroborate",
                citation_quote=citation_quote,
                cited_name=cited_name,
                cited_locator=locator,
                cited_court=cited_court,
                cited_date=cited_year,
                source_locator=locator,
                conflict={"name": "case_name", "court": "court", "date": "date"}[conflict],
                reason=f"The cited case has a conflicting {conflict}.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    date_only_docket_conflict = locator_kind == "docket" and conflict == "date"
    assert root.judgement(Question.IDENTITY).outcome == (
        "deferred_to_open_web_search" if date_only_docket_conflict else "no_match"
    )
    assert root.judgement(Question.IDENTITY).type == (None if date_only_docket_conflict else "third_party")
    if not date_only_docket_conflict:
        assert root.judgement(Question.IDENTITY).node_id == "cite-1:root_body:third_party_review"
    assert root.found is None
    assert root.authority_id is None
    assert root.fields.plaintiff == "Smith" and root.fields.defendant == "Jones"
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert trace.outcome == ("deferred" if date_only_docket_conflict else "no_match")
    assert trace.details["citation_quote"] == citation_quote
    assert trace.details["selected_evidence_index"] == 1
    assert (
        trace.details["field_comparisons"][{"name": "case_name", "court": "court", "date": "date"}[conflict]][
            "outcome"
        ]
        == "mismatch"
    )
    assert trace.details["field_comparisons"]["locator"]["outcome"] == "match"
    excerpt = trace.details["evidence"][0]["excerpt"]
    grounding = trace.details["citation_grounding"]
    assert excerpt[grounding["excerpt_span"]["start"] : grounding["excerpt_span"]["end"]] == citation_quote
    assert grounding["matched_text"] == citation_quote


@pytest.mark.parametrize(
    ("cited_reporter", "conflict", "expected"),
    [
        ("2023 WL 7654321", "date", "deferred_to_open_web_search"),
        ("2024 WL 1234567", "date", "no_match"),
        ("2023 WL 7654321", "case_name", "no_match"),
        ("2023 WL 7654321", "court", "no_match"),
    ],
)
def test_docket_third_party_date_conflict_respects_parallel_opinion_locator(
    monkeypatch, cited_reporter: str, conflict: str, expected: str
) -> None:
    cited_name = "Brown v. State" if conflict == "case_name" else "Smith v. Jones"
    cited_court = "E.D.N.Y." if conflict == "court" else "S.D.N.Y."
    cited_year = "2023" if conflict == "date" else "2024"
    citation_quote = f"{cited_name}, Case No. 1:24-cv-08760, {cited_reporter} ({cited_court} {cited_year})"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"This later filing cites {citation_quote} on the issue."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document_with_parallel_reporter()),
            client=client,
            govinfo_client=_BodyGovInfo(),
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="does_not_corroborate",
                citation_quote=citation_quote,
                cited_name=cited_name,
                cited_locator="1:24-cv-08760",
                cited_court=cited_court,
                cited_date=cited_year,
                conflict=conflict,
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]
    review = next(node for node in root.trace if node.node_id.endswith("third_party_review"))

    assert root.judgement(Question.IDENTITY).outcome == expected
    assert review.outcome == ("deferred" if expected.startswith("deferred") else "no_match")
    assert review.details["opinion_locator_comparison"] == {
        "source": ["2024wl1234567"],
        "cited": [cited_reporter.replace(" ", "").lower()],
        "shared": cited_reporter == "2024 WL 1234567",
    }
    assert review.details["field_comparisons"][conflict]["outcome"] == "mismatch"


def test_locator_mentioned_without_a_grounded_citation_does_not_create_no_match(monkeypatch) -> None:
    mention = "The clerk indexed docket number 1:24-cv-08760 among unrelated entries."
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": mention},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="does_not_corroborate",
                citation_quote=mention,
                cited_name=None,
                cited_locator="1:24-cv-08760",
                cited_court=None,
                cited_date=None,
                reason="This is a docket number mention, not a citation to a named case.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.judgement(Question.IDENTITY).type is None
    assert root.authority_id is None


def test_grounded_quote_survives_a_deferred_third_party_judgment(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Unrelated Later Filing",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": f"The filing quotes {citation} as an example of a fabricated case."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="defer",
                citation_quote=citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                reason="The later filing presents the citation as fabricated.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert trace.outcome == "deferred"
    assert trace.details["citation_quote"] == citation
    assert trace.details["citation_grounding"]["matched_text"] == citation
    assert trace.details["selected_evidence_index"] == 1


def test_body_fetch_budget_gives_each_provider_a_turn(monkeypatch) -> None:
    source_citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": str(index),
                "caseName": "Brown v. State",
                "citation": ["555 U.S. 1"],
                "opinions": [{"id": f"opinion-{index}"}],
            }
            for index in range(20)
        ],
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991"}],
            }
        ],
        opinion_text={f"opinion-{index}": "No citation here." for index in range(20)},
        recap_text={"991": f"The court follows {source_citation} on the disputed point."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=source_citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                reason="The later citation agrees on all stated fields.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.judgement(Question.IDENTITY).type == "third_party"
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert trace.details["evidence"][0]["citing_id"] == "44"


def test_third_party_review_requires_a_quote_from_citing_body(monkeypatch) -> None:
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991", "snippet": "Brown v. State, order on motion."}],
            }
        ],
        recap_text={"991": "The court cites Smith v. Jones, Case No. 1:24-cv-08760."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote="a citation not present in the later record",
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                reason="unsupported",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.authority_id is None


@pytest.mark.parametrize(
    ("body_citation", "expected_match_type"),
    [
        ("Smith v. Jones,\nCase No. 1:24-cv-08760 (S.D.N.Y. 2024)", "whitespace_relaxation"),
        ("Smith v. Jones,\nCasc No. 1:24-cv-08760 (S.D.N.Y. 2024)", "edit_distance"),
    ],
)
def test_third_party_quote_highlights_fuzzy_source_span(
    monkeypatch, body_citation: str, expected_match_type: str
) -> None:
    quoted_citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    body = f"Earlier argument. The court follows {body_citation} on this issue. Later argument."
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": body},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=quoted_citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    grounding = trace.details["citation_grounding"]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert grounding["matched_text"] == body_citation
    assert grounding["match_type"] == expected_match_type
    assert grounding["similarity_percent"] >= 90
    assert body[grounding["document_span"]["start"] : grounding["document_span"]["end"]] == body_citation


def test_third_party_review_rejects_quote_that_contains_surrounding_discussion(monkeypatch) -> None:
    citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    body = (
        f"{'The filing discusses unrelated facts. ' * 8}{citation}{' The analysis continues elsewhere.' * 8}"
    )
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Brown v. State",
                "recap_documents": [{"id": "991"}],
            }
        ],
        recap_text={"991": body},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=body,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = completed.citations[0]
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert trace.outcome == "deferred"
    assert trace.details["citation_quote"] is None
    assert trace.details["citation_grounding"] is None
    assert trace.details["selected_evidence_index"] is None
    assert root.authority_id is None


def test_third_party_quote_must_come_from_selected_document(monkeypatch) -> None:
    other_citation = "Brown v. State, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    source_citation = "Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)"
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "44",
                "docketNumber": "1:22-cv-00001",
                "caseName": "Other Filing One",
                "recap_documents": [{"id": "991"}],
            },
            {
                "docket_id": "45",
                "docketNumber": "1:22-cv-00002",
                "caseName": "Other Filing Two",
                "recap_documents": [{"id": "992"}],
            },
        ],
        recap_text={
            "991": f"This filing cites {other_citation} on the issue.",
            "992": f"This filing cites {source_citation} on the issue.",
        },
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote=source_citation,
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))
    root = completed.citations[0]
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert trace.details["citation_grounding"] is None
    assert root.authority_id is None


def test_archived_copy_of_source_filing_cannot_corroborate_itself() -> None:
    words = [f"word{index}" for index in range(500)]
    source = " ".join(words)
    archived_copy = "HEADER\n" + "\n".join(" ".join(words[index : index + 50]) for index in range(0, 500, 50))
    unrelated = " ".join(f"other{index}" for index in range(500))

    assert _same_source_document(source, archived_copy)
    assert not _same_source_document(source, unrelated)


def test_later_filing_in_trusted_source_docket_is_not_third_party_evidence() -> None:
    document = _reporter_document()
    document = document.evolve(
        source_metadata=replace(document.source_metadata, courtlistener_docket_id="own-docket"),
    )
    client = _BodyCourtListener(
        recap=[
            {
                "docket_id": "own-docket",
                "docketNumber": "1:24-cv-00001",
                "caseName": "Smith v. Jones",
                "recap_documents": [{"id": "later-filing"}],
            }
        ]
    )
    searched = asyncio.run(
        search_root_body_corroboration(_ready(document), client=client, govinfo_client=_BodyGovInfo())
    )
    restored = Document.model_validate(searched.model_dump(mode="json"))
    assert restored.source_metadata.courtlistener_docket_id == "own-docket"

    completed = asyncio.run(
        resolve_root_body_corroboration(restored, client=client, govinfo_client=_BodyGovInfo())
    )

    root = completed.citations[0]
    trace = next(node for node in root.trace if node.node_id.endswith("third_party_review"))
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert trace.details["fetches"] == [
        {
            "origin": "courtlistener_docket:own-docket",
            "outcome": "same_source_docket",
            "source_docket_id": "own-docket",
        }
    ]


def test_archived_copy_with_sparse_ocr_changes_is_still_the_source() -> None:
    words = [f"word{index}" for index in range(500)]
    copied = words.copy()
    for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
        copied[int(len(words) * fraction) + 8] = "ocrchange"

    assert _same_source_document(" ".join(words), "\n".join(copied))


def test_third_party_reporter_admission_rejects_changed_printed_number(monkeypatch) -> None:
    client = _BodyCourtListener(
        opinion=[
            {
                "cluster_id": "77",
                "caseName": "Brown v. State",
                "citation": ["555 F.3d 90"],
                "opinions": [{"id": "88", "snippet": "Brown v. State, background."}],
            }
        ],
        opinion_text={"88": "<p>Compare Smith v. Jones, 124 F.3d 456 (S.D.N.Y. 2024).</p>"},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_reporter_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote="Smith v. Jones, 124 F.3d 456 (S.D.N.Y. 2024)",
                cited_name="Smith v. Jones",
                cited_locator="124 F.3d 456",
                source_locator="123 F.3d 456",
                conflict="locator",
                reason="The cases seem to agree.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client, session=object()))

    assert completed.citations[0].judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert completed.citations[0].authority_id is None


def test_later_govinfo_package_citation_corroborates_root(monkeypatch) -> None:
    package_id = "USCOURTS-nysd-1_22-cv-00001"
    govinfo = _BodyGovInfo(
        results=[{"packageId": package_id, "title": "Brown v. State"}],
        texts={package_id: "The court follows Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)."},
    )
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_docket_document()), client=_BodyCourtListener(), govinfo_client=govinfo
        )
    )

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            _third_party_proposal(
                decision="corroborates",
                citation_quote="Smith v. Jones, Case No. 1:24-cv-08760 (S.D.N.Y. 2024)",
                cited_name="Smith v. Jones",
                cited_locator="1:24-cv-08760",
                reason="The independent package cites the same case and docket.",
            )
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(
        resolve_root_body_corroboration(
            searched, client=_BodyCourtListener(), govinfo_client=govinfo, session=object()
        )
    )
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.judgement(Question.IDENTITY).type == "third_party"
    assert root.found is not None and root.found.govinfo_package_id is None


def test_body_review_stops_courtlistener_fetches_after_api_limit(monkeypatch) -> None:
    class _RateLimitedCourtListener(_BodyCourtListener):
        def __init__(self) -> None:
            super().__init__(
                opinion=[
                    {
                        "cluster_id": str(index),
                        "caseName": f"Later Case {index}",
                        "citation": [f"555 F.3d {index}"],
                        "opinions": [{"id": str(index), "snippet": "Background."}],
                    }
                    for index in (1, 2, 3)
                ]
            )
            self.fetches = 0

        def get_opinion(self, opinion_id: str):
            del opinion_id
            self.fetches += 1
            raise CourtListenerError(
                "CourtListener rate limited this request",
                failure_type="api_limit",
                upstream_status_code=429,
                retryable=True,
            )

    waits: list[float] = []
    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.time.sleep", waits.append)
    client = _RateLimitedCourtListener()
    searched = asyncio.run(
        search_root_body_corroboration(
            _ready(_reporter_document()), client=client, govinfo_client=_BodyGovInfo()
        )
    )
    completed = asyncio.run(resolve_root_body_corroboration(searched, client=client))
    root = completed.citations[0]
    review = next(node for node in root.trace if node.node_id.endswith("third_party_review"))

    assert client.fetches == 3
    assert review.details["fetches"][0]["outcome"] == "retry"
    assert review.details["fetches"][1]["outcome"] == "retry"
    assert review.details["fetches"][2]["outcome"] == "failed"
    assert review.details["fetches"][0]["failure_type"] == "api_limit"
    assert waits == [2.0, 4.0]
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"


def test_body_fetch_recovers_from_one_transient_proxy_limit(monkeypatch) -> None:
    class _TransientCourtListener(_BodyCourtListener):
        def __init__(self) -> None:
            super().__init__(opinion_text={"88": "<p>A citation.</p>"})
            self.fetches = 0

        def get_opinion(self, opinion_id: str):
            self.fetches += 1
            if self.fetches == 1:
                raise CourtListenerError(
                    "Transient proxy limit",
                    failure_type="api_limit",
                    upstream_status_code=429,
                    retryable=True,
                )
            return super().get_opinion(opinion_id)

    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.time.sleep", lambda _seconds: None)
    client = _TransientCourtListener()
    fetches: list[dict[str, object]] = []

    text = _read_courtlistener_body(
        RootBodySearchSource.COURTLISTENER_CLUSTER,
        "88",
        client,
        origin="opinions:88:full_text",
        fetches=fetches,
    )

    assert text == "A citation."
    assert client.fetches == 2
    assert [fetch["outcome"] for fetch in fetches] == ["retry"]


def test_body_fetch_waits_for_short_retry_after_and_skips_daily_exhaustion(monkeypatch) -> None:
    class _QuotaClient(_BodyCourtListener):
        def __init__(self, wait: float) -> None:
            super().__init__(opinion_text={"88": "<p>A citation.</p>"})
            self.wait = wait
            self.fetches = 0

        def get_opinion(self, opinion_id: str):
            self.fetches += 1
            if self.fetches == 1:
                raise CourtListenerError(
                    "Proxy limit",
                    failure_type="api_limit",
                    upstream_status_code=429,
                    retryable=True,
                    retry_after_seconds=self.wait,
                )
            return super().get_opinion(opinion_id)

    waits: list[float] = []
    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.time.sleep", waits.append)
    short = _QuotaClient(12)
    fetches: list[dict[str, object]] = []
    assert (
        _read_courtlistener_body(
            RootBodySearchSource.COURTLISTENER_CLUSTER,
            "88",
            short,
            origin="opinions:88:full_text",
            fetches=fetches,
        )
        == "A citation."
    )
    assert waits == [12]
    assert fetches[0]["wait_seconds"] == 12
    assert short.fetches == 2

    daily = _QuotaClient(3600)
    fetches = []
    assert (
        _read_courtlistener_body(
            RootBodySearchSource.COURTLISTENER_CLUSTER,
            "88",
            daily,
            origin="opinions:88:full_text",
            fetches=fetches,
        )
        is None
    )
    assert daily.fetches == 1
    assert waits == [12]
    assert fetches[0]["outcome"] == "failed"


def test_body_search_retries_transient_failure_and_records_it(monkeypatch) -> None:
    class _TransientSearch(_BodyCourtListener):
        def search(self, query: str, search_type: str, cursor=None, *, semantic=False):
            if not self.calls:
                self.calls.append((query, search_type))
                raise CourtListenerError(
                    "temporarily unavailable",
                    failure_type="upstream_error",
                    upstream_status_code=503,
                    retryable=True,
                )
            return super().search(query, search_type, cursor=cursor, semantic=semantic)

    waits: list[float] = []
    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.time.sleep", waits.append)
    record = _ready(_reporter_document()).citations[0]
    client = _TransientSearch()
    node = _courtlistener_body_node(
        record,
        locator="555 F.3d 1",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert len(client.calls) == 2
    assert waits == [1.0]
    assert "after 2 attempts" in (node.status_message or "")
    assert "upstream_error (503)" in (node.status_message or "")


def test_body_search_does_not_wait_for_daily_quota_reset(monkeypatch) -> None:
    class _DailyLimit(_BodyCourtListener):
        def search(self, query: str, search_type: str, cursor=None, *, semantic=False):
            self.calls.append((query, search_type))
            raise CourtListenerError(
                "daily quota exhausted",
                failure_type="api_limit",
                upstream_status_code=429,
                retryable=True,
                retry_after_seconds=3600,
            )

    waits: list[float] = []
    monkeypatch.setattr("mellea_lrc.validation.root_identity.body.time.sleep", waits.append)
    client = _DailyLimit()
    node = _courtlistener_body_node(
        _ready(_reporter_document()).citations[0],
        locator="555 F.3d 1",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert len(client.calls) == 1
    assert waits == []
    assert node.outcome.value == "failed"
    assert "attempt 1: api_limit (429), no retry" in (node.error or "")


def _wide_opinion_page() -> list[dict[str, object]]:
    return [{"cluster_id": str(index), "caseName": f"Other Case {index}"} for index in range(1, 21)]


def test_broad_body_search_prioritizes_bounded_quoted_hits_and_round_trips() -> None:
    raw = _wide_opinion_page()
    client = _PagedBodyCourtListener(
        {
            "123 F.3d 456": (25, raw, "raw-next"),
            '"123 F.3d 456"': (
                2,
                [{"cluster_id": "gold", "caseName": "Smith v. Jones"}, raw[0]],
                None,
            ),
        }
    )
    node = _courtlistener_body_node(
        _ready(_reporter_document()).citations[0],
        locator="123 F.3d 456",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert client.calls == [("123 F.3d 456", "o"), ('"123 F.3d 456"', "o")]
    assert node.outcome.value == "exceeds_review_limit"
    assert node.candidate_count == 25
    assert node.continuation == "raw-next"
    assert [candidate["cluster_id"] for candidate in node.candidates] == [
        "gold",
        *(str(index) for index in range(1, 21)),
    ]
    assert [
        (attempt.kind, attempt.candidate_count, attempt.continuation) for attempt in node.search_attempts
    ] == [
        ("raw_locator", 25, "raw-next"),
        ("quoted_locator", 2, None),
    ]
    assert [attempt.used_for_candidates for attempt in node.search_attempts] == [True, True]

    payload = {"node_type": "RootBodySearchNode", **serialize_dataclass(node)}
    restored = deserialize_validation_node(payload)
    assert restored.search_attempts == node.search_attempts
    assert [candidate["cluster_id"] for candidate in restored.candidates] == [
        candidate["cluster_id"] for candidate in node.candidates
    ]
    del payload["search_attempts"]
    assert deserialize_validation_node(payload).search_attempts == ()


def test_broad_body_search_keeps_raw_page_when_quoted_query_is_empty() -> None:
    raw = _wide_opinion_page()
    client = _PagedBodyCourtListener({"123 F.3d 456": (25, raw, "raw-next"), '"123 F.3d 456"': (0, [], None)})
    node = _courtlistener_body_node(
        _ready(_reporter_document()).citations[0],
        locator="123 F.3d 456",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert [candidate["cluster_id"] for candidate in node.candidates] == [
        str(index) for index in range(1, 21)
    ]
    assert node.search_attempts[1].candidate_count == 0
    assert node.search_attempts[1].used_for_candidates is False
    assert node.search_attempts[1].query == '"123 F.3d 456"'


def test_broad_body_search_keeps_raw_page_when_quoted_query_also_overflows() -> None:
    raw = _wide_opinion_page()
    refined = [{"cluster_id": f"phrase-{index}"} for index in range(20)]
    client = _PagedBodyCourtListener(
        {
            "123 F.3d 456": (45, raw, "raw-next"),
            '"123 F.3d 456"': (23, refined, "phrase-next"),
        }
    )
    node = _courtlistener_body_node(
        _ready(_reporter_document()).citations[0],
        locator="123 F.3d 456",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert [candidate["cluster_id"] for candidate in node.candidates] == [
        str(index) for index in range(1, 21)
    ]
    assert node.search_attempts[1].candidate_count == 23
    assert node.search_attempts[1].continuation == "phrase-next"
    assert node.search_attempts[1].used_for_candidates is False


def test_quoted_body_query_uses_parsed_locator_despite_raw_spacing() -> None:
    client = _PagedBodyCourtListener(
        {
            "123   F.3d   456": (21, _wide_opinion_page(), "raw-next"),
            '"123 F.3d 456"': (1, [{"cluster_id": "gold"}], None),
        }
    )
    node = _courtlistener_body_node(
        _ready(_reporter_document()).citations[0],
        locator="123   F.3d   456",
        source=RootBodySearchSource.COURTLISTENER_CLUSTER,
        search_type="o",
        client=client,
    )

    assert client.calls[1] == ('"123 F.3d 456"', "o")
    assert node.candidates[0]["cluster_id"] == "gold"


def test_quoted_docket_query_uses_parsed_number_and_preserves_raw_on_failure() -> None:
    class _PhraseFailure(_PagedBodyCourtListener):
        def search(self, query: str, search_type: str, cursor=None, *, semantic=False):
            if query.startswith('"'):
                self.calls.append((query, search_type))
                raise CourtListenerError(
                    "temporary failure",
                    failure_type="upstream_error",
                    upstream_status_code=503,
                    retryable=True,
                )
            return super().search(query, search_type, cursor=cursor, semantic=semantic)

    client = _PhraseFailure({"Case No.   1:24-cv-08760": (21, [{"docket_id": "raw"}], "raw-next")})
    node = _courtlistener_body_node(
        _ready(_docket_document()).citations[0],
        locator="Case No.   1:24-cv-08760",
        source=RootBodySearchSource.COURTLISTENER_DOCKET,
        search_type="r",
        client=client,
    )

    assert client.calls[-1] == ('"1:24-cv-08760"', "r")
    assert [candidate["docket_id"] for candidate in node.candidates] == ["raw"]
    assert node.outcome.value == "exceeds_review_limit"
    assert node.search_attempts[1].status.value == "failed"
    assert node.search_attempts[1].used_for_candidates is False
    assert "CourtListenerError" in (node.search_attempts[1].error or "")
