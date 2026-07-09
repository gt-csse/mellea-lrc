"""Tests for first-layer citation retrieval."""

import asyncio

import pytest
from pydantic import ValidationError

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.core.citations import FullCaseCitation, FullLawCitation, Reporter
from mellea_lrc.core.immutable import ExtraData
from mellea_lrc.core.spans import Span
from mellea_lrc.courtlistener.remote import (
    CourtListenerAccessClient,
    CourtListenerAccessConfig,
)
from mellea_lrc.courtlistener.lookup import (
    citation_lookup_envelope_dict,
    normalize_citation_lookup_payload,
)
from mellea_lrc.courtlistener.types import CourtListenerCitationRecord, RetrievalFailureDetail
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument, ExtractionMetadata
from mellea_lrc.preprocessing import PreprocessedDocument, preprocess_plain_text_from_string
from mellea_lrc.retrieval.case_name_prepare import (
    JSON_OUTPUT_REQUIREMENT,
    _PreparedCaseName,
    _case_name_preparation_requirements,
    _proposal_from_output,
    _validate_grounded_before_locator,
    render_case_name_preparation_chat_messages,
    render_case_name_preparation_prompt,
)
from mellea_lrc.retrieval.not_found_search import _case_name_query
from mellea_lrc.retrieval.pipeline import run_retrieval, run_retrieval_async
from mellea_lrc.retrieval.types import (
    AmbiguousCitationRetrieval,
    CaseNamePreparationStatus,
    CaseNameSearchStatus,
    CaseNameSearchPreparation,
    CourtResolutionSource,
    RetrievalStatus,
)


def _client(
    response: dict[str, object],
    *,
    docket_response: dict[str, object] | None = None,
) -> CourtListenerAccessClient:
    return CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: response,
        get_json=lambda _url: docket_response or {},
    )


def _extracted_document(
    *,
    preprocessed: PreprocessedDocument,
    citations: tuple[ExtractedCitation, ...],
) -> ExtractedDocument:
    return ExtractedDocument(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        citations=citations,
        extraction_metadata=ExtractionMetadata(),
    )


def test_retrieve_full_case_found() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        citations=(
            ExtractedCitation(
                citation_id="abc123",
                span=Span(0, 28),
                matched_text="347 U.S. 483",
                citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483", court="scotus"),
            ),
        ),
    )

    result = run_retrieval(
        extraction,
        client_mode="custom",
        client=_client(
            {
                "cache": "miss",
                "request_id": "request-1",
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "clusters": [
                        {
                            "case_name": "Brown v. Board of Education",
                            "date_filed": "1954-05-17",
                            "docket_id": 191796,
                            "absolute_url": "/opinion/1/",
                        }
                    ],
                    "query_time_ms": 12,
                },
            },
            docket_response={"id": 191796, "court_id": "scotus"},
        ),
    )

    retrieval = result.retrievals[0]
    assert result.text == extraction.text
    assert result.source_metadata == extraction.source_metadata
    assert result.preprocessing_metadata == extraction.preprocessing_metadata
    assert result.extraction_metadata == extraction.extraction_metadata
    assert result.citations == extraction.citations
    assert retrieval.candidate.court_resolution.courtlistener_court_id == "scotus"
    assert retrieval.status == RetrievalStatus.FOUND
    assert retrieval.locator == "347 U.S. 483"
    assert retrieval.case_names == ("Brown v. Board of Education",)
    assert retrieval.request_trace.http_status == 200
    assert retrieval.request_trace.cache == "miss"
    assert retrieval.candidate.record == CourtListenerCitationRecord(
        case_name="Brown v. Board of Education",
        date_filed="1954-05-17",
        court_id=None,
        docket_id="191796",
        extra_data=ExtraData({"absolute_url": "/opinion/1/"}),
    )
    assert retrieval.candidate.court_resolution.courtlistener_court_id == "scotus"
    assert retrieval.candidate.court_resolution.resolved_via == CourtResolutionSource.DOCKET_LOOKUP
    assert retrieval.candidate.court_resolution.request_trace is not None
    assert retrieval.candidate.court_resolution.request_trace.http_status == 200
    assert result.found == (retrieval,)
    assert retrieval.extra_data.to_dict() == {
        "response": {"query_time_ms": 12},
        "envelope": {"request_id": "request-1"},
    }


def test_retrieve_found_docket_lookup_is_best_effort_and_deduplicated() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Example, 1 F.3d 2."),
        citations=(
            ExtractedCitation(
                citation_id="cite-1",
                span=Span(0, 18),
                matched_text="1 F.3d 2",
                citation=FullCaseCitation(volume="1", reporter=Reporter(edition_short_name="F.3d", root_short_name="F.", name="Federal Reporter", cite_type="federal", is_scotus=False, source="reporters"), page="2"),
            ),
        ),
    )
    get_urls: list[str] = []
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {
                "citation": "1 F.3d 2",
                "status": 200,
                "clusters": [
                    {"case_name": "Example A", "docket_id": 42},
                    {"case_name": "Example B", "docket_id": 42},
                ],
            }
        },
        get_json=lambda url: get_urls.append(url) or {"detail": "temporarily unavailable"},
    )

    result = run_retrieval(extraction, client_mode="custom", client=client)

    assert result.retrievals[0].status == RetrievalStatus.FOUND
    assert result.retrievals[0].candidate.record.court_id is None
    assert get_urls == ["https://cl-access.example.test/dockets/42"]


def test_retrieve_ambiguous_resolves_court_for_each_candidate() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Example, 1 F.3d 2."),
        citations=(
            ExtractedCitation(
                citation_id="cite-1",
                span=Span(0, 18),
                matched_text="1 F.3d 2",
                citation=FullCaseCitation(volume="1", reporter=Reporter(edition_short_name="F.3d", root_short_name="F.", name="Federal Reporter", cite_type="federal", is_scotus=False, source="reporters"), page="2"),
            ),
        ),
    )
    get_urls: list[str] = []

    def get_json(url: str) -> dict[str, object]:
        get_urls.append(url)
        return {"court_id": "ca1" if url.endswith("/11") else "ca2"}

    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {
                "citation": "1 F.3d 2",
                "status": 300,
                "clusters": [
                    {"case_name": "Example A", "docket_id": 11},
                    {"case_name": "Example B", "docket_id": 22},
                ],
            }
        },
        get_json=get_json,
    )

    result = run_retrieval(extraction, client_mode="custom", client=client)

    retrieval = result.retrievals[0]
    assert isinstance(retrieval, AmbiguousCitationRetrieval)
    assert [candidate.record.case_name for candidate in retrieval.candidates] == [
        "Example A",
        "Example B",
    ]
    assert [candidate.court_resolution.courtlistener_court_id for candidate in retrieval.candidates] == [
        "ca1",
        "ca2",
    ]
    assert get_urls == [
        "https://cl-access.example.test/dockets/11",
        "https://cl-access.example.test/dockets/22",
    ]


def test_retrieve_non_case_citation_is_skipped() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("See 28 U.S.C. § 636."),
        citations=(
            ExtractedCitation(
                citation_id="law",
                span=Span(4, 20),
                matched_text="28 U.S.C. § 636",
                citation=FullLawCitation(volume="28", reporter="U.S.C.", page="636"),
            ),
        ),
    )

    result = run_retrieval(extraction, client_mode="custom", client=_client({}))

    assert result.retrievals[0].status == RetrievalStatus.SKIPPED


def _not_found_extraction(citation: FullCaseCitation) -> ExtractedDocument:
    return _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Doe v. Roe, 999 U.S. 999."),
        citations=(
            ExtractedCitation(
                citation_id="nf",
                span=Span(0, 24),
                matched_text="999 U.S. 999",
                citation=citation,
            ),
        ),
    )


def test_not_found_reports_case_name_search_count() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {
            "cache": "hit",
            "key": "search-key",
            "count": 7,
            "results": [],
        },
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999", court="scotus"),
    )

    retrieval = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0]

    assert retrieval.status == RetrievalStatus.NOT_FOUND
    assert retrieval.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert retrieval.candidate_search.query == "caseName:(Doe AND Roe) AND court_id:scotus"
    assert [probe.request_trace.key for probe in retrieval.candidate_search.probes] == [
        "search-key",
        "search-key",
    ]
    assert [
        (
            probe.corpus.value,
            probe.request_trace.http_status,
            probe.request_trace.cache,
            probe.case_count,
        )
        for probe in retrieval.candidate_search.probes
    ] == [
        ("o", 200, "hit", 7),
        ("r", 200, "hit", 7),
    ]


def test_case_name_query_uses_one_meaningful_anchor_per_party() -> None:
    assert _case_name_query("Peterson", "Nelnet Diversified Sols., LLC") == "caseName:(Peterson AND Nelnet)"
    assert _case_name_query("E.E.O.C.", "Maricopa County Cmty. Coll. Dist.") == "caseName:(EEOC AND Maricopa)"
    assert _case_name_query("Estate of Martinelli", "City & Cty. of Denver") == "caseName:(Martinelli AND Denver)"


def test_case_name_query_includes_court_when_available() -> None:
    assert (
        _case_name_query("Peterson", "Nelnet", court="ca10")
        == "caseName:(Peterson AND Nelnet) AND court_id:ca10"
    )


def test_case_name_preparation_schema_never_asks_llm_for_combined_case_name() -> None:
    assert "case_name" not in _PreparedCaseName.model_fields
    assert '"case_name"' not in JSON_OUTPUT_REQUIREMENT
    assert _PreparedCaseName.model_config["extra"] == "forbid"


def test_case_name_preparation_prompt_is_compact_and_requirement_driven() -> None:
    text = "See Smith v. Jones, 999 U.S. 999."
    locator_start = text.index("999 U.S. 999")
    window = DocumentTextWindow.around(
        text,
        Span(locator_start, locator_start + len("999 U.S. 999")),
        before_chars=320,
        after_chars=0,
    )
    requirements = _case_name_preparation_requirements(window)
    prompt = render_case_name_preparation_prompt(
        local_context=window.text,
        locator="999 U.S. 999",
        extracted_plaintiff="Smith",
        extracted_defendant="Jones",
        window=window,
    )
    messages = render_case_name_preparation_chat_messages(
        local_context=window.text,
        locator="999 U.S. 999",
        extracted_plaintiff="Smith",
        extracted_defendant="Jones",
        window=window,
    )

    assert requirements[0].description == JSON_OUTPUT_REQUIREMENT
    assert requirements[0].validation_fn is not None
    assert [message.role for message in messages] == ["user"]
    assert messages[0].content == prompt
    assert "output_format_hint" not in prompt
    assert prompt.count("Return exactly one JSON object") == 1
    assert "classification must be consistent" not in prompt
    assert "plaintiff and defendant must be copied" in prompt
    assert "[local_context]" in prompt
    assert "See Smith v. Jones, 999 U.S. 999" in prompt


def test_case_name_preparation_parses_unwrapped_instruct_json() -> None:
    proposal = _proposal_from_output(
        '{"classification": "complete_case_name", '
        '"plaintiff": "Smith", "defendant": "Jones", "reason": "copied"}'
    )

    assert proposal == _PreparedCaseName(
        classification="complete_case_name",
        plaintiff="Smith",
        defendant="Jones",
        reason="copied",
    )


def test_case_name_preparation_rejects_combined_case_name_field() -> None:
    with pytest.raises(ValueError, match="case-name preparation schema"):
        _proposal_from_output(
            '{"classification": "complete_case_name", '
            '"plaintiff": "Smith", "defendant": "Jones", "case_name": "Smith v. Jones"}'
        )


def test_case_name_preparation_validation_uses_raw_instruct_output() -> None:
    text = "See Smith v. Jones, 999 U.S. 999."
    locator_start = text.index("999 U.S. 999")
    window = DocumentTextWindow.around(
        text,
        Span(locator_start, locator_start + len("999 U.S. 999")),
        before_chars=320,
        after_chars=0,
    )

    class Output:
        value = (
            '{"classification": "complete_case_name", '
            '"plaintiff": "Smith", "defendant": "Jones", "reason": "copied"}'
        )

        @property
        def parsed_repr(self) -> object:
            raise AssertionError("instruct validation must not depend on parsed_repr")

    class Context:
        def last_output(self) -> Output:
            return Output()

    assert _validate_grounded_before_locator(Context(), window).as_bool()


def test_case_name_preparation_rejects_incomplete_output_when_case_marker_is_visible() -> None:
    text = "First see Alpha v. Beta, 111 F.3d 222. But see Gamma v. Delta, 999 U.S. 999."
    locator_start = text.index("999 U.S. 999")
    window = DocumentTextWindow.around(
        text,
        Span(locator_start, locator_start + len("999 U.S. 999")),
        before_chars=320,
        after_chars=0,
    )

    class Output:
        value = (
            '{"classification": "partial_case_name", '
            '"plaintiff": "Gamma", "defendant": null, "reason": "only one party"}'
        )

    class Context:
        def last_output(self) -> Output:
            return Output()

    result = _validate_grounded_before_locator(Context(), window)

    assert not result.as_bool()
    assert result.reason is not None
    assert "v. case-name marker" in result.reason


def test_case_name_preparation_rejects_party_before_intervening_locator() -> None:
    text = "First see Alpha v. Beta, 111 F.3d 222. But see Gamma v. Delta, 999 U.S. 999."
    locator_start = text.index("999 U.S. 999")
    window = DocumentTextWindow.around(
        text,
        Span(locator_start, locator_start + len("999 U.S. 999")),
        before_chars=320,
        after_chars=0,
    )

    class Output:
        value = (
            '{"classification": "complete_case_name", '
            '"plaintiff": "Alpha", "defendant": "Beta", "reason": "parser hint"}'
        )

    class Context:
        def last_output(self) -> Output:
            return Output()

    result = _validate_grounded_before_locator(Context(), window)

    assert not result.as_bool()
    assert result.reason is not None
    assert "before another locator" in result.reason


def test_not_found_reports_zero_case_name_search_results() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"count": 0, "results": []},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999"),
    )

    retrieval = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0]

    assert retrieval.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert [probe.case_count for probe in retrieval.candidate_search.probes] == [0, 0]


def test_not_found_preserves_bounded_search_candidate_summaries() -> None:
    results = [
        {
            "caseName": f"Doe v. Roe {index}",
            "court_id": "scotus",
            "dateFiled": "2001-01-01",
            "docketNumber": str(index),
            "cluster_id": index,
        }
        for index in range(7)
    ]
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"count": 7, "results": results},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999", court="scotus"),
    )

    search = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0].candidate_search

    assert len(search.probes[0].candidates) == 5
    assert search.probes[0].candidates[0].case_name == "Doe v. Roe 0"
    assert search.probes[0].candidates[0].cluster_id == "0"


def test_not_found_preserves_failed_search_http_status() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"http_status": 503, "detail": "upstream search unavailable"},
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999"),
    )

    retrieval = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0]

    assert retrieval.candidate_search.status == CaseNameSearchStatus.SEARCH_FAILED
    assert [probe.request_trace.http_status for probe in retrieval.candidate_search.probes] == [503, 503]
    assert all(probe.case_count is None for probe in retrieval.candidate_search.probes)
    assert all(
        probe.request_trace.error_message == "upstream search unavailable"
        for probe in retrieval.candidate_search.probes
    )


def test_not_found_traces_opinion_and_recap_search_independently() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda url: (
            {"count": 3, "results": []}
            if "type=o" in url
            else {"http_status": 503, "detail": "RECAP unavailable"}
        ),
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999"),
    )

    search = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0].candidate_search

    assert search.status == CaseNameSearchStatus.PARTIAL
    assert search.probes[0].case_count == 3
    assert search.probes[1].request_trace.http_status == 503
    assert search.probes[1].request_trace.error_message == "RECAP unavailable"


def test_not_found_reads_count_from_deployed_service_raw_response() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {
            "results": [],
            "raw": {"count": 0, "next": None, "previous": None, "results": []},
        },
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", defendant="Roe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999"),
    )

    retrieval = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0]

    assert retrieval.candidate_search.status == CaseNameSearchStatus.SEARCHED
    assert [probe.case_count for probe in retrieval.candidate_search.probes] == [0, 0]
    assert all(probe.request_trace.error_message is None for probe in retrieval.candidate_search.probes)


def test_not_found_skips_search_without_both_parties() -> None:
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: pytest.fail("search must not run without both parties"),
    )
    extraction = _not_found_extraction(
        FullCaseCitation(plaintiff="Doe", volume="999", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="999"),
    )

    retrieval = run_retrieval(extraction, client_mode="custom", client=client).retrievals[0]

    assert retrieval.status == RetrievalStatus.NOT_FOUND
    assert retrieval.candidate_search.status == CaseNameSearchStatus.SKIPPED_PARTIAL_CASE_NAME
    assert retrieval.candidate_search.probes == ()


def test_async_retrieval_bounds_case_name_preparation_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    max_active = 0
    calls: list[str] = []

    class FakeSession:
        def clone(self) -> "FakeSession":
            return self

    async def fake_prepare(
        _session: object,
        *,
        document_text: str,
        citation: ExtractedCitation,
    ) -> CaseNameSearchPreparation:
        nonlocal active, max_active
        assert document_text
        active += 1
        max_active = max(max_active, active)
        calls.append(citation.citation_id)
        await asyncio.sleep(0.01)
        active -= 1
        return CaseNameSearchPreparation(
            status=CaseNamePreparationStatus.ACCEPTED,
            plaintiff=f"P{citation.citation_id}",
            defendant=f"D{citation.citation_id}",
            prepared_case_name=f"P{citation.citation_id} v. D{citation.citation_id}",
            source="llm",
        )

    monkeypatch.setattr(
        "mellea_lrc.retrieval.pipeline.prepare_case_name_for_search",
        fake_prepare,
    )
    text = "A, 999 U.S. 999. B, 999 U.S. 999. C, 999 U.S. 999. D, 999 U.S. 999."
    citations = []
    start = 0
    for index in range(4):
        locator_start = text.index("999 U.S. 999", start)
        start = locator_start + len("999 U.S. 999")
        citations.append(
            ExtractedCitation(
                citation_id=f"cite-{index}",
                span=Span(locator_start, start),
                matched_text="999 U.S. 999",
                citation=FullCaseCitation(
                    volume="999",
                    reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S."),
                    page="999",
                ),
            )
        )
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string(text),
        citations=tuple(citations),
    )
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=lambda _url, _data: {
            "response": {"citation": "999 U.S. 999", "status": 404, "clusters": []},
        },
        get_json=lambda _url: {"count": 0, "results": []},
    )

    result = asyncio.run(
        run_retrieval_async(
            extraction,
            client_mode="custom",
            client=client,
            session=FakeSession(),
            mellea_concurrency=2,
        )
    )

    assert max_active == 2
    assert sorted(calls) == [f"cite-{index}" for index in range(4)]
    assert [item.candidate_search.preparation.status for item in result.retrievals] == [
        CaseNamePreparationStatus.ACCEPTED,
    ] * 4


def test_retrieve_surfaces_typed_courtlistener_failure_detail() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Brown v. Board, 347 U.S. 483."),
        citations=(
            ExtractedCitation(
                citation_id="limited",
                span=Span(0, 28),
                matched_text="347 U.S. 483",
                citation=FullCaseCitation(volume="347", reporter=Reporter(edition_short_name="U.S.", root_short_name="U.S.", name="United States Supreme Court Reports", cite_type="federal", is_scotus=True, source="reporters"), page="483"),
            ),
        ),
    )

    result = run_retrieval(
        extraction,
        client_mode="custom",
        client=_client(
            {
                "response": {
                    "citation": "347 U.S. 483",
                    "status": 429,
                    "error_message": "CourtListener POST failed with 429",
                    "limit_detail": {
                        "failure_type": "api_limit",
                        "message": "CourtListener POST failed with 429",
                        "retryable": True,
                        "upstream_status_code": 429,
                    },
                    "clusters": [],
                },
            }
        ),
    )

    retrieval = result.retrievals[0]
    assert retrieval.status == RetrievalStatus.THROTTLED
    assert retrieval.request_trace.error_message == "CourtListener POST failed with 429"
    assert retrieval.failure_detail == RetrievalFailureDetail(
        failure_type="api_limit",
        message="CourtListener POST failed with 429",
        retryable=True,
        upstream_status_code=429,
    )


def test_courtlistener_transport_rejects_type_coercion() -> None:
    with pytest.raises(ValidationError):
        normalize_citation_lookup_payload(
            {"response": {"citation": "347 U.S. 483", "status": "200"}},
            "347",
            "U.S.",
            "483",
        )


def test_direct_courtlistener_response_does_not_duplicate_fields_as_envelope_extras() -> None:
    lookup = normalize_citation_lookup_payload(
        {
            "citation": "347 U.S. 483",
            "status": 200,
            "clusters": [],
            "query_time_ms": 12,
        },
        "347",
        "U.S.",
        "483",
    )

    assert lookup.extra_data.to_dict() == {"response": {"query_time_ms": 12}}


def test_courtlistener_service_round_trip_preserves_explicit_extra_data() -> None:
    original = normalize_citation_lookup_payload(
        {
            "request_id": "request-1",
            "response": {
                "citation": "347 U.S. 483",
                "status": 200,
                "query_time_ms": 12,
                "clusters": [{"case_name": "Brown", "absolute_url": "/opinion/1/"}],
            },
        },
        "347",
        "U.S.",
        "483",
    )

    restored = normalize_citation_lookup_payload(
        citation_lookup_envelope_dict(original),
        "347",
        "U.S.",
        "483",
    )

    assert restored == original


def test_retrieve_missing_locator_is_invalid_without_service_call() -> None:
    calls = 0

    def post_json(_url: str, _data: object) -> object:
        nonlocal calls
        calls += 1
        return {}

    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("Bad citation."),
        citations=(
            ExtractedCitation(
                citation_id="bad",
                span=Span(0, 12),
                matched_text="Bad citation",
                citation=FullCaseCitation(volume="1", reporter=None, page="2"),
            ),
        ),
    )
    client = CourtListenerAccessClient(
        CourtListenerAccessConfig(base_url="https://cl-access.example.test"),
        post_json=post_json,
    )

    result = run_retrieval(extraction, client_mode="custom", client=client)

    assert result.retrievals[0].status == RetrievalStatus.INVALID
    assert calls == 0


def test_retrieve_rejects_custom_mode_without_client() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client is required"):
        run_retrieval(extraction, client_mode="custom")


def test_retrieve_rejects_client_override_for_managed_modes() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )
    client = _client({})

    with pytest.raises(ValueError, match="client must be None"):
        run_retrieval(extraction, client_mode="deployed", client=client)

    with pytest.raises(ValueError, match="client must be None"):
        run_retrieval(extraction, client_mode="sdk", client=client)


def test_retrieve_rejects_unknown_client_mode() -> None:
    extraction = _extracted_document(
        preprocessed=preprocess_plain_text_from_string("No citations."),
        citations=(),
    )

    with pytest.raises(ValueError, match="client_mode must be one of"):
        run_retrieval(extraction, client_mode="other")  # type: ignore[arg-type]
