"""Exact reporter retrieval saves evidence for a separate rule review."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_ambiguous_rule_judgment,
    reporter_root_lookup_cluster_retrieval,
    reporter_root_lookup_docket_retrieval,
    reporter_root_lookup_unique_rule_judgment,
)
from mellea_lrc.providers.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerDocket,
    CourtListenerError,
)
from mellea_lrc.model import FullReporterCitation, Span
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome

SUBSTAGE = "validate_roots.reporter_lookup.cluster_retrieval"
DOCKET_SUBSTAGE = "validate_roots.reporter_lookup.docket_retrieval"
REVIEW_SUBSTAGE = "validate_roots.reporter_lookup.unique_rule_judgment"


class FakeLookupClient:
    def __init__(
        self,
        response: CourtListenerCitationLookup,
        *,
        dockets: dict[str, CourtListenerDocket | None] | None = None,
    ):
        self.response = response
        self.calls: list[tuple[str, str, str]] = []
        self.dockets = dockets or {}
        self.docket_calls: list[str] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.calls.append((volume, reporter, page))
        return self.response

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None:
        self.docket_calls.append(docket_id)
        return self.dockets[docket_id]


def _document(text: str = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007).") -> Document:
    return asyncio.run(grow_roots(Document.from_source(text)))


def _response(*clusters: dict[str, object], status: int = 200) -> CourtListenerCitationLookup:
    return CourtListenerCitationLookup.model_validate(
        {"citation": "550 U.S. 544", "status": status, "clusters": list(clusters)}
    )


def _matching_cluster(**changes: object) -> dict[str, object]:
    return {
        "id": 1,
        "caseName": "Bell Atlantic Corp. v. Twombly",
        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
        "court_id": "scotus",
        "dateFiled": "2007-05-21",
        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
        **changes,
    }


def _reviewed(before: Document, client: FakeLookupClient) -> Document:
    retrieved = reporter_root_lookup_cluster_retrieval(before, client=client)
    assert client.docket_calls == []
    dockets = reporter_root_lookup_docket_retrieval(retrieved, client=client)
    calls = (tuple(client.calls), tuple(client.docket_calls))
    reviewed = reporter_root_lookup_unique_rule_judgment(dockets)
    assert (tuple(client.calls), tuple(client.docket_calls)) == calls
    assert reviewed.get_substage(SUBSTAGE) == retrieved
    assert reviewed.get_substage(DOCKET_SUBSTAGE) == dockets
    return reviewed


def test_unique_lookup_records_matching_fields_and_identity_and_roundtrips() -> None:
    before = _document()
    client = FakeLookupClient(_response(_matching_cluster()))

    retrieved = reporter_root_lookup_cluster_retrieval(before, client=client)
    retrieved_root = retrieved.roots[0]
    assert retrieved.substage_runs[-1] == SUBSTAGE
    assert retrieved_root.reporter_exact_lookup is not None
    assert retrieved_root.case_name_judgments == ()
    assert retrieved_root.court_judgments == ()
    assert retrieved_root.date_judgments == ()
    assert retrieved_root.identity_judgments == ()
    assert retrieved_root.next_substage == DOCKET_SUBSTAGE
    assert client.docket_calls == []
    checkpoint = Document.model_validate_json(retrieved.model_dump_json())
    assert checkpoint == retrieved

    docket_retrieved = reporter_root_lookup_docket_retrieval(checkpoint, client=client)
    assert docket_retrieved.roots[0].next_substage == REVIEW_SUBSTAGE
    assert docket_retrieved.roots[0].case_name_judgments == ()
    after = reporter_root_lookup_unique_rule_judgment(docket_retrieved)

    assert client.calls == [("550", "U.S.", "544")]
    assert after.substage_runs[-1] == REVIEW_SUBSTAGE
    assert after.get_stage("grow_roots.root_formation") == before
    assert after.get_substage(SUBSTAGE) == retrieved
    assert after.get_substage(DOCKET_SUBSTAGE) == docket_retrieved
    assert after.get_substage(REVIEW_SUBSTAGE) == after
    (root,) = after.roots
    assert isinstance(root, FullReporterCitation)
    lookup = root.reporter_exact_lookup
    assert lookup is not None
    assert lookup.node_id == root.nodes[-3].id
    assert lookup.outcome is ReporterExactLookupOutcome.UNIQUE
    assert lookup.query is not None
    assert (lookup.query.volume, lookup.query.edition, lookup.query.page) == (550, "U.S.", "544")
    assert lookup.response == client.response
    assert lookup.response is not None
    assert lookup.response.clusters[0].case_name_full == "Bell Atlantic Corporation v. Twombly"
    for judgments, readings in (
        (root.case_name_judgments, root.case_name),
        (root.court_judgments, root.court),
        (root.date_judgments, root.date),
    ):
        (judgment,) = judgments
        assert judgment.node_id == root.nodes[-1].id
        assert judgment.reading_index == len(readings) - 1
        assert judgment.candidate_index == 0
        assert judgment.result is MatchResult.MATCH
    (identity,) = root.identity_judgments
    assert identity.node_id == root.nodes[-1].id
    assert identity.verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.next_substage is None
    assert [route.value for route in root.routes] == [DOCKET_SUBSTAGE, REVIEW_SUBSTAGE, None]
    assert all(
        record.node_id == root.nodes[-1].id
        for record in (
            *root.case_name_judgments,
            *root.court_judgments,
            *root.date_judgments,
            *root.identity_judgments,
        )
    )

    loaded = Document.model_validate_json(after.model_dump_json())
    assert loaded == after
    assert loaded.get_stage("grow_roots.root_formation") == before
    assert loaded.get_substage(SUBSTAGE) == retrieved
    assert loaded.get_substage(DOCKET_SUBSTAGE) == docket_retrieved
    assert loaded.get_substage(REVIEW_SUBSTAGE) == after
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_lookup_cluster_retrieval(after, client=client)
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_lookup_docket_retrieval(after, client=client)
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_lookup_unique_rule_judgment(after)
    with pytest.raises(ValueError, match="already recorded"):
        root.record("later_review").with_reporter_exact_lookup(lookup)


@pytest.mark.parametrize(
    ("changed_cluster", "field_log"),
    [
        ({"caseNameFull": "Bell Atlantic Corporation v. Jones"}, "case_name_judgments"),
        ({"court_id": "ca2"}, "court_judgments"),
        ({"dateFiled": "2008-05-21"}, "date_judgments"),
    ],
)
def test_unique_field_mismatch_routes_to_review(changed_cluster: dict[str, object], field_log: str) -> None:
    after = _reviewed(_document(), client=FakeLookupClient(_response(_matching_cluster(**changed_cluster))))
    root = after.roots[0]

    assert getattr(root, field_log)[0].result is MatchResult.MISMATCH
    assert root.identity_judgments == ()
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"
    assert root.routes[-1].node_id == root.nodes[-1].id


def test_missing_full_name_is_unavailable_and_routes_to_review() -> None:
    response = _response(_matching_cluster(caseNameFull=None))
    after = _reviewed(_document(), FakeLookupClient(response))
    root = after.roots[0]

    assert root.case_name_judgments[0].result is MatchResult.UNAVAILABLE
    assert root.identity_judgments == ()
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.response == response


@pytest.mark.parametrize("candidate_name", ["Bell Atlantic Corporation v. Twombly", "Other v. Party"])
def test_partial_case_name_cannot_make_a_rule_identity_judgment(candidate_name: str) -> None:
    before = _document()
    root = before.roots[0]
    fragment = "Bell Atl. Corp."
    partial = CaseName(kind=CaseNameKind.PARTIAL, partial=fragment)
    root = root.record("partial_name").with_case_name(before.text, Span(0, len(fragment)), normalized=partial)
    before = before.replace_citation(root).complete_substage("partial_name")

    after = _reviewed(
        before, client=FakeLookupClient(_response(_matching_cluster(caseNameFull=candidate_name)))
    )
    looked_up = after.roots[0]
    assert looked_up.case_name[-1].normalizable is True
    assert looked_up.case_name_judgments[-1].result is MatchResult.UNAVAILABLE
    assert looked_up.identity_judgments == ()
    assert looked_up.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_missing_provider_court_routes_inferred_court_as_unavailable() -> None:
    response = _response(_matching_cluster(court_id=None))
    after = _reviewed(_document(), FakeLookupClient(response))
    root = after.roots[0]

    assert root.court[-1].span is None
    assert root.court_judgments[-1].result is MatchResult.UNAVAILABLE
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_unique_lookup_fetches_linked_docket_and_judges_court_before_identity() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    docket = CourtListenerDocket.model_validate(
        {"id": 10, "court_id": "scotus", "court": "https://www.courtlistener.com/api/rest/v4/courts/scotus/"}
    )
    client = FakeLookupClient(response, dockets={"10": docket})

    retrieved = reporter_root_lookup_cluster_retrieval(_document(), client=client)
    assert retrieved.roots[0].reporter_exact_docket is None
    assert client.docket_calls == []
    assert retrieved.roots[0].court_judgments == ()
    dockets = reporter_root_lookup_docket_retrieval(retrieved, client=client)
    assert dockets.roots[0].reporter_exact_docket is not None
    assert dockets.roots[0].court_judgments == ()
    after = reporter_root_lookup_unique_rule_judgment(dockets)
    root = after.roots[0]

    assert client.docket_calls == ["10"]
    assert root.reporter_exact_docket is not None
    assert root.reporter_exact_docket.docket_id == "10"
    assert root.reporter_exact_docket.response == docket
    assert root.court_judgments[-1].result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.reporter_exact_docket.node_id == root.nodes[-2].id
    assert after.get_substage(DOCKET_SUBSTAGE) == dockets
    assert Document.model_validate_json(after.model_dump_json()) == after
    altered = after.model_dump(mode="json")
    altered["citations"][0]["reporter_exact_docket"]["docket_id"] = "11"
    with pytest.raises(ValueError, match="docket"):
        Document.model_validate(altered)


def test_linked_docket_court_mismatch_routes_to_review() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    docket = CourtListenerDocket.model_validate({"id": 10, "court_id": "ca2"})
    client = FakeLookupClient(response, dockets={"10": docket})

    after = _reviewed(_document(), client)
    root = after.roots[0]

    assert root.case_name_judgments[-1].result is MatchResult.MATCH
    assert root.date_judgments[-1].result is MatchResult.MATCH
    assert root.court_judgments[-1].result is MatchResult.MISMATCH
    assert root.identity_judgments == ()
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_missing_linked_docket_cannot_silently_admit_inferred_court() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    client = FakeLookupClient(response, dockets={"10": None})

    after = _reviewed(_document(), client)
    root = after.roots[0]

    assert root.reporter_exact_docket is not None
    assert root.reporter_exact_docket.response is None
    assert root.court_judgments[-1].result is MatchResult.UNAVAILABLE
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_unique_and_ambiguous_roots_share_one_linked_docket_fetch() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007). Roe v. Doe, 551 U.S. 545 (2007)."
    roots = _document(source)
    assert len(roots.roots) == 2
    docket = CourtListenerDocket.model_validate({"id": 10, "court_id": "scotus"})

    class MixedClient:
        def __init__(self) -> None:
            self.lookup_calls: list[str] = []
            self.docket_calls: list[str] = []

        def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
            self.lookup_calls.append(page)
            assert reporter == "U.S."
            if page == "544":
                assert volume == "550"
                return _response(_matching_cluster(court_id=None, docketId=10))
            assert (volume, page) == ("551", "545")
            citations = [{"volume": 551, "reporter": "U.S.", "page": "545"}]
            return CourtListenerCitationLookup.model_validate(
                {
                    "citation": "551 U.S. 545",
                    "status": 300,
                    "clusters": [
                        _matching_cluster(
                            id=2,
                            caseNameFull="Roe v. Doe",
                            court_id=None,
                            docketId=10,
                            citations=citations,
                        ),
                        _matching_cluster(
                            id=3,
                            caseNameFull="Other v. Party",
                            court_id=None,
                            docketId=10,
                            citations=citations,
                        ),
                    ],
                }
            )

        def get_docket(self, docket_id: str) -> CourtListenerDocket:
            self.docket_calls.append(docket_id)
            assert docket_id == "10"
            return docket

    client = MixedClient()
    clusters = reporter_root_lookup_cluster_retrieval(roots, client=client)
    assert client.lookup_calls == ["544", "545"]
    assert client.docket_calls == []
    assert all(root.reporter_exact_docket is None for root in clusters.roots)
    assert all(root.reporter_exact_candidate_dockets == () for root in clusters.roots)

    dockets = reporter_root_lookup_docket_retrieval(clusters, client=client)
    assert client.lookup_calls == ["544", "545"]
    assert client.docket_calls == ["10"]
    unique, ambiguous = dockets.roots
    assert unique.reporter_exact_docket is not None
    assert unique.reporter_exact_docket.response == docket
    assert unique.reporter_exact_docket.node_id == unique.nodes[-1].id
    assert [item.candidate_index for item in ambiguous.reporter_exact_candidate_dockets] == [0, 1]
    assert all(item.response == docket for item in ambiguous.reporter_exact_candidate_dockets)
    assert all(item.node_id == ambiguous.nodes[-1].id for item in ambiguous.reporter_exact_candidate_dockets)
    assert unique.next_substage == REVIEW_SUBSTAGE
    assert ambiguous.next_substage == "validate_roots.reporter_lookup.ambiguous_rule_judgment"
    assert all(not root.case_name_judgments for root in dockets.roots)
    assert Document.model_validate_json(dockets.model_dump_json()) == dockets

    judged = reporter_root_lookup_unique_rule_judgment(dockets)
    judged = reporter_root_lookup_ambiguous_rule_judgment(judged)
    assert client.lookup_calls == ["544", "545"]
    assert client.docket_calls == ["10"]


def test_missing_provider_court_routes_explicit_court_to_review() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (S.D.N.Y. 2007)."
    response = _response(_matching_cluster(court_id=None))
    after = _reviewed(_document(source), FakeLookupClient(response))
    root = after.roots[0]

    assert root.court[-1].quote == "S.D.N.Y."
    assert root.court_judgments[-1].result is MatchResult.UNAVAILABLE
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_nonmatching_listed_locator_routes_to_review_even_when_fields_match() -> None:
    response = _response(_matching_cluster(citations=[{"volume": 550, "reporter": "U.S.", "page": "545"}]))
    after = _reviewed(_document(), FakeLookupClient(response))
    root = after.roots[0]

    assert root.case_name_judgments[0].result is MatchResult.MATCH
    assert root.court_judgments[0].result is MatchResult.MATCH
    assert root.date_judgments[0].result is MatchResult.MATCH
    assert root.identity_judgments == ()
    assert root.next_substage == "validate_roots.reporter_lookup.unique_llm_judgment"


def test_multiple_results_preserve_all_clusters_and_route_to_ambiguity() -> None:
    response = _response(
        _matching_cluster(id=1, extra_provider_detail={"source": "first"}),
        _matching_cluster(id=2, caseNameFull="Bell Atlantic Corporation v. Jones"),
        status=300,
    )
    after = reporter_root_lookup_cluster_retrieval(_document(), client=FakeLookupClient(response))
    root = after.roots[0]
    lookup = root.reporter_exact_lookup

    assert lookup is not None
    assert lookup.outcome is ReporterExactLookupOutcome.AMBIGUOUS
    assert lookup.response == response
    assert lookup.response is not None
    assert [cluster.id for cluster in lookup.response.clusters] == ["1", "2"]
    assert lookup.response.clusters[0].raw_json["extra_provider_detail"] == {"source": "first"}
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert root.identity_judgments == ()
    assert root.next_substage == "validate_roots.reporter_lookup.docket_retrieval"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_no_candidate_routes_to_search_but_provider_failure_does_not_complete() -> None:
    before = _document()
    empty = reporter_root_lookup_cluster_retrieval(before, client=FakeLookupClient(_response(status=404)))
    root = empty.roots[0]
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.outcome is ReporterExactLookupOutcome.NOT_FOUND
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert root.identity_judgments == ()
    assert root.next_substage == "reporter_root_search"
    assert empty.get_stage("grow_roots.root_formation") == before

    with pytest.raises(CourtListenerError, match="item status 429"):
        reporter_root_lookup_cluster_retrieval(before, client=FakeLookupClient(_response(status=429)))
    assert before.roots[0].reporter_exact_lookup is None
    assert before.substage_runs[-1] == "grow_roots.root_formation.rule"


def test_repeated_locator_occurrences_make_one_query_for_one_root() -> None:
    before = _document("Bell Atl. Corp. v. Twombly, 550 U.S. 544. Bell Atl. Corp. v. Twombly, 550 U.S. 544.")
    client = FakeLookupClient(_response(_matching_cluster()))
    after = reporter_root_lookup_cluster_retrieval(before, client=client)

    assert len(client.calls) == 1
    assert len(after.citations) == 2
    assert len(after.roots) == 1
    assert sum(citation.reporter_exact_lookup is not None for citation in after.citations) == 1


def test_unnormalizable_root_records_search_without_request() -> None:
    source = "Unparsed reporter"
    empty = Document.from_source(source)
    citation = FullReporterCitation.from_locator(
        citation_id="reporter:0:17",
        substage="grow_roots.locator_discovery.full_reporter_locators",
        source=source,
        span=Span(0, len(source)),
    )
    before = empty.add_citation(citation).complete_substage(
        "grow_roots.locator_discovery.full_reporter_locators"
    )
    before = before.replace_citation(
        citation.record("grow_roots.root_formation.rule").with_root(citation.id)
    ).complete_substage("grow_roots.root_formation.rule")
    client = FakeLookupClient(_response())

    after = reporter_root_lookup_cluster_retrieval(before, client=client)
    root = after.roots[0]

    assert client.calls == []
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.outcome is ReporterExactLookupOutcome.UNNORMALIZABLE
    assert root.reporter_exact_lookup.query is None
    assert root.reporter_exact_lookup.response is None
    assert root.identity_judgments == ()
    assert root.next_substage == "reporter_root_search"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_judgments_use_absolute_reading_indices_and_survive_later_history() -> None:
    before = _document()
    root = before.roots[0]
    second_readings = (
        root.record("manual_review")
        .with_case_name(before.text, root.case_name[-1].span)
        .with_inferred_court("scotus")
        .with_date(before.text, root.date[-1].span)
    )
    before = before.replace_citation(second_readings).complete_substage("manual_review")
    after = _reviewed(before, FakeLookupClient(_response(_matching_cluster())))
    looked_up = after.roots[0]

    assert [
        judgment.reading_index
        for judgment in (
            looked_up.case_name_judgments[0],
            looked_up.court_judgments[0],
            looked_up.date_judgments[0],
        )
    ] == [1, 1, 1]
    assert looked_up.case_name[1].node_id == second_readings.nodes[-1].id
    assert looked_up.reporter_exact_lookup is not None
    assert looked_up.case_name_judgments[0].candidate_index == 0

    later = looked_up.record("later_review").with_route("later_review")
    final = after.replace_citation(later).complete_substage("later_review")
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    assert restored.get_substage("grow_roots.root_formation.rule") == before.get_substage(
        "grow_roots.root_formation.rule"
    )
    assert restored.get_substage(SUBSTAGE) == after.get_substage(SUBSTAGE)
    assert restored.get_substage(REVIEW_SUBSTAGE) == after
    assert restored.roots[0].case_name_judgments[0].reading_index == 1
    assert restored.roots[0].next_substage == "later_review"
    assert restored.roots[0].routes[-1].node_id == later.nodes[-1].id


@pytest.mark.parametrize(
    ("field_log", "index_name", "bad_value"),
    [
        ("case_name_judgments", "reading_index", -1),
        ("court_judgments", "reading_index", 1),
        ("date_judgments", "reading_index", 1),
        ("case_name_judgments", "candidate_index", -1),
        ("case_name_judgments", "candidate_index", 1),
    ],
)
def test_json_reload_rejects_invalid_judgment_indices(
    field_log: str, index_name: str, bad_value: int
) -> None:
    after = _reviewed(_document(), FakeLookupClient(_response(_matching_cluster())))
    altered = after.model_dump(mode="json")
    altered["citations"][0][field_log][0][index_name] = bad_value

    with pytest.raises(ValueError, match="Judgment"):
        Document.model_validate(altered)


def test_field_judgments_expose_only_three_results() -> None:
    assert {result.value for result in MatchResult} == {"match", "mismatch", "unavailable"}


@pytest.mark.parametrize("result", ["match", "mismatch"])
def test_json_reload_rejects_comparison_without_a_filing_reading(result: str) -> None:
    after = _reviewed(_document(), FakeLookupClient(_response(_matching_cluster())))
    altered = after.model_dump(mode="json")
    judgment = altered["citations"][0]["case_name_judgments"][0]
    judgment["reading_index"] = None
    judgment["result"] = result

    with pytest.raises(ValueError):
        Document.model_validate(altered)
