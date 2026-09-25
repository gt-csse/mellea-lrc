"""Exact reporter lookup retains one response and citation-local judgments."""

from __future__ import annotations

import asyncio

import pytest

from evaluations.stage_products import stage_product
from mellea_lrc.api import Document, grow_roots, reporter_root_exact_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerDocket, CourtListenerError
from mellea_lrc.model import FullReporterCitation, Span
from mellea_lrc.model.citations.judgments import IdentityNextStep, IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome

STAGE = "reporter_root_exact_lookup"


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


def test_unique_lookup_records_matching_fields_and_identity_and_roundtrips() -> None:
    before = _document()
    client = FakeLookupClient(_response(_matching_cluster()))

    after = reporter_root_exact_lookup(before, client=client)

    assert client.calls == [("550", "U.S.", "544")]
    assert after.stage_runs[-1] == STAGE
    assert after.get_stage("roots") == before
    assert after.get_stage(STAGE) == after
    (root,) = after.roots
    assert isinstance(root, FullReporterCitation)
    lookup = root.reporter_exact_lookup
    assert lookup is not None
    assert lookup.node_id == root.nodes[-1].id
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
        assert judgment.node_id == lookup.node_id
        assert judgment.reading_index == len(readings) - 1
        assert judgment.candidate_index == 0
        assert judgment.result is MatchResult.MATCH
    (identity,) = root.identity_judgments
    assert identity.node_id == lookup.node_id
    assert identity.verdict is IdentityVerdict.CORRECT_IDENTITY
    assert identity.next_step is None
    assert {item.name for item in stage_product(after, STAGE).records} == {
        "reporter_exact_lookup",
        "case_name_judgments",
        "court_judgments",
        "date_judgments",
        "identity_judgments",
    }

    loaded = Document.model_validate_json(after.model_dump_json())
    assert loaded == after
    assert loaded.get_stage("roots") == before
    assert loaded.get_stage(STAGE) == after
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_exact_lookup(after, client=client)
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
    after = reporter_root_exact_lookup(
        _document(), client=FakeLookupClient(_response(_matching_cluster(**changed_cluster)))
    )
    root = after.roots[0]

    assert getattr(root, field_log)[0].result is MatchResult.MISMATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_missing_full_name_is_undetermined_and_routes_to_review() -> None:
    response = _response(_matching_cluster(caseNameFull=None))
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    root = after.roots[0]

    assert root.case_name_judgments[0].result is MatchResult.UNDETERMINED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.response == response


def test_missing_provider_court_defers_inferred_court_as_undetermined() -> None:
    response = _response(_matching_cluster(court_id=None))
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    root = after.roots[0]

    assert root.court[-1].span is None
    assert root.court_judgments[-1].result is MatchResult.UNDETERMINED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_unique_lookup_fetches_linked_docket_and_judges_court_before_identity() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    docket = CourtListenerDocket.model_validate(
        {"id": 10, "court_id": "scotus", "court": "https://www.courtlistener.com/api/rest/v4/courts/scotus/"}
    )
    client = FakeLookupClient(response, dockets={"10": docket})

    after = reporter_root_exact_lookup(_document(), client=client)
    root = after.roots[0]

    assert client.docket_calls == ["10"]
    assert root.reporter_exact_docket is not None
    assert root.reporter_exact_docket.docket_id == "10"
    assert root.reporter_exact_docket.response == docket
    assert root.court_judgments[-1].result is MatchResult.MATCH
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert "reporter_exact_docket" in {item.name for item in stage_product(after, STAGE).records}
    assert Document.model_validate_json(after.model_dump_json()) == after
    altered = after.model_dump(mode="json")
    altered["citations"][0]["reporter_exact_docket"]["docket_id"] = "11"
    with pytest.raises(ValueError, match="docket"):
        Document.model_validate(altered)


def test_linked_docket_court_mismatch_defers_identity() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    docket = CourtListenerDocket.model_validate({"id": 10, "court_id": "ca2"})
    client = FakeLookupClient(response, dockets={"10": docket})

    after = reporter_root_exact_lookup(_document(), client=client)
    root = after.roots[0]

    assert root.case_name_judgments[-1].result is MatchResult.MATCH
    assert root.date_judgments[-1].result is MatchResult.MATCH
    assert root.court_judgments[-1].result is MatchResult.MISMATCH
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_missing_linked_docket_cannot_silently_admit_inferred_court() -> None:
    response = _response(_matching_cluster(court_id=None, docketId=10))
    client = FakeLookupClient(response, dockets={"10": None})

    after = reporter_root_exact_lookup(_document(), client=client)
    root = after.roots[0]

    assert root.reporter_exact_docket is not None
    assert root.reporter_exact_docket.response is None
    assert root.court_judgments[-1].result is MatchResult.UNDETERMINED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_missing_provider_court_routes_explicit_court_to_review() -> None:
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (S.D.N.Y. 2007)."
    response = _response(_matching_cluster(court_id=None))
    after = reporter_root_exact_lookup(_document(source), client=FakeLookupClient(response))
    root = after.roots[0]

    assert root.court[-1].quote == "S.D.N.Y."
    assert root.court_judgments[-1].result is MatchResult.UNDETERMINED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_nonmatching_listed_locator_routes_to_review_even_when_fields_match() -> None:
    response = _response(_matching_cluster(citations=[{"volume": 550, "reporter": "U.S.", "page": "545"}]))
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    root = after.roots[0]

    assert root.case_name_judgments[0].result is MatchResult.MATCH
    assert root.court_judgments[0].result is MatchResult.MATCH
    assert root.date_judgments[0].result is MatchResult.MATCH
    assert root.identity_judgments[-1].next_step is IdentityNextStep.REVIEW


def test_multiple_results_preserve_all_clusters_and_route_to_ambiguity() -> None:
    response = _response(
        _matching_cluster(id=1, extra_provider_detail={"source": "first"}),
        _matching_cluster(id=2, caseNameFull="Bell Atlantic Corporation v. Jones"),
        status=300,
    )
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    root = after.roots[0]
    lookup = root.reporter_exact_lookup

    assert lookup is not None
    assert lookup.outcome is ReporterExactLookupOutcome.AMBIGUOUS
    assert lookup.response == response
    assert lookup.response is not None
    assert [cluster.id for cluster in lookup.response.clusters] == ["1", "2"]
    assert lookup.response.clusters[0].raw_json["extra_provider_detail"] == {"source": "first"}
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_step is IdentityNextStep.AMBIGUITY
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_no_candidate_routes_to_search_but_provider_failure_does_not_complete() -> None:
    before = _document()
    empty = reporter_root_exact_lookup(before, client=FakeLookupClient(_response(status=404)))
    root = empty.roots[0]
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.outcome is ReporterExactLookupOutcome.NOT_FOUND
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert root.identity_judgments[-1].next_step is IdentityNextStep.SEARCH
    assert empty.get_stage("roots") == before

    with pytest.raises(CourtListenerError, match="item status 429"):
        reporter_root_exact_lookup(before, client=FakeLookupClient(_response(status=429)))
    assert before.roots[0].reporter_exact_lookup is None
    assert before.stage_runs[-1] == "roots"


def test_repeated_locator_occurrences_make_one_query_for_one_root() -> None:
    before = _document("Bell Atl. Corp. v. Twombly, 550 U.S. 544. Bell Atl. Corp. v. Twombly, 550 U.S. 544.")
    client = FakeLookupClient(_response(_matching_cluster()))
    after = reporter_root_exact_lookup(before, client=client)

    assert len(client.calls) == 1
    assert len(after.citations) == 2
    assert len(after.roots) == 1
    assert sum(citation.reporter_exact_lookup is not None for citation in after.citations) == 1


def test_unnormalizable_root_records_search_without_request() -> None:
    source = "Unparsed reporter"
    empty = Document.from_source(source)
    citation = FullReporterCitation.from_locator(
        citation_id="reporter:0:17", stage="full_reporter_locators", source=source, span=Span(0, len(source))
    )
    before = empty.add_citation(citation).complete("full_reporter_locators")
    before = before.replace_citation(citation.record("roots").with_root(citation.id)).complete("roots")
    client = FakeLookupClient(_response())

    after = reporter_root_exact_lookup(before, client=client)
    root = after.roots[0]

    assert client.calls == []
    assert root.reporter_exact_lookup is not None
    assert root.reporter_exact_lookup.outcome is ReporterExactLookupOutcome.UNNORMALIZABLE
    assert root.reporter_exact_lookup.query is None
    assert root.reporter_exact_lookup.response is None
    assert root.identity_judgments[-1].next_step is IdentityNextStep.SEARCH
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
    before = before.replace_citation(second_readings).complete("manual_review")
    after = reporter_root_exact_lookup(before, client=FakeLookupClient(_response(_matching_cluster())))
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

    later = looked_up.record("later_review").with_identity_judgment(
        IdentityVerdict.DEFERRED, IdentityNextStep.REVIEW
    )
    final = after.replace_citation(later).complete("later_review")
    restored = Document.model_validate_json(final.model_dump_json())
    assert restored == final
    assert restored.get_stage("roots") == before.get_stage("roots")
    assert restored.get_stage(STAGE) == after
    assert restored.roots[0].case_name_judgments[0].reading_index == 1


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
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(_response(_matching_cluster())))
    altered = after.model_dump(mode="json")
    altered["citations"][0][field_log][0][index_name] = bad_value

    with pytest.raises(ValueError, match="Judgment"):
        Document.model_validate(altered)
