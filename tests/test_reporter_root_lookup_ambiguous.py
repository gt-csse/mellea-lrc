"""Ambiguous exact lookups are assessed once, without model selection."""

from __future__ import annotations

import asyncio

import pytest

from evaluations.stage_products import stage_product
from mellea_lrc.api import (
    Document,
    grow_roots,
    reporter_root_lookup_ambiguous,
    reporter_root_lookup,
)
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerDocket
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactAmbiguityOutcome
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."


class FakeClient:
    def __init__(
        self, clusters: list[dict[str, object]], dockets: dict[str, CourtListenerDocket | None] | None = None
    ):
        self.response = CourtListenerCitationLookup.model_validate(
            {"citation": "550 U.S. 544", "status": 300, "clusters": clusters}
        )
        self.dockets = dockets or {}
        self.lookup_calls = 0
        self.docket_calls: list[str] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.lookup_calls += 1
        return self.response

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None:
        self.docket_calls.append(docket_id)
        return self.dockets[docket_id]


def cluster(id: int, **changes: object) -> dict[str, object]:
    return {
        "id": id,
        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
        "court_id": "scotus",
        "dateFiled": "2007-05-21",
        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
        **changes,
    }


def exact(client: FakeClient) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    return reporter_root_lookup(roots, client=client)


def test_unique_passing_candidate_is_admitted_and_all_comparisons_are_saved() -> None:
    client = FakeClient([cluster(1, caseNameFull="Jones v. Smith"), cluster(2)])
    before = exact(client)

    after = reporter_root_lookup_ambiguous(before, client=client)

    assert client.lookup_calls == 1
    assert client.docket_calls == []
    assert after.stage_runs[-1] == STAGE
    assert after.get_stage("reporter_root_lookup") == before
    root = after.roots[0]
    resolution = root.reporter_exact_ambiguity_resolution
    assert resolution is not None
    assert resolution.outcome is ReporterExactAmbiguityOutcome.UNIQUE_RULE_MATCH
    assert resolution.passing_candidate_indices == (1,)
    assert resolution.selected_candidate_index == 1
    assert [(item.candidate_index, item.result) for item in root.case_name_judgments] == [
        (0, MatchResult.MISMATCH),
        (1, MatchResult.MATCH),
    ]
    assert [item.candidate_index for item in root.court_judgments] == [0, 1]
    assert [item.candidate_index for item in root.date_judgments] == [0, 1]
    assert root.identity_judgments[-1].verdict is IdentityVerdict.CORRECT_IDENTITY
    assert root.identity_judgments[-1].next_stage is None
    records = stage_product(after, STAGE).records
    assert {item.name for item in records} == {
        "case_name_judgments",
        "court_judgments",
        "date_judgments",
        "reporter_exact_ambiguity_resolution",
        "identity_judgments",
    }
    restored = Document.model_validate_json(after.model_dump_json())
    assert restored == after
    assert restored.get_stage("reporter_root_lookup") == before
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_lookup_ambiguous(after, client=client)


@pytest.mark.parametrize(
    ("clusters", "passing"),
    [
        ([cluster(1, caseNameFull="Jones v. Smith"), cluster(2, caseNameFull="Doe v. Roe")], ()),
        ([cluster(1), cluster(2)], (0, 1)),
    ],
)
def test_zero_or_multiple_passing_candidates_wait_for_model_review(
    clusters: list[dict[str, object]], passing: tuple[int, ...]
) -> None:
    client = FakeClient(clusters)
    after = reporter_root_lookup_ambiguous(exact(client), client=client)
    root = after.roots[0]
    resolution = root.reporter_exact_ambiguity_resolution
    assert resolution is not None
    assert resolution.outcome is ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
    assert resolution.passing_candidate_indices == passing
    assert resolution.selected_candidate_index is None
    assert root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    assert root.identity_judgments[-1].next_stage == "reporter_root_lookup_ambiguous_llm"


def test_large_candidate_set_is_preserved_without_review_or_truncation() -> None:
    client = FakeClient([cluster(index) for index in range(20)])
    before = exact(client)

    after = reporter_root_lookup_ambiguous(before, client=client)

    root = after.roots[0]
    resolution = root.reporter_exact_ambiguity_resolution
    assert resolution is not None
    assert resolution.outcome is ReporterExactAmbiguityOutcome.CANDIDATE_LIMIT_EXCEEDED
    assert resolution.passing_candidate_indices == ()
    assert len(root.reporter_exact_lookup.response.clusters) == 20
    assert root.case_name_judgments == root.court_judgments == root.date_judgments == ()
    assert root.identity_judgments[-1].next_stage == "reporter_root_lookup_large_candidate_review"
    assert client.docket_calls == []


def test_candidate_docket_court_is_saved_and_checked_independently() -> None:
    docket = CourtListenerDocket.model_validate({"id": 10, "court_id": "scotus"})
    client = FakeClient(
        [cluster(1, court_id=None, docketId=10), cluster(2, court_id="ca2")],
        dockets={"10": docket},
    )

    after = reporter_root_lookup_ambiguous(exact(client), client=client)

    root = after.roots[0]
    assert client.docket_calls == ["10"]
    assert root.reporter_exact_candidate_dockets[0].candidate_index == 0
    assert root.reporter_exact_candidate_dockets[0].response == docket
    assert [(item.candidate_index, item.result) for item in root.court_judgments] == [
        (0, MatchResult.MATCH),
        (1, MatchResult.MISMATCH),
    ]
    assert root.reporter_exact_ambiguity_resolution.selected_candidate_index == 0
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_nonambiguous_root_does_not_gain_a_node() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    client = FakeClient([cluster(1)])
    before = reporter_root_lookup(roots, client=client)
    after = reporter_root_lookup_ambiguous(before, client=client)

    assert after.stage_runs[-1] == STAGE
    assert after.roots == before.roots
    assert after.get_stage("reporter_root_lookup") == before


def test_only_exactly_routed_roots_are_processed() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    client = FakeClient([cluster(1), cluster(2, caseNameFull="Jones v. Smith")])
    before = reporter_root_lookup(roots, client=client)
    root = before.roots[0]
    almost_stage = root.record("route_setup").with_identity_judgment(
        IdentityVerdict.DEFERRED, f"{STAGE}_review"
    )
    routed = before.replace_citation(almost_stage).complete("route_setup")

    after = reporter_root_lookup_ambiguous(routed, client=client)

    assert after.roots[0].nodes == routed.roots[0].nodes
    assert after.roots[0].reporter_exact_ambiguity_resolution is None
    assert after.roots[0].identity_judgments[-1].next_stage == f"{STAGE}_review"
