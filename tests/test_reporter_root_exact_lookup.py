"""The first reporter-validation stage keeps exact lookup separate from identity."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.api import Document, grow_roots, reporter_root_exact_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup, CourtListenerError
from mellea_lrc.model import FullReporterCitation, Span
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupOutcome


class FakeLookupClient:
    def __init__(self, response: CourtListenerCitationLookup):
        self.response = response
        self.calls: list[tuple[str, str, str]] = []

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        self.calls.append((volume, reporter, page))
        return self.response


def _document(text: str = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007).") -> Document:
    return asyncio.run(grow_roots(Document.from_source(text)))


def _response(*clusters: dict[str, object], status: int = 200) -> CourtListenerCitationLookup:
    return CourtListenerCitationLookup.model_validate(
        {"citation": "550 U.S. 544", "status": status, "clusters": list(clusters)}
    )


def test_unique_lookup_uses_both_abbreviated_parties_and_roundtrips() -> None:
    before = _document()
    client = FakeLookupClient(
        _response(
            {
                "id": 1,
                "caseName": "Bell Atlantic Corp. v. Twombly",
                "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                "court": "scotus",
                "dateFiled": "2007-05-21",
                "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
            }
        )
    )

    after = reporter_root_exact_lookup(before, client=client)

    assert client.calls == [("550", "U.S.", "544")]
    assert after.stage_runs[-1] == "reporter_root_exact_lookup"
    assert after.get_stage("roots") == before
    (root,) = after.roots
    assert isinstance(root, FullReporterCitation)
    (lookup,) = root.reporter_exact_lookup
    assert lookup.node_id == root.nodes[-1].id
    assert lookup.outcome is ReporterExactLookupOutcome.UNIQUE
    assert len(lookup.response.clusters) == 1
    (check,) = lookup.candidate_checks
    assert check.name_source == "Bell Atlantic Corporation v. Twombly"
    assert check.plaintiff_present is True
    assert check.defendant_present is True
    assert check.name_rule_passed is True
    assert check.locator_present is True
    assert check.qualifies is True
    assert not hasattr(root, "identity_judgment")
    assert Document.model_validate_json(after.model_dump_json()) == after
    with pytest.raises(ValueError, match="already completed"):
        reporter_root_exact_lookup(after, client=client)


def test_one_missing_party_stays_unqualified_without_negative_judgment() -> None:
    client = FakeLookupClient(_response({"id": 1, "caseNameFull": "Bell Atlantic Corporation v. Jones"}))
    after = reporter_root_exact_lookup(_document(), client=client)
    check = after.roots[0].reporter_exact_lookup[-1].candidate_checks[0]

    assert check.plaintiff_present is True
    assert check.defendant_present is False
    assert check.name_rule_passed is False
    assert check.qualifies is False
    assert check.locator_present is None


def test_multiple_results_are_all_preserved_for_later_ambiguity_review() -> None:
    response = _response(
        {"id": 1, "caseNameFull": "Bell Atlantic Corporation v. Twombly"},
        {"id": 2, "caseNameFull": "Bell Atlantic Corporation v. Jones"},
        status=300,
    )
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    lookup = after.roots[0].reporter_exact_lookup[-1]

    assert lookup.outcome is ReporterExactLookupOutcome.AMBIGUOUS
    assert [item.id for item in lookup.response.clusters] == ["1", "2"]
    assert [check.qualifies for check in lookup.candidate_checks] == [True, False]
    assert [check.candidate_index for check in lookup.candidate_checks] == [0, 1]


def test_no_candidate_is_distinct_from_provider_failure() -> None:
    before = _document()
    empty = reporter_root_exact_lookup(before, client=FakeLookupClient(_response(status=404)))
    assert empty.roots[0].reporter_exact_lookup[-1].outcome is ReporterExactLookupOutcome.NOT_FOUND
    assert empty.roots[0].reporter_exact_lookup[-1].candidate_checks == ()

    with pytest.raises(CourtListenerError, match="item status 429"):
        reporter_root_exact_lookup(before, client=FakeLookupClient(_response(status=429)))
    assert before.roots[0].reporter_exact_lookup == ()


def test_nonmatching_listed_locator_does_not_qualify_on_name_alone() -> None:
    response = _response(
        {
            "caseNameFull": "Bell Atlantic Corporation v. Twombly",
            "citations": [{"volume": 550, "reporter": "U.S.", "page": "545"}],
        }
    )
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    check = after.roots[0].reporter_exact_lookup[-1].candidate_checks[0]

    assert check.plaintiff_present and check.defendant_present
    assert check.locator_present is False
    assert check.qualifies is False


def test_short_database_name_does_not_stand_in_for_missing_full_name() -> None:
    response = _response({"caseName": "Bell Atlantic Corporation v. Twombly"})
    after = reporter_root_exact_lookup(_document(), client=FakeLookupClient(response))
    check = after.roots[0].reporter_exact_lookup[-1].candidate_checks[0]

    assert check.name_source is None
    assert check.name_rule_passed is None
    assert check.qualifies is False


def test_repeated_locator_occurrences_make_one_query_for_one_root() -> None:
    before = _document("Bell Atl. Corp. v. Twombly, 550 U.S. 544. Bell Atl. Corp. v. Twombly, 550 U.S. 544.")
    client = FakeLookupClient(_response({"caseNameFull": "Bell Atlantic Corporation v. Twombly"}))
    after = reporter_root_exact_lookup(before, client=client)

    assert len(client.calls) == 1
    assert len(after.citations) == 2
    assert len(after.roots) == 1
    assert sum(bool(citation.reporter_exact_lookup) for citation in after.citations) == 1


def test_unnormalizable_root_is_recorded_without_a_request() -> None:
    source = "Unparsed reporter"
    empty = Document.from_source(source)
    citation = FullReporterCitation.from_locator(
        citation_id="reporter:0:17", stage="full_reporter_locators", source=source, span=Span(0, len(source))
    )
    before = empty.add_citation(citation).complete("full_reporter_locators")
    before = before.replace_citation(citation.record("roots").with_root(citation.id)).complete("roots")
    client = FakeLookupClient(_response())

    after = reporter_root_exact_lookup(before, client=client)

    assert client.calls == []
    assert after.roots[0].reporter_exact_lookup[-1].outcome is ReporterExactLookupOutcome.UNNORMALIZABLE
