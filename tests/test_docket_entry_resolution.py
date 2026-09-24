"""Docket entry references attach only to already admitted docket locators."""

import asyncio

import pytest

from mellea_lrc.extraction import (
    find_docket_locators,
    find_full_reporter_locators,
    form_roots,
    hunt_docket_locators,
    resolve_colocations,
    resolve_docket_entries,
)
from mellea_lrc.extraction.docket_hunting import DocketSiteCandidate, DocketSiteDecision
from mellea_lrc.model import DocketEntryField, Document, FullDocketCitation

STAGE = "docket_entries"


def _rule_ready(source: str) -> Document:
    return find_docket_locators(find_full_reporter_locators(Document.from_source(source)))


def _only_docket(document: Document) -> FullDocketCitation:
    dockets = tuple(item for item in document.citations if isinstance(item, FullDocketCitation))
    assert len(dockets) == 1
    return dockets[0]


def _assert_entry(document: Document, citation: FullDocketCitation, quote: str, number: str) -> None:
    assert len(citation.docket_entry) == 1
    entry = citation.docket_entry[0]
    assert isinstance(entry, DocketEntryField)
    assert entry.quote == quote
    assert document.text[entry.span.start : entry.span.end] == quote
    assert entry.get_normalized() == number
    assert entry.node_id == citation.nodes[-1].id
    assert citation.nodes[-1].stage == STAGE


@pytest.mark.parametrize(
    ("source", "entry_quote", "entry_number"),
    [
        ("Doc. 10-1, Case No. 1:24-cv-00123.", "Doc. 10-1", "10-1"),
        ("Case No. 1:24-cv-00123, ECF No. 113.", "ECF No. 113", "113"),
        ("Case No. 1:25-bk-11282 (MG) [D.I. 21].", "D.I. 21", "21"),
    ],
)
def test_rule_docket_gets_entry_on_either_side(source: str, entry_quote: str, entry_number: str) -> None:
    before = _rule_ready(source)
    original = _only_docket(before)
    assert original.docket_entry == ()

    resolved = resolve_docket_entries(before)
    citation = _only_docket(resolved)

    assert citation.id == original.id
    assert citation.locator == original.locator
    assert citation.nodes[:1] == original.nodes
    assert len(citation.nodes) == 2
    _assert_entry(resolved, citation, entry_quote, entry_number)
    assert resolved.get_stage("docket_locators") == before
    assert resolved.get_stage(STAGE) == resolved


@pytest.mark.parametrize(
    ("source", "entry_quote", "entry_number"),
    [
        ("Doc. 10-1, No. 19 Civ. 8034.", "Doc. 10-1", "10-1"),
        ("No. 19 Civ. 8034, ECF No. 113.", "ECF No. 113", "113"),
    ],
)
def test_hunted_docket_gets_entry_on_either_side(source: str, entry_quote: str, entry_number: str) -> None:
    async def reviewer(site: DocketSiteCandidate) -> DocketSiteDecision:
        return DocketSiteDecision(
            is_docket_citation=True,
            locator=site.locator_text,
            docket_number=site.docket_number,
            reason="This is a cited case docket.",
        )

    hunted = asyncio.run(hunt_docket_locators(_rule_ready(source), reviewer=reviewer))
    original = _only_docket(hunted)
    assert original.nodes[0].stage == "docket_locator_site_hunting"
    assert original.docket_entry == ()

    resolved = resolve_docket_entries(hunted)
    citation = _only_docket(resolved)

    assert citation.id == original.id
    assert citation.locator == original.locator
    _assert_entry(resolved, citation, entry_quote, entry_number)
    assert resolved.get_stage("docket_locator_site_hunting") == hunted
    assert resolved.site_reviews == hunted.site_reviews


@pytest.mark.parametrize(
    "source",
    [
        "Case No. 1:24-cv-00123, see ECF No. 113.",
        "Case No. 1:24-cv-00123, as discussed in ECF No. 113.",
        "Case No. 1:24-cv-00123, 347 U.S. 483, ECF No. 113.",
        "Case No. 1:24-cv-00123; ECF No. 113.",
        "Case No. 1:24-cv-00123,\nECF No. 113.",
    ],
)
def test_trailing_entry_requires_immediate_adjacency(source: str) -> None:
    before = _rule_ready(source)
    resolved = resolve_docket_entries(before)

    assert _only_docket(resolved).docket_entry == ()
    assert resolved.citations == before.citations
    assert resolved.stage_runs == (*before.stage_runs, STAGE)


def test_trailing_entry_attaches_only_to_nearest_docket() -> None:
    source = "Case No. 1:24-cv-00123, Case No. 2:25-cv-00010, ECF No. 113."
    resolved = resolve_docket_entries(_rule_ready(source))
    dockets = tuple(item for item in resolved.citations if isinstance(item, FullDocketCitation))

    assert len(dockets) == 2
    assert dockets[0].docket_entry == ()
    _assert_entry(resolved, dockets[1], "ECF No. 113", "113")


def test_entry_between_two_adjacent_dockets_is_left_for_review() -> None:
    source = "Case No. 1:24-cv-00123, ECF No. 113, Case No. 2:25-cv-00010."
    resolved = resolve_docket_entries(_rule_ready(source))
    dockets = tuple(item for item in resolved.citations if isinstance(item, FullDocketCitation))

    assert len(dockets) == 2
    assert all(not citation.docket_entry for citation in dockets)


def test_entries_on_both_sides_of_one_docket_are_left_for_review() -> None:
    source = "Doc. 10, Case No. 1:24-cv-00123, ECF No. 113."
    resolved = resolve_docket_entries(_rule_ready(source))

    assert _only_docket(resolved).docket_entry == ()


def test_shared_entry_stays_unread_even_if_one_owner_has_another_candidate() -> None:
    source = "Doc. 10, Case No. 1:24-cv-00123, ECF No. 113, Case No. 2:25-cv-00010."
    resolved = resolve_docket_entries(_rule_ready(source))

    assert all(not citation.docket_entry for citation in resolved.citations)


def test_preceding_entry_does_not_cross_explanatory_text() -> None:
    resolved = resolve_docket_entries(_rule_ready("Doc. 10-1 discusses Case No. 1:24-cv-00123."))

    assert _only_docket(resolved).docket_entry == ()


def test_standalone_entry_never_creates_a_root() -> None:
    before = _rule_ready("See ECF No. 113 and Doc. 10-1.")
    assert before.citations == ()

    resolved = resolve_docket_entries(before)
    formed = form_roots(resolve_colocations(resolved))

    assert resolved.citations == ()
    assert resolved.stage_runs == (*before.stage_runs, STAGE)
    assert formed.citations == ()
    assert formed.roots == ()


def test_entry_stage_is_idempotent_and_survives_json_and_later_stages() -> None:
    before = _rule_ready("Case No. 1:24-cv-00123, ECF No. 113.")
    resolved = resolve_docket_entries(before)
    grouped = resolve_colocations(resolved)
    restored = Document.model_validate_json(grouped.model_dump_json())

    assert resolve_docket_entries(resolved) == resolved
    assert restored == grouped
    assert grouped.get_stage("docket_locators") == before
    assert grouped.get_stage(STAGE) == resolved
    assert restored.get_stage(STAGE) == resolved
