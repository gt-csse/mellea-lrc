"""Parallel short reporters share parsing boundaries, never root identity."""

import asyncio

import pytest

from mellea_lrc.api import (
    Document,
    attribute_id_citations,
    attribute_short_reporter_citations,
    find_id_citations,
    find_short_reporter_citations,
    grow_roots,
    resolve_short_reporter_case_names,
    resolve_short_reporter_colocations,
)
from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction.short_reporter_case_names import STAGE as NAME_STAGE
from mellea_lrc.extraction.short_reporter_colocations import STAGE as COLOCATION_STAGE
from mellea_lrc.extraction.short_reporter_locator import STAGE as CREATION_STAGE
from mellea_lrc.model.citations import latest

ROOTS = "Smith v. Jones, 347 U.S. 483, 74 S. Ct. 686 (1954). "
PARALLEL = "See Smith, 347 U.S. at 495, 74 S. Ct. at 690."


def created(source: str) -> Document:
    return find_short_reporter_citations(asyncio.run(grow_roots(Document.from_source(source))))


def names(source: str) -> Document:
    return resolve_short_reporter_case_names(resolve_short_reporter_colocations(created(source)))


def test_short_creation_reads_locator_and_pins_before_grouping_or_name_reading() -> None:
    document = created(ROOTS + PARALLEL)

    assert len(document.short_reporters) == 2
    assert [citation.short_locator[-1].quote for citation in document.short_reporters] == [
        "347 U.S. at 495",
        "74 S. Ct. at 690",
    ]
    assert [citation.pin_cite[-1].quote for citation in document.short_reporters] == ["495", "690"]
    for citation in document.short_reporters:
        assert citation.case_name == citation.colocation_id == citation.root_id == ()
        assert len(citation.nodes) == 1
        assert citation.nodes[0].stage == CREATION_STAGE
        assert citation.short_locator[-1].node_id == citation.pin_cite[-1].node_id == citation.nodes[0].id


@pytest.mark.parametrize("join", [", ", ",\n   ", ",\t\n\t"])
def test_parallel_short_groups_allow_literal_source_whitespace_joins(join: str) -> None:
    before = created(ROOTS + "See Smith, 347 U.S. at 495" + join + "74 S. Ct. at 690.")
    grouped = resolve_short_reporter_colocations(before)
    group = grouped.short_reporter_colocations[0]

    assert group.citation_ids == tuple(citation.id for citation in grouped.short_reporters)
    assert len(grouped.short_reporter_colocations) == 1
    assert grouped.full_locators == before.full_locators
    assert grouped.roots == before.roots
    assert grouped.colocations == before.colocations
    for previous, current in zip(before.short_reporters, grouped.short_reporters, strict=True):
        assert current.short_locator == previous.short_locator
        assert current.pin_cite == previous.pin_cite
        assert current.case_name == current.root_id == ()
        assert latest(current.colocation_id) == group.id
        assert current.nodes[-1].stage == COLOCATION_STAGE
        assert current.colocation_id[-1].node_id == current.nodes[-1].id


def test_parallel_short_names_are_grounded_before_the_first_group_member() -> None:
    source = ROOTS + PARALLEL
    document = names(source)
    first, second = document.short_reporters

    assert first.case_name[-1].quote == second.case_name[-1].quote == "Smith"
    assert first.case_name[-1].span == second.case_name[-1].span
    assert first.case_name[-1].span.end < first.site_span.start < second.site_span.start
    for citation in document.short_reporters:
        reading = citation.case_name[-1]
        assert source[reading.span.start : reading.span.end] == reading.quote
        assert reading.node_id == citation.nodes[-1].id
        assert citation.nodes[-1].stage == NAME_STAGE
        assert len(citation.case_name) == 1
        assert citation.root_id == citation.attributions == citation.reviews == ()


def test_shared_name_can_be_read_without_a_previously_known_case_alias() -> None:
    document = names("See Unknown, 347 U.S. at 495, 74 S. Ct. at 690.")

    assert len(document.short_reporter_colocations) == 1
    assert [citation.case_name[-1].quote for citation in document.short_reporters] == [
        "Unknown",
        "Unknown",
    ]


def test_duplicate_reporter_editions_split_short_groups_and_bound_name_windows() -> None:
    document = names(ROOTS + "See Smith, 347 U.S. at 495, 348 U.S. at 500.")

    assert document.short_reporter_colocations == ()
    first, second = document.short_reporters
    assert first.case_name[-1].quote == "Smith"
    assert second.case_name == ()
    assert first.short_locator[-1].quote == "347 U.S. at 495"
    assert second.short_locator[-1].quote == "348 U.S. at 500"


@pytest.mark.parametrize(
    "barrier",
    [
        "; Id.; ",
        "; 42 U.S.C. § 1983; ",
        "; Brown v. Green, 348 U.S. 500 (1955); ",
        ". Another authority: ",
        "\n\n",
    ],
)
def test_other_authorities_and_separate_citation_text_block_short_groups(barrier: str) -> None:
    document = names(ROOTS + "See Smith, 347 U.S. at 495" + barrier + "74 S. Ct. at 690.")

    assert len(document.short_reporters) == 2
    assert document.short_reporter_colocations == ()
    assert document.short_reporters[0].case_name[-1].quote == "Smith"
    assert all(
        not citation.case_name or citation.case_name[-1].quote != "Smith"
        for citation in document.short_reporters[1:]
    )


def test_short_grouping_uses_the_same_configurable_meaningful_gap_as_full_grouping() -> None:
    before = created(ROOTS + "See Smith, 347 U.S. at 495 and 74 S. Ct. at 690.")
    strict = resolve_short_reporter_colocations(
        before, rules=ExtractionRules(colocation_max_meaningful_gap=2)
    )
    lenient = resolve_short_reporter_colocations(
        before, rules=ExtractionRules(colocation_max_meaningful_gap=3)
    )

    assert strict.short_reporter_colocations == ()
    assert len(lenient.short_reporter_colocations) == 1
    assert strict.full_locators == lenient.full_locators == before.full_locators


def test_three_parallel_reporters_share_one_grounded_name_without_merging_roots() -> None:
    source = (
        "Smith v. Jones, 347 U.S. 483, 74 S. Ct. 686, 98 L. Ed. 873 (1954). "
        "See Smith, 347 U.S. at 495, 74 S. Ct. at 690, 98 L. Ed. at 875."
    )
    document = names(source)
    attributed = asyncio.run(attribute_short_reporter_citations(document, review=False))

    assert len(document.short_reporters) == 3
    assert len(document.short_reporter_colocations) == 1
    assert [citation.case_name[-1].quote for citation in document.short_reporters] == ["Smith"] * 3
    assert len(attributed.roots) == 3
    assert len({latest(citation.root_id) for citation in attributed.short_reporters}) == 3


def test_each_separate_parallel_group_gets_its_own_preceding_name() -> None:
    source = (
        ROOTS + "See Smith, 347 U.S. at 495, 74 S. Ct. at 690; " + "Brown, 348 U.S. at 500, 75 S. Ct. at 691."
    )
    document = names(source)

    assert len(document.short_reporter_colocations) == 2
    assert [citation.case_name[-1].quote for citation in document.short_reporters] == [
        "Smith",
        "Smith",
        "Brown",
        "Brown",
    ]
    first, second = document.short_reporter_colocations
    assert set(first.citation_ids).isdisjoint(second.citation_ids)


@pytest.mark.parametrize("join", ["; ", ", ", " "])
def test_independently_written_name_starts_a_new_occurrence_even_inside_the_gap(join: str) -> None:
    document = names(ROOTS + "See Smith, 347 U.S. at 495" + join + "Doe, 74 S. Ct. at 690.")

    assert document.short_reporter_colocations == ()
    assert [citation.case_name[-1].quote for citation in document.short_reporters] == ["Smith", "Doe"]


def test_intervening_name_separates_two_parallel_groups_within_the_gap() -> None:
    document = names(
        ROOTS + "See Smith, 347 U.S. at 495, 74 S. Ct. at 690; Doe, 348 U.S. at 500, 75 S. Ct. at 691."
    )

    assert len(document.short_reporter_colocations) == 2
    assert [citation.case_name[-1].quote for citation in document.short_reporters] == [
        "Smith",
        "Smith",
        "Doe",
        "Doe",
    ]


def test_short_name_stage_and_group_stage_are_independently_recoverable_after_attribution() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(ROOTS + PARALLEL)))
    before = find_short_reporter_citations(roots)
    grouped = resolve_short_reporter_colocations(before)
    named = resolve_short_reporter_case_names(grouped)
    attributed = asyncio.run(attribute_short_reporter_citations(named, review=False))
    restored = Document.model_validate_json(attributed.model_dump_json())

    assert restored == attributed
    assert restored.get_stage("10_roots") == roots.get_stage("10_roots")
    assert restored.get_stage(CREATION_STAGE) == before
    assert restored.get_stage(COLOCATION_STAGE) == grouped
    assert restored.get_stage(NAME_STAGE) == named
    assert restored.full_locators == roots.full_locators
    assert len(restored.roots) == 2
    assert len({latest(citation.root_id) for citation in restored.short_reporters}) == 2
    for initial, group, name, final in zip(
        before.short_reporters,
        grouped.short_reporters,
        named.short_reporters,
        restored.short_reporters,
        strict=True,
    ):
        assert final.short_locator == name.short_locator == group.short_locator == initial.short_locator
        assert final.pin_cite == name.pin_cite == group.pin_cite == initial.pin_cite
        assert final.case_name == name.case_name
        assert name.nodes[: len(group.nodes)] == group.nodes
        assert group.nodes[: len(initial.nodes)] == initial.nodes


def test_id_default_still_uses_the_last_preceding_parallel_short_root() -> None:
    document = names(ROOTS + PARALLEL + " Id. at 691.")
    attributed = asyncio.run(attribute_short_reporter_citations(document, review=False))
    with_id = find_id_citations(attributed)
    finished = asyncio.run(attribute_id_citations(with_id, review=False))
    id_citation = next(citation for citation in finished.short_citations if citation.kind.value == "id")

    assert latest(id_citation.root_id) == latest(finished.short_reporters[-1].root_id)
    assert id_citation.pin_cite[-1].quote == "691"
    assert finished.short_reporters == attributed.short_reporters
    assert finished.full_locators == document.full_locators


def test_short_parsing_stages_raise_for_wrong_order_and_duplicate_runs() -> None:
    roots = asyncio.run(grow_roots(Document.from_source(ROOTS + PARALLEL)))
    before = find_short_reporter_citations(roots)
    with pytest.raises(ValueError):
        resolve_short_reporter_colocations(roots)
    with pytest.raises(ValueError):
        resolve_short_reporter_case_names(before)
    with pytest.raises(ValueError):
        asyncio.run(attribute_short_reporter_citations(before, review=False))
    grouped = resolve_short_reporter_colocations(before)
    with pytest.raises(ValueError, match="already completed"):
        resolve_short_reporter_colocations(grouped)
    named = resolve_short_reporter_case_names(grouped)
    with pytest.raises(ValueError, match="already completed"):
        resolve_short_reporter_case_names(named)


def test_short_groups_must_be_valid_in_every_completed_checkpoint() -> None:
    before = created(ROOTS + PARALLEL)
    first, second = before.short_reporters
    invalid = before.replace_citation(first.record(COLOCATION_STAGE).with_colocation("single"))
    with pytest.raises(ValueError, match="A colocation group needs at least two citations"):
        invalid.complete(COLOCATION_STAGE)
    grouped = resolve_short_reporter_colocations(before)
    data = grouped.model_dump(mode="python")
    next(citation for citation in data["citations"] if citation["id"] == second.id)["colocation_id"] = ()
    with pytest.raises(ValueError, match="A colocation group needs at least two citations"):
        Document.model_validate(data)
