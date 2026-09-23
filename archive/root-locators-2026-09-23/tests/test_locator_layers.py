"""The locator spans and co-location groups are separate extraction outputs."""

from __future__ import annotations

import contextlib
import io
from dataclasses import replace

import pytest

from mellea_lrc.extraction.eyecite_extractor import extract_from_plain_text, grow_roots
from mellea_lrc.extraction.rules import stable
from mellea_lrc.extraction.structure.locator_layers import find_locators
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import Relaxation
from mellea_lrc.model.locators import LocatorLayers
from mellea_lrc.model.operations import record_colocation
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.preprocessing import preprocess


def _extract(text: str):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_from_plain_text(text, relaxation=Relaxation.FULL)


def test_a_single_locator_has_no_singleton_group() -> None:
    text = "Ashcroft v. Iqbal, 556 U.S. 662 (2009)."
    document = _extract(text)

    assert len(document.locators) == 1
    assert document.colocations == ()


def test_parallel_citations_keep_locator_spans_and_id_groups_separate() -> None:
    text = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."
    document = _extract(text)

    assert len(document.locators) == 3
    assert len(document.colocations) == 1
    assert len(document.colocations[0]) == 3
    assert len(set(document.colocations[0])) == 3


def test_find_locators_returns_the_two_layers_and_rules_none_is_eyecite() -> None:
    prepared = preprocess("See United States v. Smith, No. 1:24-cr-00123 (D. Ariz.).")
    native = grow_roots(prepared)
    stable_layers = find_locators(prepared, rules=stable())

    assert isinstance(stable_layers, LocatorLayers)
    assert native.locators == ()
    assert len(stable_layers.locators) == 1
    assert stable_layers.locators[0].text == "No. 1:24-cr-00123"
    assert stable_layers.colocations == ()


def test_find_locators_can_replay_a_document_with_a_rule_bundle() -> None:
    document = _extract("Ashcroft v. Iqbal, 556 U.S. 662 (2009).")

    layers = find_locators(document, rules=stable())

    assert len(layers.locators) == 1
    assert layers.colocations == ()


@pytest.mark.parametrize(
    ("citation", "locator"),
    [
        ("Ashcroft v. Iqbal, 556 U.S. 662 (2009).", "556 U.S. 662"),
        ("Smith v. Jones, No. 1:24-cv-123 (D. Ariz. 2024).", "No. 1:24-cv-123"),
    ],
)
def test_repeated_locators_remain_distinct_when_they_share_a_root(citation: str, locator: str) -> None:
    document = grow_roots(preprocess(f"{citation} Again, {citation}"), rules=stable())

    assert len({record.root_id for record in document.citations}) == 1
    assert [item.text for item in document.locators] == [locator, locator]
    assert document.locators[0].span != document.locators[1].span

    payload = document.model_dump(mode="json")
    assert len(payload["citations"]) == 2
    assert Document.model_validate(payload).locators == document.locators


def test_colocation_projection_retains_occurrences_sharing_roots() -> None:
    citation = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."
    document = grow_roots(preprocess(f"{citation} Again, {citation}"), rules=stable())

    assert len({record.root_id for record in document.citations}) == 3
    locator_ids = tuple(locator.citation_id for locator in document.locators)
    expected_groups = (locator_ids[:3], locator_ids[3:])
    # Supply grouping explicitly: this tests the projection, independently of
    # eyecite's full spans and the grouping reader that consumes those spans.
    assigned = {member: group[0] for group in expected_groups for member in group}
    document = document.evolve(
        citations=tuple(
            record_colocation(
                record,
                assigned[record.citation_id],
                Node(
                    f"test:group:{record.citation_id}",
                    Reads.RECORD,
                    "locator_projection_test",
                    __name__,
                    "grouped",
                ),
            )
            for record in document.citations
        ),
    )

    assert document.colocations == expected_groups
    assert [len(group) for group in document.colocations] == [3, 3]
    assert {member for group in document.colocations for member in group} == {
        locator.citation_id for locator in document.locators
    }
