"""What the tree evaluator takes out by default, and what it keeps.

Reading a bare name is a question about case names, which is answered after the
citations are read. Deferring it says what the rest of the pass looks like
without it -- but a reference that states a page is not deferred, because the
page is a claim about the opinion and every other citation's page is scored.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluations.extraction import tree
from evaluations.extraction.tree import _is_bare_name
from mellea_lrc.model.extraction_metadata import Relaxation


@pytest.mark.parametrize(
    ("kind", "pin_cite"),
    [
        ("ReferenceCitation", None),
        ("ReferenceCitation", {}),
    ],
)
def test_a_name_with_no_page_is_deferred(kind: str, pin_cite: object) -> None:
    assert _is_bare_name(kind, pin_cite)


def test_a_reference_that_states_a_page_is_not() -> None:
    """`Bell at 546` is a reference and a page claim."""
    assert not _is_bare_name("ReferenceCitation", {"start": 1, "end": 4, "quote": "546"})


@pytest.mark.parametrize(
    "kind",
    ["FullCaseCitation", "ShortCaseCitation", "IdCitation", "SupraCitation", "DocketCitation"],
)
def test_no_other_kind_is_deferred(kind: str) -> None:
    """An `Id.` states no page and no name, and is still a citation to read."""
    assert not _is_bare_name(kind, None)


def test_model_arm_reviews_leaf_sites_after_leaf_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def leaf_names(document, *, session):
        assert "leaf_growth" in document.passes
        calls.append("leaf_names")
        return document

    async def pin_sites(document, *, session):
        assert "leaf_growth" in document.passes
        calls.append("pin_sites")
        return document

    monkeypatch.setattr(tree, "hunt_leaf_case_names", leaf_names)
    monkeypatch.setattr(tree, "validate_pin_cite_sites", pin_sites)
    arm = tree.Arm(Relaxation.FULL, hunt_leaf_names=True, validate_pin_sites=True)

    asyncio.run(tree.run_document("See Ashcroft v. Iqbal, 556 U.S. 662 (2009).", arm, session=object()))

    assert calls == ["leaf_names", "pin_sites"]
