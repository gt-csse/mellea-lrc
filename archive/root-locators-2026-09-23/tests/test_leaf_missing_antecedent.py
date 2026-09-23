"""Leaf growth when eyecite points to a citation absent from the root checkpoint."""

from __future__ import annotations

import asyncio
import contextlib
import io

import pytest

from mellea_lrc.api import Document, grow_leaves, grow_roots
from mellea_lrc.extraction.eyecite_extractor import _read, _with_leaves
from mellea_lrc.extraction.rules import stable
from mellea_lrc.model.citations import CitationKind
from mellea_lrc.model.findings import FindingKind
from mellea_lrc.preprocessing import preprocess


@pytest.mark.parametrize(
    ("intervening", "leaf_text"),
    [
        ("29 U.S.C. § 794", "Id."),
        ("30 A.B.A.J. 334 (1944)", "Ibid."),
    ],
)
def test_id_after_unretained_noncase_citation_does_not_attach_to_earlier_case(
    intervening: str, leaf_text: str
) -> None:
    text = f"Brown v. Board, 347 U.S. 483 (1954). See {intervening}. {leaf_text}"
    roots = asyncio.run(grow_roots(Document.from_plain_text(text)))
    assert len(roots.citations) == 1

    grown = asyncio.run(grow_leaves(roots))

    assert [record.citation_id for record in grown.citations] == [roots.citations[0].citation_id]
    assert [finding.kind for finding in grown.findings] == [FindingKind.UNGROWN_LEAF]
    assert grown.findings[0].citation is not None
    assert grown.findings[0].citation.matched_text == leaf_text
    assert Document.model_validate(grown.model_dump(mode="json")) == grown


def test_short_form_uses_stated_root_when_parser_antecedent_is_missing() -> None:
    text = "Brown v. Board, 347 U.S. 483 (1954). Later, Brown, 347 U.S. at 495."
    roots = asyncio.run(grow_roots(Document.from_plain_text(text)))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _, leaves = _read(preprocess(text), stable())
    assert len(leaves) == 1
    leaf_id, citation, _parser_antecedent = leaves[0]
    assert citation.kind is CitationKind.SHORT_CASE

    grown = _with_leaves(roots, [(leaf_id, citation, "excluded-parser-citation")])

    leaf = next(record for record in grown.citations if record.citation_id == leaf_id)
    assert leaf.root_id == roots.citations[0].citation_id
    assert leaf.resolves_to is None
    assert Document.model_validate(grown.model_dump(mode="json")) == grown
