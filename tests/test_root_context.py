"""The identity-repair LLM sees one target citation at a time."""

from __future__ import annotations

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.root_context import masked_root_context


def test_masked_root_context_keeps_only_the_target_citation_readable() -> None:
    text = "Alpha v. Beta, 100 U.S. 1 (1900). Gamma v. Delta, 200 U.S. 2 (1901)."
    document = grow_roots(preprocess(text), rules=stable())
    roots = [record for record in document.active_citations if isinstance(record.stated, FullCaseCitation)]

    context = masked_root_context(document, roots[0], before=len(text), after=len(text))

    assert context.start == 0
    assert context.end == len(text)
    assert context.text[roots[0].locator_span.start : roots[0].locator_span.end] == "100 U.S. 1"
    assert "200 U.S. 2" not in context.text
    assert "Gamma v. Delta" not in context.text
    assert len(context.text) == len(text)
