"""Root formation scores conditional groups and complete annotated roots."""

from __future__ import annotations

from evaluations.score_stages import score_document
from mellea_lrc.model import Document, FullReporterCitation, Span


def _root_case(*, include_toa: bool) -> tuple[Document, tuple[dict, ...], str, str]:
    source = "TABLE OF AUTHORITIES\nBrown v. Board, 347 U.S. 483.\n\nBrown v. Board, 347 U.S. 483."
    text = "347 U.S. 483"
    first = source.index(text)
    second = source.index(text, first + 1)
    spans = (Span(first, first + len(text)), Span(second, second + len(text)))
    citations = tuple(
        FullReporterCitation.from_locator(
            citation_id=f"reporter:{index}", stage="full_reporter_locators", source=source, span=span
        )
        for index, span in enumerate(spans, 1)
    )
    body = {
        "id": "gold-body",
        "root_id": "gold-root",
        "is_root": False,
        "kind": "FullCaseCitation",
        "locator": {"start": second, "end": second + len(text), "quote": text},
        "in_table_of_authorities": False,
    }
    toa = {
        "id": "gold-root",
        "root_id": "gold-root",
        "is_root": True,
        "kind": "FullCaseCitation",
        "locator": {"start": first, "end": first + len(text), "quote": text},
        "in_table_of_authorities": True,
    }
    base = Document.from_source(source)
    doc = Document.model_validate(
        {**base.model_dump(mode="python"), "index_spans": (Span(0, source.index("\n\n")),)}
    )
    for citation in citations if include_toa else citations[1:]:
        doc = doc.add_citation(citation)
    return doc.complete("full_reporter_locators"), (toa, body), citations[0].id, citations[1].id


def test_missing_toa_root_locator_keeps_conditional_group_but_marks_global_root_incomplete() -> None:
    document, gold, _toa_id, body_id = _root_case(include_toa=False)
    body = document.full_locators[0]
    document = document.replace_citation(body.record("roots").with_root(body_id)).complete("roots")

    counts, details = score_document(document, "roots", gold)
    root = next(item for item in details if item.get("product") == "gold_root")

    assert counts["exact_groups"] == 1
    assert counts["global_gold_roots"] == 1
    assert counts["global_exact_root_groups"] == 0
    assert counts["toa_gold_roots"] == 1
    assert counts["toa_canonical_root_locators_found"] == 0
    assert counts["index_gold_locators"] == 1
    assert counts["index_locators_found"] == 0
    assert counts["alternate_representative_roots"] == 1
    assert root["exact_conditional_group"] is True
    assert root["exact_global_group"] is False
    assert root["outcome"] == "canonical_locator_missing"


def test_found_toa_and_body_group_has_exact_global_root_and_correct_anchor() -> None:
    document, gold, toa_id, _body_id = _root_case(include_toa=True)
    for citation in document.full_locators:
        document = document.replace_citation(citation.record("roots").with_root(toa_id))
    document = document.complete("roots")

    counts, details = score_document(document, "roots", gold)
    root = next(item for item in details if item.get("product") == "gold_root")

    assert counts["global_gold_roots"] == counts["global_exact_root_groups"] == 1
    assert counts["canonical_root_locators_found"] == 1
    assert counts["canonical_root_anchors_correct"] == 1
    assert counts["index_gold_locators"] == counts["index_locators_found"] == 1
    assert root["exact_global_group"] is True
    assert root["canonical_anchor_correct"] is True
    assert root["outcome"] == "exact_root"


def test_complete_root_requires_canonical_occurrence_as_anchor() -> None:
    document, gold, _toa_id, body_id = _root_case(include_toa=True)
    for citation in document.full_locators:
        document = document.replace_citation(citation.record("roots").with_root(body_id))
    document = document.complete("roots")

    counts, details = score_document(document, "roots", gold)
    root = next(item for item in details if item.get("product") == "gold_root")

    assert counts["exact_groups"] == 1
    assert counts["global_exact_root_groups"] == 0
    assert counts["canonical_root_locators_found"] == 1
    assert counts["canonical_root_anchors_correct"] == 0
    assert counts["root_outcome_wrong_anchor"] == 1
    assert root["exact_conditional_group"] is True
    assert root["exact_global_group"] is False
    assert root["outcome"] == "wrong_anchor"
