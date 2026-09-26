"""A saved final Document scores each stage's own durable product."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from evaluations import score_stages
from evaluations.score_stages import _summary, score_document
from evaluations.stage_products import stage_product
from mellea_lrc.api import grow_roots
from mellea_lrc.model import (
    Document,
    FullDocketCitation,
    FullReporterCitation,
    PreprocessingMetadata,
    SourceFormat,
    SourceMetadata,
    Span,
)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], ("primary",)),
        (["--set", "hallucination-set-1"], ("hallucination-set-1",)),
        (["--set", "primary", "--set", "hallucination-set-1"], ("primary", "hallucination-set-1")),
    ],
)
def test_cli_set_selection_defaults_to_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], expected: tuple[str, ...]
) -> None:
    selected: list[tuple[str, ...]] = []

    def fake_evaluate(_data_root: Path, _run_dir: Path, _stage: str, sets: tuple[str, ...]) -> dict:
        selected.append(sets)
        return {"occurrences": {}}

    monkeypatch.setattr(score_stages, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys, "argv", ["score_stages", "--run-dir", str(tmp_path), "--stage", score_stages.STAGES[0], *options]
    )
    score_stages.main()
    assert selected == [expected]


def _document(source: str) -> Document:
    return Document(
        source_metadata=SourceMetadata(format=SourceFormat.TEXT),
        text=source,
        preprocessing_metadata=PreprocessingMetadata(),
    )


def _span(source: str, quote: str) -> Span:
    start = source.index(quote)
    return Span(start, start + len(quote))


def _gold_span(value: Span) -> dict[str, int]:
    return {"start": value.start, "end": value.end}


def _docket_document(source: str, locator_text: str = "Case No. 1:24-cv-00123") -> Document:
    locator = _span(source, locator_text)
    number = _span(source, "1:24-cv-00123")
    citation = FullDocketCitation.from_locator(
        citation_id="docket:1",
        stage="docket_locators",
        source=source,
        span=locator,
        number_span=number,
    )
    return _document(source).add_citation(citation).complete("docket_locators")


def test_toa_reporter_locator_is_scored_but_its_pin_is_not_checkable() -> None:
    source = "TABLE OF AUTHORITIES\nBrown v. Board of Education, 347 U.S. 483, 489 (1954).\n"
    initial = Document.model_validate(
        {**_document(source).model_dump(mode="python"), "index_spans": (Span(0, len(source)),)}
    )
    final = asyncio.run(grow_roots(initial, hunt_dockets=False))
    citation = final.roots[0]
    assert isinstance(citation, FullReporterCitation)
    assert len(citation.pin_cite) == 1
    gold = (
        {
            "id": "toa-root",
            "kind": "FullCaseCitation",
            "locator": _gold_span(citation.locator_span),
            "pin_cite": _gold_span(citation.pin_cite[-1].span),
        },
    )

    locator_counts, _ = score_document(final, "full_reporter_locators", gold)
    pin_counts, _ = score_document(final, "pin_cites", gold)

    assert locator_counts["eligible_gold"] == locator_counts["exact_spans"] == 1
    assert pin_counts["eligible_gold"] == pin_counts["predicted"] == 0


def test_stage_product_inherits_previous_checkpoint_without_reclaiming_its_citations() -> None:
    source = "Alpha v. Beta, 347 U.S. 483; Smith v. Jones, Case No. 1:24-cv-00123."
    reporter_span = _span(source, "347 U.S. 483")
    reporter = FullReporterCitation.from_locator(
        citation_id="reporter:1",
        stage="full_reporter_locators",
        source=source,
        span=reporter_span,
    )
    reporter_checkpoint = _document(source).add_citation(reporter).complete("full_reporter_locators")
    docket_span = _span(source, "Case No. 1:24-cv-00123")
    docket = FullDocketCitation.from_locator(
        citation_id="docket:1",
        stage="docket_locators",
        source=source,
        span=docket_span,
        number_span=_span(source, "1:24-cv-00123"),
    )
    final = reporter_checkpoint.add_citation(docket).complete("docket_locators")

    product = stage_product(final, "docket_locators")
    assert product.before == reporter_checkpoint
    assert product.after == final
    assert [citation.id for citation in product.created] == ["docket:1"]
    assert [(item.citation.id, item.name) for item in product.fields] == [("docket:1", "locator")]

    gold = (
        {"id": "gold-reporter", "kind": "FullCaseCitation", "locator": _gold_span(reporter_span)},
        {
            "id": "gold-docket",
            "kind": "DocketCitation",
            "locator": _gold_span(docket_span),
            "identifier": {"kind": "docket", "docket_number": "1:24-cv-00123"},
        },
    )
    counts, details = score_document(final, "docket_locators", gold)
    assert counts["gold_fields"] == counts["eligible_gold"] == counts["predicted"] == 1
    assert counts["exact_spans"] == counts["normalization_correct"] == 1
    assert [detail["citation_id"] for detail in details] == ["docket:1"]


def test_later_field_update_does_not_change_earlier_stage_product_or_score() -> None:
    source = "Alpha v. Beta; Smith v. Jones, Case No. 1:24-cv-00123."
    located = _docket_document(source)
    first_name = _span(source, "Alpha v. Beta")
    second_name = _span(source, "Smith v. Jones")
    citation = located.citations[0]
    named = located.replace_citation(citation.record("case_names").with_case_name(source, first_name))
    named = named.complete("case_names")
    revised = named.replace_citation(
        named.citations[0].record("later_review").with_case_name(source, second_name)
    ).complete("later_review")

    assert revised.citations[0].case_name[-1].span == second_name
    early_product = stage_product(named, "case_names")
    late_product = stage_product(revised, "case_names")
    assert late_product == early_product
    assert [(item.name, item.reading.span) for item in late_product.fields] == [("case_name", first_name)]
    gold = (
        {
            "id": "gold-docket",
            "kind": "DocketCitation",
            "locator": _gold_span(located.citations[0].locator_span),
            "case_name": {**_gold_span(first_name), "normalized": "Alpha v. Beta"},
        },
    )
    assert score_document(revised, "case_names", gold) == score_document(named, "case_names", gold)
    counts, _ = score_document(revised, "case_names", gold)
    assert counts["exact_spans"] == 1
    assert counts["normalization_checked"] == counts["normalization_correct"] == 1
    assert counts["predicted"] == 1


def test_inferred_court_scores_normalized_id_without_a_source_span() -> None:
    source = "Alpha v. Beta, 347 U.S. 483."
    locator = _span(source, "347 U.S. 483")
    citation = FullReporterCitation.from_locator(
        citation_id="reporter:1",
        stage="full_reporter_locators",
        source=source,
        span=locator,
    )
    located = _document(source).add_citation(citation).complete("full_reporter_locators")
    inferred = located.replace_citation(citation.record("courts").with_inferred_court("scotus"))
    inferred = inferred.complete("courts")
    gold = (
        {
            "id": "gold-reporter",
            "kind": "FullCaseCitation",
            "locator": _gold_span(locator),
            "court": {"how": "reporter", "id": "scotus", "name": "Supreme Court of the United States"},
        },
    )

    product = stage_product(inferred, "courts")
    assert product.fields[0].reading.span is None
    assert product.fields[0].reading.quote is None
    counts, details = score_document(inferred, "courts", gold)
    assert counts["gold_without_span"] == counts["matched_without_span"] == 1
    assert counts["exact_spans"] == 0
    assert counts["normalization_checked"] == counts["normalization_correct"] == 1
    assert details[0]["span"] is None


def test_entry_span_and_normalization_are_scored_separately() -> None:
    source = "Case No. 1:24-cv-00123, Doc. 10-1."
    located = _docket_document(source)
    entry_span = _span(source, "Doc. 10-1")
    citation = located.citations[0]
    with_entry = located.replace_citation(
        citation.record("docket_entries").with_docket_entry(source, entry_span)
    ).complete("docket_entries")
    gold = (
        {
            "id": "gold-docket",
            "kind": "DocketCitation",
            "locator": _gold_span(citation.locator_span),
            "docket_entry": {**_gold_span(entry_span), "quote": "Doc. 10-1", "number": "11"},
        },
    )

    counts, details = score_document(with_entry, "docket_entries", gold)
    assert counts["exact_spans"] == 1
    assert counts["normalizable"] == 1
    assert counts["normalization_checked"] == counts["normalization_wrong"] == 1
    assert counts["normalization_correct"] == 0
    assert details[0]["normalized"] == "10-1"


def test_unrun_stage_raises_instead_of_silently_scoring_nothing() -> None:
    located = _docket_document("Case No. 1:24-cv-00123.")
    with pytest.raises(KeyError, match="Stage has not run"):
        stage_product(located, "docket_entries")
    with pytest.raises(KeyError, match="Stage has not run"):
        score_document(located, "docket_entries", ())


def test_completed_empty_stage_counts_its_missed_gold() -> None:
    source = "Case No. 1:24-cv-00123, Doc. 10-1."
    located = _docket_document(source)
    completed = located.complete("docket_entries")
    gold = (
        {
            "id": "gold-docket",
            "kind": "DocketCitation",
            "locator": _gold_span(located.citations[0].locator_span),
            "docket_entry": _gold_span(_span(source, "Doc. 10-1")),
        },
    )
    counts, _ = score_document(completed, "docket_entries", gold)
    assert counts["eligible_gold"] == counts["missed_gold"] == 1
    assert counts["predicted"] == counts["exact_spans"] == 0


def test_restored_final_document_recovers_the_same_stage_product_and_score() -> None:
    source = "Case No. 1:24-cv-00123, Doc. 10-1."
    located = _docket_document(source)
    citation = located.citations[0]
    entry_span = _span(source, "Doc. 10-1")
    final = located.replace_citation(
        citation.record("docket_entries").with_docket_entry(source, entry_span)
    ).complete("docket_entries")
    final = final.complete("empty_checkpoint")
    restored = Document.model_validate_json(final.model_dump_json())
    gold = (
        {
            "id": "gold-docket",
            "kind": "DocketCitation",
            "locator": _gold_span(citation.locator_span),
            "docket_entry": {**_gold_span(entry_span), "quote": "Doc. 10-1", "number": "10-1"},
        },
    )

    assert restored == final
    assert stage_product(restored, "docket_entries") == stage_product(final, "docket_entries")
    assert score_document(restored, "docket_entries", gold) == score_document(final, "docket_entries", gold)
    empty = stage_product(restored, "empty_checkpoint")
    assert empty.after == final
    assert empty.before == final.get_stage("docket_entries")
    assert not (empty.created or empty.fields or empty.relationships or empty.reviews)


def test_colocation_and_root_scores_use_only_their_relationship_stage() -> None:
    source = "Case No. 1:24-cv-00123; Case No. 2:24-cv-00456."
    document = _document(source)
    for index, text in enumerate(("Case No. 1:24-cv-00123", "Case No. 2:24-cv-00456"), 1):
        locator = _span(source, text)
        number = _span(source, text.removeprefix("Case No. "))
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=f"docket:{index}",
                stage="docket_locators",
                source=source,
                span=locator,
                number_span=number,
            )
        )
    document = document.complete("docket_locators")
    gold = tuple(
        {
            "id": f"gold-{index}",
            "kind": "DocketCitation",
            "locator": _gold_span(citation.locator_span),
            "colocation_id": "gold-parallel",
            "root_id": "gold-1",
        }
        for index, citation in enumerate(document.citations, 1)
    )
    for citation in document.citations:
        document = document.replace_citation(
            citation.record("colocations").with_colocation("predicted-parallel")
        )
    document = document.complete("colocations")
    colocation_checkpoint = document
    for citation in document.citations:
        document = document.replace_citation(citation.record("roots").with_root(citation.id))
    document = document.complete("roots")

    assert stage_product(document, "colocations").after == colocation_checkpoint
    colocations, _ = score_document(document, "colocations", gold)
    assert colocations["gold_groups"] == colocations["predicted_groups"] == 1
    assert colocations["exact_groups"] == colocations["correct_links"] == 1
    colocation_summary = _summary(colocations, "colocations")
    assert colocation_summary["gold_links"] == colocation_summary["predicted_links"] == 1
    assert colocation_summary["link_precision"] == colocation_summary["link_recall"] == 1.0

    roots, _ = score_document(document, "roots", gold)
    assert roots["gold_groups"] == 1
    assert roots["predicted_groups"] == 2
    assert roots["exact_groups"] == 0
    root_summary = _summary(roots, "roots")
    assert root_summary["exact_group_recall"] == 0.0
    link_keys = {
        "gold_links",
        "predicted_links",
        "correct_links",
        "link_precision",
        "link_recall",
    }
    assert link_keys.isdisjoint(roots)
    assert link_keys.isdisjoint(root_summary)
