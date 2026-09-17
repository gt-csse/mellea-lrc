"""Each layer evaluator consumes the shared grow-roots corpus independently."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluations.extraction.eval_colocation import eval_colocation
from evaluations.extraction.eval_docket_audit import eval_docket_audit
from evaluations.extraction.eval_locators import eval_locators
from evaluations.extraction.locator_eval_common import GrownAnnotation, grow_annotated_corpus


def _grow(tmp_path: Path, text: str, citations: list[dict[str, Any]]) -> tuple[GrownAnnotation, ...]:
    text_path = tmp_path / "corpus" / "documents_txt" / "sample.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(text, encoding="utf-8")

    header = {
        "unit": "header",
        "document": "sample.txt",
        "text": {"path": "corpus/documents_txt/sample.txt"},
    }
    annotation_path = tmp_path / "annotations"
    annotation_path.mkdir()
    (annotation_path / "sample.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [header, *citations]) + "\n",
        encoding="utf-8",
    )
    return grow_annotated_corpus(annotation_path, tmp_path)


def test_locator_and_colocation_scores_are_separate(tmp_path: Path) -> None:
    text = "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."
    spans = [
        (text.index(locator), text.index(locator) + len(locator))
        for locator in ("390 U.S. 727", "88 S.Ct. 1323", "20 L.Ed.2d 262")
    ]
    citations = [
        {
            "unit": "citation",
            "kind": "FullCaseCitation",
            "is_root": True,
            "colocation_id": "parallel-group",
            "locator": {"start": start, "end": end},
        }
        for start, end in spans
    ]
    corpus = _grow(tmp_path, text, citations)
    locator_result = eval_locators(corpus)
    colocation_result = eval_colocation(corpus)

    assert locator_result["locator_spans"]["tp"] == 3
    assert locator_result["locator_spans"]["fp"] == 0
    assert set(locator_result) == {"dataset", "documents", "locator_spans", "locator_spans_by_kind"}
    assert colocation_result["colocation_groups"]["tp"] == 1
    assert colocation_result["colocation_groups"]["fp"] == 0
    assert set(colocation_result) == {"dataset", "documents", "colocation_groups"}


def test_a_repeated_locator_counts_even_when_it_resolves_to_an_existing_root(tmp_path: Path) -> None:
    citation = "Ashcroft v. Iqbal, 556 U.S. 662 (2009)."
    locator = "556 U.S. 662"
    text = f"{citation} Again, {citation}"
    starts = (text.index(locator), text.rindex(locator))
    rows = [
        {
            "unit": "citation",
            "kind": "FullCaseCitation",
            "is_root": index == 0,
            "locator": {"start": start, "end": start + len(locator)},
        }
        for index, start in enumerate(starts)
    ]
    corpus = _grow(tmp_path, text, rows)
    sample = corpus[0]
    assert len({record.root_id for record in sample.document.citations}) == 1
    score = eval_locators(corpus)["locator_spans"]
    assert (score["gold"], score["tp"], score["fp"], score["fn"]) == (2, 2, 0, 0)

    # Losing the repeated occurrence is a locator miss even though the root survives.
    missing_repeat = replace(
        sample, document=replace(sample.document, citations=sample.document.citations[:1])
    )
    score = eval_locators((missing_repeat,))["locator_spans"]
    assert (score["gold"], score["tp"], score["fp"], score["fn"]) == (2, 1, 0, 1)


def test_audit_scores_admission_without_hiding_raw_locator_false_positives(tmp_path: Path) -> None:
    text = "Case No. 1:24-cv-00123\n\nSmith v. Jones, No. 1:23-cv-00456 (D. Ariz. 2023)."
    locator = "No. 1:23-cv-00456"
    start = text.index(locator)
    corpus = _grow(
        tmp_path,
        text,
        [
            {
                "unit": "citation",
                "kind": "DocketCitation",
                "locator": {"start": start, "end": start + len(locator)},
            }
        ],
    )

    raw = eval_locators(corpus)["locator_spans"]
    audit = eval_docket_audit(corpus)["docket_audit"]
    assert (raw["tp"], raw["fp"]) == (1, 1)
    assert (audit["candidates"], audit["accepted"], audit["withdrawn"]) == (2, 1, 1)
    assert (audit["accepted_spans"]["tp"], audit["accepted_spans"]["fp"]) == (1, 0)
