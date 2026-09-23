"""Each layer evaluator consumes the shared grow-roots corpus independently."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from evaluations.extraction.eval_colocation import eval_colocation
from evaluations.extraction.eval_locators import eval_locators
from evaluations.extraction.locator_eval_common import GrownAnnotation, grow_annotated_corpus
from mellea_lrc.model.document import Document


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

    assert locator_result["reporter_locators"]["tp"] == 3
    assert locator_result["reporter_locators"]["fp"] == 0
    assert set(locator_result) == {"dataset", "documents", "reporter_locators", "docket_locators"}
    assert colocation_result["colocation"]["tp"] == 1
    assert colocation_result["colocation"]["fp"] == 0
    assert set(colocation_result) == {"dataset", "documents", "colocation"}


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
    score = eval_locators(corpus)["reporter_locators"]
    assert (score["gold"], score["tp"], score["fp"], score["fn"]) == (2, 2, 0, 0)

    # A separate counterfactual run that never found the repeat is a locator
    # miss. A later checkpoint of the real run cannot delete a created record.
    first_only = Document.from_plain_text(text).evolve(citations=sample.document.citations[:1])
    missing_repeat = replace(sample, document=first_only)
    score = eval_locators((missing_repeat,))["reporter_locators"]
    assert (score["gold"], score["tp"], score["fp"], score["fn"]) == (2, 1, 0, 1)


def test_docket_locator_score_uses_only_audit_admitted_records(tmp_path: Path) -> None:
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

    score = eval_locators(corpus)["docket_locators"]
    assert (score["gold"], score["predicted"], score["tp"], score["fp"], score["fn"]) == (1, 1, 1, 0, 0)
