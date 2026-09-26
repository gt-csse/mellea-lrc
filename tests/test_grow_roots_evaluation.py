"""The first two extraction stages have the same precision contract."""

from __future__ import annotations

import pytest

from evaluations.grow_roots import DOCKET_STAGE, REPORTER_STAGE, Precision, render_markdown, score_document
from mellea_lrc.api import Document, find_docket_locators, find_full_reporter_locators, resolve_colocations


def _row(document: Document, quote: str, kind: str, identifier: dict | None) -> dict:
    start = document.text.index(quote)
    return {
        "unit": "citation",
        "kind": kind,
        "locator": {"start": start, "end": start + len(quote), "quote": quote},
        "identifier": identifier,
    }


def test_each_stage_scores_only_its_own_decisions_and_own_gold() -> None:
    source = "See 550 U.S. 544 and Case No. 1:24-cv-08705."
    document = find_docket_locators(find_full_reporter_locators(Document.from_source(source)))
    rows = (
        _row(
            document,
            "550 U.S. 544",
            "FullCaseCitation",
            {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
        ),
        _row(
            document,
            "Case No. 1:24-cv-08705",
            "DocketCitation",
            {"kind": "docket", "docket_number": "1:24-cv-08705"},
        ),
    )

    reporter = score_document(document, rows, REPORTER_STAGE)
    docket = score_document(document, rows, DOCKET_STAGE)

    assert reporter.span == reporter.normalization == Precision(1, 1)
    assert docket.span == docket.normalization == Precision(1, 1)


def test_normalization_precision_requires_an_exact_span_and_same_row_target() -> None:
    document = find_full_reporter_locators(Document.from_source("See 550 U.S. 544."))
    row = _row(document, "550 U.S. 544", "FullCaseCitation", None)
    assert score_document(document, (row,), REPORTER_STAGE).span == Precision(1, 1)
    assert score_document(document, (row,), REPORTER_STAGE).normalization == Precision(0, 0)

    row["identifier"] = {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "545"}
    assert score_document(document, (row,), REPORTER_STAGE).normalization == Precision(0, 1)

    row["locator"]["start"] += 1
    row["locator"]["end"] += 1
    row["locator"]["quote"] = document.text[row["locator"]["start"] : row["locator"]["end"]]
    result = score_document(document, (row,), REPORTER_STAGE)
    assert result.span == Precision(0, 1)
    assert result.normalization == Precision(0, 0)


def test_missing_checkpoint_and_duplicate_gold_fail_loudly() -> None:
    document = find_full_reporter_locators(Document.from_source("See 550 U.S. 544."))
    row = _row(document, "550 U.S. 544", "FullCaseCitation", None)
    with pytest.raises(KeyError, match="Stage has not run"):
        score_document(document, (row,), DOCKET_STAGE)
    with pytest.raises(ValueError, match="Duplicate annotated"):
        score_document(document, (row, row), REPORTER_STAGE)


def test_later_checkpoint_and_json_reload_keep_the_same_stage_scores() -> None:
    document = find_docket_locators(
        find_full_reporter_locators(Document.from_source("See 550 U.S. 544 and Case No. 1:24-cv-08705."))
    )
    rows = (
        _row(
            document,
            "550 U.S. 544",
            "FullCaseCitation",
            {"kind": "reporter", "volume": "550", "reporter": "U.S.", "page": "544"},
        ),
        _row(
            document,
            "Case No. 1:24-cv-08705",
            "DocketCitation",
            {"kind": "docket", "docket_number": "1:24-cv-08705"},
        ),
    )
    later = Document.model_validate_json(resolve_colocations(document).model_dump_json())
    for stage in (REPORTER_STAGE, DOCKET_STAGE):
        assert score_document(later, rows, stage) == score_document(document, rows, stage)


def test_report_contains_only_the_two_precision_columns() -> None:
    result = [
        {
            "set": "primary",
            "stages": [
                {
                    "stage": REPORTER_STAGE,
                    "span": Precision(1, 2).as_dict(),
                    "normalization": Precision().as_dict(),
                }
            ],
        }
    ]
    report = render_markdown(result)
    assert "Span precision | Normalization precision" in report
    assert "1/2 (50.0%)" in report
    assert "—" in report
    assert "recall" not in report.casefold()
