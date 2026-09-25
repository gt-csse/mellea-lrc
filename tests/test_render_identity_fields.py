"""Field reports expose comparison accuracy separately from extraction and routing gaps."""

from __future__ import annotations

import pytest

from evaluations.render_identity_report import render_identity_report


def _field(
    *,
    stated: int,
    unique: int,
    aligned: int,
    correct: int,
    decided: int,
    missing: int = 0,
    misaligned: int = 0,
    wrong: int = 0,
    undetermined: int = 0,
    omitted: int = 0,
    unscored: int = 0,
    later_occurrence: int = 0,
) -> dict:
    return {
        "gold_stated": stated,
        "gold_not_stated": 0,
        "unique_gold": unique,
        "eligible_gold": aligned,
        "correct_predictions": correct,
        "correct_gold": correct,
        "decided": decided,
        "missing_reading": missing,
        "misaligned_reading": misaligned,
        "incorrect_predictions": wrong,
        "undetermined": undetermined,
        "omitted_gold": omitted,
        "unscored_judgments": unscored,
        "changed_occurrence_judgments": later_occurrence,
        "confusion": {
            "agrees": {"match": 0, "mismatch": 0, "undetermined": 0},
            "disagrees": {"match": 0, "mismatch": 0, "undetermined": 0},
        },
    }


def _summary() -> dict:
    fields = {
        "case_name": _field(
            stated=7,
            unique=6,
            aligned=4,
            correct=2,
            decided=3,
            missing=1,
            misaligned=1,
            wrong=1,
            undetermined=1,
            unscored=2,
            later_occurrence=1,
        ),
        "court": _field(
            stated=5,
            unique=4,
            aligned=4,
            correct=0,
            decided=0,
            undetermined=2,
            omitted=2,
            unscored=1,
            later_occurrence=1,
        ),
        "date": _field(stated=3, unique=3, aligned=3, correct=3, decided=3),
    }
    fields["case_name"]["confusion"]["agrees"] = {"match": 2, "mismatch": 1, "undetermined": 1}
    fields["court"]["confusion"]["agrees"]["undetermined"] = 2
    fields["date"]["confusion"]["agrees"]["match"] = 3
    row = {"documents": 1, "gold_reporter_roots": 8, "field_gold_roots": 7, "fields": fields}
    return {
        "stage": "reporter_root_exact_lookup",
        "sets": {"primary": row},
        "totals": row,
    }


def test_field_report_shows_accuracy_coverage_and_failure_categories() -> None:
    report = render_identity_report(_summary(), source_label="saved/summary.json")

    assert "saved/summary.json" in report
    assert "Decision precision" in report
    assert "later citation is unscored for fields" in report
    assert "7 of 8 identity-labeled reporter roots" in report
    assert "`not_stated` is excluded" in report
    assert "| primary | Case name | 7 | 6 | 4 | 2/3 (66.7%) | 3/4 (75.0%) | 2/7 (28.6%) |" in report
    assert "| primary | Court | 5 | 4 | 4 | — | 0/4 (0.0%) | 0/5 (0.0%) |" in report
    assert "| Total | Date | 3 | 3 | 3 | 3/3 (100.0%) | 3/3 (100.0%) | 3/3 (100.0%) |" in report
    assert "| primary | Case name | 1 | 1 | 1 | 1 | 1 | 0 | 2 |" in report
    assert "| Total | Court | 1 | 0 | 0 | 0 | 2 | 2 | 1 |" in report
    assert "| Case name | 2 | 1 | 0 | 0 | 1 | 0 |" in report
    assert "1 case name, 1 court, 0 date" in report


def test_field_report_rejects_impossible_unique_lookup_count() -> None:
    result = _summary()
    result["sets"]["primary"]["fields"]["date"]["unique_gold"] = 4

    with pytest.raises(ValueError, match="unique lookup count exceeds"):
        render_identity_report(result, source_label="saved/summary.json")
