"""The identity Markdown report shows only three field judgments per set."""

from __future__ import annotations

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
        "precision": round(correct / decided, 4) if decided else None,
        "conditional_recall": round(correct / aligned, 4) if aligned else None,
        "global_recall": round(correct / stated, 4) if stated else None,
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
        "stage": "reporter_root_lookup",
        "prediction_run": {
            "root_rules": "stable",
            "hunt_dockets": False,
            "court_docket_fetch": False,
        },
        "sets": {"primary": row},
        "totals": row,
    }


def test_field_report_has_only_judgment_precision_and_recall() -> None:
    report = render_identity_report(_summary(), source_label="saved/summary.json")

    assert "saved/summary.json" in report
    table = [line for line in report.splitlines() if line.startswith("|")]
    assert table[0] == "| Set | Field | Judgment precision | Judgment recall |"
    assert all(set(cell.strip()) <= {"-", ":"} for cell in table[1].strip("|").split("|"))
    assert table[2:] == [
        "| primary | Case name | 66.7% | 50.0% |",
        "| primary | Court | — | 0.0% |",
        "| primary | Date | 100.0% | 100.0% |",
        "| Total | Case name | 66.7% | 50.0% |",
        "| Total | Court | — | 0.0% |",
        "| Total | Date | 100.0% | 100.0% |",
    ]
    for old_metric in (
        "Decision precision",
        "Decision recall",
        "Admission precision",
        "Admission recall",
        "Judgment coverage",
        "Comparison accuracy",
        "Missing reading",
        "Misaligned reading",
        "Comparison directions",
        "locator membership",
    ):
        assert old_metric.lower() not in report.lower()


def test_field_report_uses_saved_rates_without_rescoring() -> None:
    result = _summary()
    result["sets"]["primary"]["fields"]["case_name"]["precision"] = 0.1234
    result["sets"]["primary"]["fields"]["case_name"]["conditional_recall"] = 0.5678

    report = render_identity_report(result, source_label="saved/summary.json")

    assert "| primary | Case name | 12.3% | 56.8% |" in report


def test_field_report_includes_three_rows_for_each_selected_set() -> None:
    result = _summary()
    result["sets"]["hallucination-set-1"] = {
        "documents": 1,
        "fields": {
            field: _field(stated=0, unique=0, aligned=0, correct=0, decided=0)
            for field in ("case_name", "court", "date")
        },
    }
    result["totals"] = {**result["totals"], "documents": 2}

    report = render_identity_report(result, source_label="saved/summary.json")
    table = [line for line in report.splitlines() if line.startswith("|")]

    assert len(table) == 2 + 3 * (len(result["sets"]) + 1)
    assert table[5:8] == [
        "| hallucination-set-1 | Case name | — | — |",
        "| hallucination-set-1 | Court | — | — |",
        "| hallucination-set-1 | Date | — | — |",
    ]
