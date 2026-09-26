"""Reporter validation reports contain only the requested incremental field scores."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from evaluations.evaluate_run import render_report as render_run_report
from evaluations.render_identity_report import render_identity_report
from evaluations.render_reporter_lookup_ambiguous import render_report as render_ambiguity_report
from evaluations.render_reporter_lookup_ambiguous_llm import render_report as render_ambiguous_llm_report
from evaluations.render_reporter_lookup_unique_llm import render_report as render_unique_llm_report
from mellea_lrc.validation.reporter_root_lookup import STAGE as LOOKUP_STAGE
from mellea_lrc.validation.reporter_root_lookup_ambiguous import STAGE as AMBIGUITY_STAGE
from mellea_lrc.validation.reporter_root_lookup_unique_llm import STAGE as UNIQUE_LLM_STAGE
from mellea_lrc.validation.stage_names import REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM as AMBIGUOUS_LLM_STAGE

FIELD_COUNTS = {
    "case_name": (1, 2, 4),
    "court": (0, 0, 3),
    "date": (3, 3, 4),
}
EXPECTED_PRIMARY_ROWS = [
    "| primary | Case name | 50.0% | 25.0% |",
    "| primary | Court | — | 0.0% |",
    "| primary | Date | 100.0% | 75.0% |",
]
REPORTERS = (
    (LOOKUP_STAGE, render_identity_report),
    (AMBIGUITY_STAGE, render_ambiguity_report),
    (UNIQUE_LLM_STAGE, render_unique_llm_report),
    (AMBIGUOUS_LLM_STAGE, render_ambiguous_llm_report),
)


def _stage_summary(stage: str) -> dict[str, Any]:
    fields = {}
    for name, (correct, scored, eligible) in FIELD_COUNTS.items():
        precision = round(correct / scored, 4) if scored else None
        recall = round(correct / eligible, 4)
        fields[name] = {
            "correct": correct,
            "correct_gold": correct,
            "correct_predictions": correct,
            "scored": scored,
            "decided": scored,
            "eligible": eligible,
            "eligible_gold": eligible,
            "precision": precision,
            "recall": recall,
            "conditional_recall": recall,
            "global_recall": 0.1111,
            "value": precision,
        }
    row = {
        "documents": 1,
        "fields": fields,
        "field_precision": {
            name: {"correct": value["correct"], "scored": value["scored"], "value": value["precision"]}
            for name, value in fields.items()
        },
        "field_recall": {
            name: {"correct": value["correct"], "eligible": value["eligible"], "value": value["recall"]}
            for name, value in fields.items()
        },
    }
    return {"stage": stage, "sets": {"primary": row}, "totals": row}


def _assert_only_requested_field_table(markdown: str) -> None:
    table = [line for line in markdown.splitlines() if line.startswith("|")]
    assert table[0].lower() in {
        "| set | field | precision | recall |",
        "| set | field | judgment precision | judgment recall |",
    }
    assert all(set(cell.strip()) <= {"-", ":"} for cell in table[1].strip("|").split("|"))
    assert [line for line in table[2:] if line.startswith("| primary |")][0:3] == EXPECTED_PRIMARY_ROWS
    assert len([line for line in table[2:] if line.startswith("| primary |")]) == 3
    assert all(len(line.strip("|").split("|")) == 4 for line in table)
    assert not re.search(r"\b\d+/\d+\b", markdown)
    for unwanted in (
        "locators found",
        "toa roots",
        "root coverage",
        "lookup cluster",
        "population",
        "admission precision",
        "decision precision",
        "diagnostic counts",
    ):
        assert unwanted not in markdown.lower()


@pytest.mark.parametrize(("stage", "renderer"), REPORTERS)
def test_each_reporter_stage_shows_only_three_incremental_precision_recall_pairs(
    stage: str, renderer: Callable[..., str]
) -> None:
    report = renderer(_stage_summary(stage), source_label="saved/summary.json")

    _assert_only_requested_field_table(report)
    assert "11.1%" not in report  # The scorer's global recall is not this stage's recall.


def test_composed_report_omits_unrequested_extraction_root_and_population_scores() -> None:
    projected = {
        name: {
            "precision": round(correct / scored, 4) if scored else None,
            "recall": round(correct / eligible, 4),
        }
        for name, (correct, scored, eligible) in FIELD_COUNTS.items()
    }
    stages = {stage: {"sets": {"primary": projected}, "totals": projected} for stage, _ in REPORTERS}
    result = {
        "sets": ["primary"],
        "stage_order": [
            LOOKUP_STAGE,
            AMBIGUITY_STAGE,
            AMBIGUOUS_LLM_STAGE,
            UNIQUE_LLM_STAGE,
        ],
        "stages": stages,
    }

    report = render_run_report(result, source_label="saved/summary.json")

    table = [line for line in report.splitlines() if line.startswith("|")]
    assert table[:2] == [
        "| Stage | Set | Field | Precision | Recall |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    assert table[2:] == [
        f"| {stage} | primary | {field} | {precision} | {recall} |"
        for stage in result["stage_order"]
        for field, precision, recall in (
            ("case name", "50.0%", "25.0%"),
            ("court", "—", "0.0%"),
            ("date", "100.0%", "75.0%"),
        )
    ]
    assert not re.search(r"\b\d+/\d+\b", report)
    for unwanted in (
        "full locators found",
        "toa roots found",
        "overall reporter-root field judgments",
        "lookup cluster",
        "docket site proposal",
    ):
        assert unwanted not in report.lower()
