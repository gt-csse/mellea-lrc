"""Markdown rendering consumes saved stage scores without running evaluation."""

from __future__ import annotations

from copy import deepcopy

from evaluations.annotations import SETS
from evaluations.render_stage_report import REPORT_ORDER, STAGE_NAMES, render_stage_report
from evaluations.score_stages import ELIGIBILITY, FIELD_STAGES, STAGES


def _result(
    stage: str,
    counts: dict[str, object],
    *,
    sets: dict[str, dict[str, object]] | None = None,
) -> dict:
    return {
        "stage": stage,
        "basis": "Only readings or relationships written by this stage",
        "eligibility": ELIGIBILITY[stage],
        "sets": sets if sets is not None else {"primary": counts},
        "totals": counts,
        "occurrences": {"primary/filing.txt": [{"outcome": "matched"}]},
    }


def _empty_result(stage: str, *, documents: int = 1) -> dict:
    if stage in FIELD_STAGES:
        counts = {
            "documents": documents,
            "eligible_gold": 0,
            "predicted": 0,
            "gold_with_span": 0,
            "gold_with_normalization": 0,
            "exact_spans": 0,
            "normalization_checked": 0,
            "normalization_correct": 0,
        }
    else:
        counts = {
            "documents": documents,
            "gold_locators": 0,
            "eligible_locators": 0,
            "gold_groups": 0,
            "predicted_groups": 0,
            "exact_groups": 0,
        }
        if stage == "colocations":
            counts.update({"gold_links": 0, "predicted_links": 0, "correct_links": 0})
    return _result(stage, counts)


def _all_results(
    stage: str | None = None,
    counts: dict[str, object] | None = None,
    *,
    documents: int = 1,
) -> dict[str, dict]:
    results = {name: _empty_result(name, documents=documents) for name in STAGES}
    if stage is not None:
        assert counts is not None
        results[stage] = _result(stage, counts)
    return results


def _section(markdown: str, stage: str) -> str:
    return markdown.split(f"### {STAGE_NAMES[stage]}\n", 1)[1].split("\n### ", 1)[0]


def test_stage_and_set_order_follow_declared_order_not_json_insertion_order() -> None:
    ordered = _all_results()
    reordered = dict(reversed(tuple(deepcopy(ordered).items())))
    for result in reordered.values():
        counts = result["totals"]
        result["sets"] = dict.fromkeys(reversed(SETS), counts)
    for result in ordered.values():
        counts = result["totals"]
        result["sets"] = dict.fromkeys(SETS, counts)

    first = render_stage_report(ordered, source_label="local/evaluations/root-stage-eval")
    second = render_stage_report(reordered, source_label="local/evaluations/root-stage-eval")

    assert first == second
    assert "local/evaluations/root-stage-eval" in first
    headings = [line for line in first.splitlines() if line.startswith("### ")]
    assert headings == [f"### {STAGE_NAMES[stage]}" for stage in REPORT_ORDER]
    assert REPORT_ORDER == (
        "full_reporter_locators",
        "docket_locators",
        "docket_locator_site_hunting",
        "docket_entries",
        "colocations",
        "case_names",
        "courts",
        "dates",
        "pin_cites",
        "roots",
        "short_reporter_citations",
    )
    primary = first.index("| primary (1) |")
    hallucination_1 = first.index("| hallucination-set-1 (1) |")
    hallucination_2 = first.index("| hallucination-set-2 (1) |")
    reliable_high = first.index("| reliable-high-profile (1) |")
    reliable_low = first.index("| reliable-low-profile (1) |")
    assert primary < hallucination_1 < hallucination_2 < reliable_high < reliable_low


def test_field_denominators_exclude_unlabeled_predictions_and_null_is_dash() -> None:
    counts = {
        "documents": 2,
        "eligible_gold": 4,
        "predicted": 5,
        "gold_with_span": 4,
        "gold_with_normalization": 0,
        "exact_spans": 2,
        "wrong_span": 1,
        "unlabeled_field": 2,
        "normalization_checked": 0,
        "normalization_correct": 0,
        "span_precision_on_scored_predictions": 0.6667,
        "span_recall": 0.5,
        "normalization_accuracy_on_matched_evidence": None,
        "normalization_recall": None,
    }
    markdown = render_stage_report(
        _all_results("case_names", counts, documents=2), source_label="saved summaries"
    )

    section = _section(markdown, "case_names")
    assert "| primary (2) | 4 | 5 | 2/3 (66.7%) | 2/4 (50.0%) | — | — |" in section
    assert "2/5 (40.0%)" not in section
    assert "0/0" not in section
    assert "None" not in markdown


def test_field_normalization_accuracy_and_recall_use_distinct_denominators() -> None:
    counts = {
        "documents": 1,
        "eligible_gold": 4,
        "predicted": 3,
        "gold_with_span": 4,
        "gold_with_normalization": 3,
        "exact_spans": 3,
        "normalization_checked": 2,
        "normalization_correct": 1,
        "normalization_accuracy_on_matched_evidence": 0.5,
        "normalization_recall": 0.3333,
    }
    markdown = render_stage_report(_all_results("docket_entries", counts), source_label="saved summaries")

    assert "| primary (1) | 4 | 3 | 3/3 (100.0%) | 3/4 (75.0%) | 1/2 (50.0%) | 1/3 (33.3%) |" in _section(
        markdown, "docket_entries"
    )


def test_root_report_shows_group_recall_without_link_columns() -> None:
    counts = {
        "documents": 66,
        "gold_locators": 2385,
        "eligible_locators": 2381,
        "gold_groups": 2053,
        "predicted_groups": 2092,
        "exact_groups": 2036,
        "exact_group_recall": 0.9917,
    }
    markdown = render_stage_report(
        _all_results("roots", counts, documents=66), source_label="saved summaries"
    )

    section = _section(markdown, "roots")
    assert "| Set (documents) | Eligible full locators | Exact groups / gold groups |" in section
    assert "| **Total** (66) | 2381/2385 | 2036/2053 (99.2%) |" in section
    assert "links" not in section.lower()
    assert "| Root formation | 2381/2385 | 2036/2053 (99.2%) |" in markdown
    assert "Root link recall" not in markdown
    assert "Root formation misses" not in markdown


def test_colocation_report_keeps_pair_scores() -> None:
    counts = {
        "documents": 3,
        "gold_locators": 7,
        "eligible_locators": 6,
        "gold_groups": 2,
        "predicted_groups": 2,
        "exact_groups": 1,
        "gold_links": 3,
        "predicted_links": 2,
        "correct_links": 1,
    }
    markdown = render_stage_report(
        _all_results("colocations", counts, documents=3), source_label="saved summaries"
    )

    section = _section(markdown, "colocations")
    assert "| Correct colocated pairs / predicted pairs | Correct colocated pairs / gold pairs |" in section
    assert "| **Total** (3) | 6/7 | 1/2 (50.0%) | 1/2 (50.0%) | 1/3 (33.3%) |" in section


def test_occurrence_details_do_not_change_the_report() -> None:
    results = _all_results()
    without_occurrences = deepcopy(results)
    del without_occurrences["roots"]["occurrences"]

    assert render_stage_report(results, source_label="saved summaries") == render_stage_report(
        without_occurrences, source_label="saved summaries"
    )
