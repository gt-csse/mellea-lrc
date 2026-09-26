"""Saved unique-review scores use only its own field judgments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluations import evaluate_run
from evaluations.score_reporter_lookup_unique_llm import _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model import Span
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import ReporterUniqueReviewDecision
from mellea_lrc.validation._support.reporter_unique_llm import ReporterUniqueReviewOutcome
from mellea_lrc.validation.reporter_root_lookup_unique_llm import (
    STAGE,
    reporter_root_lookup_unique_llm,
)

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."


class _LookupClient:
    def __init__(self, *, full_name: str | None = None) -> None:
        self.full_name = full_name

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": f"{volume} {reporter} {page}",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseName": "Bell Atlantic Corporation v. Twombly",
                        "caseNameFull": self.full_name,
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )


class _Reviewer:
    def __init__(self, *, case_name_result: str = "match") -> None:
        self.case_name_result = case_name_result

    async def __call__(self, _context: object) -> ReporterUniqueReviewDecision:
        return ReporterUniqueReviewDecision.model_validate(
            {
                "case_name": {
                    "propose_replacement": True,
                    "quote": "Bell Atl. Corp. v. Twombly",
                    "normalized": {
                        "kind": "adversarial",
                        "plaintiff": "Bell Atl. Corp.",
                        "defendant": "Twombly",
                    },
                    "result": self.case_name_result,
                    "reason": "The cited parties were checked against the opinion record.",
                },
                "court": {
                    "propose_replacement": False,
                    "quote": None,
                    "result": "match",
                    "reason": "The reporter and record identify the same court.",
                },
                "date": {
                    "propose_replacement": True,
                    "quote": "2007",
                    "result": "match",
                    "reason": "The cited year agrees with the opinion date.",
                },
                "reason": "Each field was reviewed against the saved opinion record.",
            }
        )


class _FailedReviewer:
    async def __call__(self, _context: object) -> ReporterUniqueReviewOutcome:
        return ReporterUniqueReviewOutcome(decision=None, failure_reason="The review could not be completed")


def _lookup(*, incorrect_case_name: bool = False, full_name: str | None = None) -> Document:
    roots = asyncio.run(grow_roots(Document.from_source(SOURCE), hunt_dockets=False))
    if incorrect_case_name:
        root = roots.roots[0]
        start = SOURCE.index("Corp. v.")
        end = SOURCE.index(", 550")
        roots = roots.replace_citation(
            root.record("test_incorrect_reading").with_case_name(SOURCE, Span(start, end))
        ).complete("test_incorrect_reading")
    return reporter_root_lookup(roots, client=_LookupClient(full_name=full_name))


def _gold(document: Document, *, case_name_label: str = "agrees") -> dict:
    root = document.roots[0]
    case_name = root.case_name[-1]
    court = root.court[-1]
    date = root.date[-1]
    return {
        "id": "gold-1",
        "root_id": "gold-1",
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter"},
        "locator": {
            "start": root.locator_span.start,
            "end": root.locator_span.end,
            "quote": document.text[root.locator_span.start : root.locator_span.end],
        },
        "case_name": {
            "start": case_name.span.start,
            "end": case_name.span.end,
            "quote": case_name.quote,
        },
        "court": {"id": court.get_normalized().id},
        "date": {
            "start": date.span.start,
            "end": date.span.end,
            "quote": date.quote,
            "normalized": "2007",
        },
        "validation": {
            "identity": {
                "label": "CORRECT_IDENTITY",
                "fields": {
                    "case_name": {"label": case_name_label},
                    "court": {"label": "agrees"},
                    "date": {"label": "agrees"},
                },
            }
        },
    }


def test_corrected_latest_reading_is_scored_from_unique_stage() -> None:
    before = _lookup(incorrect_case_name=True)
    assert before.roots[0].identity_judgments[-1].next_stage == STAGE
    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=_Reviewer()))
    assert after.roots[0].case_name[-1].quote == "Bell Atl. Corp. v. Twombly"

    counts, details = score_document(after, (_gold(after),))
    summary = _summary(counts)

    assert counts["routed_roots"] == 1
    for field in ("case_name", "court", "date"):
        assert summary["fields"][field] == {
            "correct": 1,
            "scored": 1,
            "eligible": 1,
            "correct_gold": 1,
            "precision": 1.0,
            "recall": 1.0,
        }
    case_name = next(item for item in details if item["field"] == "case_name")
    assert case_name["latest_reading_index"] == len(after.roots[0].case_name) - 1
    assert case_name["reading_index"] == case_name["latest_reading_index"]
    assert case_name["latest_reading"]["quote"] == "Bell Atl. Corp. v. Twombly"
    assert case_name["outcome"] == "correct"


def test_rule_stage_judgments_are_not_included_when_unique_stage_did_not_review() -> None:
    before = _lookup(full_name="Bell Atlantic Corporation v. Twombly")
    assert before.roots[0].case_name_judgments
    assert before.roots[0].identity_judgments[-1].next_stage is None
    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=_Reviewer()))

    counts, details = score_document(after, (_gold(after),))

    assert details == []
    assert counts["routed_roots"] == 0
    assert all(
        _summary(counts)["fields"][field]
        == {
            "correct": 0,
            "scored": 0,
            "eligible": 0,
            "correct_gold": 0,
            "precision": None,
            "recall": None,
        }
        for field in ("case_name", "court", "date")
    )


def test_new_judgment_overrides_prior_rule_match_for_field_precision() -> None:
    before = _lookup(full_name="Bell Atlantic Corporation v. Twombly")
    assert before.roots[0].case_name_judgments[-1].result.value == "match"
    root = before.roots[0].record("test_route").with_identity_judgment(IdentityVerdict.DEFERRED, STAGE)
    routed = before.replace_citation(root).complete("test_route")
    after = asyncio.run(
        reporter_root_lookup_unique_llm(routed, reviewer=_Reviewer(case_name_result="mismatch"))
    )

    counts, details = score_document(after, (_gold(after, case_name_label="disagrees"),))

    assert _summary(counts)["fields"]["case_name"]["correct"] == 1
    case_name = next(item for item in details if item["field"] == "case_name")
    assert case_name["result"] == "mismatch"
    assert case_name["gold_label"] == "disagrees"
    assert case_name["outcome"] == "correct"


def test_field_judgment_uses_root_label_across_reading_and_occurrence_changes() -> None:
    before = _lookup()
    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=_Reviewer()))
    gold = _gold(after)
    wrong_reading = {**gold, "case_name": {"start": 0, "end": 9, "quote": "Bell Atl."}}
    counts, details = score_document(after, (wrong_reading,))
    assert _summary(counts)["fields"]["case_name"] == {
        "correct": 1,
        "scored": 1,
        "eligible": 1,
        "correct_gold": 1,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert next(item for item in details if item["field"] == "case_name")["outcome"] == "correct"


def test_failed_review_has_per_field_gaps_and_recall_misses() -> None:
    before = _lookup()
    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=_FailedReviewer()))

    counts, details = score_document(after, (_gold(after),))

    assert all(
        _summary(counts)["fields"][field]
        == {
            "correct": 0,
            "scored": 0,
            "eligible": 1,
            "correct_gold": 0,
            "precision": None,
            "recall": 0.0,
        }
        for field in ("case_name", "court", "date")
    )
    assert [item["field"] for item in details] == ["case_name", "court", "date"]
    assert all(item["product"] == "field_gap" and item["outcome"] == "review_failed" for item in details)
    assert all(item["review_failure_reason"] == "The review could not be completed" for item in details)


def test_cumulative_cli_writes_unique_stage_field_scores_and_occurrences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _lookup()
    after = asyncio.run(reporter_root_lookup_unique_llm(before, reviewer=_Reviewer()))
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    filename = "sample.txt"
    digest = hashlib.sha256(after.text.encode("utf-8")).hexdigest()
    manifest_path = data_root / "primary" / "documents.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"documents": {filename: {"sha256": digest, "length": len(after.text)}}}),
        encoding="utf-8",
    )
    annotation_path = data_root / "primary" / "documents" / "sample.jsonl"
    annotation_path.parent.mkdir(parents=True)
    header = {"unit": "header", "document": filename, "text": {"sha256": digest}}
    gold = {"unit": "citation", **_gold(after)}
    annotation_path.write_text("\n".join(json.dumps(row) for row in (header, gold)) + "\n", encoding="utf-8")
    artifact_path = run_dir / "documents" / "primary" / "sample.txt.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(after.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_run",
            "--data-root",
            str(data_root),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    evaluate_run.main()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    occurrences = json.loads((output_dir / "occurrences.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert summary["stage_order"][-1] == STAGE
    unique = summary["stages"][STAGE]
    assert set(unique["totals"]) == {"case_name", "court", "date"}
    for field in ("case_name", "court", "date"):
        assert unique["totals"][field] == {"precision": 1.0, "recall": 1.0}
    assert [item["field"] for item in occurrences[STAGE]["primary/sample.txt"]] == [
        "case_name",
        "court",
        "date",
    ]
    assert f"| {STAGE} | primary | case name | 100.0% | 100.0% |" in report
    assert f"| {STAGE} | primary | court | 100.0% | 100.0% |" in report
    assert f"| {STAGE} | primary | date | 100.0% | 100.0% |" in report
    assert "admission" not in report.lower()
    assert "decision precision" not in report.lower()
