"""Ambiguous model scores use only selected, aligned stage judgments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluations import evaluate_run
from evaluations import run_reporter_root_lookup_ambiguous_llm as review_runner
from evaluations.render_reporter_lookup_ambiguous_llm import render_report
from evaluations.score_reporter_lookup_ambiguous_llm import _summary, score_document
from mellea_lrc.api import Document, grow_roots, reporter_root_lookup, reporter_root_lookup_ambiguous
from mellea_lrc.courtlistener import CourtListenerCitationLookup
from mellea_lrc.model.citations.reporter_lookup import ReporterAmbiguousReviewDecision
from mellea_lrc.validation._support.reporter_ambiguous_llm import ReporterAmbiguousReviewOutcome
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm import STAGE, reporter_root_lookup_ambiguous_llm

SOURCE = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."


class _LookupClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": f"{volume} {reporter} {page}",
                "status": 300,
                "clusters": [
                    {
                        "id": index,
                        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                    for index in (11, 22)
                ],
            }
        )


class _Reviewer:
    def __init__(self, selected: int | None = 1, *, failed: bool = False) -> None:
        self.selected = selected
        self.failed = failed

    async def __call__(
        self, _context: object
    ) -> ReporterAmbiguousReviewDecision | ReporterAmbiguousReviewOutcome:
        if self.failed:
            return ReporterAmbiguousReviewOutcome(
                decision=None, failure_reason="The review could not be completed"
            )
        result = "match" if self.selected is not None else "undetermined"
        return ReporterAmbiguousReviewDecision.model_validate(
            {
                "selected_candidate_index": self.selected,
                **{
                    field: {
                        "propose_replacement": False,
                        "quote": None,
                        "result": result,
                        "reason": "Compared with the saved candidate.",
                    }
                    for field in ("case_name", "court", "date")
                },
                "case_name": {
                    "propose_replacement": False,
                    "quote": None,
                    "normalized": {
                        "kind": "adversarial",
                        "plaintiff": "Bell Atl. Corp.",
                        "defendant": "Twombly",
                    },
                    "result": result,
                    "reason": "Compared with the saved candidate.",
                },
                "reason": "The candidate evidence supports this selection."
                if self.selected is not None
                else "No candidate can be selected.",
            }
        )


def _before(source: str | Path = SOURCE) -> Document:
    client = _LookupClient()
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    return reporter_root_lookup_ambiguous(reporter_root_lookup(roots, client=client), client=client)


def _gold(
    document: Document,
    *,
    case_name_label: str = "agrees",
    cluster_ids: tuple[str | None, ...] = ("22",),
) -> dict:
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
        "case_name": {"start": case_name.span.start, "end": case_name.span.end, "quote": case_name.quote},
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
                "evidence": [{"source": {"kind": "cluster", "id": cluster_id}} for cluster_id in cluster_ids],
                "fields": {
                    "case_name": {"label": case_name_label},
                    "court": {"label": "agrees"},
                    "date": {"label": "agrees"},
                },
            }
        },
    }


def test_selected_candidate_field_precision_and_occurrence_details() -> None:
    before = _before()
    assert before.roots[0].identity_judgments[-1].next_stage == STAGE
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=_Reviewer()))

    counts, details = score_document(after, (_gold(after),))

    assert counts["routed_roots"] == 1
    for field in ("case_name", "court", "date"):
        assert _summary(counts)["field_precision"][field] == {"correct": 1, "scored": 1, "value": 1.0}
    assert [item["field"] for item in details] == ["case_name", "court", "date"]
    assert all(item["selected_candidate_index"] == 1 for item in details)
    assert all(str(item["selected_cluster_id"]) == "22" for item in details)
    assert all(item["outcome"] == "correct" for item in details)


def test_selected_candidate_is_scored_without_cluster_evidence_gate() -> None:
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(_before(), reviewer=_Reviewer(0)))

    counts, details = score_document(after, (_gold(after, cluster_ids=("22",)),))

    assert all(
        _summary(counts)["field_precision"][field] == {"correct": 1, "scored": 1, "value": 1.0}
        for field in ("case_name", "court", "date")
    )
    assert [item["outcome"] for item in details] == ["correct"] * 3


def test_missing_or_null_cluster_evidence_does_not_block_field_scoring() -> None:
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(_before(), reviewer=_Reviewer()))

    for evidence_ids in ((), (None,)):
        counts, details = score_document(after, (_gold(after, cluster_ids=evidence_ids),))
        assert all(
            _summary(counts)["field_precision"][field]["scored"] == 1
            for field in ("case_name", "court", "date")
        )
        assert [item["outcome"] for item in details] == ["correct"] * 3


def test_canonical_root_supplies_cluster_evidence_by_root_id() -> None:
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(_before(), reviewer=_Reviewer()))
    canonical = _gold(after)
    occurrence_without_evidence = {
        **canonical,
        "validation": {
            "identity": {
                **canonical["validation"]["identity"],
                "evidence": [],
            }
        },
    }

    counts, details = score_document(after, (occurrence_without_evidence,), identity_roots=(canonical,))

    assert all(
        _summary(counts)["field_precision"][field]["scored"] == 1 for field in ("case_name", "court", "date")
    )
    assert all(item["outcome"] == "correct" for item in details)


def test_unselected_and_failed_reviews_save_per_field_gaps() -> None:
    before = _before()
    for reviewer, outcome in (
        (_Reviewer(None), "no_candidate_selected"),
        (_Reviewer(failed=True), "review_failed"),
    ):
        after = asyncio.run(reporter_root_lookup_ambiguous_llm(before, reviewer=reviewer))
        counts, details = score_document(after, (_gold(after),))
        assert all(
            _summary(counts)["field_precision"][field]["scored"] == 0
            for field in ("case_name", "court", "date")
        )
        assert [item["outcome"] for item in details] == [outcome] * 3
        assert all(item["selected_candidate_index"] is None for item in details)
        assert all(item["product"] == "field_gap" for item in details)
        if outcome == "review_failed":
            assert all(
                item["review_failure_reason"] == "The review could not be completed" for item in details
            )
        else:
            assert all(
                item["selection_exclusion_reason"] == "No candidate can be selected." for item in details
            )


def test_misaligned_root_reading_still_uses_canonical_field_label() -> None:
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(_before(), reviewer=_Reviewer()))
    gold = _gold(after)
    wrong_reading = {**gold, "case_name": {"start": 0, "end": 9, "quote": "Bell Atl."}}

    counts, details = score_document(after, (wrong_reading,))

    assert _summary(counts)["field_precision"]["case_name"]["scored"] == 1
    assert next(item for item in details if item["field"] == "case_name")["outcome"] == "correct"
    assert _summary(counts)["field_precision"]["court"]["scored"] == 1


def test_cumulative_cli_writes_only_field_precision_for_ambiguous_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    after = asyncio.run(reporter_root_lookup_ambiguous_llm(_before(), reviewer=_Reviewer()))
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
    annotation_path.write_text(
        "\n".join(json.dumps(row) for row in (header, {"unit": "citation", **_gold(after)})) + "\n",
        encoding="utf-8",
    )
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
    ambiguous_llm = summary["stages"][STAGE]
    assert set(ambiguous_llm["totals"]) == {"field_precision"}
    assert all(
        ambiguous_llm["totals"]["field_precision"][field] == {"correct": 1, "scored": 1, "value": 1.0}
        for field in ("case_name", "court", "date")
    )
    assert len(occurrences[STAGE]["primary/sample.txt"]) == 3
    section = report.split(f"## Field judgment precision at `{STAGE}`", 1)[1]
    assert "| primary | 1/1 (100.0%) | 1/1 (100.0%) | 1/1 (100.0%) |" in section
    assert "admission" not in section.lower()
    assert "decision precision" not in section.lower()


def test_review_runner_resumes_from_ambiguous_rule_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    source_path = data_root / "primary" / "documents_txt" / "sample.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(SOURCE, encoding="utf-8")
    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    (data_root / "primary" / "documents.json").write_text(
        json.dumps({"documents": {"sample.txt": {"sha256": digest, "length": len(SOURCE)}}}),
        encoding="utf-8",
    )
    before = _before(source_path)
    input_dir = tmp_path / "ambiguity"
    output_dir = tmp_path / "review"
    input_path = input_dir / "documents" / "primary" / "sample.txt.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(before.model_dump_json(), encoding="utf-8")
    (input_dir / "run.json").write_text(
        json.dumps({"source_data_root": str(data_root.resolve()), "checkpoints": list(before.stage_runs)}),
        encoding="utf-8",
    )

    async def fake_review(document: Document) -> Document:
        return document.complete(STAGE)

    monkeypatch.setattr(review_runner, "reporter_root_lookup_ambiguous_llm", fake_review)
    assert asyncio.run(review_runner.run_documents(data_root, input_dir, output_dir)) == {
        "review_created": 1,
        "review_reused": 0,
    }
    saved = Document.model_validate_json(
        (output_dir / "documents" / "primary" / "sample.txt.json").read_text(encoding="utf-8")
    )
    assert saved.get_stage(before.stage_runs[-1]) == before

    async def reject_repeat(_document: Document) -> Document:
        raise AssertionError("Completed review must not run again")

    monkeypatch.setattr(review_runner, "reporter_root_lookup_ambiguous_llm", reject_repeat)
    assert asyncio.run(review_runner.run_documents(data_root, input_dir, output_dir)) == {
        "review_created": 0,
        "review_reused": 1,
    }


def test_report_shows_precision_with_denominators_only() -> None:
    fields = {
        "case_name": {"value": 1.0, "correct": 3, "scored": 3},
        "court": {"value": None, "correct": 0, "scored": 0},
        "date": {"value": 0.5, "correct": 1, "scored": 2},
    }
    report = render_report(
        {
            "stage": STAGE,
            "sets": {"primary": {"field_precision": fields}},
            "totals": {"field_precision": fields},
        },
        source_label="saved/summary.json",
    )

    assert "| primary | 3/3 (100.0%) | 0/0 (—) | 1/2 (50.0%) |" in report
    assert "admission" not in report.lower()
    assert "decision precision" not in report.lower()
