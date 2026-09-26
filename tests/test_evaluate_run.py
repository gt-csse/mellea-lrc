"""One saved run produces cumulative stage scores and a field-level report."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

from evaluations import evaluate_run
from mellea_lrc.api import Document, grow_roots, reporter_root_exact_lookup
from mellea_lrc.courtlistener import CourtListenerCitationLookup


def _save_document(
    data_root: Path,
    run_dir: Path,
    name: str,
    document: Document,
    *,
    rows: tuple[dict, ...] = (),
) -> None:
    filename = "sample.txt"
    digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    source_dir = "documents_txt" if name == "primary" else "filings_txt"
    source_path = data_root / name / source_dir / filename
    source_path.parent.mkdir(parents=True)
    source_path.write_text(document.text, encoding="utf-8")
    (data_root / name / "documents.json").write_text(
        json.dumps({"documents": {filename: {"sha256": digest, "length": len(document.text)}}}),
        encoding="utf-8",
    )
    annotation_path = data_root / name / "documents" / "sample.jsonl"
    annotation_path.parent.mkdir(parents=True)
    header = {"unit": "header", "document": filename, "text": {"sha256": digest}}
    annotation_path.write_text("\n".join(json.dumps(row) for row in (header, *rows)) + "\n", encoding="utf-8")
    artifact_path = run_dir / "documents" / name / f"{filename}.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(document.model_dump_json(), encoding="utf-8")


def _empty_document(*stages: str) -> Document:
    document = Document.from_source("No citations appear here.")
    for stage in stages:
        document = document.complete(stage)
    return document


def test_evaluate_discovers_only_completed_stages_from_saved_document(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _save_document(
        data_root,
        run_dir,
        "primary",
        _empty_document("full_reporter_locators", "docket_locators"),
    )

    result = evaluate_run.evaluate(data_root, run_dir, ("primary",))

    assert result["sets"] == ["primary"]
    assert result["stage_order"] == ["full_reporter_locators", "docket_locators"]
    assert list(result["stages"]) == result["stage_order"]
    assert list(result["occurrences"]) == result["stage_order"]
    assert result["diagnostics"]["docket_site_proposals"]["totals"]["documents"] == 1
    for stage in result["stage_order"]:
        assert result["stages"][stage]["totals"]["documents"] == 1
        assert "occurrences" not in result["stages"][stage]
        assert result["occurrences"][stage] == {"primary/sample.txt": []}


@pytest.mark.parametrize(
    ("options", "selected"),
    [
        ([], ["primary"]),
        (["--set", "hallucination-set-1"], ["hallucination-set-1"]),
        (["--set", "primary", "--set", "hallucination-set-1"], ["primary", "hallucination-set-1"]),
    ],
)
def test_cli_selects_sets_and_writes_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], selected: list[str]
) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    for name in ("primary", "hallucination-set-1"):
        _save_document(data_root, run_dir, name, _empty_document("full_reporter_locators"))

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
            *options,
        ],
    )
    evaluate_run.main()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    occurrences = json.loads((output_dir / "occurrences.json").read_text(encoding="utf-8"))
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert summary["sets"] == selected
    assert summary["stage_order"] == ["full_reporter_locators"]
    assert set(summary["stages"]["full_reporter_locators"]["sets"]) == set(selected)
    assert set(occurrences["full_reporter_locators"]) == {f"{name}/sample.txt" for name in selected}
    assert "Full reporter locators" in report
    assert all(name in report for name in selected)


def test_cli_defaults_output_to_run_evaluation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _save_document(data_root, run_dir, "primary", _empty_document("full_reporter_locators"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_run", "--data-root", str(data_root), "--run-dir", str(run_dir)],
    )

    evaluate_run.main()

    output_dir = run_dir / "evaluation"
    assert {path.name for path in output_dir.iterdir()} == {
        "summary.json",
        "occurrences.json",
        "report.md",
    }


def test_evaluate_rejects_different_stage_sequences_across_documents(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _save_document(data_root, run_dir, "primary", _empty_document("full_reporter_locators"))
    _save_document(
        data_root,
        run_dir,
        "hallucination-set-1",
        _empty_document("full_reporter_locators", "docket_locators"),
    )

    with pytest.raises(ValueError, match="stage"):
        evaluate_run.evaluate(data_root, run_dir, ("primary", "hallucination-set-1"))


def test_evaluate_does_not_skip_manifest_document_without_saved_result(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    _save_document(data_root, run_dir, "primary", _empty_document("full_reporter_locators"))
    manifest_path = data_root / "primary" / "documents.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"]["missing.txt"] = {"sha256": "0" * 64, "length": 1}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"missing\.txt\.json"):
        evaluate_run.evaluate(data_root, run_dir, ("primary",))


class _LookupClient:
    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        assert (volume, reporter, page) == ("550", "U.S.", "544")
        return CourtListenerCitationLookup.model_validate(
            {
                "citation": "550 U.S. 544",
                "status": 200,
                "clusters": [
                    {
                        "id": 1,
                        "caseNameFull": "Bell Atlantic Corporation v. Twombly",
                        "court_id": "scotus",
                        "dateFiled": "2007-05-21",
                        "citations": [{"volume": 550, "reporter": "U.S.", "page": "544"}],
                    }
                ],
            }
        )


def test_exact_identity_field_judgments_appear_in_cumulative_json_and_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "evaluation"
    source = "Bell Atl. Corp. v. Twombly, 550 U.S. 544 (2007)."
    roots = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    document = reporter_root_exact_lookup(roots, client=_LookupClient())
    (root,) = document.roots
    case_name = root.case_name[-1]
    date = root.date[-1]
    gold = {
        "unit": "citation",
        "id": "gold-1",
        "root_id": "gold-1",
        "is_root": True,
        "kind": "FullCaseCitation",
        "identifier": {"kind": "reporter", "volume": 550, "reporter": "U.S.", "page": "544"},
        "locator": {
            "start": root.locator_span.start,
            "end": root.locator_span.end,
            "quote": source[root.locator_span.start : root.locator_span.end],
        },
        "case_name": {
            "start": case_name.span.start,
            "end": case_name.span.end,
            "quote": case_name.quote,
        },
        "court": {"id": "scotus", "how": "reporter"},
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
                    "case_name": {"label": "agrees"},
                    "court": {"label": "agrees"},
                    "date": {"label": "agrees"},
                },
            }
        },
    }
    _save_document(data_root, run_dir, "primary", document, rows=(gold,))

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
    identity = summary["stages"]["reporter_root_exact_lookup"]
    assert summary["stage_order"][-1] == "reporter_root_exact_lookup"
    for field in ("case_name", "court", "date"):
        assert identity["totals"]["fields"][field]["gold_stated"] == 1
        assert identity["totals"]["fields"][field]["correct_predictions"] == 1
    assert any(
        item["product"] == "field_judgment" and item["field"] == "court"
        for item in occurrences["reporter_root_exact_lookup"]["primary/sample.txt"]
    )
    assert "| Set | Field | Judgment precision | Judgment recall |" in report
    assert "| primary | Case name | 100.0% | 100.0% |" in report
    assert "| primary | Court | 100.0% | 100.0% |" in report
    assert "| primary | Date | 100.0% | 100.0% |" in report
    assert "Decision precision" not in report
