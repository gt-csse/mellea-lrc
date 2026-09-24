"""The live runner checkpoints predictions before reading annotation gold."""

import asyncio
import hashlib
import json
from pathlib import Path

from mellea_lrc.extraction.docket_hunting import STAGE, DocketSiteDecision
from mellea_lrc.model import Document, Span
from scripts import run_docket_site_hunt as runner


def test_runner_saves_native_document_then_scores_and_resumes(tmp_path: Path, monkeypatch) -> None:
    source = (
        "INDEX: Case No. 035547/2021\n"
        "See Case No. 1:24-cv-00123. Later No. 19 Civ. 8034; then No. 20 Civ. 9000; Misc 77/4."
    )
    data_root = tmp_path / "data"
    root = data_root / "primary"
    text_dir = root / "documents_txt"
    annotations_dir = root / "documents"
    text_dir.mkdir(parents=True)
    annotations_dir.mkdir()
    filename = "001__example.txt"
    (text_dir / filename).write_text(source, encoding="utf-8")
    # This unlisted text must never enter the selection.
    (text_dir / "001__example (before clean).txt").write_text("No. 88 Civ. 9999", encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    index_end = source.index("\n")
    (root / "documents.json").write_text(
        json.dumps(
            {
                "documents": {
                    filename: {
                        "sha256": digest,
                        "length": len(source),
                        "backend": "docling",
                        "index_spans": [{"start": 0, "end": index_end}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def docket_row(quote: str, *, parallel_wl: bool = False) -> dict[str, object]:
        start = source.index(quote)
        return {
            "unit": "citation",
            "kind": "DocketCitation",
            "identifier": None if parallel_wl else {"docket_number": quote},
            "locator": {"start": start, "end": start + len(quote), "quote": quote},
        }

    rows = [
        {"unit": "header", "document": filename, "text": {"sha256": digest}},
        docket_row("Case No. 035547/2021"),
        docket_row("Case No. 1:24-cv-00123"),
        docket_row("No. 19 Civ. 8034", parallel_wl=True),
        docket_row("77/4"),
    ]
    (annotations_dir / "001__example.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    items = runner.select_documents(data_root, ("primary",), ("001",))
    assert [(name, namefile) for name, namefile, _ in items] == [("primary", filename)]
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "documents" / "primary" / f"{filename}.json"
    original_gold_spans = runner._gold_spans

    def guarded_gold(*args, **kwargs):
        assert checkpoint.exists()
        saved = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
        assert STAGE in saved.stage_runs
        return original_gold_spans(*args, **kwargs)

    monkeypatch.setattr(runner, "_gold_spans", guarded_gold)
    reviewed: list[str] = []

    async def reviewer(site):
        reviewed.append(site.locator_text)
        return DocketSiteDecision(
            is_docket_citation=True,
            locator=site.locator_text,
            docket_number=site.docket_number,
            reason="Test reviewer accepted the source span.",
        )

    first = asyncio.run(runner.evaluate(data_root, output_dir, items, reviewer=reviewer))
    assert reviewed == ["No. 19 Civ. 8034", "No. 20 Civ. 9000"]
    assert first["status"] == "complete"
    assert first["resumed_documents"] == 0
    assert first["totals"] == {
        "documents": 1,
        "gold_docket_locators": 3,
        "rule_locators": 1,
        "rule_exact": 1,
        "reviewed_sites": 2,
        "hunt_admissions": 2,
        "hunt_exact": 1,
        "hunt_nonmatching": 1,
        "hunt_declined": 0,
        "hunt_failed": 0,
        "combined_locators": 3,
        "combined_exact": 2,
        "combined_nonmatching": 1,
        "remaining_gold_misses": 1,
    }
    saved = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
    assert saved.index_spans == (Span(0, index_end),)
    assert saved == Document.model_validate_json(saved.model_dump_json())
    per_document = json.loads(
        (output_dir / "summaries" / "primary" / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert per_document["counts"] == first["totals"]

    async def forbidden_reviewer(_site):
        raise AssertionError("Resume must not make review calls")

    second = asyncio.run(runner.evaluate(data_root, output_dir, items, reviewer=forbidden_reviewer))
    assert second["resumed_documents"] == 1
    assert second["totals"] == first["totals"]
