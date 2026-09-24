"""The live runner checkpoints predictions without reading annotation gold."""

import asyncio
import hashlib
import json
from pathlib import Path

from mellea_lrc.extraction.docket_hunting import STAGE, DocketSiteDecision
from mellea_lrc.model import Document, Span
from scripts import run_docket_site_hunt as runner


def test_runner_saves_native_document_without_gold_and_resumes(tmp_path: Path) -> None:
    source = (
        "INDEX: Case No. 035547/2021\n"
        "See Case No. 1:24-cv-00123. Later No. 19 Civ. 8034; then No. 20 Civ. 9000; Misc 77/4."
    )
    data_root = tmp_path / "data"
    root = data_root / "primary"
    text_dir = root / "documents_txt"
    text_dir.mkdir(parents=True)
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

    items = runner.select_documents(data_root, ("primary",), ("001",))
    assert [(name, namefile) for name, namefile, _ in items] == [("primary", filename)]
    output_dir = tmp_path / "run"
    checkpoint = output_dir / "documents" / "primary" / f"{filename}.json"
    # There is deliberately no annotation directory. A runner that tries to
    # score gold here will fail instead of producing this checkpoint.
    assert not (root / "documents").exists()
    reviewed: list[str] = []

    async def reviewer(site):
        reviewed.append(site.locator_text)
        return DocketSiteDecision(
            is_docket_citation=True,
            locator=site.locator_text,
            docket_number=site.docket_number,
            reason="Test reviewer accepted the source span.",
        )

    first = asyncio.run(runner.run(data_root, output_dir, items, reviewer=reviewer))
    assert reviewed == ["No. 19 Civ. 8034", "No. 20 Civ. 9000"]
    assert first["status"] == "complete"
    assert first["resumed_documents"] == 0
    assert first["totals"] == {
        "documents": 1,
        "reviewed_sites": 2,
        "hunt_admissions": 2,
        "hunt_declined": 0,
        "hunt_failed": 0,
    }
    saved = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
    assert STAGE in saved.stage_runs
    assert saved.index_spans == (Span(0, index_end),)
    assert saved == Document.model_validate_json(saved.model_dump_json())
    per_document = json.loads(
        (output_dir / "summaries" / "primary" / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert per_document["counts"] == first["totals"]

    async def forbidden_reviewer(_site):
        raise AssertionError("Resume must not make review calls")

    second = asyncio.run(runner.run(data_root, output_dir, items, reviewer=forbidden_reviewer))
    assert second["resumed_documents"] == 1
    assert second["totals"] == first["totals"]
