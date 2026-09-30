"""Checkpoint-resume behavior for the leaf workflow runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from evaluations import run_grow_leaves as runner
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations
from mellea_lrc.extraction.supra_citations import find_supra_citations


def test_runner_resumes_from_stage_30_without_replaying_discovery_or_calling_models(
    tmp_path: Path, monkeypatch
) -> None:
    text = "Smith v. Jones, 347 U.S. 483 (1954). See Smith, 347 U.S. at 495. Id. at 496."
    source = tmp_path / "filing.txt"
    source.write_text(text, encoding="utf-8")
    document = asyncio.run(grow_roots(Document.from_source(source)))
    document = find_short_reporter_citations(document)
    document = find_supra_citations(document)
    document = find_id_citations(document)
    assert document.stage_runs[-1] == "30_id_citations"

    input_run = tmp_path / "input-run"
    input_documents = input_run / "documents"
    input_documents.mkdir(parents=True)
    filename = "filing.txt"
    serialized = document.model_dump_json()
    (input_documents / f"{filename}.json").write_text(serialized, encoding="utf-8")
    (input_run / "run.json").write_text(
        json.dumps({"set": "primary", "status": "complete", "filings": [filename]}),
        encoding="utf-8",
    )

    calls = []
    stages = []
    for name, _ in runner._STAGES:
        if name.endswith("_llm"):
            async def fake_model_stage(current, stage=name):
                calls.append(stage)
                return current.complete(stage)

            stages.append((name, fake_model_stage))
        else:
            def fake_rule_stage(current, stage=name):
                calls.append(stage)
                return current.complete(stage)

            stages.append((name, fake_rule_stage))
    monkeypatch.setattr(runner, "_STAGES", tuple(stages))
    monkeypatch.setattr(runner, "_RESULTS_ROOT", tmp_path / "runs")

    result = asyncio.run(runner.run(input_documents, input_stage="30_id_citations"))

    expected = [name for name, _ in stages if name not in {
        "28_short_reporter_citations",
        "29_supra_citations",
        "30_id_citations",
    }]
    assert calls == expected
    assert not {"28_short_reporter_citations", "29_supra_citations", "30_id_citations"} & set(calls)
    output = Document.model_validate_json(
        (result / "documents" / f"{filename}.json").read_text(encoding="utf-8")
    )
    assert output.stage_runs[-len(expected) :] == tuple(expected)
    record = json.loads((result / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["input_sha256"][filename] == hashlib.sha256(serialized.encode()).hexdigest()
