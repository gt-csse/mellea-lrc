"""The locator-checkpoint run artifact is a recoverable stage chain."""

from __future__ import annotations

import asyncio
import json

from evaluations.extraction.run_locator_checkpoint_chain import (
    CHECKPOINTS,
    Corpus,
    _colocation_score,
    _read_artifact,
    _stored_document,
    run_document,
)
from mellea_lrc.model.citations import FullCaseCitation


def test_run_document_persists_locator_stages_and_colocation_then_can_resume(tmp_path) -> None:
    data = tmp_path / "data"
    text_path = data / "primary" / "documents_txt" / "example.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("Ashcroft v. Iqbal, 556 U.S. 662 (2009).", encoding="utf-8")
    artifact_path = tmp_path / "run" / "documents" / "example.json"

    artifact = asyncio.run(run_document(text_path, data=data, artifact_path=artifact_path, resume=True))

    assert tuple(artifact["checkpoints"]) == CHECKPOINTS
    saved = _read_artifact(artifact_path)
    assert saved is not None
    stages = [_stored_document(saved, stage) for stage in CHECKPOINTS]
    assert all(stage is not None for stage in stages)
    assert [stage.passes[-1] for stage in stages] == list(CHECKPOINTS)  # type: ignore[union-attr]
    assert all(
        any(node.stage == "full_reporter_locator_rule" for node in record.trace)
        for record in stages[0].citations  # type: ignore[union-attr]
        if isinstance(record.fields, FullCaseCitation)
    )
    assert stages[2].nodes[0].outcome == "not_run"  # type: ignore[union-attr]
    assert stages[-1].colocations == ()  # type: ignore[union-attr]

    resumed = asyncio.run(run_document(text_path, data=data, artifact_path=artifact_path, resume=True))
    assert resumed == artifact
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload["checkpoints"]) == set(CHECKPOINTS)


def test_colocation_score_reads_the_persisted_layer(tmp_path) -> None:
    text = "St. Amant v. Thompson, 390 U.S. 727, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."
    data = tmp_path / "data"
    text_path = data / "primary" / "documents_txt" / "parallel.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(text, encoding="utf-8")
    artifact_path = tmp_path / "run" / "documents" / "parallel.json"
    artifact = asyncio.run(run_document(text_path, data=data, artifact_path=artifact_path, resume=True))

    locators = ("390 U.S. 727", "88 S.Ct. 1323", "20 L.Ed.2d 262")
    annotations = data / "primary" / "documents"
    annotations.mkdir()
    rows = [
        {"unit": "header", "document": "parallel.txt"},
        *[
            {
                "unit": "citation",
                "kind": "FullCaseCitation",
                "colocation_id": "parallel",
                "locator": {"start": text.index(locator), "end": text.index(locator) + len(locator)},
            }
            for locator in locators
        ],
    ]
    (annotations / "parallel.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    score = _colocation_score(
        [artifact], Corpus(name="primary", texts=text_path.parent, annotations=annotations)
    )
    assert (score["gold"], score["predicted"], score["tp"], score["fp"], score["fn"]) == (1, 1, 1, 0, 0)
