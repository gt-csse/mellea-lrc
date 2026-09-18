"""The locator-checkpoint run artifact is a recoverable stage chain."""

from __future__ import annotations

import asyncio
import json

from evaluations.extraction.run_locator_checkpoint_chain import (
    CHECKPOINTS,
    _read_artifact,
    _stored_document,
    run_document,
)
from mellea_lrc.core.citations import FullCaseCitation


def test_run_document_persists_four_cumulative_documents_and_can_resume(tmp_path) -> None:
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
        if isinstance(record.stated, FullCaseCitation)
    )
    assert stages[2].nodes[0].outcome == "not_run"  # type: ignore[union-attr]

    resumed = asyncio.run(run_document(text_path, data=data, artifact_path=artifact_path, resume=True))
    assert resumed == artifact
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(payload["checkpoints"]) == set(CHECKPOINTS)
