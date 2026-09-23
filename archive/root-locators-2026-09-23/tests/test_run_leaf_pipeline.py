"""The leaf runner resumes a saved root checkpoint without retrieval."""

from __future__ import annotations

import asyncio
import json

from mellea_lrc.api import Document, grow_roots
from scripts.run_leaf_pipeline import run


def test_leaf_runner_persists_each_selected_stage(tmp_path) -> None:
    roots = asyncio.run(grow_roots(Document.from_plain_text("A filing without citations.")))
    checkpoint = tmp_path / "roots.json"
    checkpoint.write_text(json.dumps(roots.model_dump(mode="json")), encoding="utf-8")

    result = asyncio.run(
        run(
            checkpoint,
            artifacts=tmp_path / "artifacts",
            hunt_leaf_names=True,
            validate_pin_sites=True,
        )
    )

    assert result.passes[-3:] == (
        "leaf_growth",
        "leaf_case_name_hunting",
        "pin_cite_site_validation",
    )
    for stage in result.passes[-3:]:
        (artifact,) = (tmp_path / "artifacts" / stage).glob("*.json")
        restored = Document.model_validate(json.loads(artifact.read_text(encoding="utf-8")))
        assert stage in restored.passes
    assert roots.passes[-1] == "root_formation"
