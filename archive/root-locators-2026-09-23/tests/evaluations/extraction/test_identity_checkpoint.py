"""Leaf evaluation consumes the whole native root-identity checkpoint."""

import pytest

from evaluations.extraction.identity import settled
from mellea_lrc.api import Document


def test_settled_resumes_complete_document(tmp_path) -> None:
    original = Document.from_plain_text("A citation.")
    saved = original.evolve(passes=("root_formation", "root_identity"))
    artifact = tmp_path / "identity.json"
    artifact.write_text(saved.model_dump_json(), encoding="utf-8")

    assert settled(original, artifact) == saved

    with pytest.raises(ValueError, match="different document text"):
        settled(Document.from_plain_text("Another filing."), artifact)
