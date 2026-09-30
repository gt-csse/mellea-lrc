"""Case-name representation changes preserve saved citation histories."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mellea_lrc.api import Document, grow_roots
from mellea_lrc.model import Span
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind

STAGE23 = "23_locator_body_review"
SAVED_STAGE23 = (
    Path(__file__).resolve().parents[1] / "evaluations/results/primary/2026-09-29T15-14-26Z/documents"
)


def test_saved_stage23_documents_load_with_legacy_case_name_shape() -> None:
    """The local saved run predates the optional ``partial`` value slot."""
    if not SAVED_STAGE23.is_dir():
        pytest.skip("Saved stage-23 evaluation artifacts are not present")

    paths = sorted(SAVED_STAGE23.glob("*.json"))
    assert paths
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["stage_runs"][-1] == STAGE23
        saved_readings = [
            reading for citation in saved["citations"] for reading in citation.get("case_name", [])
        ]
        assert all(
            "partial" not in reading["normalized"]
            for reading in saved_readings
            if reading["normalized"] is not None
        )

        document = Document.model_validate(saved)
        assert document.get_stage(STAGE23) == document
        loaded_readings = [
            reading for citation in document.citations for reading in getattr(citation, "case_name", ())
        ]
        assert [reading.quote for reading in loaded_readings] == [
            reading["quote"] for reading in saved_readings
        ]
        assert [reading.node_id for reading in loaded_readings] == [
            reading["node_id"] for reading in saved_readings
        ]
        assert Document.model_validate_json(document.model_dump_json()) == document


def test_failed_historical_reading_survives_later_partial_reading() -> None:
    source = "Smith v. Jones, 550 U.S. 544."
    original = asyncio.run(grow_roots(Document.from_source(source), hunt_dockets=False))
    root = original.roots[0]
    span = Span(0, len("Smith"))
    recorded = root.record(STAGE23).with_case_name(source, span)
    stage23 = original.replace_citation(recorded).complete(STAGE23)

    # Mimic a checkpoint written before the parser understood partial names.
    old_json = stage23.model_dump(mode="json")
    for citation in old_json["citations"]:
        for reading in citation.get("case_name", []):
            if reading["normalized"] is not None:
                reading["normalized"].pop("partial", None)
    saved_reading = next(
        citation["case_name"][-1] for citation in old_json["citations"] if citation["id"] == root.id
    )
    saved_reading.update(
        normalizable=False,
        normalized=None,
        normalization_error="Legacy parser could not assign a case-name kind",
    )

    restored = Document.model_validate(old_json)
    historical = restored.roots[0].case_name[-1]
    assert historical.quote == "Smith"
    assert historical.span == span
    assert historical.normalizable is False
    assert historical.normalization_error == saved_reading["normalization_error"]
    assert restored.get_stage("10_roots") == original

    partial = CaseName.from_quote("Smith")
    assert partial.kind is CaseNameKind.PARTIAL
    updated_root = (
        restored.roots[0].record("24_case_name_review").with_case_name(source, span, normalized=partial)
    )
    updated = restored.replace_citation(updated_root).complete("24_case_name_review")
    reloaded = Document.model_validate_json(updated.model_dump_json())

    assert reloaded == updated
    assert reloaded.get_stage(STAGE23) == restored
    assert reloaded.get_stage("10_roots") == original
    assert reloaded.roots[0].case_name[-2] == historical
    assert reloaded.roots[0].case_name[-1].get_normalized() == partial
    assert reloaded.roots[0].case_name[-1].normalized_by == "model"
