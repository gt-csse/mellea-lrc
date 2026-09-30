"""Case-name representation changes preserve saved citation histories."""

from __future__ import annotations

import asyncio
from mellea_lrc.api import Document, grow_roots
from mellea_lrc.model import Span
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameKind

STAGE23 = "23_locator_body_llm_judgment"


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
        restored.roots[0]
        .record("27_intended_case_llm_selection")
        .with_case_name(source, span, normalized=partial)
    )
    updated = restored.replace_citation(updated_root).complete("27_intended_case_llm_selection")
    reloaded = Document.model_validate_json(updated.model_dump_json())

    assert reloaded == updated
    assert reloaded.get_stage(STAGE23) == restored
    assert reloaded.get_stage("10_roots") == original
    assert reloaded.roots[0].case_name[-2] == historical
    assert reloaded.roots[0].case_name[-1].get_normalized() == partial
    assert reloaded.roots[0].case_name[-1].normalized_by == "model"
