"""Exercise the native root workflow at its public checkpoint boundaries."""

import pytest

from mellea_lrc.config.extraction import ExtractionRules
from mellea_lrc.extraction import find_full_reporter_locators, resolve_colocations
from mellea_lrc.model import Document, PreprocessedDocument, PreprocessingMetadata, SourceMetadata
from mellea_lrc.model.citations import latest
from mellea_lrc.model.execution import GROW_ROOTS_RUNS
from mellea_lrc.workflows import grow_roots
from mellea_lrc.workflows.grow_roots import discover_root_locators, form_root_groups, read_root_fields


def _document(text: str) -> Document:
    return Document.from_preprocessed(
        PreprocessedDocument(
            text=text,
            source_metadata=SourceMetadata(path="filing.txt"),
            preprocessing_metadata=PreprocessingMetadata(),
        )
    )


def test_grow_roots_composes_existing_writers_and_replays_each_checkpoint():
    initial = _document(
        "Alpha v. Beta, 123 F.3d 456, 459 (2d Cir. 2000). Later, 123 F.3d 456. "
        "Smith v. Jones, No. 1:24-cv-00123, ECF No. 113 (D. Ariz. 2024)."
    )
    checkpoints = []
    result = grow_roots(initial, checkpoint=checkpoints.append)
    first, second, docket = result.full_locators
    assert result.runs == GROW_ROOTS_RUNS
    assert result.stage_runs == (
        "grow_roots.discover_root_locators",
        "grow_roots.read_root_fields",
        "grow_roots.form_root_groups",
    )
    assert result.roots == (first, docket)
    assert latest(second.root_id) == first.id
    assert first.locator_span != second.locator_span
    assert first.case_name[-1].quote == "Alpha v. Beta"
    assert first.pin_cite[-1].quote == "459"
    assert docket.docket_entry[-1].quote == "ECF No. 113"
    assert docket.court[-1].quote == "D. Ariz."

    restored = Document.model_validate_json(result.model_dump_json())
    for index, (run, expected) in enumerate(zip(GROW_ROOTS_RUNS, checkpoints, strict=True)):
        replay = restored.get_stage if run.kind == "stage" else restored.get_substage
        assert replay(run.name) == expected
        resumed = []
        assert grow_roots(replay(run.name), checkpoint=resumed.append) == result
        assert resumed == checkpoints[index + 1 :]


def test_grow_roots_can_resume_legacy_discovery_and_preserve_empty_boundaries():
    initial = _document("No authorities are cited here.")
    first_step = find_full_reporter_locators(initial)
    restored = Document.model_validate_json(first_step.model_dump_json())

    assert restored.stage_runs == ()
    result = grow_roots(restored)
    assert result.runs == GROW_ROOTS_RUNS
    assert result.get_substage(first_step.substage_runs[-1]) == first_step
    assert result.citations == result.roots == ()


def test_stage_composition_passes_the_requested_field_rules():
    initial = _document("Alpha v. Beta, 123 F.3d 456, 459 (2d Cir. 2000).")
    discovered = discover_root_locators(initial)
    rules = ExtractionRules(post_locator_window=1)
    read = read_root_fields(discovered, rules)

    assert read.full_locators[0].pin_cite is None
    assert form_root_groups(read) == grow_roots(initial, rules=rules)
    with pytest.raises(ValueError, match="already completed"):
        discover_root_locators(discovered)
    with pytest.raises(ValueError, match="preceding stages"):
        form_root_groups(discovered)


def test_independent_substage_history_is_replayable_and_out_of_order_resume_is_rejected():
    initial = _document("See 123 F.3d 456.")
    reporters = find_full_reporter_locators(initial)
    independent = resolve_colocations(reporters)
    restored = Document.model_validate_json(independent.model_dump_json())
    assert restored.get_substage(reporters.substage_runs[-1]) == reporters

    callbacks = []
    with pytest.raises(ValueError, match="ordered workflow checkpoint"):
        grow_roots(restored, checkpoint=callbacks.append)
    assert callbacks == []


def test_stage_marker_requires_its_completed_substages_and_rejects_forged_history():
    initial = _document("See 123 F.3d 456.")
    reporters = find_full_reporter_locators(initial)
    discovery = "grow_roots.discover_root_locators"
    bad_marker = {"kind": "stage", "name": discovery}

    with pytest.raises(ValueError, match="grow_roots workflow order"):
        reporters.complete_stage(discovery)
    payload = reporters.model_dump(mode="json")
    payload["runs"].append(bad_marker)
    with pytest.raises(ValueError, match="grow_roots workflow order"):
        Document.model_validate(payload)
