"""One optional inference checkpoint feeds independently configured field readers."""

import asyncio
import json
from importlib import import_module

import pytest
from pydantic import ValidationError

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.predict_root_fields_ml import SUBSTAGE as ROOT_ML
from mellea_lrc.extraction.predict_short_reporter_fields_ml import SUBSTAGE as SHORT_ML
from mellea_lrc.extraction.resolve_case_names import resolve_case_names
from mellea_lrc.extraction.resolve_courts import resolve_courts
from mellea_lrc.extraction.resolve_dates import resolve_dates
from mellea_lrc.extraction.resolve_pin_cites import resolve_pin_cites
from mellea_lrc.extraction.resolve_short_reporter_case_names import resolve_short_reporter_case_names
from mellea_lrc.field_extraction.models import MLFieldPrediction
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span
from mellea_lrc.workflows import validate_workflow_checkpoint
from mellea_lrc.workflows.grow_leaves import grow_leaves
from mellea_lrc.workflows.grow_leaves.grow_short_reporter_leaves import grow_short_reporter_leaves
from mellea_lrc.workflows.grow_roots import grow_roots
from mellea_lrc.workflows.grow_roots.read_root_fields import read_root_fields

SOURCE = "Alpha v. Beta, 123 F.3d 456, 460 (2d Cir. 1997). See Alpha, 123 F.3d at 462."
RULE = FieldExtractionConfig(ml_enabled=False, case_name="rule", court="rule", date="rule", pin_cite="rule")
ML = FieldExtractionConfig(
    ml_enabled=True,
    case_name="ml",
    court="ml",
    date="ml",
    pin_cite="ml",
    model_id="test/joint-bert",
    revision="a" * 40,
    variant="bert_augmented",
    device="cpu",
    context_before=256,
    context_after=256,
    batch_size=8,
)
ROOT_CASE_NAMES = "grow_roots.read_root_fields.resolve_case_names"
ROOT_COURTS = "grow_roots.read_root_fields.resolve_courts"
ROOT_DATES = "grow_roots.read_root_fields.resolve_dates"
ROOT_PIN_CITES = "grow_roots.read_root_fields.resolve_pin_cites"
ROOT_FIELDS = (ROOT_CASE_NAMES, ROOT_COURTS, ROOT_DATES, ROOT_PIN_CITES)
SHORT_CASE_NAMES = "grow_leaves.grow_short_reporter_leaves.resolve_case_names"
SHORT_STAGE = "grow_leaves.grow_short_reporter_leaves"
OMITTED_ROOT_OPTIONS = (
    "grow_roots.discover_root_locators.hunt_docket_locators",
    "grow_roots.form_root_groups.reassign_docket_roots_llm",
)


class FakeReader:
    def __init__(self):
        self.calls = []

    def predict_many(self, source, locator_spans):
        self.calls.append((source, locator_spans))
        predictions = []
        for locator in locator_spans:
            short = " at " in source[locator.start : locator.end]
            name = "Alpha" if short else "Alpha v. Beta"
            name_start = source.rfind(name, 0, locator.start)
            fields = {"case_name": Span(name_start, name_start + len(name))}
            # Deliberately different from the root's written pin to prove each
            # reader chooses its configured source rather than the raw cache.
            for field, quote in (("court", "2d Cir."), ("date", "1997"), ("pin_cite", "462")):
                start = source.index(quote)
                fields[field] = Span(start, start + len(quote))
            predictions.append(
                MLFieldPrediction(
                    model_id=ML.model_id,
                    revision=ML.revision,
                    variant=ML.variant,
                    context_span=Span(0, len(source)),
                    **fields,
                )
            )
        return tuple(predictions)


@pytest.fixture
def reader(monkeypatch):
    fake = FakeReader()
    for module_name in (
        "mellea_lrc.field_extraction.ml_cite",
        "mellea_lrc.extraction.predict_root_fields_ml",
        "mellea_lrc.extraction.predict_short_reporter_fields_ml",
    ):
        monkeypatch.setattr(import_module(module_name), "get_ml_reader", lambda config: fake)
    return fake


def root_run(document, config, snapshots=None):
    return asyncio.run(
        grow_roots(
            document,
            field_extraction=config,
            checkpoint=None if snapshots is None else snapshots.append,
        )
    )


def short_run(document, config, snapshots=None):
    return asyncio.run(
        grow_short_reporter_leaves(
            document,
            field_extraction=config,
            review_leaves=False,
            checkpoint=None if snapshots is None else snapshots.append,
        )
    )


def reload(document):
    return Document.model_validate_json(document.model_dump_json())


def test_inference_and_field_writes_have_separate_recoverable_checkpoints(reader):
    root_snapshots, short_snapshots = [], []
    roots = root_run(Document.from_source(SOURCE), ML, root_snapshots)
    result = short_run(roots, ML, short_snapshots)
    assert [len(locators) for _, locators in reader.calls] == [1, 1]
    inference = roots.get_substage(ROOT_ML)
    inferred = inference.full_locators[0]
    assert inferred.ml_field_reading is not None
    assert inferred.case_name == inferred.court == inferred.date == ()
    assert inferred.pin_cite is None
    root = result.roots[0]
    assert root.get_ml_field_prediction() == inferred.get_ml_field_prediction()
    for field, substage in zip(("case_name", "court", "date", "pin_cite"), ROOT_FIELDS, strict=True):
        field_node = next(node for node in root.nodes if node.substage == substage)
        assert getattr(root, field)[-1].node_id == field_node.id
        assert field_node.id != root.ml_field_reading.node_id
    assert root.case_name[-1].quote == "Alpha v. Beta"
    assert root.court[-1].get_normalized().id == "ca2"
    assert root.date[-1].get_normalized().year == 1997
    assert root.pin_cite[-1].get_normalized()[0].first == 462
    assert set(ROOT_FIELDS).issubset(result.substage_runs)
    assert SHORT_CASE_NAMES in result.substage_runs
    short = result.short_reporters[0]
    assert short.case_name[-1].quote == "Alpha"
    assert short.case_name[-1].node_id != short.ml_field_reading.node_id
    assert short.pin_cite[-1].quote == "462"
    assert short.pin_cite[-1].node_id == short.nodes[0].id
    assert not hasattr(short, "court") and not hasattr(short, "date")
    restored = reload(result)
    for snapshot in (*root_snapshots, *short_snapshots):
        event = snapshot.runs[-1]
        recovered = (
            restored.get_stage(event.name) if event.kind == "stage" else restored.get_substage(event.name)
        )
        assert recovered.model_dump_json().encode() == snapshot.model_dump_json().encode()


def test_each_partial_ml_checkpoint_resumes_without_repeating_inference(reader):
    root_snapshots, short_snapshots = [], []
    roots = root_run(Document.from_source(SOURCE), ML, root_snapshots)
    expected = short_run(roots, ML, short_snapshots)
    for snapshot in root_snapshots:
        reader.calls.clear()
        assert root_run(reload(snapshot), ML) == roots
        assert len(reader.calls) == (0 if ROOT_ML in snapshot.substage_runs else 1)
    for snapshot in short_snapshots:
        if SHORT_STAGE in snapshot.stage_runs:
            continue
        reader.calls.clear()
        assert short_run(reload(snapshot), ML) == expected
        assert len(reader.calls) == (0 if SHORT_ML in snapshot.substage_runs else 1)


def test_rule_workflows_run_all_field_checkpoints_without_loading_ml(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Rule extraction must not load or call the ML runtime")

    for module_name in (
        "mellea_lrc.field_extraction.ml_cite",
        "mellea_lrc.extraction.predict_root_fields_ml",
        "mellea_lrc.extraction.predict_short_reporter_fields_ml",
    ):
        monkeypatch.setattr(import_module(module_name), "get_ml_reader", unexpected)
    monkeypatch.setattr(import_module("mellea_lrc.field_extraction.ml_cite"), "_load_runtime", unexpected)
    roots = root_run(Document.from_source(SOURCE), RULE)
    result = short_run(roots, RULE)
    assert set(ROOT_FIELDS).issubset(result.substage_runs)
    assert SHORT_CASE_NAMES in result.substage_runs
    assert ROOT_ML not in result.substage_runs and SHORT_ML not in result.substage_runs
    assert all(citation.ml_field_reading is None for citation in result.citations)
    assert roots.roots[0].pin_cite[-1].quote == "460"
    assert result.short_reporters[0].pin_cite[-1].quote == "462"


def test_mixed_sources_use_rule_pin_and_saved_ml_spans_without_extra_inference(reader):
    mixed = ML.model_copy(update={"pin_cite": "rule"})
    roots = root_run(Document.from_source(SOURCE), mixed)
    assert len(reader.calls) == 1
    root = roots.roots[0]
    assert root.get_ml_field_prediction().pin_cite != root.pin_cite[-1].span
    assert root.pin_cite[-1].quote == "460"
    assert root.case_name[-1].quote == "Alpha v. Beta"
    reader.calls.clear()
    replay = root_run(reload(roots.get_substage(ROOT_ML)), ML)
    assert reader.calls == []
    assert replay.roots[0].pin_cite[-1].quote == "462"
    assert replay.roots[0].ml_field_reading == root.ml_field_reading


def test_enabled_inference_can_be_kept_even_when_all_fields_choose_rules(reader):
    config = ML.model_copy(update=dict.fromkeys(("case_name", "court", "date", "pin_cite"), "rule"))
    roots = root_run(Document.from_source(SOURCE), config)
    assert len(reader.calls) == 1
    assert ROOT_ML in roots.substage_runs
    assert roots.roots[0].ml_field_reading is not None
    assert roots.roots[0].pin_cite[-1].quote == "460"


def test_completed_stage_requires_the_same_four_field_checkpoints():
    source = Document.from_source(SOURCE)
    roots = root_run(source, RULE)
    before = roots.get_substage("grow_roots.read_root_fields.resolve_colocations")
    incomplete = before.model_dump(mode="json")
    incomplete["runs"].append({"kind": "stage", "name": "grow_roots.read_root_fields"})
    with pytest.raises(ValidationError, match="unfinished required substages"):
        Document.model_validate_json(json.dumps(incomplete))


def test_saved_checkpoint_validation_does_not_require_current_ml_configuration(reader, monkeypatch):
    roots = root_run(Document.from_source(SOURCE), ML)
    leaves = short_run(roots, ML)
    monkeypatch.setenv("MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED", "false")
    validate_workflow_checkpoint(reload(roots), "grow_roots", omitted_substages=OMITTED_ROOT_OPTIONS)
    validate_workflow_checkpoint(reload(leaves), "grow_leaves")


@pytest.mark.parametrize(
    ("setting", "replacement"),
    (("model_id", "test/different-model"), ("revision", "b" * 40), ("variant", "bert_plain")),
)
@pytest.mark.parametrize(
    "entrypoint", ("grow_roots", "read_root_fields", "grow_leaves", "grow_short_reporter_leaves")
)
def test_changed_cached_model_is_rejected_before_runtime_loading(
    reader, monkeypatch, setting, replacement, entrypoint
):
    roots = root_run(Document.from_source(SOURCE), ML)
    short = entrypoint in {"grow_leaves", "grow_short_reporter_leaves"}
    saved = short_run(roots, ML) if short else roots
    saved = reload(saved.get_substage(SHORT_ML if short else ROOT_ML))
    serialized = saved.model_dump_json()
    changed = ML.model_copy(update={setting: replacement})
    reader.calls.clear()

    def unexpected(config):
        pytest.fail("Detect cached model drift before loading the newly configured model")

    monkeypatch.setattr(FieldExtractionConfig, "validate_runtime", unexpected)
    with pytest.raises(ValueError, match="match the configured model"):
        if entrypoint == "grow_roots":
            root_run(saved, changed)
        elif entrypoint == "read_root_fields":
            read_root_fields(saved, field_extraction=changed)
        elif entrypoint == "grow_leaves":
            asyncio.run(grow_leaves(saved, field_extraction=changed, review_leaves=False))
        else:
            short_run(saved, changed)
    assert reader.calls == []
    assert saved.model_dump_json() == serialized


@pytest.mark.parametrize(
    "field_reader",
    (resolve_case_names, resolve_courts, resolve_dates, resolve_pin_cites, resolve_short_reporter_case_names),
)
@pytest.mark.parametrize("marked", (False, True), ids=("missing_checkpoint", "missing_citation_prediction"))
def test_ml_field_readers_require_checkpoint_and_citation_prediction(field_reader, marked):
    roots = root_run(Document.from_source(SOURCE), RULE)
    short = field_reader is resolve_short_reporter_case_names
    document = (
        short_run(roots, RULE).get_substage("grow_leaves.grow_short_reporter_leaves.resolve_colocations")
        if short
        else roots.get_substage("grow_roots.read_root_fields.resolve_colocations")
    )
    if marked:
        document = document.complete_substage(SHORT_ML if short else ROOT_ML)
    serialized = document.model_dump_json()
    with pytest.raises(ValueError, match="prediction is unavailable" if marked else "Predict ML"):
        field_reader(document, field_extraction=ML)
    assert document.model_dump_json() == serialized


@pytest.mark.parametrize(
    "field_reader",
    (resolve_case_names, resolve_courts, resolve_dates, resolve_pin_cites, resolve_short_reporter_case_names),
)
def test_direct_ml_field_readers_reject_predictions_from_another_checkpoint(reader, field_reader):
    roots = root_run(Document.from_source(SOURCE), ML)
    short = field_reader is resolve_short_reporter_case_names
    document = short_run(roots, ML).get_substage(SHORT_ML) if short else roots.get_substage(ROOT_ML)
    serialized = document.model_dump_json()
    with pytest.raises(ValueError, match="match the configured model"):
        field_reader(document, field_extraction=ML.model_copy(update={"revision": "b" * 40}))
    assert document.model_dump_json() == serialized
