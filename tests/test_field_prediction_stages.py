"""Joint inference is a raw checkpoint before ordinary field readers."""

from importlib import import_module

import pytest
from pydantic import ValidationError

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.extraction.find_docket_locators import find_docket_locators
from mellea_lrc.extraction.find_full_reporter_locators import find_full_reporter_locators
from mellea_lrc.extraction.find_short_reporter_citations import find_short_reporter_citations
from mellea_lrc.extraction.form_roots import form_roots
from mellea_lrc.extraction.predict_root_fields_ml import SUBSTAGE as ROOT_FIELDS
from mellea_lrc.extraction.predict_root_fields_ml import predict_root_fields_ml
from mellea_lrc.extraction.predict_short_reporter_fields_ml import SUBSTAGE as SHORT_FIELDS
from mellea_lrc.extraction.predict_short_reporter_fields_ml import predict_short_reporter_fields_ml
from mellea_lrc.extraction.resolve_colocations import resolve_colocations
from mellea_lrc.extraction.resolve_short_reporter_colocations import resolve_short_reporter_colocations
from mellea_lrc.model.citations.ml_field_reading import MLFieldPrediction, MLFieldReading
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span

CONFIG = FieldExtractionConfig(
    ml_enabled=True,
    case_name="ML",
    court="ML",
    date="ML",
    pin_cite="ML",
    model_id="test/joint-model",
    revision="a" * 40,
    variant="bert_augmented",
    device="cpu",
    context_before=256,
    context_after=256,
    batch_size=8,
)
RULE = FieldExtractionConfig(ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE")


class StubReader:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict_many(self, source, locator_spans):
        self.calls.append((source, locator_spans))
        return self.predictions


def prediction(source, **fields):
    return MLFieldPrediction(
        model_id=CONFIG.model_id,
        revision=CONFIG.revision,
        variant=CONFIG.variant,
        context_span=Span(0, len(source)),
        **{
            name: Span(source.index(quote), source.index(quote) + len(quote))
            for name, quote in fields.items()
        },
    )


def before_root_fields(source):
    return resolve_colocations(
        find_docket_locators(find_full_reporter_locators(Document.from_source(source)))
    )


def before_short_fields(source):
    return resolve_short_reporter_colocations(
        find_short_reporter_citations(form_roots(before_root_fields(source)))
    )


def test_root_prediction_runs_all_heads_once_without_writing_normalized_fields():
    source = "Smith v. Jones, No. 1:24-cv-00123, 2024 WL 1234567, at *3 (D. Ariz. Jan. 1, 2024)."
    before = before_root_fields(source)
    raw = prediction(source, case_name="Smith v. Jones", court="D. Ariz.", date="Jan. 1, 2024", pin_cite="*3")
    reader = StubReader((raw,) * len(before.full_locators))
    result = predict_root_fields_ml(before, CONFIG, reader=reader)
    assert reader.calls == [(source, tuple(c.locator_span for c in before.full_locators))]
    assert len(result.full_locators) == len(before.full_locators) == 2
    for previous, current in zip(before.full_locators, result.full_locators, strict=True):
        assert current.locator == previous.locator
        assert current.colocation_id == previous.colocation_id
        assert current.root_id == previous.root_id
        assert len(current.nodes) == len(previous.nodes) + 1
        assert current.nodes[-1].substage == ROOT_FIELDS
        assert current.ml_field_reading == MLFieldReading(node_id=current.nodes[-1].id, prediction=raw)
        assert current.get_ml_field_prediction() == raw
        for field in ("case_name", "court", "date", "pin_cite"):
            assert getattr(current, field) == getattr(previous, field)
    restored = Document.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.get_substage(before.substage_runs[-1]) == before
    assert restored.get_substage(ROOT_FIELDS) == result


def test_prediction_does_not_infer_court_or_fall_back_to_rules():
    source = "Alpha v. Beta, 556 U.S. 662 (2009)."
    result = predict_root_fields_ml(
        before_root_fields(source), CONFIG, reader=StubReader((prediction(source),))
    )
    citation = result.full_locators[0]
    assert citation.case_name == citation.court == citation.date == ()
    assert citation.pin_cite is None
    assert citation.get_ml_field_prediction().court is None


def test_short_prediction_retains_four_raw_fields_and_preserves_locator_pin():
    source = "Alpha v. Beta, 556 U.S. 662 (2009). See Alpha, 556 U.S. at 670 (2d Cir. 2010)."
    before = before_short_fields(source)
    target = before.short_reporters[0]
    raw = prediction(source, case_name="Alpha", pin_cite="2010", court="2d Cir.", date="2010")
    reader = StubReader((raw,))
    result = predict_short_reporter_fields_ml(before, CONFIG, reader=reader)
    citation = result.short_reporters[0]
    assert reader.calls == [(source, (target.short_locator_span,))]
    assert citation.short_locator == target.short_locator
    assert citation.colocation_id == target.colocation_id
    assert citation.root_id == target.root_id
    assert citation.case_name == target.case_name == ()
    assert citation.pin_cite == target.pin_cite
    assert citation.pin_cite[-1].quote == "670"
    assert not hasattr(citation, "court") and not hasattr(citation, "date")
    assert citation.get_ml_field_prediction() == raw
    assert len(citation.nodes) == len(target.nodes) + 1
    assert result.full_locators == before.full_locators
    restored = Document.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.get_substage(before.substage_runs[-1]) == before


def test_prediction_requires_structure_and_rejects_repeated_calls():
    source = "Alpha v. Beta, 556 U.S. 662 (2009)."
    reader = StubReader(())
    with pytest.raises(ValueError, match="Resolve colocations"):
        predict_root_fields_ml(Document.from_source(source), CONFIG, reader=reader)
    before = before_root_fields(source)
    result = predict_root_fields_ml(before, CONFIG, reader=StubReader((prediction(source),)))
    with pytest.raises(ValueError, match="already completed"):
        predict_root_fields_ml(result, CONFIG, reader=reader)
    with pytest.raises(ValueError, match="before forming roots"):
        predict_root_fields_ml(form_roots(before), CONFIG, reader=reader)
    short_source = source + " Alpha, 556 U.S. at 670."
    created = find_short_reporter_citations(form_roots(before_root_fields(short_source)))
    with pytest.raises(ValueError, match="Resolve short reporter colocations"):
        predict_short_reporter_fields_ml(created, CONFIG, reader=reader)
    grouped = resolve_short_reporter_colocations(created)
    result = predict_short_reporter_fields_ml(grouped, CONFIG, reader=StubReader((prediction(short_source),)))
    with pytest.raises(ValueError, match="already completed"):
        predict_short_reporter_fields_ml(result, CONFIG, reader=reader)
    assert reader.calls == []


def test_empty_prediction_checkpoints_check_availability_without_inference(monkeypatch):
    loads = []
    model = StubReader(())

    def load(config):
        loads.append(config)
        return model

    for module in ("predict_root_fields_ml", "predict_short_reporter_fields_ml"):
        monkeypatch.setattr(import_module(f"mellea_lrc.extraction.{module}"), "get_ml_reader", load)
    roots = predict_root_fields_ml(before_root_fields("No case citation here."), CONFIG)
    assert ROOT_FIELDS in roots.substage_runs
    leaves = resolve_short_reporter_colocations(find_short_reporter_citations(form_roots(roots)))
    assert SHORT_FIELDS in predict_short_reporter_fields_ml(leaves, CONFIG).substage_runs
    assert len(loads) == 2
    assert model.calls == []


@pytest.mark.parametrize("failure", ["count", "context", "source", "model_id", "revision", "variant"])
def test_invalid_batch_is_rejected_before_any_citation_changes(failure):
    source = "Alpha v. Beta, 556 U.S. 662 (2009)."
    before = before_root_fields(source)
    serialized = before.model_dump_json()
    raw = prediction(source)
    if failure == "count":
        outputs, message = (), "one prediction"
    elif failure == "context":
        outputs, message = (
            (raw.model_copy(update={"context_span": Span(0, 2)}),),
            "contain the target locator",
        )
    elif failure == "source":
        outputs, message = (
            (raw.model_copy(update={"context_span": Span(0, len(source) + 10)}),),
            "outside the source",
        )
    else:
        outputs, message = (raw.model_copy(update={failure: "other"}),), "match the configured model"
    with pytest.raises(ValueError, match=message):
        predict_root_fields_ml(before, CONFIG, reader=StubReader(outputs))
    assert before.model_dump_json() == serialized


def test_prediction_is_optional_not_reassignable_and_recovered_by_node():
    source = "Alpha v. Beta, 556 U.S. 662 (2009)."
    with pytest.raises(ValidationError, match="inside the prediction context"):
        MLFieldPrediction(model_id="m", revision="r", variant="v", context_span=Span(0, 2), date=Span(2, 4))
    citation = before_root_fields(source).full_locators[0]
    with pytest.raises(ValueError, match="unavailable"):
        citation.get_ml_field_prediction()
    citation = citation.record(ROOT_FIELDS)
    raw = prediction(source)
    with pytest.raises(ValueError, match="current decision node"):
        citation.with_ml_field_reading(MLFieldReading(node_id=citation.nodes[0].id, prediction=raw))
    reading = MLFieldReading(node_id=citation.nodes[-1].id, prediction=raw)
    recorded = citation.with_ml_field_reading(reading)
    with pytest.raises(ValueError, match="already recorded"):
        recorded.with_ml_field_reading(reading)
    later = recorded.record("grow_roots.read_root_fields.resolve_case_names")
    with pytest.raises(ValueError, match="already recorded"):
        later.with_ml_field_reading(MLFieldReading(node_id=later.nodes[-1].id, prediction=raw))
    assert recorded._through_node_count(len(citation.nodes) - 1).ml_field_reading is None
    assert recorded._through_node_count(len(citation.nodes)).get_ml_field_prediction() == raw


@pytest.mark.parametrize("short", [False, True])
def test_disabled_ml_is_rejected_before_inference(short):
    source = "Alpha v. Beta, 556 U.S. 662 (2009). See Alpha, 556 U.S. at 670."
    before = before_short_fields(source) if short else before_root_fields(source)
    stage = predict_short_reporter_fields_ml if short else predict_root_fields_ml
    reader = StubReader((prediction(source),))
    with pytest.raises(ValueError, match="requires ML_ENABLED=true"):
        stage(before, RULE, reader=reader)
    assert reader.calls == []
