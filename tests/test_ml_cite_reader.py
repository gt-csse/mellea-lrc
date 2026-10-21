"""Production ML spans stay source-grounded; no weights or provider calls in tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mellea_lrc.config.field_extraction import FieldExtractionConfig
from mellea_lrc.field_extraction import ml_cite
from mellea_lrc.parsing.span import Span

_PREFIX = "MELLEA_LRC_FIELD_EXTRACTION_"
_REVISION = "881db5474fafe756474034ef6bf9c86a39e9630c"
_SETTINGS = {
    "ML_ENABLED": "true",
    "CASE_NAME": "ML",
    "COURT": "ML",
    "DATE": "ML",
    "PIN_CITE": "ML",
    "MODEL_ID": "gt-csse/ML-cite",
    "REVISION": _REVISION,
    "VARIANT": "ML-cite-Bert-augmented",
    "DEVICE": "cpu",
    "CONTEXT_BEFORE": "260",
    "CONTEXT_AFTER": "260",
    "BATCH_SIZE": "2",
}


def _config(**changes):
    settings = {_PREFIX + key: value for key, value in (_SETTINGS | changes).items()}
    return FieldExtractionConfig.from_env(environ=settings)


class _RepresentationError(ValueError):
    pass


class _Checkpoint:
    def __init__(self, *, quotes=None, limit=None):
        self.quotes = quotes or {}
        self.limit = limit
        self.encoded = []
        self.batches = []
        self.mutate = None

    def encode(self, record):
        self.encoded.append(record)
        assert set(record) == {"context_text", "locator"}
        if self.limit is not None and len(record["context_text"]) > self.limit:
            raise _RepresentationError("input exceeds limit")
        return record

    def predict(self, examples, contexts):
        self.batches.append((examples, contexts))
        result = []
        for context in contexts:
            fields = {}
            for field in ("case_name", "court", "date", "pin_cite"):
                quote = self.quotes.get(field)
                span = None
                if quote:
                    start = context.index(quote)
                    span = {"start": start, "end": start + len(quote), "quote": quote}
                fields[field] = SimpleNamespace(present=quote is not None, span=span)
            result.append(fields)
        return self.mutate(result) if self.mutate else result


@pytest.fixture
def checkpoint(monkeypatch):
    model = _Checkpoint()
    calls = []

    def load_model(**settings):
        calls.append(settings)
        return model

    runtime = SimpleNamespace(RepresentationError=_RepresentationError, load_model=load_model)
    monkeypatch.setattr(ml_cite, "_load_runtime", lambda: runtime)
    ml_cite.get_ml_reader.cache_clear()
    yield model, calls
    ml_cite.get_ml_reader.cache_clear()


def test_reader_reuses_explicit_checkpoint_and_batches_all_four_fields(checkpoint):
    model, loads = checkpoint
    model.quotes = {"case_name": "Alice v. Bob", "court": "X", "date": "2001", "pin_cite": "at 5"}
    source = "prefix Ω😊 Alice v. Bob, 1 U.S. 2, at 5 (X 2001). tail"
    start = source.index("1 U.S. 2")
    locator = Span(start, start + len("1 U.S. 2"))
    config = _config()
    reader = ml_cite.get_ml_reader(config.ml_reader_config())
    assert reader is ml_cite.get_ml_reader(config.ml_reader_config())
    predictions = reader.predict_many(source, (locator, locator, locator))
    assert len(loads) == 1
    assert loads[0] == dict(repo_id=config.model_id, variant=config.variant, revision=_REVISION, device="cpu")
    assert [len(batch[0]) for batch in model.batches] == [2, 1]
    for result in predictions:
        assert result.context_span == Span(0, len(source))
        assert result.revision == _REVISION
        for field, quote in model.quotes.items():
            span = getattr(result, field)
            assert source[span.start : span.end] == quote
    assert predictions[0].case_name.start == source.index("Alice")


def test_context_is_source_only_and_preserves_absolute_unicode_offsets(checkpoint):
    model, _ = checkpoint
    model.quotes = {"date": "2001"}
    source = "Ω😊" * 20 + "1 U.S. 2 (2001)" + " tail" * 20
    start = source.index("1 U.S. 2")
    locator = Span(start, start + len("1 U.S. 2"))
    result = ml_cite.get_ml_reader(
        _config(CONTEXT_BEFORE="3", CONTEXT_AFTER="10").ml_reader_config()
    ).predict_many(source, (locator,))[0]
    assert result.context_span.start == start - 3
    assert result.date == Span(source.index("2001"), source.index("2001") + 4)
    assert result.case_name is None and result.court is None and result.pin_cite is None
    record = model.encoded[0]
    assert record["context_text"] == source[result.context_span.start : result.context_span.end]
    assert record["locator"]["quote"] == source[locator.start : locator.end]


def test_overlong_context_shrinks_without_shrinking_locator(checkpoint):
    model, _ = checkpoint
    model.limit = 30
    source = "a" * 100 + "1 U.S. 2" + "b" * 100
    locator = Span(100, 108)
    result = ml_cite.get_ml_reader(_config().ml_reader_config()).predict_many(source, (locator,))[0]
    assert result.context_span.start <= locator.start < locator.end <= result.context_span.end
    assert result.context_span.end - result.context_span.start <= 30
    assert len(model.encoded) > 1
    assert len(model.batches) == 1
    assert all(record["locator"]["quote"] == "1 U.S. 2" for record in model.encoded)


def test_anchor_over_token_limit_raises_and_never_runs_inference(checkpoint):
    model, _ = checkpoint
    model.limit = 2
    with pytest.raises(ValueError, match="Locator cannot fit"):
        ml_cite.get_ml_reader(_config().ml_reader_config()).predict_many("1 U.S. 2", (Span(0, 8),))
    assert not model.batches


@pytest.mark.parametrize("span", [Span(0, 0), Span(0, 30), Span(20, 21)])
def test_invalid_anchor_is_not_silently_ignored(checkpoint, span):
    with pytest.raises(ValueError, match="Target locator"):
        ml_cite.get_ml_reader(_config().ml_reader_config()).predict_many("1 U.S. 2", (span,))


@pytest.mark.parametrize(
    "corruption", ["quote", "offset", "empty", "presence", "absent_span", "inventory", "batch"]
)
def test_malformed_predictions_raise_instead_of_becoming_absence(checkpoint, corruption):
    model, _ = checkpoint
    model.quotes = {"case_name": "Alpha"}

    def mutate(rows):
        prediction = rows[0]["case_name"]
        if corruption == "quote":
            prediction.span["quote"] = "not the source"
        elif corruption == "offset":
            prediction.span["start"] = -1
        elif corruption == "empty":
            prediction.span = None
        elif corruption == "presence":
            prediction.present = 1
        elif corruption == "absent_span":
            prediction.present = False
        elif corruption == "inventory":
            del rows[0]["court"]
        elif corruption == "batch":
            return []
        return rows

    model.mutate = mutate
    with pytest.raises(ValueError):
        ml_cite.get_ml_reader(_config().ml_reader_config()).predict_many("Alpha, 1 U.S. 2", (Span(7, 15),))


def test_empty_batch_does_not_run_inference_and_rule_config_rejects_ml_reader(checkpoint):
    model, _ = checkpoint
    assert ml_cite.get_ml_reader(_config().ml_reader_config()).predict_many("", ()) == ()
    assert not model.encoded and not model.batches
    with pytest.raises(ValueError, match="requires ML_ENABLED=true"):
        ml_cite.get_ml_reader(
            FieldExtractionConfig(
                ml_enabled=False, case_name="RULE", court="RULE", date="RULE", pin_cite="RULE"
            ).ml_reader_config()
        )


def test_ml_accepts_the_same_structural_context_without_rewriting_predictions(checkpoint):
    from mellea_lrc.field_extraction import FieldContext

    model, _ = checkpoint
    source = "Alpha v. Beta, No. 12-cv-1234, 2006 WL 1984362 (E.D. La. 2006)."
    start = source.index("2006 WL")
    locator = Span(start, start + len("2006 WL 1984362"))
    docket_start = source.index("No.")
    context = FieldContext(
        kind="reporter",
        name_before=Span(0, docket_start),
        court_date_after=Span(locator.end, len(source)),
        pin_region=Span(locator.end, len(source)),
    )
    model.quotes = {"case_name": "Alpha v. Beta", "court": "E.D. La.", "date": "2006"}
    reader = ml_cite.get_ml_reader(_config().ml_reader_config())
    assert reader.predict(source, locator, context=context) == reader.predict(source, locator)
    assert reader.predict_many(source, (locator,), contexts=(context,)) == (reader.predict(source, locator),)


def test_ml_validates_structural_context_before_inference(checkpoint):
    from mellea_lrc.field_extraction import FieldContext

    model, _ = checkpoint
    reader = ml_cite.get_ml_reader(_config().ml_reader_config())
    with pytest.raises(ValueError, match="exactly one FieldContext"):
        reader.predict_many("1 U.S. 2", (Span(0, 8),), contexts=())
    context = FieldContext(
        kind="short_reporter", name_before=Span(0, 0), court_date_after=Span(8, 8), pin_region=Span(0, 9)
    )
    with pytest.raises(ValueError, match="outside the source"):
        reader.predict("1 U.S. 2", Span(0, 8), context=context)
    assert not model.encoded and not model.batches
