"""ML receives exactly the same prepared context as rules, without recropping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mellea_lrc.field_extraction import LocatorContext, MLReaderConfig, RuleFieldReader, ml_cite
from mellea_lrc.parsing.span import Span

_REVISION = "881db5474fafe756474034ef6bf9c86a39e9630c"


def _config(**changes):
    return MLReaderConfig(
        **(
            dict(
                model_id="gt-csse/ML-cite",
                revision=_REVISION,
                variant="ML-cite-Bert-augmented",
                device="cpu",
                batch_size=2,
            )
            | changes
        )
    )


def _context(source, locator_quote="1 U.S. 2", *, left=0, right=None, **changes):
    right = len(source) if right is None else right
    start = source.index(locator_quote)
    locator = Span(start, start + len(locator_quote))
    return LocatorContext(
        **(
            dict(
                text=source[left:right],
                source_span=Span(left, right),
                locator=locator,
                kind="reporter",
                name_before=Span(left, locator.start),
                court_date_after=Span(locator.end, right),
                pin_region=Span(locator.end, right),
            )
            | changes
        )
    )


class _RepresentationError(ValueError):
    pass


class _Checkpoint:
    def __init__(self):
        self.quotes = {}
        self.limit = None
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


def test_reader_reuses_checkpoint_and_batches_all_four_fields(checkpoint):
    model, loads = checkpoint
    model.quotes = {"case_name": "Alice v. Bob", "court": "X", "date": "2001", "pin_cite": "at 5"}
    source = "prefix Ω😊 Alice v. Bob, 1 U.S. 2, at 5 (X 2001). tail"
    prepared = _context(source)
    config = _config()
    reader = ml_cite.get_ml_reader(config)
    assert reader is ml_cite.get_ml_reader(config)
    predictions = reader.predict_many((prepared, prepared, prepared))
    assert len(loads) == 1
    assert loads[0] == dict(repo_id=config.model_id, variant=config.variant, revision=_REVISION, device="cpu")
    assert [len(batch[0]) for batch in model.batches] == [2, 1]
    for result in predictions:
        assert result.context_span == prepared.source_span
        assert result.revision == _REVISION
        for field, quote in model.quotes.items():
            span = getattr(result, field)
            assert source[span.start : span.end] == quote
    assert predictions[0].case_name.start == source.index("Alice")


def test_supplied_excerpt_is_used_byte_identically_with_absolute_unicode_offsets(checkpoint):
    model, _ = checkpoint
    model.quotes = {"date": "2001"}
    source = "Ω😊" * 20 + "1 U.S. 2 (2001)" + " tail" * 20
    start = source.index("1 U.S. 2")
    prepared = _context(source, left=start - 3, right=start + len("1 U.S. 2 (2001)"))
    result = ml_cite.get_ml_reader(_config()).predict(prepared)
    assert result.context_span == prepared.source_span
    assert result.date == Span(source.index("2001"), source.index("2001") + 4)
    assert result.case_name is None and result.court is None and result.pin_cite is None
    assert model.encoded == [
        {
            "context_text": prepared.text,
            "locator": {"start": 3, "end": 11, "quote": "1 U.S. 2"},
        }
    ]
    assert model.batches[0][1] == [prepared.text]
    assert model.encoded[0]["context_text"].encode() == prepared.text.encode()


def test_overlong_shared_context_raises_without_cropping_retry_or_inference(checkpoint):
    model, _ = checkpoint
    model.limit = 30
    source = "a" * 100 + "1 U.S. 2" + "b" * 100
    prepared = _context(source)
    with pytest.raises(ValueError, match="will not crop or change the shared context"):
        ml_cite.get_ml_reader(_config()).predict(prepared)
    assert len(model.encoded) == 1
    assert model.encoded[0]["context_text"] == prepared.text
    assert not model.batches


def test_invalid_batch_raises_before_any_encoding(checkpoint):
    model, _ = checkpoint
    reader = ml_cite.get_ml_reader(_config())
    valid = _context("1 U.S. 2")
    invalid = valid.model_copy(update={"locator": Span(0, 0)})
    with pytest.raises(ValueError, match="Target locator"):
        reader.predict_many((valid, invalid))
    with pytest.raises(TypeError, match="LocatorContexts"):
        reader.predict_many(("1 U.S. 2",))
    assert not model.encoded and not model.batches


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
        ml_cite.get_ml_reader(_config()).predict(_context("Alpha, 1 U.S. 2"))


def test_empty_batch_does_not_run_inference(checkpoint):
    model, _ = checkpoint
    assert ml_cite.get_ml_reader(_config()).predict_many(()) == ()
    assert not model.encoded and not model.batches


def test_both_readers_receive_one_identical_prepared_context(checkpoint):
    model, _ = checkpoint
    source = "outside | Alpha v. Beta, No. 12-cv-1234, 2006 WL 1984362 (E.D. La. 2007)."
    locator = "2006 WL 1984362"
    left = source.index("Alpha")
    prepared = _context(source, locator, left=left, name_before=Span(left, source.index("No.")))
    rule = RuleFieldReader().predict(prepared)
    model.quotes = {
        field: source[value.start : value.end]
        for field in ("case_name", "court", "date", "pin_cite")
        if (value := getattr(rule, field)) is not None
    }
    ml = ml_cite.get_ml_reader(_config()).predict(prepared)
    assert ml.context_span == rule.context_span == prepared.source_span
    assert {name: getattr(ml, name) for name in ("case_name", "court", "date", "pin_cite")} == {
        name: getattr(rule, name) for name in ("case_name", "court", "date", "pin_cite")
    }
    assert model.encoded[0]["context_text"].encode() == prepared.text.encode()


def test_reader_config_cannot_construct_a_second_context():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        _config(context_before=260, context_after=260)
