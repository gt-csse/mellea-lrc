"""Exercise local span decoding without checkpoint weights or network access."""

from __future__ import annotations

from importlib import import_module

import pytest

from mellea_lrc.extraction.field_readers.ml_cite.reader import MLCiteReader
from mellea_lrc.parsing.span import Span

torch = pytest.importorskip("torch", reason="ML-cite is an optional extraction dependency")
pytest.importorskip("transformers", reason="ML-cite is an optional extraction dependency")
pytest.importorskip("huggingface_hub", reason="ML-cite is an optional extraction dependency")
pytest.importorskip("safetensors.torch", reason="ML-cite is an optional extraction dependency")

_runtime = import_module("mellea_lrc.extraction.field_readers.ml_cite._runtime")


def _decode_selected_range(field: str, selected: str, *, rescanned: bool = False):
    context = selected + " Alpha"
    offsets = [(0, len(selected)), (len(selected) + 1, len(context))]
    example = _runtime.ModelInput(input_ids=[1, 2], positions=[0, 1], offsets=offsets)
    outputs = []
    for name in _runtime.FIELDS:
        if name == field:
            # In the reversed case, independent argmax selects (1, 0), but
            # ordered rescanning must select the higher-scoring pair (0, 0).
            start = [9.0, 10.0] if rescanned else [10.0, 0.0]
            end = [10.0, 0.0]
        else:
            start = end = [0.0, 10.0]
        outputs.append(tuple(torch.tensor(scores) for scores in (start, end, [2.0, 2.0])))
    return context, offsets, _runtime.decode_outputs(outputs, example, context)


@pytest.mark.parametrize("field", _runtime.FIELDS)
@pytest.mark.parametrize("selected", [".", "-", " \t "])
@pytest.mark.parametrize("rescanned", [False, True])
def test_high_presence_range_trimmed_to_empty_becomes_absent(field, selected, rescanned):
    context, offsets, predictions = _decode_selected_range(field, selected, rescanned=rescanned)
    prediction = predictions[field]
    assert prediction.present is False
    assert prediction.span is None
    assert prediction.rescanned is rescanned
    context_span = Span(100, 100 + len(context))
    assert MLCiteReader._absolute_span(prediction, context, context_span, field) is None

    for name in _runtime.FIELDS:
        if name == field:
            continue
        other = predictions[name]
        assert other.present is True
        assert other.span == {"start": offsets[1][0], "end": offsets[1][1], "quote": "Alpha"}
        assert other.rescanned is False
        assert MLCiteReader._absolute_span(other, context, context_span, name) == Span(
            100 + offsets[1][0], 100 + offsets[1][1]
        )


@pytest.mark.parametrize("field", _runtime.FIELDS)
@pytest.mark.parametrize("selected,start,end,quote", [(".Alpha,", 1, 6, "Alpha"), (":", 0, 1, ":")])
def test_nonempty_trimmed_range_preserves_grounded_span(field, selected, start, end, quote):
    context, _, predictions = _decode_selected_range(field, selected)
    prediction = predictions[field]
    assert prediction.present is True
    assert prediction.span == {"start": start, "end": end, "quote": quote}
    assert prediction.rescanned is False
    assert MLCiteReader._absolute_span(prediction, context, Span(100, 100 + len(context)), field) == Span(
        100 + start, 100 + end
    )
