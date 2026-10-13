"""Token counters retain provider meaning through native IVR serialization."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mellea_lrc.llm.ivr import _provider_response
from mellea_lrc.model.ivr import IvrAttempt, TokenUsage


def test_stream_usage_is_retained_without_mutating_the_reconstructed_response():
    response = {"id": "completion-1", "choices": [], "usage": None}
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 60},
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    generation = SimpleNamespace(
        _generate_log=SimpleNamespace(model_output=response),
        generation=SimpleNamespace(usage=usage),
    )
    attempt = IvrAttempt(output="answer", requirements=(), response=_provider_response(generation))
    usage["prompt_tokens"] = 999
    assert response["usage"] is None
    restored = IvrAttempt.model_validate_json(attempt.model_dump_json())
    assert restored.usage == TokenUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        cached_input_tokens=60,
        reasoning_output_tokens=5,
    )


def test_final_stream_usage_takes_precedence_over_a_partial_response_counter():
    generation = SimpleNamespace(
        _generate_log=SimpleNamespace(model_output={"usage": {"completion_tokens": 2}}),
        generation=SimpleNamespace(usage={"prompt_tokens": 10, "completion_tokens": 5}),
    )
    assert TokenUsage.from_response(_provider_response(generation)) == TokenUsage(
        input_tokens=10,
        output_tokens=5,
    )


def test_usage_can_survive_without_a_reconstructed_stream_body():
    generation = SimpleNamespace(generation=SimpleNamespace(usage={"completion_tokens": 7}))
    assert TokenUsage.from_response(_provider_response(generation)) == TokenUsage(output_tokens=7)


@pytest.mark.parametrize("response", [None, {}, {"usage": None}, {"usage": {}}, {"usage": {"metrics": None}}])
def test_missing_usage_is_unknown(response):
    assert IvrAttempt(output="", requirements=(), response=response).usage is None


def test_zero_usage_is_reported_and_does_not_invent_cache_counters():
    attempt = IvrAttempt(
        output="",
        requirements=(),
        response={
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": None,
            },
        },
    )
    assert attempt.usage == TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)


def test_partial_usage_preserves_unknown_fields_without_deriving_a_total():
    attempt = IvrAttempt(
        output="",
        requirements=(),
        response={
            "usage": {"completion_tokens": 21, "completion_tokens_details": {"reasoning_tokens": 6}},
        },
    )
    assert attempt.usage == TokenUsage(output_tokens=21, reasoning_output_tokens=6)


@pytest.mark.parametrize("count", [True, "100", 1.5, -1])
def test_malformed_present_counters_raise(count):
    with pytest.raises(ValidationError):
        TokenUsage.from_response({"usage": {"prompt_tokens": count}})


@pytest.mark.parametrize("usage", [False, [], {"prompt_tokens_details": False}])
def test_malformed_usage_objects_raise(usage):
    with pytest.raises(ValueError):
        TokenUsage.from_response({"usage": usage})
