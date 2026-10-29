"""Provider-native usage and failure classification remain inspectable after JSON save."""

import pytest
from pydantic import ValidationError

from mellea_lrc.model.ivr import (
    SCHEMA_REQUIREMENT_DESCRIPTION,
    IvrAttempt,
    IvrFailureKind,
    IvrRequirementAttempt,
    IvrRun,
    TokenUsage,
)


def _requirement(description, *, passed=False):
    return IvrRequirementAttempt(description=description, passed=passed, reason="Diagnostic.", score=None)


def _run(*attempts, success=False):
    return IvrRun(
        success=success,
        selected_attempt=len(attempts) - 1,
        attempts=attempts,
        backend="offline",
        model="test",
        model_options={},
        instruction="Review.",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )


def test_responses_usage_keeps_standard_details_without_estimating_totals():
    usage = TokenUsage.from_response(
        {
            "usage": {
                "input_tokens": 150,
                "output_tokens": 35,
                "input_tokens_details": {"cached_tokens": 100},
                "output_tokens_details": {"reasoning_tokens": 20},
            }
        }
    )

    assert usage == TokenUsage(
        input_tokens=150, output_tokens=35, cached_input_tokens=100, reasoning_output_tokens=20
    )
    assert usage.total_tokens is None
    assert usage.cache_write_input_tokens is None


def test_equal_chat_and_responses_aliases_count_once_and_leave_raw_response_intact():
    response = {
        "usage": {
            "prompt_tokens": 100,
            "input_tokens": 100,
            "completion_tokens": 40,
            "output_tokens": 40,
            "total_tokens": 140,
            "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 5},
            "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 10},
            "cost_details": {"provider_extra": "preserved"},
        }
    }
    attempt = IvrAttempt(output="{}", requirements=(), response=response)

    assert attempt.usage == TokenUsage(
        input_tokens=100,
        output_tokens=40,
        total_tokens=140,
        cached_input_tokens=30,
        cache_write_input_tokens=5,
        reasoning_output_tokens=10,
    )
    restored = IvrAttempt.model_validate_json(attempt.model_dump_json())
    assert restored.response == response
    assert restored.usage == attempt.usage


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 100, "input_tokens": 99},
        {"completion_tokens": 40, "output_tokens": 41},
        {"prompt_tokens_details": {"cached_tokens": 30}, "input_tokens_details": {"cached_tokens": 29}},
        {
            "completion_tokens_details": {"reasoning_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 9},
        },
        {
            "prompt_tokens_details": {"cache_write_tokens": 5},
            "input_tokens_details": {"cache_write_tokens": 4},
        },
    ],
)
def test_conflicting_provider_aliases_raise(usage):
    with pytest.raises(ValueError, match="Conflicting provider counters"):
        TokenUsage.from_response({"usage": usage})


@pytest.mark.parametrize("counter", ["12", 1.5, True, -1])
def test_reported_token_counts_are_strict_nonnegative_integers(counter):
    with pytest.raises(ValidationError):
        TokenUsage.from_response({"usage": {"input_tokens": counter}})


@pytest.mark.parametrize(
    "response", [None, {}, {"usage": None}, {"usage": {}}, {"usage": {"input_tokens": None}}]
)
def test_unreported_usage_remains_unknown(response):
    assert TokenUsage.from_response(response) is None


def test_reported_zero_is_distinct_from_unknown():
    usage = TokenUsage.from_response(
        {"usage": {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}}}
    )

    assert usage.input_tokens == 0
    assert usage.cached_input_tokens == 0
    assert usage.output_tokens is None
    assert usage.total_tokens is None


@pytest.mark.parametrize("usage", [[], {"input_tokens_details": []}, {"prompt_tokens_details": "unknown"}])
def test_malformed_usage_is_not_silently_ignored(usage):
    with pytest.raises(ValueError, match="must be"):
        TokenUsage.from_response({"usage": usage})


@pytest.mark.parametrize(
    ("output", "kind"),
    [
        (' {"decision": ', IvrFailureKind.INVALID_JSON),
        ('{"decision":"wrong type"}', IvrFailureKind.SCHEMA_VALIDATION),
    ],
)
def test_schema_failure_distinguishes_invalid_json_from_invalid_shape(output, kind):
    attempt = IvrAttempt(output=output, requirements=(_requirement(SCHEMA_REQUIREMENT_DESCRIPTION),))
    assert attempt.failure_kinds == (kind,)


def test_requirement_failure_is_retained_even_when_a_later_repair_succeeds():
    failed = IvrAttempt(output="{}", requirements=(_requirement("Quote must be grounded in the source."),))
    repaired = IvrAttempt(
        output="{}", requirements=(_requirement("Quote must be grounded in the source.", passed=True),)
    )
    run = _run(failed, repaired, success=True)

    assert run.repaired and run.recovered_after_repair
    assert not run.first_attempt_success
    assert run.failure_kinds == (IvrFailureKind.REQUIREMENT_VALIDATION,)
    assert run.final_failure_kinds == ()
    assert run.elapsed_seconds is None
    restored = IvrRun.model_validate_json(run.model_dump_json())
    assert restored.failure_kinds == run.failure_kinds
    assert restored.recovered_after_repair


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"finish_reason": "length"}]},
        {"incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
def test_truncation_and_invalid_json_are_overlapping_observed_events(response):
    attempt = IvrAttempt(
        output='{"decision":', response=response, requirements=(_requirement(SCHEMA_REQUIREMENT_DESCRIPTION),)
    )
    assert attempt.failure_kinds == (IvrFailureKind.INVALID_JSON, IvrFailureKind.OUTPUT_TRUNCATED)
    assert _run(attempt).final_failure_kinds == attempt.failure_kinds


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"refusal": "Refused."}}]},
        {"output": [{"content": [{"type": "refusal", "refusal": "Refused."}]}]},
    ],
)
def test_provider_refusal_is_observed_from_retained_response(response):
    assert IvrAttempt(output="", response=response, requirements=()).failure_kinds == (
        IvrFailureKind.REFUSAL,
    )


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"finish_reason": "content_filter"}]},
        {"incomplete_details": {"reason": "content_filter"}},
    ],
)
def test_provider_filtering_is_not_mislabeled_as_a_model_refusal(response):
    assert IvrAttempt(output="", response=response, requirements=()).failure_kinds == (
        IvrFailureKind.CONTENT_FILTER,
    )


def test_first_attempt_success_is_distinct_from_recovery():
    run = _run(IvrAttempt(output="{}", requirements=()), success=True)
    assert run.first_attempt_success
    assert not run.repaired
    assert not run.recovered_after_repair
    assert run.failure_kinds == ()
