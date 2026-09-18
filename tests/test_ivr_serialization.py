"""Tests for preserving Mellea's own IVR attempt history."""

import json
from types import SimpleNamespace

from mellea.core import ValidationResult
from pydantic import BaseModel

from mellea_lrc.llm.ivr import InstructIvrSpec, _to_ivr_run
from mellea_lrc.serialization.ivr import deserialize_ivr_run, serialize_ivr_run


class _Output(BaseModel):
    decision: bool


class _Requirement:
    def __init__(self, description: str) -> None:
        self.description = description


class _Generation:
    def __init__(self, value: str) -> None:
        self.value = value


class _Sampling:
    result_index = 1
    success = False
    sample_generations = [_Generation('{"decision": true}'), _Generation("")]
    sample_validations = [
        [(_Requirement("Return JSON."), ValidationResult(result=True))],
        [(_Requirement("Return JSON."), ValidationResult(result=False, reason="JSON required"))],
    ]


def test_ivr_run_preserves_every_generation_and_validation() -> None:
    run = _to_ivr_run(
        SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        InstructIvrSpec(
            description="Decide.",
            user_variables={"candidate": "No. 25-11030"},
            output_format=_Output,
        ),
        {"max_tokens": 1200},
        _Sampling(),
    )

    assert not run.success
    assert run.output == ""
    assert run.failure_reason == "JSON required"
    assert run.attempts[0].requirements[0].passed
    assert run.attempts[1].requirements[0].reason == "JSON required"

    payload = serialize_ivr_run(run)

    assert json.loads(json.dumps(payload)) == payload
    assert deserialize_ivr_run(payload) == run


def test_ivr_run_serializes_each_provider_request_and_repair_feedback() -> None:
    """A later attempt retains the exact Mellea feedback that prompted it."""
    repair_feedback = (
        "The following requirements have not been met:\n"
        "* locator is required\n"
        "Please try again to fulfill the requirements."
    )
    first_request = [
        {"role": "system", "content": "Static contract"},
        {"role": "user", "content": "Inspect No. 21-381"},
    ]
    repaired_request = [
        *first_request,
        {"role": "assistant", "content": '{"complete_locator":"No. 21-381"}'},
        {"role": "user", "content": repair_feedback},
    ]
    first_response = {
        "id": "first",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"completion_tokens": 21},
    }
    repaired_response = {
        "id": "second",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"completion_tokens": 18},
    }
    sampling = SimpleNamespace(
        success=True,
        result_index=1,
        sample_generations=[
            SimpleNamespace(
                value='{"complete_locator":"No. 21-381"}',
                _generate_log=SimpleNamespace(
                    prompt=first_request,
                    model_output=first_response,
                ),
            ),
            SimpleNamespace(
                value='{"locator":"No. 21-381"}',
                _generate_log=SimpleNamespace(
                    prompt=repaired_request,
                    model_output=repaired_response,
                ),
            ),
        ],
        sample_validations=[
            [(_Requirement("Return JSON."), ValidationResult(result=False, reason="locator is required"))],
            [(_Requirement("Return JSON."), ValidationResult(result=True))],
        ],
    )

    run = _to_ivr_run(
        SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        InstructIvrSpec(description="Decide."),
        {"max_tokens": 1200},
        sampling,
    )

    assert run.attempts[0].request == first_request
    assert run.attempts[1].request == repaired_request
    assert run.attempts[1].request[-1] == {"role": "user", "content": repair_feedback}
    assert run.attempts[0].response == first_response
    assert run.attempts[1].response == repaired_response
    assert deserialize_ivr_run(serialize_ivr_run(run)) == run
