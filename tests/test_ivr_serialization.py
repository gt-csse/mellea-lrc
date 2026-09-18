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
