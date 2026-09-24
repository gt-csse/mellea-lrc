"""Tests for preserving Mellea's own IVR attempt history."""

import asyncio
import json
from types import SimpleNamespace

from mellea.backends import ModelOption
from mellea.core import ValidationResult
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from mellea_lrc.llm.ivr import (
    InstructIvrSpec,
    _requirements_for,
    _schema_requirement,
    _timed_out_ivr_run,
    _to_ivr_run,
    run_instruct_ivr,
)
from mellea_lrc.model.ivr import IvrRun


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


def _context(value: str) -> SimpleNamespace:
    return SimpleNamespace(last_output=lambda: SimpleNamespace(value=value))


def test_output_format_installs_one_schema_requirement_with_actionable_feedback() -> None:
    requirement = _schema_requirement(_Output)

    result = requirement.validation_fn(_context('{"complete_locator":"No. 21-381"}'))

    assert result is not None
    assert not result.as_bool()
    assert result.reason == (
        "Return exactly one JSON object matching the required output schema. "
        "`decision` is required. Remove unsupported field `complete_locator`."
    )


def test_schema_requirement_reports_incomplete_json_without_pydantic_noise() -> None:
    requirement = _schema_requirement(_Output)

    result = requirement.validation_fn(_context('{"decision":'))

    assert result is not None
    assert not result.as_bool()
    assert result.reason == (
        "Return exactly one JSON object matching the required output schema. "
        "The previous response was incomplete or invalid JSON."
    )
    assert "pydantic.dev" not in result.reason


def test_schema_requirement_prevents_domain_parser_from_running_on_bad_json() -> None:
    called = False

    def domain_validation(_ctx: object) -> ValidationResult:
        nonlocal called
        called = True
        raise AssertionError("the schema guard should have returned first")

    requirements = _requirements_for(
        InstructIvrSpec(
            description="Return a decision.",
            output_format=_Output,
            requirements=[req("A domain-specific condition.", validation_fn=domain_validation)],
        )
    )

    assert len(requirements) == 2
    guarded_result = requirements[1].validation_fn(_context('{"complete_locator":"No. 21-381"}'))
    assert guarded_result is not None
    assert guarded_result.as_bool()
    assert not called


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

    payload = run.model_dump(mode="json")

    assert json.loads(json.dumps(payload)) == payload
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


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
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def test_prefix_is_sent_as_system_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_instruct(description: str, **kwargs: object) -> object:
        captured["description"] = description
        captured["options"] = kwargs["model_options"]
        return SimpleNamespace(
            success=True,
            result_index=0,
            sample_generations=[_Generation('{"decision":true}')],
            sample_validations=[[]],
        )

    monkeypatch.setattr(mfuncs, "instruct", fake_instruct)
    session = SimpleNamespace(backend=SimpleNamespace(model_id="test-model"))
    run = asyncio.run(
        run_instruct_ivr(
            session,
            InstructIvrSpec(description="Decide.", prefix="Static instructions", output_format=_Output),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options={"max_tokens": 64},
        )
    )

    assert run.success
    assert captured["description"] == "Decide."
    assert captured["options"][ModelOption.SYSTEM_PROMPT] == "Static instructions"
    assert run.prefix == "Static instructions"


def test_timeout_is_a_durable_failed_run() -> None:
    run = _timed_out_ivr_run(
        SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        InstructIvrSpec(description="Decide."),
        {"max_tokens": 64},
        2.5,
    )

    assert not run.success
    assert "2.5-second timeout" in (run.failure_reason or "")
    assert IvrRun.model_validate_json(run.model_dump_json()) == run
