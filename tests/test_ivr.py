"""Tests for preserving Mellea's own IVR attempt history."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from mellea.backends import ModelOption
from mellea.core import GenerateLog, ModelOutputThunk, ValidationResult
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from mellea_lrc.llm.config import OUTPUT_MODE_OPTION, PROFILE_OPTION, LlmOutputMode
from mellea_lrc.llm.ivr import (
    InstructIvrSpec,
    _output_transport,
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

    async def fake_instruct(description: str, **kwargs: object) -> object:
        captured["description"] = description
        captured["options"] = kwargs["model_options"]
        return SimpleNamespace(
            success=True,
            result_index=0,
            sample_generations=[_Generation('{"decision":true}')],
            sample_validations=[[]],
        )

    monkeypatch.setattr(mfuncs, "ainstruct", fake_instruct)
    session = SimpleNamespace(backend=SimpleNamespace(model_id="test-model"))
    run = asyncio.run(
        run_instruct_ivr(
            session,
            InstructIvrSpec(description="Decide.", prefix="Static instructions", output_format=_Output),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options={
                "max_tokens": 64,
                ModelOption.STREAM_TIMEOUT: 120,
                OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA,
            },
        )
    )

    assert run.success
    assert captured["description"] == "Decide."
    assert captured["options"][ModelOption.SYSTEM_PROMPT] == "Static instructions"
    assert run.prefix == "Static instructions"


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_output_transport_preserves_schema_and_domain_repair_at_instruct_boundary(monkeypatch, mode) -> None:
    captured = {}
    domain_values = []

    def domain_validation(ctx) -> ValidationResult:
        output = _Output.model_validate_json(ctx.last_output().value)
        domain_values.append(output.decision)
        return ValidationResult(
            result=output.decision, reason=None if output.decision else "Decision must be true."
        )

    async def fake_instruct(description: str, **kwargs):
        captured.update(description=description, **kwargs)
        requirements = kwargs["requirements"]
        rendered = description
        for name, value in kwargs["user_variables"].items():
            rendered = rendered.replace("{{" + name + "}}", value)
        messages = [
            {"role": "system", "content": kwargs["model_options"][ModelOption.SYSTEM_PROMPT]},
            {"role": "user", "content": rendered},
        ]
        generations = []
        validations = []
        for index, text in enumerate(['{"decision":', '{"decision":false}', '{"decision":true}']):
            results = [
                (requirement, requirement.validation_fn(_context(text))) for requirement in requirements
            ]
            generations.append(
                SimpleNamespace(
                    value=text,
                    _generate_log=SimpleNamespace(
                        prompt=[dict(message) for message in messages],
                        model_output={
                            "id": f"attempt-{index}",
                            "model": "served-model",
                            "choices": [{"finish_reason": "stop"}],
                        },
                    ),
                )
            )
            validations.append(results)
            if all(result.as_bool() for _, result in results):
                break
            repair, _ = kwargs["strategy"].repair(kwargs["context"], kwargs["context"], [], [], validations)
            messages.extend(
                [{"role": "assistant", "content": text}, {"role": "user", "content": repair.content}]
            )
        return SimpleNamespace(
            success=True, result_index=2, sample_generations=generations, sample_validations=validations
        )

    monkeypatch.setattr(mfuncs, "ainstruct", fake_instruct)
    spec = InstructIvrSpec(
        description="Review {{case}}.",
        prefix="Stable source text",
        user_variables={"case": "Example case"},
        grounding_context={"source": "Evidence source"},
        output_format=_Output,
        requirements=[req("Decision must be true.", validation_fn=domain_validation)],
    )
    options = {"max_tokens": 64, "temperature": 0, OUTPUT_MODE_OPTION: mode, ModelOption.STREAM_TIMEOUT: 120}
    run = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=SimpleNamespace(model_id="configured-alias")),
            spec,
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=options,
        )
    )

    assert captured["return_sampling_results"] is True
    assert captured["await_result"] is True
    assert len(captured["requirements"]) == 2
    assert captured["grounding_context"] == {"source": "Evidence source"}
    assert OUTPUT_MODE_OPTION not in captured["model_options"]
    assert options[OUTPUT_MODE_OPTION] is mode
    assert captured["model_options"][ModelOption.SYSTEM_PROMPT] == "Stable source text"
    if mode is LlmOutputMode.JSON_SCHEMA:
        assert captured["format"] is _Output
        assert captured["description"] == spec.description
        assert captured["user_variables"] == {"case": "Example case"}
        assert "response_format" not in captured["model_options"]
    else:
        assert captured["format"] is None
        assert "{{ivr_output_schema}}" in captured["description"]
        assert json.loads(captured["user_variables"]["ivr_output_schema"]) == _Output.model_json_schema()
        assert captured["user_variables"]["case"] == "Example case"
        if mode is LlmOutputMode.JSON_OBJECT:
            assert captured["model_options"]["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in captured["model_options"]
    assert domain_values == [False, True]
    assert run.success
    assert run.output == '{"decision":true}'
    assert not run.attempts[0].requirements[0].passed
    assert run.attempts[0].requirements[1].passed
    assert run.attempts[1].requirements[0].passed
    assert not run.attempts[1].requirements[1].passed
    assert "incomplete or invalid JSON" in run.attempts[1].request[-1]["content"]
    assert "Decision must be true." in run.attempts[2].request[-1]["content"]
    assert run.output_schema == _Output.model_json_schema()
    assert run.model_options[OUTPUT_MODE_OPTION] == mode.value
    assert run.model == "configured-alias"
    assert run.attempts[-1].response["model"] == "served-model"
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_output_transport_rejects_response_format_conflicts_before_instruct(monkeypatch, mode) -> None:
    async def unexpected_instruct(*_args, **_kwargs):
        raise AssertionError("Invalid transport must fail before generation")

    monkeypatch.setattr(mfuncs, "ainstruct", unexpected_instruct)
    with pytest.raises(ValueError, match="Configure output_mode"):
        asyncio.run(
            run_instruct_ivr(
                SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
                InstructIvrSpec(description="Decide.", output_format=_Output),
                strategy=MultiTurnStrategy(loop_budget=1),
                model_options={
                    OUTPUT_MODE_OPTION: mode,
                    "response_format": {"type": "json_object"},
                    ModelOption.STREAM_TIMEOUT: 120,
                },
            )
        )


@pytest.mark.parametrize("mode", [LlmOutputMode.JSON_OBJECT, LlmOutputMode.PROMPT])
@pytest.mark.parametrize("variable_source", ["user_variables", "grounding_context"])
def test_prompt_schema_variable_collisions_are_rejected(mode, variable_source) -> None:
    spec = InstructIvrSpec(
        description="Decide.",
        output_format=_Output,
        **{variable_source: {"ivr_output_schema": "Caller content"}},
    )

    with pytest.raises(ValueError, match="ivr_output_schema is reserved"):
        _output_transport(spec, {OUTPUT_MODE_OPTION: mode})


@pytest.mark.parametrize("mode", [LlmOutputMode.JSON_OBJECT, LlmOutputMode.PROMPT])
def test_non_native_transport_requires_a_schema_contract(mode) -> None:
    with pytest.raises(ValueError, match="requires a Pydantic output_format"):
        _output_transport(InstructIvrSpec(description="Decide."), {OUTPUT_MODE_OPTION: mode})


def test_unknown_transport_mode_does_not_fall_back() -> None:
    with pytest.raises(ValueError):
        _output_transport(
            InstructIvrSpec(description="Decide.", output_format=_Output), {OUTPUT_MODE_OPTION: "automatic"}
        )


def test_missing_transport_mode_does_not_use_an_implicit_schema_default() -> None:
    with pytest.raises(ValueError, match="explicit output_mode"):
        _output_transport(InstructIvrSpec(description="Decide.", output_format=_Output), {})


@pytest.mark.parametrize("timeout", [None, True, 0, -1, float("nan"), float("inf"), "120"])
def test_missing_or_invalid_deadline_raises_before_generation(monkeypatch, timeout) -> None:
    def unexpected_instruct(*args, **kwargs):
        raise AssertionError("Missing configuration must fail before generation is started")

    monkeypatch.setattr(mfuncs, "ainstruct", unexpected_instruct)
    with pytest.raises(ValueError, match="stream_timeout"):
        asyncio.run(
            run_instruct_ivr(
                SimpleNamespace(backend=object()),
                InstructIvrSpec(description="Decide."),
                strategy=MultiTurnStrategy(loop_budget=1),
                model_options={
                    ModelOption.STREAM_TIMEOUT: timeout,
                    OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA,
                },
            )
        )


def test_endpoint_rejection_does_not_retry_with_a_different_transport(monkeypatch) -> None:
    formats = []

    async def reject_schema(_description, **kwargs):
        formats.append(kwargs["format"])
        raise RuntimeError("Endpoint rejected json_schema")

    monkeypatch.setattr(mfuncs, "ainstruct", reject_schema)
    with pytest.raises(RuntimeError, match="Endpoint rejected json_schema"):
        asyncio.run(
            run_instruct_ivr(
                SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
                InstructIvrSpec(description="Decide.", output_format=_Output),
                strategy=MultiTurnStrategy(loop_budget=3),
                model_options={
                    OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA,
                    ModelOption.STREAM_TIMEOUT: 120,
                },
            )
        )

    assert formats == [_Output]


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


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_timeout_preserves_completed_repairs_without_mutating_caller_strategy(monkeypatch, mode) -> None:
    """A stalled third request must not erase two completed failed reviews."""
    release = asyncio.Event()
    captured = {}
    expected_requests = []
    expected_responses = []
    domain_reasons = []
    strategy = MultiTurnStrategy(loop_budget=3)
    strategy_state = dict(vars(strategy))
    native_repair = strategy.repair

    def domain_validation(_ctx):
        reason = f"Quote {len(domain_reasons) + 1} is not grounded in the supplied source."
        domain_reasons.append(reason)
        return ValidationResult(result=False, reason=reason, score=0.0)

    async def stalled_instruct(description, **kwargs):
        captured.update(description=description, **kwargs)
        rendered = description
        for name, value in kwargs["user_variables"].items():
            rendered = rendered.replace("{{" + name + "}}", value)
        messages = [
            {"role": "system", "content": kwargs["model_options"][ModelOption.SYSTEM_PROMPT]},
            {"role": "user", "content": rendered},
        ]
        generations = []
        validations = []
        for index, output in enumerate(['{"decision":false}', '{"decision": false}']):
            request = [dict(message) for message in messages]
            response = {
                "id": f"completed-{index}",
                "model": "served-model",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"completion_tokens": 10 + index},
            }
            expected_requests.append(json.loads(json.dumps(request)))
            expected_responses.append(json.loads(json.dumps(response)))
            generations.append(
                SimpleNamespace(
                    value=output,
                    _generate_log=SimpleNamespace(prompt=request, model_output=response),
                )
            )
            validations.append([(r, r.validation_fn(_context(output))) for r in kwargs["requirements"]])
            repair, repair_context = kwargs["strategy"].repair(
                kwargs["context"], kwargs["context"], [], generations, validations
            )
            assert repair_context is kwargs["context"]
            messages.extend(
                [{"role": "assistant", "content": output}, {"role": "user", "content": repair.content}]
            )
        # Later native changes cannot alter an already retained attempt.
        generations[0]._generate_log.prompt[0]["content"] = "Mutated after repair"
        generations[0]._generate_log.model_output["usage"]["completion_tokens"] = 999
        await release.wait()
        return SimpleNamespace(
            success=False, result_index=1, sample_generations=generations, sample_validations=validations
        )

    monkeypatch.setattr(mfuncs, "ainstruct", stalled_instruct)
    spec = InstructIvrSpec(
        description="Review the supplied source.",
        prefix="Original source",
        output_format=_Output,
        requirements=(req("Ground the quoted evidence.", validation_fn=domain_validation),),
    )

    async def invoke():
        try:
            return await run_instruct_ivr(
                SimpleNamespace(backend=SimpleNamespace(model_id="configured-alias")),
                spec,
                strategy=strategy,
                model_options={ModelOption.STREAM_TIMEOUT: 0.2, OUTPUT_MODE_OPTION: mode},
            )
        finally:
            release.set()

    run = asyncio.run(invoke())

    assert captured["strategy"] is not strategy
    assert captured["strategy"].loop_budget == strategy.loop_budget
    assert vars(strategy) == strategy_state
    assert strategy.repair is native_repair
    assert MultiTurnStrategy.repair is native_repair
    assert not run.success
    assert run.selected_attempt == 2
    assert len(run.attempts) == 3
    for index, attempt in enumerate(run.attempts[:2]):
        assert attempt.request == expected_requests[index]
        assert attempt.response == expected_responses[index]
        assert attempt.requirements[0].passed
        assert not attempt.requirements[1].passed
        assert attempt.requirements[1].reason == domain_reasons[index]
        assert attempt.requirements[1].score == 0.0
    assert domain_reasons[0] in run.attempts[1].request[-1]["content"]
    assert run.attempts[-1].request is None
    assert run.attempts[-1].response is None
    assert run.attempts[-1].output == ""
    assert "0.2-second timeout" in run.failure_reason
    assert run.output_schema == _Output.model_json_schema()
    assert run.model_options[OUTPUT_MODE_OPTION] == mode.value
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def test_native_async_timeout_cancels_sampling_before_another_repair_request() -> None:
    """Exercise public ainstruct and the native sampler with an offline backend."""

    class OfflineBackend:
        model_id = "offline-model"

        def __init__(self):
            self.actions = []
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.pending_loop = None

        async def generate_from_context(self, action, *, ctx, **_kwargs):
            index = len(self.actions)
            self.actions.append(action)
            if index == 2:
                self.pending_loop = asyncio.get_running_loop()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    raise
            output = ModelOutputThunk('{"decision":false}')
            output._generate_log = GenerateLog(
                prompt=[{"role": "user", "content": f"Request {index}"}],
                model_output={"id": f"completed-{index}", "choices": [{"finish_reason": "stop"}]},
                action=action,
                result=output,
            )
            return output, ctx.add(action).add(output)

    backend = OfflineBackend()
    strategy = MultiTurnStrategy(loop_budget=4)
    original_state = dict(vars(strategy))
    spec = InstructIvrSpec(
        description="Review the source.",
        output_format=_Output,
        requirements=(
            req(
                "Ground the quoted evidence.",
                validation_fn=lambda _ctx: ValidationResult(
                    result=False, reason="The quoted evidence is not grounded in the source."
                ),
            ),
        ),
    )

    async def invoke():
        try:
            run = await run_instruct_ivr(
                SimpleNamespace(backend=backend),
                spec,
                strategy=strategy,
                model_options={
                    ModelOption.STREAM_TIMEOUT: 0.2,
                    OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA,
                },
            )
            assert backend.cancelled.is_set()
        finally:
            # Also release a regressed thread-based call so this test fails
            # without leaving a blocked worker behind.
            if backend.pending_loop is not None:
                backend.pending_loop.call_soon_threadsafe(backend.release.set)
        # A surviving producer could now return another failed answer and
        # dispatch the fourth repair generation.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return run

    run = asyncio.run(invoke())

    assert len(backend.actions) == 3
    assert "not grounded in the source" in backend.actions[1].content
    assert "not grounded in the source" in backend.actions[2].content
    assert vars(strategy) == original_state
    assert not run.success
    assert run.selected_attempt == 2
    assert len(run.attempts) == 3
    assert [attempt.response["id"] for attempt in run.attempts[:2]] == ["completed-0", "completed-1"]
    assert all(
        any(
            not requirement.passed and "not grounded" in (requirement.reason or "")
            for requirement in attempt.requirements
        )
        for attempt in run.attempts[:2]
    )
    assert run.attempts[-1].response is None
    assert "0.2-second timeout" in run.failure_reason
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def _provider_finish_run(finish_reason: str | None, *, success: bool = False) -> IvrRun:
    response = {
        "choices": [{"finish_reason": finish_reason}],
        "usage": {"completion_tokens": 0, "prompt_tokens": 0},
    }
    sampling = SimpleNamespace(
        success=success,
        result_index=-1,
        sample_generations=[
            SimpleNamespace(
                value='{"',
                _generate_log=SimpleNamespace(
                    prompt=[],
                    model_output={"choices": [{"finish_reason": "content_filter"}]},
                ),
            ),
            SimpleNamespace(
                value='{"decision":true}' if success else '{"',
                _generate_log=SimpleNamespace(prompt=[], model_output=response),
            ),
        ],
        sample_validations=[
            [(_Requirement("Return JSON."), ValidationResult(result=False, reason="Invalid JSON"))],
            [
                (
                    _Requirement("Return JSON."),
                    ValidationResult(result=success, reason=None if success else "Invalid JSON"),
                )
            ],
        ],
    )
    return _to_ivr_run(
        SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        InstructIvrSpec(description="Decide."),
        {"max_tokens": 6000},
        sampling,
    )


@pytest.mark.parametrize("finish_reason", ["content_filter", "length", "error"])
def test_failed_ivr_exposes_provider_finish_reason_alongside_schema_failure(finish_reason) -> None:
    run = _provider_finish_run(finish_reason)

    assert run.failure_reason == f"Provider finish reason: {finish_reason}. Invalid JSON"
    assert run.output == '{"'
    assert "failure_reason" not in run.model_dump()
    restored = IvrRun.model_validate_json(run.model_dump_json())
    assert restored.failure_reason == run.failure_reason


@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_normal_or_missing_provider_finish_reason_preserves_schema_failure(finish_reason) -> None:
    assert _provider_finish_run(finish_reason).failure_reason == "Invalid JSON"


def test_successful_repair_does_not_inherit_an_earlier_provider_interruption() -> None:
    run = _provider_finish_run("stop", success=True)

    assert run.success
    assert run.failure_reason is None
    assert run.output == '{"decision":true}'


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_profile_metadata_is_saved_without_reaching_the_provider_or_exposing_credentials(monkeypatch, mode):
    captured = {}
    metadata = {"name": "test-profile", "endpoint": "https://provider.invalid/v1"}

    async def fake_instruct(_description, **kwargs):
        captured.update(kwargs["model_options"])
        return SimpleNamespace(
            success=True,
            result_index=0,
            sample_generations=[_Generation('{"decision":true}')],
            sample_validations=[[]],
        )

    monkeypatch.setattr(mfuncs, "ainstruct", fake_instruct)
    session = SimpleNamespace(backend=SimpleNamespace(model_id="test-model", api_key="private-test-secret"))
    options = {
        "max_tokens": 32,
        OUTPUT_MODE_OPTION: mode,
        PROFILE_OPTION: metadata,
        ModelOption.STREAM_TIMEOUT: 120,
    }
    run = asyncio.run(
        run_instruct_ivr(
            session,
            InstructIvrSpec(description="Decide.", output_format=_Output),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options=options,
        )
    )

    assert run.success
    assert PROFILE_OPTION not in captured
    assert OUTPUT_MODE_OPTION not in captured
    assert run.model_options[PROFILE_OPTION] == {**metadata, "max_attempts": 1}
    assert "private-test-secret" not in run.model_dump_json()
    assert options[PROFILE_OPTION] == metadata
    assert IvrRun.model_validate_json(run.model_dump_json()) == run
