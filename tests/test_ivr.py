"""Tests for preserving Mellea's own IVR attempt history."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from mellea.backends import ModelOption
from mellea.core import ValidationResult
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from openai import APITimeoutError
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm.config import (
    OUTPUT_MODE_OPTION,
    PROFILE_OPTION,
    LlmApiConfig,
    LlmOutputMode,
    start_mellea_session,
)
from mellea_lrc.llm.ivr import (
    InstructIvrSpec,
    InvalidJsonResponseError,
    _http_timeout_ivr_run,
    _output_transport,
    _parse_schema,
    _requirements_for,
    _schema_requirement,
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
    assert "PLEASE CHECK IF TOKEN LIMIT VIOLATED" not in result.reason


@pytest.mark.parametrize("value", ['{"decision":', '{"decision": yes}'])
def test_json_syntax_failure_has_specific_exception_and_parser_detail(value) -> None:
    with pytest.raises(InvalidJsonResponseError) as caught:
        _parse_schema(_context(value), _Output)

    error = caught.value
    assert isinstance(error.__cause__, ValidationError)
    assert error.parser_error == error.__cause__.errors(include_url=False)[0]["msg"]
    assert error.parser_error in str(error)
    assert "PLEASE CHECK IF TOKEN LIMIT VIOLATED" in str(error)


def test_valid_json_schema_failure_remains_pydantic_validation_error() -> None:
    with pytest.raises(ValidationError) as caught:
        _parse_schema(_context('{"decision": "not a boolean"}'), _Output)

    assert caught.value.errors(include_url=False)[0]["type"] == "bool_parsing"


def test_schema_requirement_reports_specific_incomplete_json_repair() -> None:
    requirement = _schema_requirement(_Output)

    result = requirement.validation_fn(_context('{"decision":'))

    assert result is not None
    assert not result.as_bool()
    assert result.reason == (
        "Return exactly one JSON object matching the required output schema. "
        "The previous response was incomplete or invalid JSON. "
        "Parser error: Invalid JSON: EOF while parsing a value at line 1 column 12 "
        "PLEASE CHECK IF TOKEN LIMIT VIOLATED. "
        "Return a complete JSON object with all required fields and closing delimiters. "
        "Keep reasons concise, quote only necessary evidence, and avoid repetitive text."
    )
    assert "pydantic.dev" not in result.reason


@pytest.mark.parametrize("value", ['{"decision":', '{"complete_locator":"No. 21-381"}'])
def test_schema_requirement_prevents_domain_parser_from_running_on_bad_json(value) -> None:
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
    guarded_result = requirements[1].validation_fn(_context(value))
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
                ModelOption.STREAM_TIMEOUT: None,
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
    options = {"max_tokens": 64, "temperature": 0, OUTPUT_MODE_OPTION: mode, ModelOption.STREAM_TIMEOUT: None}
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
    json_error = run.attempts[0].requirements[0].reason
    assert "Invalid JSON: EOF while parsing a value" in json_error
    assert "PLEASE CHECK IF TOKEN LIMIT VIOLATED" in json_error
    assert run.attempts[0].requirements[1].passed
    assert run.attempts[1].requirements[0].passed
    assert not run.attempts[1].requirements[1].passed
    assert "incomplete or invalid JSON" in run.attempts[1].request[-1]["content"]
    assert json_error in run.attempts[1].request[-1]["content"]
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
                    ModelOption.STREAM_TIMEOUT: None,
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
                    ModelOption.STREAM_TIMEOUT: None,
                },
            )
        )

    assert formats == [_Output]


@pytest.fixture
def offline_openai_session(monkeypatch):
    """Use the real Mellea backend and OpenAI SDK, without a network request."""
    import httpx2
    import mellea.backends.openai as backend_module
    import openai

    monkeypatch.setattr(backend_module, "is_vllm_server_with_structured_output", lambda **_kwargs: False)
    clients = []
    sessions = []

    async def create(handler, *, timeout_seconds=0.08, mode=LlmOutputMode.JSON_SCHEMA):
        config = LlmApiConfig(
            model="offline-model",
            api_base="https://provider.invalid/v1",
            api_key="offline-secret",
            temperature=0,
            timeout_seconds=timeout_seconds,
            output_mode=mode,
        )
        session = start_mellea_session(config)
        await session.backend._async_client.close()
        client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout_seconds,
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )
        session.backend._client_cache.put(id(asyncio.get_running_loop()), client)
        clients.append(client)
        sessions.append(session)
        return session, config.mellea_call_options(max_tokens=64)

    yield create
    for client in clients:
        asyncio.run(client.close())
    for session in sessions:
        session.backend._client.close()


def _http_response(request, index, output):
    import httpx2

    return httpx2.Response(
        200,
        request=request,
        json={
            "id": f"response-{index}",
            "model": "offline-model",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}
            ],
        },
    )


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_cumulative_repairs_can_exceed_individual_http_timeout(offline_openai_session, mode) -> None:
    requests = []

    async def respond(request):
        assert request.extensions["timeout"]["read"] == 0.08
        requests.append(json.loads(request.content))
        # Each response is inside its HTTP timeout; three responses together
        # exceed that timeout. No review-wide deadline should cancel repairs.
        await asyncio.sleep(0.04)
        return _http_response(request, len(requests), json.dumps({"decision": len(requests) == 3}))

    async def invoke():
        session, options = await offline_openai_session(respond, mode=mode)
        started = time.monotonic()
        run = await run_instruct_ivr(
            session,
            InstructIvrSpec(
                description="Review the source.",
                output_format=_Output,
                requirements=(
                    req(
                        "Decision must be true.",
                        validation_fn=lambda ctx: ValidationResult(
                            result=_Output.model_validate_json(ctx.last_output().value).decision,
                            reason="Decision must be true.",
                        ),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=options,
        )
        return run, time.monotonic() - started

    run, elapsed = asyncio.run(invoke())
    assert run.success
    assert elapsed > 0.08
    assert len(requests) == len(run.attempts) == 3
    assert run.attempts[-1].requirements[-1].passed
    assert "Decision must be true." in requests[1]["messages"][-1]["content"]
    assert run.model_options[ModelOption.STREAM_TIMEOUT] is None
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def test_validation_duration_is_not_limited_by_http_timeout(offline_openai_session) -> None:
    def slow_validation(_ctx):
        time.sleep(0.06)
        return ValidationResult(result=True)

    async def respond(request):
        return _http_response(request, 0, '{"decision":true}')

    async def invoke():
        session, options = await offline_openai_session(respond, timeout_seconds=0.02)
        started = time.monotonic()
        run = await run_instruct_ivr(
            session,
            InstructIvrSpec(
                description="Review.",
                output_format=_Output,
                requirements=(req("Check the source.", validation_fn=slow_validation),),
            ),
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options=options,
        )
        return run, time.monotonic() - started

    run, elapsed = asyncio.run(invoke())
    assert run.success
    assert elapsed > 0.02
    assert len(run.attempts) == 1


@pytest.mark.parametrize("completed", [0, 2])
@pytest.mark.parametrize("mode", LlmOutputMode)
def test_http_timeout_preserves_completed_repairs_and_stops_requests(
    offline_openai_session, completed, mode
) -> None:
    import httpx2

    requests = []
    strategy = MultiTurnStrategy(loop_budget=4)
    original_state = dict(vars(strategy))
    native_repair = strategy.repair

    async def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == completed + 1:
            raise httpx2.ReadTimeout("The offline HTTP transport stopped responding.", request=request)
        return _http_response(request, len(requests), '{"decision":false}')

    async def invoke():
        session, options = await offline_openai_session(respond, mode=mode)
        return await run_instruct_ivr(
            session,
            InstructIvrSpec(
                description="Review the source.",
                output_format=_Output,
                requirements=(
                    req(
                        "Ground the quoted evidence.",
                        validation_fn=lambda _ctx: ValidationResult(
                            result=False,
                            reason="The quote is not grounded in the source.",
                            score=0.0,
                        ),
                    ),
                ),
            ),
            strategy=strategy,
            model_options=options,
        )

    run = asyncio.run(invoke())
    assert not run.success
    assert len(requests) == completed + 1
    assert len(run.attempts) == completed + 1
    assert run.selected_attempt == completed
    assert vars(strategy) == original_state
    assert strategy.repair is native_repair
    for index, attempt in enumerate(run.attempts[:completed]):
        assert attempt.request == requests[index]["messages"]
        assert attempt.response["id"] == f"response-{index + 1}"
        assert any(not r.passed and "not grounded" in (r.reason or "") for r in attempt.requirements)
    if completed:
        assert "not grounded" in requests[-1]["messages"][-1]["content"]
    assert run.attempts[-1].response is None
    assert run.attempts[-1].output == ""
    assert run.attempts[-1].requirements[-1].description == (
        "The provider HTTP request completed within its configured timeout."
    )
    assert "APITimeoutError" in run.failure_reason
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def test_http_timeout_helper_is_a_durable_failed_run() -> None:
    import httpx2

    error = APITimeoutError(request=httpx2.Request("POST", "https://provider.invalid/v1/chat/completions"))
    run = _http_timeout_ivr_run(
        SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
        InstructIvrSpec(description="Decide."),
        {"max_tokens": 64},
        error,
    )
    assert not run.success
    assert run.failure_reason == f"{type(error).__name__}: {error}"
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


def test_http_timeout_during_validation_retains_generated_answer(monkeypatch) -> None:
    import httpx2

    generation = SimpleNamespace(
        value='{"decision":true}',
        _generate_log=SimpleNamespace(
            prompt=[{"role": "user", "content": "Review the source."}],
            model_output={"id": "completed-before-validation", "choices": [{"finish_reason": "stop"}]},
        ),
    )
    error = APITimeoutError(request=httpx2.Request("POST", "https://provider.invalid/v1/chat/completions"))

    def fail_validation(_ctx):
        raise error

    async def generated_before_timeout(_description, **kwargs):
        context = SimpleNamespace(last_output=lambda: generation)
        for requirement in kwargs["requirements"]:
            requirement.validation_fn(context)
        raise AssertionError("HTTP failure must propagate from the validator")

    monkeypatch.setattr(mfuncs, "ainstruct", generated_before_timeout)
    run = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
            InstructIvrSpec(
                description="Review.",
                output_format=_Output,
                requirements=(req("Check the remote source.", validation_fn=fail_validation),),
            ),
            strategy=MultiTurnStrategy(loop_budget=2),
            model_options={OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA, ModelOption.STREAM_TIMEOUT: None},
        )
    )
    assert not run.success
    assert len(run.attempts) == 1
    assert run.output == generation.value
    assert run.attempts[0].request == generation._generate_log.prompt
    assert run.attempts[0].response == generation._generate_log.model_output
    assert run.attempts[0].requirements[0].passed
    assert "APITimeoutError" in run.failure_reason
    assert IvrRun.model_validate_json(run.model_dump_json()) == run


@pytest.mark.parametrize(
    "error", [TimeoutError("Validator timeout"), RuntimeError("Invalid validator state")]
)
def test_non_http_errors_are_not_mislabeled_as_http_timeout(monkeypatch, error) -> None:
    async def fail_instruct(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(mfuncs, "ainstruct", fail_instruct)
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(
            run_instruct_ivr(
                SimpleNamespace(backend=SimpleNamespace(model_id="test-model")),
                InstructIvrSpec(description="Review.", output_format=_Output),
                strategy=MultiTurnStrategy(loop_budget=1),
                model_options={
                    OUTPUT_MODE_OPTION: LlmOutputMode.JSON_SCHEMA,
                    ModelOption.STREAM_TIMEOUT: None,
                },
            )
        )


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
        ModelOption.STREAM_TIMEOUT: None,
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
