"""Exercise conditional JSON repair through Mellea's native offline sampler."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from mellea.backends import Backend, ModelOption
from mellea.core import GenerateLog, ModelOutputThunk, ValidationResult
from mellea.formatters import TemplateFormatter
from mellea.stdlib.components import Message
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from mellea_lrc.llm.config import OUTPUT_MODE_OPTION, LlmOutputMode
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.model.ivr import IvrRun


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: bool
    quote: str


_SOURCE = "The court affirmed."
_PREFIX = "Original opinion and instructions: literal {{source}} must remain intact."
_GOOD = '{"decision":true,"quote":"The court affirmed."}'
_SEMANTIC_FAILURE = '{"decision":false,"quote":"The court affirmed."}'
_INCOMPLETE = '{"decision":true,"quote":'
_MALFORMED = '{"decision":tru,"quote":"The court affirmed."}'
_GROUNDING_REASON = "The quoted evidence is not grounded in the supplied source."


class _OfflineBackend(Backend):
    """Record the native rendered context; return scripted answers without I/O."""

    model_id = "offline-model"
    _model_id = model_id
    _provider = "offline"

    def __init__(self, outputs):
        self.outputs = outputs
        self.requests = []
        self.responses = []
        self.formats = []
        self.options = []
        self.cancelled = False
        self.formatter = TemplateFormatter(model_id=self.model_id)

    async def _generate_from_context(self, action, ctx, *, format=None, model_options=None, **_kwargs):
        index = len(self.requests)
        assert index < len(self.outputs), "Native sampling exceeded the scripted generation budget."
        options = model_options or {}
        messages = self.formatter.to_chat_messages([*ctx.view_for_generation(), action])
        request = [{"role": message.role, "content": message.content} for message in messages]
        if ModelOption.SYSTEM_PROMPT in options:
            request.insert(0, {"role": "system", "content": options[ModelOption.SYSTEM_PROMPT]})
        self.requests.append(request)
        self.formats.append(format)
        self.options.append(deepcopy(options))
        value = self.outputs[index]
        if value is None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        output = ModelOutputThunk(value)
        response = {
            "id": f"offline-{index}",
            "model": "offline-served-model",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": value}}],
            "usage": {"prompt_tokens": 100 + index, "completion_tokens": 10 + index},
        }
        self.responses.append(response)
        output._generate_log = GenerateLog(
            prompt=request, model_output=response, action=action, result=output
        )
        return output, ctx.add(action).add(output)

    async def _generate_from_raw(self, *_args, **_kwargs):
        raise AssertionError("IVR should use native context generation.")


def _spec(*, requirements=(), output_format=_Output):
    return InstructIvrSpec(
        description="Review {{case}} using the supplied source.",
        prefix=_PREFIX,
        user_variables={"case": "Example v. State"},
        grounding_context={"source": _SOURCE},
        output_format=output_format,
        requirements=requirements,
    )


def _run(outputs, mode, *, spec=None, strategy=None, timeout=5):
    backend = _OfflineBackend(outputs)
    strategy = strategy or MultiTurnStrategy(loop_budget=len(outputs))
    original_state = dict(vars(strategy))
    original_repair = strategy.repair
    class_repair = MultiTurnStrategy.repair
    run = asyncio.run(
        run_instruct_ivr(
            SimpleNamespace(backend=backend),
            spec or _spec(),
            strategy=strategy,
            model_options={
                OUTPUT_MODE_OPTION: mode,
                ModelOption.STREAM_TIMEOUT: timeout,
                "max_tokens": 256,
            },
        )
    )
    assert vars(strategy) == original_state
    assert strategy.repair is original_repair
    assert MultiTurnStrategy.repair is class_repair
    return run, backend


def _failed_reason(run, index):
    failures = [requirement for requirement in run.attempts[index].requirements if not requirement.passed]
    assert len(failures) == 1
    assert failures[0].reason
    return failures[0].reason


def _feedback(reason):
    return (
        "The following requirements have not been met:\n"
        f"* {reason}\n"
        "Please try again to fulfill the requirements."
    )


def _assert_durable_trace(run, backend, outputs):
    assert [attempt.output for attempt in run.attempts] == outputs
    assert [attempt.request for attempt in run.attempts] == backend.requests
    assert [attempt.response for attempt in run.attempts] == backend.responses
    assert all(attempt.requirements for attempt in run.attempts)
    restored = IvrRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.output == run.output


def _assert_transport(backend, mode):
    assert all(OUTPUT_MODE_OPTION not in options for options in backend.options)
    if mode is LlmOutputMode.JSON_SCHEMA:
        assert all(format is _Output for format in backend.formats)
    else:
        assert all(format is None for format in backend.formats)
        assert '"decision"' in backend.requests[0][1]["content"]
        assert '"quote"' in backend.requests[0][1]["content"]
    for options in backend.options:
        if mode is LlmOutputMode.JSON_OBJECT:
            assert options["response_format"] == {"type": "json_object"}
        else:
            assert "response_format" not in options


@pytest.mark.parametrize("mode", LlmOutputMode)
@pytest.mark.parametrize("bad_output", [_INCOMPLETE, _MALFORMED], ids=["incomplete", "malformed"])
def test_bad_json_is_omitted_from_repair_request_but_retained_in_native_run(mode, bad_output):
    run, backend = _run([bad_output, _GOOD], mode)

    assert run.success
    assert run.selected_attempt == 1
    assert run.output == _GOOD
    first_request, repaired_request = backend.requests
    assert first_request[0] == {"role": "system", "content": _PREFIX}
    assert "Review Example v. State" in first_request[1]["content"]
    assert _SOURCE in first_request[1]["content"]
    assert repaired_request[:-1] == first_request
    assert not any(message["role"] == "assistant" for message in repaired_request)
    reason = _failed_reason(run, 0)
    assert "incomplete or invalid JSON" in reason
    assert "Invalid JSON:" in reason
    assert "PLEASE CHECK IF TOKEN LIMIT VIOLATED" in reason
    assert repaired_request[-1] == {"role": "user", "content": _feedback(reason)}
    _assert_transport(backend, mode)
    _assert_durable_trace(run, backend, [bad_output, _GOOD])


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_late_bad_json_preserves_prior_answer_and_its_repair_instruction(mode):
    domain_values = []

    def validate_decision(ctx):
        output = _Output.model_validate_json(ctx.last_output().value)
        domain_values.append(output.decision)
        return ValidationResult(
            result=output.decision, reason=None if output.decision else "Decision must be true."
        )

    run, backend = _run(
        [_SEMANTIC_FAILURE, _INCOMPLETE, _GOOD],
        mode,
        spec=_spec(requirements=(req("Decision must be true.", validation_fn=validate_decision),)),
    )

    assert run.success
    assert run.selected_attempt == 2
    assert domain_values == [False, True]
    first, second, third = backend.requests
    semantic_feedback = {"role": "user", "content": _feedback("Decision must be true.")}
    assert second == [*first, {"role": "assistant", "content": _SEMANTIC_FAILURE}, semantic_feedback]
    assert third[:-1] == second
    assert [message["content"] for message in third if message["role"] == "assistant"] == [_SEMANTIC_FAILURE]
    assert third[-2] == semantic_feedback
    assert third[-1] == {"role": "user", "content": _feedback(_failed_reason(run, 1))}
    _assert_durable_trace(run, backend, [_SEMANTIC_FAILURE, _INCOMPLETE, _GOOD])


@pytest.mark.parametrize("mode", LlmOutputMode)
@pytest.mark.parametrize("failure", ["wrong-keys", "grounding"])
def test_valid_json_schema_and_grounding_failures_keep_native_assistant_context(mode, failure):
    domain_values = []

    def validate_grounding(ctx):
        output = _Output.model_validate_json(ctx.last_output().value)
        domain_values.append(output.quote)
        passed = output.quote in _SOURCE
        return ValidationResult(
            result=passed, reason=None if passed else _GROUNDING_REASON, score=float(passed)
        )

    bad_output = (
        '{"complete_locator":"No. 21-381"}'
        if failure == "wrong-keys"
        else '{"decision":true,"quote":"The court reversed."}'
    )
    run, backend = _run(
        [bad_output, _GOOD],
        mode,
        spec=_spec(requirements=(req("Ground the quoted evidence.", validation_fn=validate_grounding),)),
    )

    assert run.success
    reason = _failed_reason(run, 0)
    assert "incomplete or invalid JSON" not in reason
    if failure == "wrong-keys":
        assert "`decision` is required." in reason
        assert "Remove unsupported field `complete_locator`." in reason
        assert domain_values == [_SOURCE]
    else:
        assert reason == _GROUNDING_REASON
        assert domain_values == ["The court reversed.", _SOURCE]
    assert backend.requests[1] == [
        *backend.requests[0],
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": _feedback(reason)},
    ]
    _assert_durable_trace(run, backend, [bad_output, _GOOD])


@pytest.mark.parametrize("mode", LlmOutputMode)
def test_native_exhaustion_selects_last_bad_output_and_keeps_all_attempts(mode):
    outputs = [_INCOMPLETE, _MALFORMED, ""]
    run, backend = _run(outputs, mode)

    assert not run.success
    assert run.selected_attempt == -1
    assert run.output == ""
    assert len(backend.requests) == 3
    assert run.failure_reason == _failed_reason(run, 2)
    for index in range(1, len(outputs)):
        assert backend.requests[index][:-1] == backend.requests[index - 1]
        assert backend.requests[index][-1] == {
            "role": "user",
            "content": _feedback(_failed_reason(run, index - 1)),
        }
        assert not any(message["role"] == "assistant" for message in backend.requests[index])
    _assert_durable_trace(run, backend, outputs)


def test_unstructured_instruction_keeps_native_context_even_for_json_like_bad_text():
    bad_output = _INCOMPLETE

    def validate_text(ctx):
        passed = ctx.last_output().value == "Complete prose answer."
        return ValidationResult(result=passed, reason=None if passed else "Return a complete prose answer.")

    run, backend = _run(
        [bad_output, "Complete prose answer."],
        LlmOutputMode.JSON_SCHEMA,
        spec=_spec(
            output_format=None,
            requirements=(req("Return complete prose.", validation_fn=validate_text),),
        ),
    )

    assert run.success
    assert run.output_schema is None
    assert backend.formats == [None, None]
    assert backend.requests[1] == [
        *backend.requests[0],
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": _feedback("Return a complete prose answer.")},
    ]
    _assert_durable_trace(run, backend, [bad_output, "Complete prose answer."])


@pytest.mark.parametrize(
    ("bad_output", "retained_messages"),
    [
        (_INCOMPLETE, []),
        (
            '{"complete_locator":"No. 21-381"}',
            [
                {"role": "assistant", "content": '{"complete_locator":"No. 21-381"}'},
                {"role": "user", "content": "Custom native context extension."},
            ],
        ),
    ],
    ids=["invalid-json", "schema-failure"],
)
def test_custom_native_repair_feedback_is_preserved_with_conditional_context(bad_output, retained_messages):
    class CustomStrategy(MultiTurnStrategy):
        @staticmethod
        def repair(old_ctx, new_ctx, past_actions, past_results, past_val):
            action, context = MultiTurnStrategy.repair(old_ctx, new_ctx, past_actions, past_results, past_val)
            return (
                Message(role="user", content=action.content + "\nCustom repair instruction."),
                context.add(Message(role="user", content="Custom native context extension.")),
            )

    run, backend = _run(
        [bad_output, _GOOD],
        LlmOutputMode.JSON_SCHEMA,
        strategy=CustomStrategy(loop_budget=2),
    )

    assert run.success
    assert backend.requests[1] == [
        *backend.requests[0],
        *retained_messages,
        {"role": "user", "content": _feedback(_failed_reason(run, 0)) + "\nCustom repair instruction."},
    ]
    _assert_durable_trace(run, backend, [bad_output, _GOOD])


def test_timeout_after_bad_json_retains_durable_failed_attempts_and_cancels_native_sampling():
    run, backend = _run([_INCOMPLETE, _MALFORMED, None], LlmOutputMode.PROMPT, timeout=0.2)

    assert not run.success
    assert backend.cancelled
    assert len(backend.requests) == 3
    assert run.selected_attempt == 2
    assert [attempt.output for attempt in run.attempts] == [_INCOMPLETE, _MALFORMED, ""]
    assert [attempt.request for attempt in run.attempts[:2]] == backend.requests[:2]
    assert [attempt.response for attempt in run.attempts[:2]] == backend.responses
    assert all("incomplete or invalid JSON" in _failed_reason(run, index) for index in range(2))
    assert all(not any(message["role"] == "assistant" for message in request) for request in backend.requests)
    assert run.attempts[-1].request is None
    assert run.attempts[-1].response is None
    assert "0.2-second timeout" in run.failure_reason
    assert IvrRun.model_validate_json(run.model_dump_json()) == run
