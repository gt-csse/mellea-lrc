---
tags: [mellea, llm, assessment, ivr, requirements, generative, multiturn]
status: active
created: 2026-06-27
---

# Mellea Usage

This document covers how we use Mellea in this project, the patterns we discovered, and the design decisions behind them. New direct Mellea IVR workflows should go through `src/mellea_lrc/llm/ivr.py`; older `@generative` notes below are historical guidance for understanding earlier implementation decisions.

---

## What Mellea Provides

Mellea is a framework for building structured LLM workflows. The core abstraction is the **Instruct → Validate → Repair (IVR) loop**: you define what the model should produce, specify requirements the output must satisfy, and choose a strategy for what happens when requirements are not met. Mellea handles the rest — prompt rendering, output parsing, requirement checking, and repair orchestration.

In this project, Mellea replaces manual retry loops and prompt error appending. We keep project-owned parsing and validation explicit for direct `instruct` workflows so framework artifacts do not leak into the model-facing prompt.

---

## Project Standard: Direct `instruct` IVR

For new Mellea-backed nodes, use `mellea_lrc.llm.ivr` as the boundary module:

- domain modules build an `InstructIvrSpec`;
- `run_instruct_ivr(...)` executes with a fresh `ChatContext`;
- `render_instruct_prompt(...)`, `render_instruct_chat_messages(...)`, and `visualize_instruct_chat_messages(...)` visualize the exact instruction used by the call;
- domain modules parse raw output into their own Pydantic models;
- Pydantic parsing/schema validation should be the first requirement validation when `instruct` is not using Mellea's `format=` parser.

Example:

```python
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict

from mellea_lrc.llm import InstructIvrSpec, render_instruct_prompt, run_instruct_ivr


class MyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


spec = InstructIvrSpec(
    description="Extract the answer from local_context.",
    grounding_context={"local_context": local_context},
    user_variables={},
    requirements=[
        req("Return exactly one JSON object with shape {\"result\":{\"answer\":\"...\"}}.", validation_fn=_validate_schema),
        check("answer must be copied from local_context", validation_fn=_validate_grounding),
    ],
)

print(render_instruct_prompt(spec))  # debugging/fixture/snapshot helper
result = await run_instruct_ivr(
    session,
    spec,
    strategy=MultiTurnStrategy(loop_budget=3),
    model_options=structured_model_options(max_tokens=512),
)
proposal = _parse_output(result.result.value)
```

Design rules:

- Keep the prompt compact. Put task semantics in `description`; put only useful model-facing postconditions in `req`.
- Use `check` for hidden deterministic constraints that should not add prompt noise.
- Do not add post-call “final validation” that duplicates requirements. If a condition decides acceptance, it belongs in the requirement list.
- Do not ask the model to return framework or app-derived fields. Return the minimal copied facts; construct derived fields locally.
- Use `render_instruct_prompt` / `render_instruct_chat_messages` / `visualize_instruct_chat_messages` before changing prompts or tests. If the rendered prompt looks noisy, the implementation is probably noisy too.

---

## Historical Note: The `@generative` Pattern

`@generative` is the decorator for structured LLM calls that return a Pydantic model. The function body is the docstring — Mellea renders it as the prompt, and the return type annotation drives JSON schema enforcement.

```python
from mellea import generative
from pydantic import BaseModel

class MyOutput(BaseModel):
    available: bool
    case_name: str | None = None

@generative
async def my_function(local_context: str, hint: str) -> MyOutput:
    """Extract the case name from local_context.

    Return exactly one JSON object with key "result" containing:
      - "available": true if a case name can be found, false otherwise
      - "case_name": copied exactly from local_context, or null

    Example: {"result": {"available": true, "case_name": "Smith v. Jones"}}
    """
```

**What the framework handles automatically:**

- The response schema is derived from `MyOutput` and sent to the provider. Mellea wraps it in `{"result": {...}}`, so the model always returns `{"result": {"available": ..., "case_name": ...}}`.
- `action.parse()` calls `model_validate_json` on the response and raises `ComponentParseError` if it fails. This propagates before requirement validation ever runs — you do not need to handle parse errors in your `validation_fn`.
- `parsed_repr` is set on the `ModelOutputThunk` before any requirement is checked. This means `ctx.last_output().parsed_repr` in a `validation_fn` is always a valid, fully-typed Pydantic instance — never `None`.

**What you must do explicitly:**

- Write the docstring carefully. For providers that do not fully honour `json_schema` (e.g. DeepSeek), include an explicit `{"result": ...}` example in the docstring. The schema enforcement is a fallback; the docstring instruction is the reliable path.
- Business-logic checks (grounding, consistency) must be expressed as requirements — they are not automatic.

---

## Requirements: `req` vs `check`

Requirements are validated after each generation attempt. They come in two flavours.

```python
from mellea.stdlib.requirements import check, req
```

### `req(description, *, validation_fn=None)`

`req` bakes its `description` into the initial prompt as a postcondition. Use it when the description is a useful natural-language instruction that genuinely helps the model produce correct output, and when it also serves as useful repair feedback if the check fails.

```python
req(
    "case_name must be a string copied exactly from local_context",
    validation_fn=lambda ctx: _validate_grounding(ctx, document_context),
)
```

### `check(description, *, validation_fn=None)`

`check` is `req` with `check_only=True`. The description is **not** injected into the initial prompt. Use it when the constraint is a structural or logical invariant that is difficult to express clearly in natural language — injecting it would add noise and potentially confuse the model. The description is still used by `MultiTurnStrategy` in repair messages.

```python
check(
    "available must be true when case_name is provided, and false when case_name is null",
    validation_fn=_validate_availability_consistency,
)
```

**Rule of thumb:** if the description makes the model perform better when it sees it, use `req`. If the description would confuse the model but is still useful for repair feedback, use `check`. If it would be useless even for repair, omit the description entirely (both `req` and `check` accept `description=None`).

### `validation_fn` — programmatic checking

When `validation_fn` is provided, it is **mutually exclusive** with LLM-as-a-judge. The framework does not combine them. Validation is purely programmatic.

```python
from mellea.core import ValidationResult
from mellea.core.base import Context

def _validate_grounding(ctx: Context, document_context: str) -> ValidationResult:
    proposal: MyOutput = ctx.last_output().parsed_repr  # always a valid Pydantic instance here
    if not proposal.available:
        return ValidationResult(True)   # grounding only applies when available
    if is_in_context(proposal.case_name, document_context):
        return ValidationResult(True)
    return ValidationResult(
        False,
        reason=f"case_name={proposal.case_name!r} was not copied exactly from local_context",
    )
```

`ctx.last_output()` walks backward through the context to find the most recent `ModelOutputThunk`. In a multi-turn conversation, this always returns the latest attempt's output. The `parsed_repr` attribute is the typed Pydantic object.

To pass extra arguments (like `document_context` above), use a lambda or `functools.partial` at the call site:

```python
validation_fn=lambda ctx: _validate_grounding(ctx, document_context)
```

---

## Sampling Strategies

All three built-in strategies share `loop_budget: int` (default 1) — the total number of generate → validate iterations allowed.

### `RejectionSamplingStrategy`

Re-runs the identical prompt from the identical context. At `temperature=0`, the model produces exactly the same output every time. **Effectively useless for repair** — only meaningful at high temperatures where sampling variance gives different outputs.

### `RepairTemplateStrategy`

Appends the failure message to the original prompt via a `{% if repair %}` block in `Instruction.jinja2`. The model retries with a stronger hint but no memory of what it said before. **Does not work with `@generative`** — `GenerativeSlot.jinja2` has no `{% if repair %}` block, so it falls through to identical behaviour as rejection sampling.

### `MultiTurnStrategy` ← correct choice for `@generative`

On failure, appends a new user turn to the conversation:

```
"The following requirements have not been met:
* case_name must be a string copied exactly from local_context
Please try again to fulfill the requirements."
```

The model's prior (wrong) output is already in the conversation context. It can compare what it said against the instruction and self-correct. Even at `temperature=0`, the input has changed — the model receives different tokens and can produce a different output.

**Requires `ChatContext`.** Pass `(ChatContext(), session.backend, ...)` instead of `(session, ...)`. A fresh `ChatContext` per call prevents state bleed-over between concurrent requests. The return type changes to `tuple[R, Context]` — unpack accordingly.

**Exhaustion behaviour.** When `loop_budget` is exhausted, the strategy returns the last attempt's result rather than raising an exception. Always add a post-call acceptance check.

---

## Full Usage Example

This is the production pattern from `src/mellea_lrc/assessment/llm/reextract.py`.

```python
from mellea import MelleaSession, generative
from mellea.core import ValidationResult
from mellea.core.base import Context, ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel


class _ReextractionProposal(BaseModel):
    available: bool
    case_name: str | None = None


@generative
async def _propose_case_name_reextraction(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> _ReextractionProposal:
    """Extract the case name that actually appears in local_context.

    Your task is faithful extraction — copy whatever case name is present in
    local_context, even if it differs from retrieved_case_name or
    extracted_case_name. Do not try to correct toward retrieved_case_name.
    The downstream system handles whether names match.

    Return exactly one JSON object with key "result" containing two fields:
      - "available": true if a case name can be identified in local_context
      - "case_name": copied exactly from local_context, or null

    Example: {"result": {"available": true, "case_name": "Smith v. Jones"}}
    """


def _validate_availability_consistency(ctx: Context) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if proposal.available == (proposal.case_name is not None):
        return ValidationResult(True)
    if proposal.available:
        return ValidationResult(False, reason="available is true but case_name is null")
    return ValidationResult(False, reason="available is false but case_name is not null")


def _validate_grounding(ctx: Context, document_context: str) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if not proposal.available:
        return ValidationResult(True)
    if is_in_context(proposal.case_name, document_context):
        return ValidationResult(True)
    return ValidationResult(
        False,
        reason=f"case_name={proposal.case_name!r} was not copied exactly from local_context",
    )


async def reextract_case_name(session: MelleaSession, *, document_context: str, ...) -> ...:
    final_ctx: Context | None = None
    try:
        proposal, final_ctx = await _propose_case_name_reextraction(
            ChatContext(),          # fresh context per call — no state bleed-over
            session.backend,
            local_context=document_context,
            extracted_case_name=...,
            retrieved_case_name=...,
            requirements=[
                # Structural invariant: not expressed in the prompt, purely post-generation.
                check(
                    "available must be true when case_name is provided, and false when null",
                    validation_fn=_validate_availability_consistency,
                ),
                # Grounding instruction: appears in prompt AND drives repair messages.
                req(
                    "case_name must be a string copied exactly from local_context",
                    validation_fn=lambda ctx: _validate_grounding(ctx, document_context),
                ),
            ],
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=512),
        )
    except Exception as exc:
        return failed_result(str(exc))

    # Strategy exhaustion returns the last attempt — must check acceptance explicitly.
    if not proposal.available:
        return empty_result(chat_history=_chat_history_from_context(final_ctx))
    if not is_in_context(proposal.case_name, document_context):
        return failed_result(
            f"exhausted retries: {proposal.case_name!r} not grounded",
            chat_history=_chat_history_from_context(final_ctx),
        )
    return accepted_result(proposal.case_name)
```

---

## Design Decisions

### Explicit `available` field instead of `null`-as-proxy

An earlier design used `case_name: str | None` alone — `null` meant "not available." This conflates two distinct things: the model's opinion ("I believe no case name is present") and a missing value. It is also unreliable, because a model returning `null` cannot be distinguished from a model that failed to produce a value.

Adding `available: bool` forces the model to express its opinion explicitly. This makes outputs inspectable, makes repair messages meaningful, and separates the consistency check (is `available` coherent with `case_name`?) from the grounding check (is `case_name` actually in the document?).

### `check` for consistency, `req` for grounding

The consistency rule (`available ↔ case_name is not None`) is a logical invariant. Expressing it in the prompt as a natural-language instruction tends to add confusing noise — the model struggles to internalize it as a precondition on its JSON output. It is more reliable as a silent post-generation check that triggers repair if violated.

The grounding rule (`case_name` must be copied exactly from `local_context`) is a substantive task instruction. It helps the model produce correct output when it sees it in the prompt, and it produces meaningful repair feedback when it fails. This belongs in `req`.

### `MultiTurnStrategy` over rejection sampling

At `temperature=0`, re-running the same prompt yields the same output. `MultiTurnStrategy` changes the input on each retry by adding a user turn with the failure reason and the model's prior wrong output. The model can compare its mistake against the original instruction and self-correct. This is the only strategy that can repair `@generative` outputs at zero temperature.

### Capturing chat history on failure

When re-extraction fails or returns empty, `final_ctx.as_list()` is walked to reconstruct the conversation turns. `Message` components carry `.role` and `.content` directly. `ModelOutputThunk` items are tagged as `"assistant"`. This history is stored in `ReextractionResult.chat_history` and serialized into the assessment snapshot, making it possible to diagnose failures without re-running the pipeline.

---

## Async Context Warning

Mellea emits a warning when `ChatContext` is used with async calls:

```
Not using a SimpleContext with asynchronous requests could cause unexpected results
due to stale contexts. Ensure you await between requests.
```

This warning is a false positive for our usage pattern. The risk it guards against is sharing a single `ChatContext` across concurrent coroutines — concurrent mutations corrupt the context's linked-list state. We create a **fresh `ChatContext()` per call** and always `await` the full call before returning. There is no shared state and no concurrent mutation. The warning cannot be suppressed through the current genslot API (`silence_context_type_warning` is not exposed as a call-site parameter).
