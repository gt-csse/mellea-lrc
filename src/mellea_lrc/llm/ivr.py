"""Project-owned Mellea instruct/validate/repair helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mellea.stdlib import functional as mfuncs
from mellea.stdlib.components import Instruction
from mellea.stdlib.context import ChatContext
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mellea import MelleaSession
    from mellea.core.requirement import Requirement
    from mellea.core.sampling import SamplingResult
    from mellea.formatters.chat_formatter import ChatFormatter
    from mellea.stdlib.sampling import MultiTurnStrategy


@dataclass(frozen=True, slots=True)
class RenderedChatMessage:
    """Rendered chat message sent to a chat-completion backend."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class InstructIvrSpec:
    """Complete project-level specification for one Mellea IVR instruction."""

    description: str
    grounding_context: Mapping[str, str] = field(default_factory=dict)
    user_variables: Mapping[str, str] = field(default_factory=dict)
    requirements: Sequence[Requirement] = field(default_factory=tuple)

    def to_instruction(self) -> Instruction:
        """Build the exact Mellea ``Instruction`` used for rendering and execution."""
        return Instruction(
            self.description,
            grounding_context=dict(self.grounding_context),
            user_variables=dict(self.user_variables),
            requirements=list(self.requirements),
        )


class MelleaRequirementsExhaustedError(RuntimeError):
    """All IVR repair turns failed; no invalid selected output may be consumed."""

    def __init__(self, result: SamplingResult[str]) -> None:
        failed = [
            f"{requirement.description}: {validation.reason or 'requirement not met'}"
            for requirement, validation in result.result_validations
            if not validation
        ]
        detail = "; ".join(failed) or "Mellea reported unsuccessful sampling."
        super().__init__(f"Mellea requirements exhausted after repair budget: {detail}")
        self.result = result


async def run_instruct_ivr(
    session: MelleaSession,
    spec: InstructIvrSpec,
    *,
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> SamplingResult[str]:
    """Run one IVR instruction with a fresh chat context.

    Keep this wrapper as the standard project entrypoint for direct Mellea
    ``instruct`` usage. Domain modules should construct an ``InstructIvrSpec``
    and keep parsing/validation domain-specific.
    """
    result = await asyncio.to_thread(
        mfuncs.instruct,
        spec.description,
        context=ChatContext(),
        backend=session.backend,
        grounding_context=dict(spec.grounding_context),
        user_variables=dict(spec.user_variables),
        requirements=list(spec.requirements),
        strategy=strategy,
        return_sampling_results=True,
        model_options=model_options,
    )
    if not result.success:
        raise MelleaRequirementsExhaustedError(result)
    return result


def render_instruct_prompt(
    spec: InstructIvrSpec,
    *,
    session: MelleaSession | None = None,
    formatter: ChatFormatter | None = None,
) -> str:
    """Render the single prompt string produced by Mellea's formatter."""
    resolved_formatter = _formatter(session=session, formatter=formatter)
    return resolved_formatter.print(spec.to_instruction())


def render_instruct_chat_messages(
    spec: InstructIvrSpec,
    *,
    session: MelleaSession | None = None,
    formatter: ChatFormatter | None = None,
) -> tuple[RenderedChatMessage, ...]:
    """Render the chat messages that a chat-completion backend receives."""
    resolved_formatter = _formatter(session=session, formatter=formatter)
    messages = resolved_formatter.to_chat_messages([spec.to_instruction()])
    return tuple(
        RenderedChatMessage(role=message.role, content=message.content)
        for message in messages
    )


def format_rendered_chat_messages(messages: Sequence[RenderedChatMessage]) -> str:
    """Format rendered chat messages for prompt inspection logs or snapshots."""
    return "\n\n".join(f"[{message.role}]\n{message.content}" for message in messages)


def visualize_instruct_chat_messages(
    spec: InstructIvrSpec,
    *,
    session: MelleaSession | None = None,
    formatter: ChatFormatter | None = None,
) -> str:
    """Render and format the chat messages that enter the LLM backend."""
    return format_rendered_chat_messages(
        render_instruct_chat_messages(spec, session=session, formatter=formatter)
    )


def _formatter(
    *,
    session: MelleaSession | None,
    formatter: ChatFormatter | None,
) -> ChatFormatter:
    if formatter is not None:
        return formatter
    if session is not None:
        backend_formatter = getattr(session.backend, "formatter", None)
        if backend_formatter is not None:
            return backend_formatter
    from mellea.formatters.template_formatter import TemplateFormatter  # noqa: PLC0415

    return TemplateFormatter("default")
