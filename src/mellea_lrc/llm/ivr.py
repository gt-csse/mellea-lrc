"""Project-owned Mellea instruct/validate/repair helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mellea import MelleaSession
    from mellea.core.requirement import Requirement
    from mellea.core.sampling import SamplingResult
    from mellea.stdlib.sampling import MultiTurnStrategy


@dataclass(frozen=True, slots=True)
class InstructIvrSpec:
    """Complete project-level specification for one Mellea IVR instruction."""

    description: str
    grounding_context: Mapping[str, str] = field(default_factory=dict)
    user_variables: Mapping[str, str] = field(default_factory=dict)
    requirements: Sequence[Requirement] = field(default_factory=tuple)


async def run_instruct_ivr(
    session: MelleaSession,
    spec: InstructIvrSpec,
    *,
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> SamplingResult[str]:
    """Run one IVR instruction with a fresh Mellea chat context."""
    return await asyncio.to_thread(
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
