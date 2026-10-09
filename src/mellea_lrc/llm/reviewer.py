"""Shared runtime binding for substage-specific IVR reviewers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from mellea_lrc.llm.config import start_mellea_session
from mellea_lrc.llm.profiles import LlmProfile

if TYPE_CHECKING:
    from mellea import MelleaSession


@dataclass(frozen=True, slots=True)
class IvrReviewer:
    """A bound session, model options, and repair budget for one substage."""

    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int

    @classmethod
    def from_profile(cls, profile: LlmProfile) -> Self:
        """Bind a complete profile while preserving the concrete reviewer type."""
        config = profile.resolve()
        return cls(
            session=start_mellea_session(config),
            model_options=config.mellea_call_options(max_tokens=profile.max_tokens),
            max_attempts=profile.max_attempts,
        )
