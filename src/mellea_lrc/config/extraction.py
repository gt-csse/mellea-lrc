"""Small, explicit rules for the first extraction pass."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractionRules(BaseModel):
    """Parameters governing locator grouping and bounded context reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    colocation_max_meaningful_gap: int = Field(default=5, ge=0)
    case_name_window: int = Field(default=160, ge=1)
    post_locator_window: int = Field(default=120, ge=1)


def stable() -> ExtractionRules:
    """The conservative, reproducible first-pass rules."""
    return ExtractionRules()
