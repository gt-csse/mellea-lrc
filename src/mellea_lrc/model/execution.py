"""Durable completion events for atomic document checkpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CheckpointRun(BaseModel):
    """One completed substage, including a run that changed no citations."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["substage"] = "substage"
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,2}$")
