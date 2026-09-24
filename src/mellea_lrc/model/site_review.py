"""Durable reviews of proposed citation sites that may never become citations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span


class SiteReview(BaseModel):
    """A proposed site and its outcome, including a refusal or failed review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str
    candidate_span: Span
    candidate_text: str
    outcome: Literal["accepted", "declined", "failed"]
    reason: str
    proposed_locator: str | None = None
    proposed_identifier: str | None = None
    citation_id: str | None = None
    ivr: IvrRun | None = None
