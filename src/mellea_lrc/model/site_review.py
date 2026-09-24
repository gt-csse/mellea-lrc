"""Durable reviews of proposed citation sites that may never become citations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from mellea_lrc.model.span import Span


class ReviewAttempt(BaseModel):
    """One model response and the feedback used for a possible repair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    response: str | None = None
    feedback: str | None = None
    request_json: str | None = None
    response_json: str | None = None
    provider_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


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
    attempts: tuple[ReviewAttempt, ...] = ()
