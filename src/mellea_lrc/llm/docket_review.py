"""Strict, repairable OpenRouter review of one proposed docket site."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from mellea_lrc.extraction.docket_hunting import (
    DocketReviewOutcome,
    DocketSiteCandidate,
    DocketSiteDecision,
    _grounded,
)
from mellea_lrc.model.site_review import ReviewAttempt

_PREFIX = """Decide whether a proposed span in a legal filing is a docket locator cited for a court case. A docket number is a court-assigned identifier for a case or proceeding.

Quote the complete proposed locator and its docket-number portion exactly as written. Do not add nearby court, date, pinpoint, or case-name text. An exhibit number, docket-entry number, statute, filing reference, or internal cross-reference is not a case docket citation. Give a short reason for either decision.

Return only the requested structured fields: is_docket_citation, locator, docket_number, reason. If this is not a docket citation, set the two quoted fields to null."""

_INSTRUCTION = """Proposed span: {locator}

Decide whether that span cites a case docket. Keep the quoted fields within the proposed span.

Filing window:
{window}"""


class DocketReviewServiceError(RuntimeError):
    """A provider failure that must not be mistaken for a model refusal."""


@dataclass(frozen=True, slots=True)
class OpenRouterDocketReviewer:
    """One focused model decision with bounded schema and grounding repairs."""

    api_key: str
    base_url: str
    model: str
    service_tier: str | None = None
    temperature: float | None = None
    max_tokens: int = 1800
    max_attempts: int = 3
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> OpenRouterDocketReviewer:
        load_dotenv(override=False)
        key = os.getenv("MELLEA_LRC_LLM_API_KEY")
        base = os.getenv("MELLEA_LRC_LLM_API_BASE")
        model = os.getenv("MELLEA_LRC_LLM_MODEL")
        if not key or not base or not model:
            raise RuntimeError("Docket site hunting needs model key, base URL, and model in the environment")
        temperature = os.getenv("MELLEA_LRC_LLM_TEMPERATURE")
        return cls(
            api_key=key,
            base_url=base,
            model=model,
            service_tier=os.getenv("MELLEA_LRC_LLM_SERVICE_TIER") or None,
            temperature=float(temperature) if temperature else None,
        )

    async def __call__(self, candidate: DocketSiteCandidate) -> DocketReviewOutcome:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PREFIX},
            {
                "role": "user",
                "content": _INSTRUCTION.format(locator=candidate.locator_text, window=candidate.context),
            },
        ]
        attempts: list[ReviewAttempt] = []
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for _ in range(self.max_attempts):
                payload: dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "docket_site_decision",
                            "strict": True,
                            "schema": DocketSiteDecision.model_json_schema(),
                        },
                    },
                    "session_id": "mellea-lrc-docket-site-hunting-v1",
                }
                if self.service_tier:
                    payload["service_tier"] = self.service_tier
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
                response = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                if error := body.get("error"):
                    message = str(error.get("message", error)) if isinstance(error, dict) else str(error)
                    raise DocketReviewServiceError(f"Docket review provider error: {message}")
                if not body.get("choices"):
                    raise DocketReviewServiceError("Docket review provider returned no choices")
                choice = body.get("choices", [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                provider_id = body.get("id")
                raw_usage = body.get("usage") or {}
                usage = {key: value for key, value in raw_usage.items() if isinstance(value, int)}
                feedback: str | None = None
                try:
                    decision = DocketSiteDecision.model_validate_json(content)
                except (ValidationError, ValueError) as exc:
                    feedback = f"Return the required JSON object. Schema validation: {str(exc)[:1000]}"
                else:
                    if decision.is_docket_citation and not _grounded(candidate, decision):
                        feedback = (
                            "The quoted locator or docket number does not match the proposed span. "
                            "Copy its source characters exactly; spacing differences are permitted."
                        )
                attempts.append(
                    ReviewAttempt(
                        model=self.model,
                        response=content,
                        feedback=feedback,
                        request_json=json.dumps(payload, ensure_ascii=False),
                        response_json=response.text,
                        provider_id=str(provider_id) if provider_id is not None else None,
                        finish_reason=choice.get("finish_reason"),
                        usage=usage or None,
                    )
                )
                if feedback is None:
                    return DocketReviewOutcome(decision=decision, attempts=tuple(attempts))
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": feedback})
        return DocketReviewOutcome(
            decision=None,
            attempts=tuple(attempts),
            failure_reason=attempts[-1].feedback if attempts else "No model response",
        )
