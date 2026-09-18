"""JSON round-trip support for model instruct/validate/repair runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span
from mellea_lrc.llm.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.serialization._json import JsonValue, require_list, require_mapping, serialize_dataclass

if TYPE_CHECKING:
    from mellea_lrc.extraction.adjudication.types import Candidate, SiteReview


@dataclass(frozen=True, slots=True)
class SiteReviewTrace:
    """The durable evidence for one reviewed candidate.

    The semantic answer belongs on the citation record it admitted or changed.
    This trace instead retains the candidate, the concise reason, and every
    model attempt so a declined or failed review remains inspectable without a
    second call.
    """

    candidate: Candidate
    reason: str | None
    ivr: IvrRun


def serialize_ivr_run(run: IvrRun) -> dict[str, JsonValue]:
    """Write every Mellea-visible answer and validation result as JSON data."""
    return serialize_dataclass(run)


def deserialize_ivr_run(payload: Mapping[str, object]) -> IvrRun:
    """Recover a serialised IVR run without recreating a live Mellea session."""
    fields = require_mapping(payload, name="ivr_run")
    attempts = tuple(_read_attempt(item) for item in require_list(fields.get("attempts"), name="ivr_run.attempts"))
    selected_attempt = _required_int(fields.get("selected_attempt"), name="ivr_run.selected_attempt")
    if not attempts or not -len(attempts) <= selected_attempt < len(attempts):
        msg = "ivr_run.selected_attempt must identify one recorded attempt"
        raise ValueError(msg)
    return IvrRun(
        success=_required_bool(fields.get("success"), name="ivr_run.success"),
        selected_attempt=selected_attempt,
        attempts=attempts,
        backend=_required_string(fields.get("backend"), name="ivr_run.backend"),
        model=_optional_string(fields.get("model"), name="ivr_run.model"),
        model_options=dict(require_mapping(fields.get("model_options"), name="ivr_run.model_options")),
        instruction=_required_string(fields.get("instruction"), name="ivr_run.instruction"),
        prefix=_optional_string(fields.get("prefix"), name="ivr_run.prefix"),
        grounding_context=_string_mapping(fields.get("grounding_context"), name="ivr_run.grounding_context"),
        user_variables=_string_mapping(fields.get("user_variables"), name="ivr_run.user_variables"),
        output_schema=(
            dict(require_mapping(fields["output_schema"], name="ivr_run.output_schema"))
            if fields.get("output_schema") is not None
            else None
        ),
    )


def serialize_site_review(candidate: Candidate, review: SiteReview[object]) -> dict[str, JsonValue]:
    """Serialize the durable trace shared by every site-review node."""
    return serialize_site_review_trace(
        SiteReviewTrace(candidate=candidate, reason=review.reason, ivr=review.run)
    )


def serialize_site_review_trace(trace: SiteReviewTrace) -> dict[str, JsonValue]:
    """Write a granular site-review trace for a node's opaque details field."""
    return {
        "candidate": serialize_dataclass(trace.candidate),
        "reason": trace.reason,
        "ivr": serialize_ivr_run(trace.ivr),
    }


def deserialize_site_review_trace(payload: Mapping[str, object]) -> SiteReviewTrace:
    """Recover a granular site-review trace from document-node details."""
    fields = require_mapping(payload, name="site_review")
    return SiteReviewTrace(
        candidate=_read_candidate(fields.get("candidate")),
        reason=_optional_string(fields.get("reason"), name="site_review.reason"),
        ivr=deserialize_ivr_run(require_mapping(fields.get("ivr"), name="site_review.ivr")),
    )


def _read_attempt(value: object) -> IvrAttempt:
    fields = require_mapping(value, name="ivr_run.attempt")
    return IvrAttempt(
        output=_required_string(fields.get("output"), name="ivr_run.attempt.output"),
        requirements=tuple(
            _read_requirement(item)
            for item in require_list(fields.get("requirements"), name="ivr_run.attempt.requirements")
        ),
    )


def _read_candidate(value: object) -> Candidate:
    from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind

    fields = require_mapping(value, name="site_review.candidate")
    return Candidate(
        generator=_required_string(fields.get("generator"), name="site_review.candidate.generator"),
        kind=CandidateKind(_required_string(fields.get("kind"), name="site_review.candidate.kind")),
        span=_read_span(fields.get("span"), name="site_review.candidate.span"),
        window=_read_span(fields.get("window"), name="site_review.candidate.window"),
        note=_required_string(fields.get("note"), name="site_review.candidate.note"),
        about=_optional_string(fields.get("about"), name="site_review.candidate.about"),
    )


def _read_span(value: object, *, name: str) -> Span:
    fields = require_mapping(value, name=name)
    return Span(
        start=_required_int(fields.get("start"), name=f"{name}.start"),
        end=_required_int(fields.get("end"), name=f"{name}.end"),
    )


def _read_requirement(value: object) -> IvrRequirementAttempt:
    fields = require_mapping(value, name="ivr_run.requirement")
    score = fields.get("score")
    if score is not None and (not isinstance(score, int | float) or isinstance(score, bool)):
        msg = "ivr_run.requirement.score must be a number or null"
        raise ValueError(msg)
    return IvrRequirementAttempt(
        description=_optional_string(fields.get("description"), name="ivr_run.requirement.description"),
        passed=_required_bool(fields.get("passed"), name="ivr_run.requirement.passed"),
        reason=_optional_string(fields.get("reason"), name="ivr_run.requirement.reason"),
        score=float(score) if score is not None else None,
    )


def _string_mapping(value: object, *, name: str) -> dict[str, str]:
    return {
        key: _required_string(item, name=f"{name}.{key}")
        for key, item in require_mapping(value, name=name).items()
    }


def _required_string(value: object, *, name: str) -> str:
    if isinstance(value, str):
        return value
    msg = f"{name} must be a string"
    raise ValueError(msg)


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, name=name)


def _required_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    msg = f"{name} must be a boolean"
    raise ValueError(msg)


def _required_int(value: object, *, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"{name} must be an integer"
    raise ValueError(msg)
