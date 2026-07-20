"""Pin-cite content assessment for a locator-found citation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import (
    InstructIvrSpec,
    RenderedChatMessage,
    render_instruct_chat_messages,
    render_instruct_prompt,
    run_instruct_ivr,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context
    from mellea.core.requirement import Requirement

    from mellea_lrc.extraction.types import ExtractedCitation

PIN_CITE_INFERENCE_MAX_TOKENS = 512


class PinCiteInferenceStatus(str, Enum):
    """Execution status of pin-cite inference."""

    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PinCiteValid:
    """Reasoned judgment about content attributed to an exact pinpoint."""

    reason: str
    pinpoint_marker: str | None
    pinpoint_excerpt: str | None
    pin_cite_match: bool | None


@dataclass(frozen=True, slots=True)
class PinCiteInference:
    """Complete pin-cite inference outcome."""

    status: PinCiteInferenceStatus
    validation: PinCiteValid | None = None
    error_message: str | None = None


class _PinCiteProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Field order is deliberate: the model must state its evidence-based reason
    # before committing to the nullable validation judgment.
    reason: str = Field(min_length=1)
    pinpoint_marker: str | None
    pinpoint_excerpt: str | None
    pin_cite_match: bool | None

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only reasoning."""
        if not value.strip():
            message = "reason must not be blank"
            raise ValueError(message)
        return value


PIN_CITE_INSTRUCTION = """
CITED_DOCUMENT:
{{cited_document}}

Read citing_context to identify the content or proposition attributed to the
citation. Then locate cited_pin_cite inside cited_document and read the text
at that exact page, page range, paragraph, or footnote. Decide whether the
pinpointed text contains or substantively supports the attributed content.

This is pinpoint-content validation. Do not rely on discussion elsewhere in
cited_document. Do not decide whether the document is the correct case or
whether citing the authority is otherwise legally appropriate.

Recognize the document's own location scheme, including printed reporter page
numbers, bracketed or starred page markers, paragraph numbers, page ranges, and
pinpoint forms such as "at *3", "*6-7", "1142", or "¶ 21". Formatting may differ
between cited_pin_cite and the marker printed in cited_document.

Set pin_cite_match to true when the exact pinpoint is locatable and its text
contains or substantively supports the content attributed in citing_context. Set
it to false when the exact pinpoint is locatable but its text does not contain or
support that attributed content. Set it to null when the pinpoint cannot be
located reliably, pagination was stripped, or citing_context does not identify a
content claim clear enough to test. Content found only on another page does not
make the pinpoint match. A matching passage somewhere in a document that does
not expose the cited page, paragraph, footnote, or star-page boundary is not
enough: return null because its exact location is unverified. Never infer that a
passage is at the requested pinpoint merely because the document is short or
because the passage matches the attributed content.

Example of the null boundary: if cited_pin_cite is "at *5" and cited_document
contains words supporting the attribution but contains no printed "*5" location
marker, return pinpoint_marker=null, pinpoint_excerpt=null, and
pin_cite_match=null. The matching words cannot be assigned to *5.

First state a concise reason describing the attributed content and what the exact
pinpoint says. Then copy the exact location label printed in cited_document as
pinpoint_marker (for example, "*3", "[1142]", or "¶ 21") and one short verbatim
pinpoint_excerpt from the text governed by that marker. Use null for both fields
when the pinpoint cannot be located. Finally state the judgment. Do not reveal
hidden chain-of-thought; provide only the short evidence-based explanation needed
to audit the result.

Return exactly one JSON object in this field order:
{"reason":"...","pinpoint_marker":"... or null","pinpoint_excerpt":"... or null","pin_cite_match":true_or_false_or_null}

cited_pin_cite:
{{pin_cite}}

full_citation_text:
{{citation_text}}

base_locator:
{{locator}}

citing_context:
{{citing_context}}
""".strip()


def pin_cite_requirements(cited_document: str) -> list[Requirement]:
    """Build the schema/order and semantic requirements for one pinpoint."""
    return [
        req(
            "output must satisfy the PinCiteValid schema and place reason first, "
            "pinpoint_marker second, pinpoint_excerpt third, and pin_cite_match last",
            validation_fn=_validate_output_schema_and_order,
        ),
        req(
            "a true or false judgment requires non-empty pinpoint_marker and pinpoint_excerpt "
            "copied verbatim from cited_document; a null judgment requires both to be null",
            validation_fn=lambda ctx: _validate_pinpoint_grounding(ctx, cited_document),
        ),
    ]


async def _propose_pin_cite_validation(
    session: MelleaSession,
    *,
    cited_document: str,
    pin_cite: str,
    matched_citation_text: str,
    matched_locator_text: str,
    citing_context: str,
    requirements: list[Requirement],
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> _PinCiteProposal:
    """Run the direct Mellea instruct/validate/repair interaction."""
    result = await run_instruct_ivr(
        session,
        _pin_cite_spec(
            cited_document=cited_document,
            pin_cite=pin_cite,
            matched_citation_text=matched_citation_text,
            matched_locator_text=matched_locator_text,
            citing_context=citing_context,
            requirements=requirements,
        ),
        strategy=strategy,
        model_options=model_options,
    )
    return _proposal_from_output(result.result.value)


async def assess_found_locator_pin_cite(
    session: MelleaSession,
    *,
    citing_document: str,
    citation: ExtractedCitation,
    cited_document: str,
) -> PinCiteInference:
    """Assess the upstream citation's pinpoint against the retrieved opinion.

    This narrow operation is intended for a locator-found candidate. Citation
    text, locator, pinpoint, and citing context are derived from the extraction
    artifact rather than reconstructed by the caller.
    """
    if not cited_document.strip():
        return PinCiteInference(
            status=PinCiteInferenceStatus.FAILED,
            error_message="cited_document must not be empty",
        )
    if not isinstance(citation.citation, FullCaseCitation):
        return PinCiteInference(
            status=PinCiteInferenceStatus.NOT_APPLICABLE,
            error_message="pin-cite assessment requires a full case citation",
        )
    pin_cite = citation.citation.pin_cite
    if not pin_cite or not pin_cite.strip():
        return PinCiteInference(
            status=PinCiteInferenceStatus.NOT_APPLICABLE,
            error_message="citation does not contain a pin cite",
        )
    try:
        citing_context = DocumentTextWindow.around(
            citing_document,
            citation.citation_span,
        ).text
    except ValueError as exc:
        return PinCiteInference(
            status=PinCiteInferenceStatus.FAILED,
            error_message=str(exc),
        )
    source_slice = citing_document[citation.citation_span.start : citation.citation_span.end]
    if source_slice != citation.matched_citation_text:
        return PinCiteInference(
            status=PinCiteInferenceStatus.FAILED,
            error_message="matched_citation_text does not match citation_span in citing_document",
        )
    try:
        proposal = await _propose_pin_cite_validation(
            session,
            cited_document=cited_document,
            pin_cite=pin_cite,
            matched_citation_text=citation.matched_citation_text,
            matched_locator_text=citation.matched_locator_text,
            citing_context=citing_context,
            requirements=pin_cite_requirements(cited_document),
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=PIN_CITE_INFERENCE_MAX_TOKENS),
        )
    except Exception as exc:
        return PinCiteInference(
            status=PinCiteInferenceStatus.FAILED,
            error_message=str(exc),
        )
    return PinCiteInference(
        status=PinCiteInferenceStatus.COMPLETED,
        validation=PinCiteValid(
            reason=proposal.reason,
            pinpoint_marker=proposal.pinpoint_marker,
            pinpoint_excerpt=proposal.pinpoint_excerpt,
            pin_cite_match=proposal.pin_cite_match,
        ),
    )


def render_pin_cite_prompt(
    *,
    citing_document: str,
    citation: ExtractedCitation,
    cited_document: str,
) -> str:
    """Render the raw Mellea prompt without calling a model."""
    pin_cite, citing_context = _citation_prompt_inputs(citing_document, citation)
    return render_instruct_prompt(
        _pin_cite_spec(
            cited_document=cited_document,
            pin_cite=pin_cite,
            matched_citation_text=citation.matched_citation_text,
            matched_locator_text=citation.matched_locator_text,
            citing_context=citing_context,
            requirements=pin_cite_requirements(cited_document),
        )
    )


def render_pin_cite_chat_messages(
    *,
    citing_document: str,
    citation: ExtractedCitation,
    cited_document: str,
) -> tuple[RenderedChatMessage, ...]:
    """Render the exact chat messages presented to the LLM backend."""
    pin_cite, citing_context = _citation_prompt_inputs(citing_document, citation)
    return render_instruct_chat_messages(
        _pin_cite_spec(
            cited_document=cited_document,
            pin_cite=pin_cite,
            matched_citation_text=citation.matched_citation_text,
            matched_locator_text=citation.matched_locator_text,
            citing_context=citing_context,
            requirements=pin_cite_requirements(cited_document),
        )
    )


def _pin_cite_spec(
    *,
    cited_document: str,
    pin_cite: str,
    matched_citation_text: str,
    matched_locator_text: str,
    citing_context: str,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=PIN_CITE_INSTRUCTION,
        # Keep the opinion in the task body so the generation model reads it
        # before answering; grounding validation remains function-based below.
        grounding_context={},
        user_variables={
            "pin_cite": pin_cite,
            "citation_text": matched_citation_text,
            "locator": matched_locator_text,
            "citing_context": citing_context,
            "cited_document": cited_document,
        },
        requirements=requirements,
    )


def _citation_prompt_inputs(
    citing_document: str,
    citation: ExtractedCitation,
) -> tuple[str, str]:
    if not isinstance(citation.citation, FullCaseCitation):
        message = "pin-cite assessment requires a full case citation"
        raise TypeError(message)
    pin_cite = citation.citation.pin_cite
    if not pin_cite or not pin_cite.strip():
        message = "citation does not contain a pin cite"
        raise ValueError(message)
    source_slice = citing_document[citation.citation_span.start : citation.citation_span.end]
    if source_slice != citation.matched_citation_text:
        message = "matched_citation_text does not match citation_span in citing_document"
        raise ValueError(message)
    context = DocumentTextWindow.around(citing_document, citation.citation_span).text
    return pin_cite, context


def _validate_output_schema_and_order(ctx: Context) -> ValidationResult:
    try:
        _proposal_from_context(ctx, require_reason_first=True)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_pinpoint_grounding(ctx: Context, cited_document: str) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    if proposal.pin_cite_match is None:
        if proposal.pinpoint_marker is None and proposal.pinpoint_excerpt is None:
            return ValidationResult(result=True)
        return ValidationResult(
            result=False,
            reason=("pinpoint_marker and pinpoint_excerpt must be null when pin_cite_match is null"),
        )
    marker = proposal.pinpoint_marker
    excerpt = proposal.pinpoint_excerpt
    if marker is None or not marker.strip():
        return ValidationResult(
            result=False,
            reason="true or false pin_cite_match requires a non-empty pinpoint_marker",
        )
    if marker not in cited_document:
        return ValidationResult(
            result=False,
            reason=(
                "the proposed pinpoint_marker does not occur in cited_document, so the exact "
                "pinpoint is unverified; return null for pinpoint_marker, pinpoint_excerpt, "
                "and pin_cite_match"
            ),
        )
    if excerpt is None or not excerpt.strip():
        return ValidationResult(
            result=False,
            reason="true or false pin_cite_match requires a non-empty pinpoint_excerpt",
        )
    if excerpt not in cited_document:
        return ValidationResult(
            result=False,
            reason="pinpoint_excerpt was not copied verbatim from cited_document",
        )
    return ValidationResult(result=True)


def _proposal_from_context(
    ctx: Context,
    *,
    require_reason_first: bool = False,
) -> _PinCiteProposal:
    return _proposal_from_output(
        ctx.last_output().value,
        require_reason_first=require_reason_first,
    )


def _proposal_from_output(
    output: str | object,
    *,
    require_reason_first: bool = False,
) -> _PinCiteProposal:
    if not isinstance(output, str):
        message = f"LLM output was not text: {type(output).__name__}"
        raise ValueError(message)  # noqa: TRY004 - validators share one parse-failure path
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        message = f"LLM output was not valid JSON: {exc}"
        raise ValueError(message) from exc
    if not isinstance(payload, dict):
        message = "LLM output JSON was not an object"
        raise ValueError(message)  # noqa: TRY004 - validators share one parse-failure path
    try:
        proposal = _PinCiteProposal.model_validate(payload)
    except ValidationError as exc:
        message = f"LLM output did not match PinCiteValid schema: {exc}"
        raise ValueError(message) from exc
    if require_reason_first and list(payload) != [
        "reason",
        "pinpoint_marker",
        "pinpoint_excerpt",
        "pin_cite_match",
    ]:
        message = "LLM output must place reason before pinpoint_marker, pinpoint_excerpt, and pin_cite_match"
        raise ValueError(message)
    return proposal
