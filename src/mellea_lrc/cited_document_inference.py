"""LLM inference for deciding whether a document is the cited authority."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.assessment.model_options import structured_model_options
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

CITED_DOCUMENT_INFERENCE_MAX_TOKENS = 768
MISSING_CUE = "<UNKNOWN>"


class CitedDocumentIdentity(str, Enum):
    """Relationship between a candidate document and the cited authority."""

    SAME_DOCUMENT = "same_document"
    DIFFERENT_DOCUMENT = "different_document"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CitedDocumentInferenceStatus(str, Enum):
    """Execution status of cited-document inference."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CitedDocumentReference:
    """Identity cues copied from or recovered for one citation."""

    citation_text: str
    case_name: str | None = None
    locator: str | None = None
    court: str | None = None
    decision_date: str | None = None
    docket_number: str | None = None


@dataclass(frozen=True, slots=True)
class CitedDocumentInference:
    """Grounded identity judgment for one candidate document."""

    status: CitedDocumentInferenceStatus
    identity: CitedDocumentIdentity | None = None
    document_case_name: str | None = None
    evidence: tuple[str, ...] = ()
    reason: str | None = None
    error_message: str | None = None


class _CitedDocumentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["same_document", "different_document", "insufficient_evidence"]
    document_case_name: str | None
    evidence: list[str]
    reason: str


CITED_DOCUMENT_INSTRUCTION = """
Read the entire candidate_document and decide whether it is the judicial document
identified by the cited reference below.

This is document identity analysis only. Do not decide whether the citation is
proper, whether the document supports a proposition, or whether the authority is
published or precedential.

Compare affirmative identity signals such as the parties or case name, court,
decision date, docket number, and any locator printed in the document. Treat
formatting differences, abbreviations, and reordered party captions as possible
representations of the same case rather than automatic conflicts.

Reporter and Westlaw locators are external publication identifiers. The cited
locator is supplied as a corroborating cue if the document happens to print it;
it is not content the document is expected to contain. Never treat an absent
locator, an unprinted locator, or the document's lack of a reporter citation as
identity conflict. In particular, when the case name, court, date, and docket
identity align, classify same_document even if the cited locator does not appear
anywhere in candidate_document.

Return same_document only when the document contains affirmative identity
evidence that aligns with the cited reference and no material identity conflict.
Return different_document when the document contains affirmative identity
evidence that materially conflicts with the cited reference. Return
insufficient_evidence when the document does not expose enough identity evidence
to decide reliably.

Every evidence item must be a non-empty verbatim excerpt copied from
candidate_document. Evidence should contain the shortest useful text supporting
the classification. Do not copy identity cues from the cited reference into
evidence unless they also occur verbatim in candidate_document.

Return exactly one JSON object with fields classification, document_case_name,
evidence, and reason. classification must be same_document, different_document,
or insufficient_evidence. document_case_name must be the case name shown by the
document or null. evidence must be a JSON array of copied excerpts.

cited_reference_text:
{{citation_text}}

cited_case_name:
{{case_name}}

cited_locator:
{{locator}}

cited_court:
{{court}}

cited_decision_date:
{{decision_date}}

cited_docket_number:
{{docket_number}}
""".strip()


def _cited_document_spec(
    *,
    candidate_document: str,
    reference: CitedDocumentReference,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=CITED_DOCUMENT_INSTRUCTION,
        grounding_context={"candidate_document": candidate_document},
        user_variables={
            "citation_text": reference.citation_text,
            "case_name": _cue(reference.case_name),
            "locator": _cue(reference.locator),
            "court": _cue(reference.court),
            "decision_date": _cue(reference.decision_date),
            "docket_number": _cue(reference.docket_number),
        },
        requirements=requirements,
    )


def cited_document_requirements(candidate_document: str) -> list[Requirement]:
    """Build schema, consistency, and source-grounding requirements."""
    return [
        req(
            "output must satisfy the cited-document inference schema",
            validation_fn=_validate_output_schema,
        ),
        check(
            "same_document and different_document require at least one evidence excerpt",
            validation_fn=_validate_evidence_availability,
        ),
        req(
            "classification must follow the identity rubric: it must rely on affirmative "
            "document identity evidence; an absent or unprinted reporter/Westlaw locator is "
            "neutral and cannot support different_document",
        ),
        req(
            "every evidence item must be copied verbatim from candidate_document",
            validation_fn=lambda ctx: _validate_evidence_grounding(ctx, candidate_document),
        ),
    ]


async def _propose_cited_document_inference(
    session: MelleaSession,
    *,
    candidate_document: str,
    reference: CitedDocumentReference,
    requirements: list[Requirement],
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> _CitedDocumentProposal:
    """Run the direct Mellea instruct/validate/repair interaction."""
    spec = _cited_document_spec(
        candidate_document=candidate_document,
        reference=reference,
        requirements=requirements,
    )
    result = await run_instruct_ivr(
        session,
        spec,
        strategy=strategy,
        model_options=model_options,
    )
    return _proposal_from_output(result.result.value)


async def infer_cited_document(
    session: MelleaSession,
    *,
    candidate_document: str,
    reference: CitedDocumentReference,
) -> CitedDocumentInference:
    """Infer whether ``candidate_document`` is the authority identified by ``reference``."""
    if not candidate_document.strip():
        return CitedDocumentInference(
            status=CitedDocumentInferenceStatus.FAILED,
            error_message="candidate_document must not be empty",
        )
    try:
        proposal = await _propose_cited_document_inference(
            session,
            candidate_document=candidate_document,
            reference=reference,
            requirements=cited_document_requirements(candidate_document),
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=CITED_DOCUMENT_INFERENCE_MAX_TOKENS),
        )
    except Exception as exc:
        return CitedDocumentInference(
            status=CitedDocumentInferenceStatus.FAILED,
            error_message=str(exc),
        )
    return CitedDocumentInference(
        status=CitedDocumentInferenceStatus.COMPLETED,
        identity=CitedDocumentIdentity(proposal.classification),
        document_case_name=proposal.document_case_name,
        evidence=tuple(proposal.evidence),
        reason=proposal.reason,
    )


def render_cited_document_prompt(
    *,
    candidate_document: str,
    reference: CitedDocumentReference,
) -> str:
    """Render the raw Mellea prompt for inspection without calling a model."""
    return render_instruct_prompt(
        _cited_document_spec(
            candidate_document=candidate_document,
            reference=reference,
            requirements=cited_document_requirements(candidate_document),
        )
    )


def render_cited_document_chat_messages(
    *,
    candidate_document: str,
    reference: CitedDocumentReference,
) -> tuple[RenderedChatMessage, ...]:
    """Render the exact chat messages presented to the LLM backend."""
    return render_instruct_chat_messages(
        _cited_document_spec(
            candidate_document=candidate_document,
            reference=reference,
            requirements=cited_document_requirements(candidate_document),
        )
    )


def _validate_output_schema(ctx: Context) -> ValidationResult:
    try:
        _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_evidence_availability(ctx: Context) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    if proposal.classification == CitedDocumentIdentity.INSUFFICIENT_EVIDENCE.value:
        return ValidationResult(result=True)
    if proposal.evidence:
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=f"classification={proposal.classification!r} requires evidence",
    )


def _validate_evidence_grounding(ctx: Context, candidate_document: str) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    invalid = [item for item in proposal.evidence if not item.strip() or item not in candidate_document]
    if invalid:
        return ValidationResult(
            result=False,
            reason=f"evidence was not copied verbatim from candidate_document: {invalid!r}",
        )
    return ValidationResult(result=True)


def _proposal_from_context(ctx: Context) -> _CitedDocumentProposal:
    return _proposal_from_output(ctx.last_output().value)


def _proposal_from_output(output: str | object) -> _CitedDocumentProposal:
    if not isinstance(output, str):
        message = f"LLM output was not text: {type(output).__name__}"
        raise ValueError(message)  # noqa: TRY004 - validators share one parse-failure path
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        message = f"LLM output was not valid JSON: {exc}"
        raise ValueError(message) from exc
    try:
        return _CitedDocumentProposal.model_validate(payload)
    except ValidationError as exc:
        message = f"LLM output did not match cited-document inference schema: {exc}"
        raise ValueError(message) from exc


def _cue(value: str | None) -> str:
    return value if value and value.strip() else MISSING_CUE
