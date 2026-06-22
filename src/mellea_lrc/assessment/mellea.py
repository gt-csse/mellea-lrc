"""Mellea-backed semantic assessment for citations."""

import json
import os
from typing import Literal, cast

from mellea import MelleaSession, generative
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import TypeAdapter, ValidationError

from mellea_lrc.assessment.case_name import assess_case_name_exact_match
from mellea_lrc.llm import llm_provider_config_from_env
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    ModifiedExtractedCitationProposal,
    is_in_context,
)

CaseNameVerdict = Literal["match", "different_case", "irregular_form"]
MODIFIED_EXTRACTION_ADAPTER = TypeAdapter(ModifiedExtractedCitationProposal)
CASE_NAME_VERDICT_MAX_TOKENS = 16
MODIFIED_EXTRACTION_MAX_TOKENS = 512
MODIFIED_EXTRACTION_STRATEGY = RejectionSamplingStrategy(loop_budget=3)


@generative
def classify_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> CaseNameVerdict:
    """Classify how an extracted legal case name relates to the retrieved record.

    You are given the case name pulled from a citation (extracted_case_name), the
    case name of the record retrieved for that citation (retrieved_case_name), and
    the surrounding document text (local_context). Judge ONLY the case name. Do not
    consider volume, reporter, page, pin cite, court, or year.

    Return one of:
    - "match": the two names denote the SAME case and the extracted form is a
      normal way to cite it. Treat these as acceptable (still "match"): using only
      party surnames and dropping given or middle names (e.g. "United States v.
      Golden" for "United States v. Bobby Ray Golden"), abbreviation, "et al.",
      dropped institutional suffixes (such as "Inc." or "Co."), and ordinary
      citation style. As long as BOTH sides of the "v." are represented by a
      recognizable party, prefer "match".
    - "different_case": the extracted name denotes a DIFFERENT, unrelated case than
      the retrieved record. A differing retrieved name is NOT automatically the
      extractor's fault; report "different_case" and do not assume the extraction
      is wrong.
    - "irregular_form": the names denote the SAME case, but the extracted name is
      genuinely incomplete or garbled BEYOND normal shortening — for example a
      whole party is missing (only one side of the "v." is present), the parties
      are in the wrong order, or the text is broken by stray characters or line
      breaks.
    """


@generative
def propose_corrected_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> ModifiedExtractedCitationProposal:
    """Re-extract a corrected case name from local_context, only when warranted.

    Propose corrected plaintiff and defendant strings, or a single corrected
    case_name string, ONLY when local_context shows a more complete or more correct
    case name than extracted_case_name. Copy strings EXACTLY from local_context.
    Never use retrieved_case_name to invent text that is not present in
    local_context.

    Return an EMPTY ModifiedExtractedCitationProposal (all fields null) when the
    current extraction is already the best reading supported by local_context —
    including when the retrieved record simply names a different case. Do not
    propose a change merely because extracted_case_name and retrieved_case_name
    differ.
    """


@generative
def propose_corrected_case_name_json(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> str:
    """Re-extract a corrected case name from local_context as compact JSON.

    Return exactly one JSON object with these keys: plaintiff, defendant,
    case_name. Each value must be a string copied exactly from local_context
    or null. Do not add markdown, comments, or explanatory text.

    Use case_name when the corrected full case name appears as one visible string.
    Use plaintiff and defendant when the corrected parties appear separately.
    Propose a correction ONLY when local_context shows a more complete or more
    correct case name than extracted_case_name. If the current extraction is
    already the best grounded reading, return:
    {"plaintiff": null, "defendant": null, "case_name": null}
    """


def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Assess one case name, using Mellea only when exact equality is not enough."""
    exact_result = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return CaseNameAssessmentRun(assessment=exact_result)
    if not courtlistener_case_name:
        return CaseNameAssessmentRun(assessment=exact_result)

    try:
        return _assess_case_name_with_mellea_after_exact(
            session,
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Mellea case-name assessment failed: {exc}"
        raise RuntimeError(msg) from exc


def _assess_case_name_with_mellea_after_exact(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Classify the case name, then re-extract from local context when warranted.

    Stage 1 classifies the original extraction as ``match`` / ``different_case`` /
    ``irregular_form``. A ``match`` is final. Otherwise Stage 2 revisits the local
    context window and only re-extracts when a more complete grounded case name is
    available; a re-extraction emits a separate reassessment alongside the
    first-pass verdict, while the first-pass verdict is preserved.
    """
    # Stage 1: classify the original extraction (when there is one to classify).
    if extracted_case_name:
        verdict = classify_case_name(
            session,
            local_context=document_context,
            extracted_case_name=extracted_case_name,
            retrieved_case_name=courtlistener_case_name,
            model_options=_structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
        )
        status = CaseNameAssessmentStatus(cast("str", verdict))
        if status == CaseNameAssessmentStatus.MATCH:
            return CaseNameAssessmentRun(
                assessment=_case_name_assessment(
                    citation_id, status, extracted_case_name, courtlistener_case_name
                ),
            )
    else:
        # Nothing was extracted: treat the form as deficient and try to recover it.
        status = CaseNameAssessmentStatus.IRREGULAR_FORM

    # Stage 2: revisit local context and re-extract only when warranted.
    proposal = _propose_corrected_case_name(
        session,
        document_context=document_context,
        extracted_case_name=extracted_case_name or "",
        courtlistener_case_name=courtlistener_case_name,
    )
    first_pass = _case_name_assessment(
        citation_id, status, extracted_case_name, courtlistener_case_name
    )
    if not proposal.valid(document_context):
        # Re-extraction is not warranted; keep the first-pass verdict.
        return CaseNameAssessmentRun(assessment=first_pass)

    reassessment = _assess_corrected_case_name(
        session,
        citation_id=citation_id,
        corrected_case_name=cast("str", proposal.extracted_case_name),
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )
    return CaseNameAssessmentRun(
        assessment=first_pass,
        modified_citation=proposal,
        reassessment=reassessment,
    )


def _assess_corrected_case_name(
    session: MelleaSession,
    *,
    citation_id: str,
    corrected_case_name: str,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    """Assess a re-extracted case name, exact-first then model-backed."""
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return exact
    verdict = classify_case_name(
        session,
        local_context=document_context,
        extracted_case_name=corrected_case_name,
        retrieved_case_name=courtlistener_case_name,
        model_options=_structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
    )
    status = CaseNameAssessmentStatus(cast("str", verdict))
    return _case_name_assessment(
        citation_id, status, corrected_case_name, courtlistener_case_name
    )


def _case_name_assessment(
    citation_id: str,
    status: CaseNameAssessmentStatus,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
) -> CaseNameAssessment:
    return CaseNameAssessment(
        citation_id=citation_id,
        status=status,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message=_message_for_status(status),
    )


def modified_extracted_citation_is_in_context_requirement(document_context: str) -> Requirement:
    """Build the Mellea requirement that prevents ungrounded modified citations."""

    def validate(output: str) -> tuple[bool, str]:
        modified = _modified_extracted_citation_from_output(output)
        if modified is None:
            return False, "Output could not be parsed as ModifiedExtractedCitationProposal."
        values = tuple(
            value for value in (modified.plaintiff, modified.defendant, modified.case_name) if value
        )
        if all(is_in_context(value, document_context) for value in values):
            return True, ""
        return False, "Modified citation fields must appear in document_context."

    return Requirement(
        "Every non-empty field in ModifiedExtractedCitationProposal must be copied from document_context.",
        validation_fn=simple_validate(validate),
    )


def _propose_corrected_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> ModifiedExtractedCitationProposal:
    requirement = modified_extracted_citation_is_in_context_requirement(document_context)
    try:
        return propose_corrected_case_name(
            session,
            local_context=document_context,
            extracted_case_name=extracted_case_name,
            retrieved_case_name=courtlistener_case_name,
            requirements=[requirement],
            strategy=MODIFIED_EXTRACTION_STRATEGY,
            model_options=_structured_model_options(max_tokens=MODIFIED_EXTRACTION_MAX_TOKENS),
        )
    except Exception:
        raw = propose_corrected_case_name_json(
            session,
            local_context=document_context,
            extracted_case_name=extracted_case_name,
            retrieved_case_name=courtlistener_case_name,
            requirements=[requirement],
            strategy=MODIFIED_EXTRACTION_STRATEGY,
            model_options={"temperature": 0, "max_tokens": MODIFIED_EXTRACTION_MAX_TOKENS},
        )
        modified = _modified_extracted_citation_from_output(str(raw))
        if modified is not None:
            return modified
        return ModifiedExtractedCitationProposal()


def _modified_extracted_citation_from_output(output: str) -> ModifiedExtractedCitationProposal | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    try:
        return MODIFIED_EXTRACTION_ADAPTER.validate_python(result)
    except ValidationError:
        return None


def _structured_model_options(*, max_tokens: int) -> dict[str, object]:
    return llm_provider_config_from_env(os.environ).mellea_call_options(max_tokens=max_tokens)


_STATUS_MESSAGES = {
    CaseNameAssessmentStatus.EXACT_MATCH: "Extracted case name exactly matches CourtListener.",
    CaseNameAssessmentStatus.MATCH: "Extracted case name matches the retrieved case.",
    CaseNameAssessmentStatus.DIFFERENT_CASE: (
        "Extracted case name refers to a different case than the retrieved record."
    ),
    CaseNameAssessmentStatus.IRREGULAR_FORM: (
        "Extracted case name uses an unusual or incomplete form for this case."
    ),
    CaseNameAssessmentStatus.NEEDS_ASSESSMENT: "Case name has not been assessed.",
}


def _message_for_status(status: CaseNameAssessmentStatus) -> str:
    return _STATUS_MESSAGES.get(status, "Case name has not been assessed.")
