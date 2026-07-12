"""LLM-backed preparation for not-found case-name candidate search."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

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
from mellea_lrc.retrieval.types import (
    CaseNamePreparationStatus,
    CaseNameSearchPreparation,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context
    from mellea.core.requirement import Requirement

    from mellea_lrc.extraction.types import ExtractedCitation

PREPARATION_CONTEXT_BEFORE_CHARS = 320
PREPARATION_CONTEXT_AFTER_CHARS = 160
PREPARATION_MAX_TOKENS = 512
COMPLETE_PARTY_COUNT = 2
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_QUERY_TERM = re.compile(r"^[A-Za-z0-9 .,'&-]{1,160}$")
_OCR_DIGITS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"})
_MONTH_PREFIXES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
JSON_OUTPUT_REQUIREMENT = (
    'Return exactly one JSON object with shape '
    '{"classification":"...","plaintiff":"... or null","defendant":"... or null",'
    '"decision_date":"YYYY-MM-DD or null","reason":"..."}.'
)
CASE_NAME_PREPARATION_INSTRUCTION = """
Prepare one bounded case-name search attempt for the citation marked by locator.

First examine and correct the copied parties. Use only parties bound to this
locator. The relevant copied case name usually appears before the locator. If
local_context contains multiple citations, do not borrow parties from another
citation.

Treat the locator string as the boundary marker inside local_context. Prefer the
nearest copied "plaintiff v. defendant" name before that locator over parser
hints. Parser hints may come from a different citation in the same window.
If parser hints conflict with the nearest copied name attached to locator, ignore
the hints and return the locator-bound copied parties. A wrong parser hint is not
a reason to answer no_case_name.

When local_context contains an earlier citation and then another copied case name
right before locator, choose the later copied name bound to locator. For example,
in "... Alpha v. Beta, 111 F.3d 222. ... Gamma v. Delta, 999 U.S. 999", the
parties for locator "999 U.S. 999" are Gamma and Delta.

Reporter-like citations between the case name and locator can be parallel
citations for the same authority, not necessarily a boundary. Do not borrow
parties from an earlier separate citation, but do not reject a case name merely
because the same copied citation includes multiple reporter locators.

Use classification "complete_case_name" when both parties are present,
"partial_case_name" when exactly one party is present, and "no_case_name" when
no bound party is present.

Then examine the asserted decision date. Return decision_date only as ISO
YYYY-MM-DD when a complete written date is copied in the same citation
parenthetical after this locator. Do not infer a missing month/day from the
year, and do not borrow a date from a neighboring citation. The parser's date
hint is evidence, not authority; correct it when the copied citation says so.
Extra spaces, punctuation variation, and recognizable OCR substitutions do
not make a copied complete date absent. Do not return null merely because that
date needs normalization. YYYY-MM-DD is the normalized output format; it is
never required to occur verbatim in the copied citation.

locator:
{{locator}}

Parser hints, which may be missing or wrong:
plaintiff={{extracted_plaintiff}}
defendant={{extracted_defendant}}
Parser date hint={{extracted_decision_date}}
""".strip()

QUERY_OUTPUT_REQUIREMENT = (
    'Return exactly one JSON object with shape '
    '{"query_plaintiff":"...","query_defendant":"...","reason":"..."}.'
)
QUERY_PLANNING_INSTRUCTION = """
Plan exactly one CourtListener case-name search from already validated parties.
query_plaintiff and query_defendant may normalize harmless formatting, spacing,
punctuation, or a clear abbreviation expansion. They are plain search terms,
not corrected evidence. Do not return CourtListener syntax, operators, fields,
quotes, or additional attempts. The program constructs the query.

validated plaintiff={{plaintiff}}
validated defendant={{defendant}}
validated decision date={{decision_date}}
court={{court}}
locator={{locator}}
""".strip()


class _PreparedCaseName(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: str
    plaintiff: str | None = None
    defendant: str | None = None
    decision_date: str | None = None
    reason: str | None = None


class _PlannedCaseNameQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_plaintiff: str
    query_defendant: str
    reason: str | None = None


async def _prepare_case_name(
    session: MelleaSession,
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str,
    extracted_defendant: str,
    extracted_decision_date: str,
    requirements: list[Requirement],
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> tuple[_PreparedCaseName, object]:
    """Prepare parties for case-name search through Mellea instruct/validate/repair."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        extracted_decision_date=extracted_decision_date,
        requirements=requirements,
    )
    result = await run_instruct_ivr(session, spec, strategy=strategy, model_options=model_options)
    proposal = _proposal_from_output(result.result.value)
    return proposal, result.result_ctx


def _case_name_preparation_spec(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str,
    extracted_defendant: str,
    extracted_decision_date: str,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=CASE_NAME_PREPARATION_INSTRUCTION,
        grounding_context={"local_context": local_context},
        user_variables={
            "locator": locator,
            "extracted_plaintiff": extracted_plaintiff or "<EMPTY>",
            "extracted_defendant": extracted_defendant or "<EMPTY>",
            "extracted_decision_date": extracted_decision_date or "<EMPTY>",
        },
        requirements=requirements,
    )


def _query_planning_spec(
    *,
    preparation: _PreparedCaseName,
    court: str | None,
    locator: str,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=QUERY_PLANNING_INSTRUCTION,
        user_variables={
            "plaintiff": preparation.plaintiff or "<EMPTY>",
            "defendant": preparation.defendant or "<EMPTY>",
            "decision_date": preparation.decision_date or "<EMPTY>",
            "court": court or "<EMPTY>",
            "locator": locator,
        },
        requirements=requirements,
    )


def render_case_name_preparation_prompt(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str = "",
    extracted_defendant: str = "",
    extracted_decision_date: str = "",
    window: DocumentTextWindow,
) -> str:
    """Render the raw prompt for a case-name preparation instruction."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        extracted_decision_date=extracted_decision_date,
        requirements=_case_name_preparation_requirements(window, locator),
    )
    return render_instruct_prompt(spec)


def render_case_name_preparation_chat_messages(
    *,
    local_context: str,
    locator: str,
    extracted_plaintiff: str = "",
    extracted_defendant: str = "",
    extracted_decision_date: str = "",
    window: DocumentTextWindow,
) -> tuple[RenderedChatMessage, ...]:
    """Render the raw chat messages for a case-name preparation instruction."""
    spec = _case_name_preparation_spec(
        local_context=local_context,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        extracted_decision_date=extracted_decision_date,
        requirements=_case_name_preparation_requirements(window, locator),
    )
    return render_instruct_chat_messages(spec)


def _case_name_preparation_requirements(window: DocumentTextWindow, locator: str) -> list[Requirement]:
    return [
        req(JSON_OUTPUT_REQUIREMENT, validation_fn=_validate_output_schema),
        check(
            "classification must be consistent with party availability",
            validation_fn=_validate_classification_consistency,
        ),
        req(
            "plaintiff and defendant must be copied from local_context before the locator",
            validation_fn=lambda ctx: _validate_grounded_before_locator(ctx, window, locator),
        ),
        req(
            "decision_date must normalize a copied complete month/day/year after this locator; ISO is not required verbatim",
            validation_fn=lambda ctx: _validate_date_grounding(ctx, window, locator),
        ),
    ]


async def _plan_case_name_query(
    session: MelleaSession,
    *,
    preparation: _PreparedCaseName,
    court: str | None,
    locator: str,
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> _PlannedCaseNameQuery:
    spec = _query_planning_spec(
        preparation=preparation,
        court=court,
        locator=locator,
        requirements=[req(QUERY_OUTPUT_REQUIREMENT, validation_fn=_validate_query_output)],
    )
    result = await run_instruct_ivr(session, spec, strategy=strategy, model_options=model_options)
    return _query_proposal_from_output(result.result.value)


async def prepare_case_name_for_search(
    session: MelleaSession,
    *,
    document_text: str,
    citation: ExtractedCitation,
) -> CaseNameSearchPreparation:
    """Prepare party anchors for a not-found citation's candidate search."""
    if not isinstance(citation.citation, FullCaseCitation):
        return CaseNameSearchPreparation(status=CaseNamePreparationStatus.EMPTY)

    # ``ExtractedCitation.citation_span`` is the full eyecite span around the
    # authority; ``matched_locator_text`` is the reporter/WL locator used for
    # exact lookup and as the boundary marker inside the local window.
    citation_span = citation.citation_span
    matched_locator_text = citation.matched_locator_text
    window = DocumentTextWindow.around(
        document_text,
        citation_span,
        before_chars=PREPARATION_CONTEXT_BEFORE_CHARS,
        # Eyecite can omit a whitespace-damaged court/date parenthetical from
        # its full span. Keep a bounded tail so grounded preparation can repair
        # that extraction without seeing an unbounded next citation.
        after_chars=PREPARATION_CONTEXT_AFTER_CHARS,
    )
    extracted_plaintiff = citation.citation.plaintiff or ""
    extracted_defendant = citation.citation.defendant or ""
    extracted_decision_date = citation.asserted_decision_date or ""
    try:
        requirements = _case_name_preparation_requirements(window, matched_locator_text)
        proposal, _final_ctx = await _prepare_case_name(
            session,
            local_context=window.text,
            locator=matched_locator_text,
            extracted_plaintiff=extracted_plaintiff,
            extracted_defendant=extracted_defendant,
            extracted_decision_date=extracted_decision_date,
            requirements=requirements,
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=PREPARATION_MAX_TOKENS),
        )
    except Exception as exc:
        return CaseNameSearchPreparation(
            status=CaseNamePreparationStatus.FAILED,
            original_case_name=_original_case_name(citation.citation),
            plaintiff=extracted_plaintiff or None,
            defendant=extracted_defendant or None,
            prepared_case_name=None,
            extracted_decision_date=extracted_decision_date or None,
            court=citation.citation.court,
            locator=matched_locator_text,
            source="llm",
            error_message=str(exc),
        )

    status = _status_from_classification(proposal.classification)
    date_basis = _date_grounding_basis(
        decision_date=proposal.decision_date,
        extracted_decision_date=extracted_decision_date or None,
        window=window,
        locator=matched_locator_text,
    )
    if status is not CaseNamePreparationStatus.ACCEPTED:
        return CaseNameSearchPreparation(
            status=status,
            original_case_name=_original_case_name(citation.citation),
            plaintiff=proposal.plaintiff,
            defendant=proposal.defendant,
            prepared_case_name=_prepared_case_name(proposal.plaintiff, proposal.defendant),
            extracted_decision_date=extracted_decision_date or None,
            decision_date=proposal.decision_date,
            decision_date_basis=date_basis,
            court=citation.citation.court,
            locator=matched_locator_text,
            source="llm",
            llm_classification=proposal.classification,
            llm_reason=proposal.reason,
        )
    try:
        query_plan = await _plan_case_name_query(
            session.clone(),
            preparation=proposal,
            court=citation.citation.court,
            locator=matched_locator_text,
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=PREPARATION_MAX_TOKENS),
        )
    except Exception as exc:
        return CaseNameSearchPreparation(
            status=CaseNamePreparationStatus.FAILED,
            original_case_name=_original_case_name(citation.citation),
            plaintiff=proposal.plaintiff,
            defendant=proposal.defendant,
            prepared_case_name=_prepared_case_name(proposal.plaintiff, proposal.defendant),
            extracted_decision_date=extracted_decision_date or None,
            decision_date=proposal.decision_date,
            decision_date_basis=date_basis,
            court=citation.citation.court,
            locator=matched_locator_text,
            source="llm",
            llm_classification=proposal.classification,
            llm_reason=proposal.reason,
            error_message=f"query planning failed: {exc}",
        )
    return CaseNameSearchPreparation(
        status=status,
        original_case_name=_original_case_name(citation.citation),
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
        prepared_case_name=_prepared_case_name(proposal.plaintiff, proposal.defendant),
        extracted_decision_date=extracted_decision_date or None,
        decision_date=proposal.decision_date,
        decision_date_basis=date_basis,
        query_plaintiff=query_plan.query_plaintiff,
        query_defendant=query_plan.query_defendant,
        query_reason=query_plan.reason,
        court=citation.citation.court,
        locator=matched_locator_text,
        source="llm",
        llm_classification=proposal.classification,
        llm_reason=proposal.reason,
    )


def _proposal_from_output(output: str | object) -> _PreparedCaseName:
    if not isinstance(output, str):
        msg = f"LLM output was not text: {type(output).__name__}"
        raise TypeError(msg)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        msg = f"LLM output was not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "LLM output JSON was not an object"
        raise TypeError(msg)
    try:
        return _PreparedCaseName.model_validate(payload)
    except ValidationError as exc:
        msg = f"LLM output did not match case-name preparation schema: {exc}"
        raise ValueError(msg) from exc


def _proposal_from_context(ctx: Context) -> _PreparedCaseName:
    return _proposal_from_output(ctx.last_output().value)


def _query_proposal_from_output(output: str | object) -> _PlannedCaseNameQuery:
    if not isinstance(output, str):
        msg = f"LLM output was not text: {type(output).__name__}"
        raise TypeError(msg)
    try:
        return _PlannedCaseNameQuery.model_validate_json(output)
    except ValidationError as exc:
        msg = f"LLM output did not match query-planning schema: {exc}"
        raise ValueError(msg) from exc


def _validate_query_output(ctx: Context) -> ValidationResult:
    try:
        proposal = _query_proposal_from_output(ctx.last_output().value)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    values = (proposal.query_plaintiff, proposal.query_defendant)
    if any(_QUERY_TERM.fullmatch(value) is None for value in values):
        return ValidationResult(result=False, reason="query terms were not safe plain text")
    return ValidationResult(result=True)


def _validate_output_schema(ctx: Context) -> ValidationResult:
    try:
        _proposal_from_context(ctx)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_classification_consistency(ctx: Context) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    party_count = sum(bool(value) for value in (proposal.plaintiff, proposal.defendant))
    if proposal.classification == "complete_case_name" and party_count == COMPLETE_PARTY_COUNT:
        return ValidationResult(result=True)
    if proposal.classification == "partial_case_name" and party_count == 1:
        return ValidationResult(result=True)
    if proposal.classification == "no_case_name" and party_count == 0:
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason="classification did not match plaintiff/defendant availability",
    )


def _validate_grounded_before_locator(
    ctx: Context,
    window: DocumentTextWindow,
    locator: str,
) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    locator_grounded = window.locate(locator)
    locator_global_start = (
        locator_grounded.global_span.start
        if locator_grounded is not None
        else window.anchor_global_span.start
    )
    for label, value in (("plaintiff", proposal.plaintiff), ("defendant", proposal.defendant)):
        if not value:
            continue
        grounded = window.locate(value)
        if grounded is None:
            return ValidationResult(
                result=False,
                reason=f"{label}={value!r} was not copied from local_context",
            )
        if grounded.global_span.end > locator_global_start:
            return ValidationResult(
                result=False,
                reason=f"{label}={value!r} did not appear before the locator",
            )
    return ValidationResult(result=True)


def _validate_date_grounding(
    ctx: Context,
    window: DocumentTextWindow,
    locator: str,
) -> ValidationResult:
    try:
        proposal = _proposal_from_context(ctx)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    basis = _date_grounding_basis(
        decision_date=proposal.decision_date,
        extracted_decision_date=None,
        window=window,
        locator=locator,
    )
    if proposal.decision_date is None:
        if _complete_copied_dates(_citation_parenthetical(window, locator)):
            return ValidationResult(
                result=False,
                reason="a complete copied date exists after the locator; normalize it instead of returning null",
            )
        return ValidationResult(result=True)
    if basis is None:
        return ValidationResult(result=False, reason="decision_date was not copied after the locator")
    return ValidationResult(result=True)


def _citation_parenthetical(window: DocumentTextWindow, locator: str) -> str:
    locator_grounded = window.locate(locator)
    locator_global_end = (
        locator_grounded.global_span.end
        if locator_grounded is not None
        else window.anchor_global_span.end
    )
    locator_window_end = window.global_offset_to_window(locator_global_end)
    citation_window_end = window.text.find(")", locator_window_end)
    if citation_window_end == -1:
        citation_window_end = window.global_offset_to_window(window.anchor_global_span.end)
    else:
        citation_window_end += 1
    return window.text[locator_window_end:citation_window_end]


def _date_grounding_basis(
    *,
    decision_date: str | None,
    extracted_decision_date: str | None,
    window: DocumentTextWindow,
    locator: str,
) -> str | None:
    if decision_date is None:
        return None
    if _ISO_DATE.fullmatch(decision_date) is None:
        return None
    parenthetical = _citation_parenthetical(window, locator)
    if _parenthetical_contains_date(parenthetical, decision_date):
        return "eyecite_hint_confirmed" if decision_date == extracted_decision_date else "mechanically_parsed"
    if _parenthetical_contains_ocr_date(parenthetical, decision_date):
        return "llm_repaired"
    return None


def _parenthetical_contains_date(text: str, expected_iso_date: str) -> bool:
    """Parse date-shaped substrings in the bounded citation parenthetical.

    This is evidence validation, not a retrieval fallback: it proves an LLM
    proposal corresponds to copied text and never manufactures a date.
    """
    formats = ("%b %d, %Y", "%B %d, %Y")
    for start in range(len(text)):
        for end in range(start + 8, min(len(text), start + 24) + 1):
            candidate = " ".join(text[start:end].replace(".", "").split())
            candidate = candidate.replace("Sept ", "Sep ")
            for format_string in formats:
                if _parsed_date_matches(candidate, format_string, expected_iso_date):
                    return True
    return False


def _complete_copied_dates(text: str) -> tuple[str, ...]:
    found: set[str] = set()
    formats = ("%b %d, %Y", "%B %d, %Y")
    for start in range(len(text)):
        for end in range(start + 8, min(len(text), start + 24) + 1):
            candidate = " ".join(text[start:end].replace(".", "").split())
            candidate = candidate.replace("Sept ", "Sep ")
            for format_string in formats:
                parsed = _parse_copied_date(candidate, format_string)
                if parsed is not None:
                    found.add(parsed)
    return tuple(sorted(found))


def _parsed_date_matches(candidate: str, format_string: str, expected_iso_date: str) -> bool:
    parsed = _parse_copied_date(candidate, format_string)
    return parsed == expected_iso_date


def _parse_copied_date(candidate: str, format_string: str) -> str | None:
    try:
        parsed = datetime.strptime(candidate, format_string)  # noqa: DTZ007 - date-only citation text
    except ValueError:
        return None
    return parsed.date().isoformat()


def _parenthetical_contains_ocr_date(text: str, expected_iso_date: str) -> bool:
    """Accept only a proposed date traceable to a common-OCR date fragment."""
    target = date.fromisoformat(expected_iso_date)
    tokens = [token.strip(".,;:()[]") for token in text.split()]
    for index, month_token in enumerate(tokens[:-2]):
        month = _MONTH_PREFIXES.get(month_token.lower().rstrip(".")[:3])
        if month != target.month:
            continue
        day = _ocr_int(tokens[index + 1])
        year = _ocr_int(tokens[index + 2])
        if day == target.day and year == target.year:
            return True
    return False


def _ocr_int(value: str) -> int | None:
    normalized = value.translate(_OCR_DIGITS)
    return int(normalized) if normalized.isdecimal() else None




def _status_from_classification(classification: str) -> CaseNamePreparationStatus:
    if classification == "complete_case_name":
        return CaseNamePreparationStatus.ACCEPTED
    if classification in {"partial_case_name", "no_case_name"}:
        return CaseNamePreparationStatus.EMPTY
    return CaseNamePreparationStatus.FAILED


def _original_case_name(citation: FullCaseCitation) -> str | None:
    return _prepared_case_name(citation.plaintiff, citation.defendant)


def _prepared_case_name(plaintiff: str | None, defendant: str | None) -> str | None:
    if plaintiff and defendant:
        return f"{plaintiff} v. {defendant}"
    return None
