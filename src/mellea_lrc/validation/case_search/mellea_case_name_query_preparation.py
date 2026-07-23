"""Mellea preparation of a CourtListener case-name search query."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.types import (
    CitationValidation,
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    MelleaCaseNameReextractionNode,
    MelleaCaseNameReextractionOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

MAX_TOKENS = 128
MAX_REPAIR_TURNS = 2
INSTRUCTION = """
Prepare broad but faithful party-name search terms for a legal case search.

You receive the plaintiff and defendant names from one case citation. A downstream
program will add search syntax and a court filter. Your only task is to provide
one term for each party in the structured output fields ``query_plaintiff`` and
``query_defendant``.

Prefer the shortest distinctive term likely to retrieve the same party. You may:

- normalize whitespace or punctuation;
- expand an unambiguous legal abbreviation;
- remove generic organizational or governmental wording when the remaining term
  preserves the party's identifying core;
- simplify reordered forms of the same identifying core.

Example:

- plaintiff: "City of New York"
- defendant: "New York City"
- query_plaintiff: "New York"
- query_defendant: "New York"

Another example:

- plaintiff: "Methodist Hosp. of Sacramento"
- defendant: "Shalala"
- query_plaintiff: "Methodist Hospital of Sacramento"
- query_defendant: "Shalala"

Do not use outside knowledge to correct or replace a name. Do not invent missing
words, combine the two parties, omit either party, add alternative terms, or
include quotes, operators, field names, court identifiers, or search-query
syntax. Return exactly one non-empty term for each output field.

plaintiff:
{{plaintiff}}

defendant:
{{defendant}}
""".strip()


class _QueryTermsProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_plaintiff: str
    query_defendant: str


async def run_mellea_case_name_query_preparation(
    validation: CitationValidation,
    *,
    reextraction: MelleaCaseNameReextractionNode,
    session: MelleaSession | None = None,
) -> MelleaCaseNameQueryPreparationNode:
    """Prepare query terms, then construct one CourtListener query locally."""
    court_id = _court_id(validation)
    if (
        reextraction.outcome is not MelleaCaseNameReextractionOutcome.COMPLETE
        or reextraction.plaintiff is None
        or reextraction.defendant is None
        or court_id is None
    ):
        return _node(
            validation,
            reextraction,
            ValidationNodeStatus.SKIPPED,
            MelleaCaseNameQueryPreparationOutcome.UNAVAILABLE,
            court_id=court_id,
        )
    try:
        resolved_session = session or start_mellea_session_from_env()
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "plaintiff": reextraction.plaintiff,
                "defendant": reextraction.defendant,
            },
            output_format=_QueryTermsProposal,
            requirements=[req("Return valid case-name query terms.", validation_fn=_valid_schema)],
        )
        result = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
        if not result.success:
            return _node(
                validation,
                reextraction,
                ValidationNodeStatus.FAILED,
                MelleaCaseNameQueryPreparationOutcome.FAILED,
                court_id=court_id,
                error="Case-name query preparation exhausted its repair budget",
            )
        terms = _proposal(result.result.value)
        query = _query(terms, court_id)
    except Exception as exc:
        return _node(
            validation,
            reextraction,
            ValidationNodeStatus.FAILED,
            MelleaCaseNameQueryPreparationOutcome.FAILED,
            court_id=court_id,
            error=f"{type(exc).__name__}: {exc}",
        )
    return _node(
        validation,
        reextraction,
        ValidationNodeStatus.SUCCEEDED,
        MelleaCaseNameQueryPreparationOutcome.PREPARED,
        court_id=court_id,
        query=query,
        query_plaintiff=terms.query_plaintiff,
        query_defendant=terms.query_defendant,
    )


def _node(
    validation: CitationValidation,
    reextraction: MelleaCaseNameReextractionNode,
    status: ValidationNodeStatus,
    outcome: MelleaCaseNameQueryPreparationOutcome,
    *,
    court_id: str | None,
    query: str | None = None,
    query_plaintiff: str | None = None,
    query_defendant: str | None = None,
    error: str | None = None,
) -> MelleaCaseNameQueryPreparationNode:
    return MelleaCaseNameQueryPreparationNode(
        node_id=f"{validation.citation_id}:mellea_case_name_query_preparation",
        status=status,
        outcome=outcome,
        query=query,
        query_plaintiff=query_plaintiff,
        query_defendant=query_defendant,
        court_id=court_id,
        depends_on=(reextraction.node_id,),
        error=error,
    )


def _court_id(validation: CitationValidation) -> str | None:
    citation = validation.citation.citation
    return citation.court if isinstance(citation, FullCaseCitation) else None


def _proposal(value: object) -> _QueryTermsProposal:
    try:
        proposal = _QueryTermsProposal.model_validate_json(value)
    except ValidationError as exc:
        msg = f"Invalid case-name query preparation output: {exc}"
        raise ValueError(msg) from exc
    if not proposal.query_plaintiff.strip() or not proposal.query_defendant.strip():
        msg = "Case-name query terms must not be blank"
        raise ValueError(msg)
    return proposal


def _valid_schema(ctx: Context) -> ValidationResult:
    try:
        _proposal(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _query(terms: _QueryTermsProposal, court_id: str) -> str:
    """Construct CourtListener query syntax outside the Mellea response."""
    return (
        f"caseName:({_term(terms.query_plaintiff)} AND {_term(terms.query_defendant)}) "
        f"AND court_id:{court_id}"
    )


def _term(value: str) -> str:
    """Return one quoted query term with CourtListener-special characters escaped."""
    escaped = value.strip().replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
