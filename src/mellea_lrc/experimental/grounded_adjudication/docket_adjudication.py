"""Confirm a suspected docket number and pick the court that identifies it.

Both halves are grounded the same way the reporter locators are: the model
quotes each verbatim and the quote is resolved back into the document, so an
answer that was tidied or invented fails to resolve rather than producing a
span.

The court is the harder half. Candidate strings come from ``courts-db``, and a
window routinely offers more than one -- a preceding citation's ``8th Cir.``
alongside this case's ``M.D.N.C.``, or an unrelated ``Alaska`` beside
``D. Nev.``. Choosing between them is a reading task, so the prompt names the
candidates and what each resolves to, and the model picks; it is not asked to
recall what an abbreviation means, and a court it did not choose from the list
cannot be grounded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.core.spans import Span
from mellea_lrc.experimental.grounded_adjudication.docket_hunting import docket_context
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.experimental.grounded_adjudication.docket_hunting import SuspectedDocket

MAX_TOKENS = 260
MAX_REPAIR_TURNS = 2

INSTRUCTION = """
Below is a window of text from a legal filing. A docket-number-shaped string
was found in it: {{docket}}

{{court_context}}

Decide whether that string is a docket number cited for a COURT CASE, and if
so which court it belongs to.

Rules:
- Quote the docket number exactly as written in the window, character for
  character, including any damage. Do not repair it.
- Report the court by quoting one of the candidate strings listed above,
  exactly as written. If none of them is the court of THIS docket number,
  report court as null. Never name a court that is not written in the window.
- A window often contains a court from a NEARBY, different citation. Choose the
  court that belongs to this docket number, not the closest one.
- Set is_docket_citation=false if the string is not citing a case: an exhibit
  or docket-entry number, a statute, a filing reference, or an internal
  cross-reference.

window:
{{window}}
""".strip()


class _DocketProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_docket_citation: bool
    docket: str | None = None
    court: str | None = None


@dataclass(frozen=True, slots=True)
class AdjudicatedDocket:
    """One confirmed docket citation, with both halves grounded in the text."""

    docket_span: Span
    docket_text: str
    court_span: Span | None
    court_text: str | None
    court_id: str | None
    court_name: str | None


def _parse(value: object) -> _DocketProposal:
    return _DocketProposal.model_validate_json(str(value))


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValidationError as error:
        return ValidationResult(result=False, reason=str(error))
    return ValidationResult(result=True)


def _validate_docket(ctx: Context, site: SuspectedDocket) -> ValidationResult:
    """A confirmed citation must quote the docket number found in the window."""
    proposal = _parse(ctx.last_output().value)
    if not proposal.is_docket_citation:
        return ValidationResult(result=True)
    if not proposal.docket or proposal.docket not in site.window:
        return ValidationResult(
            result=False,
            reason=(
                f"{proposal.docket!r} is not written in the window. Quote the docket "
                f"number exactly as it appears, character for character."
            ),
        )
    return ValidationResult(result=True)


def _validate_court(ctx: Context, site: SuspectedDocket) -> ValidationResult:
    """The court must be one of the candidates, not one the model recalled."""
    proposal = _parse(ctx.last_output().value)
    if proposal.court is None:
        return ValidationResult(result=True)
    if not any(proposal.court == candidate.text for candidate in site.courts):
        offered = ", ".join(repr(candidate.text) for candidate in site.courts) or "none"
        return ValidationResult(
            result=False,
            reason=(
                f"{proposal.court!r} is not one of the court strings written near this "
                f"docket number. Choose exactly one of: {offered}; or report null."
            ),
        )
    return ValidationResult(result=True)


async def adjudicate_docket(
    document_text: str,
    site: SuspectedDocket,
    *,
    session: MelleaSession | None = None,
) -> AdjudicatedDocket | None:
    """Confirm one suspected docket site, or return None if it is not a citation."""
    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "docket": site.docket_text,
                "court_context": docket_context(site),
                "window": site.window,
            },
            output_format=_DocketProposal,
            requirements=[
                req("Return a valid docket proposal.", validation_fn=_validate_schema),
                req(
                    "Quote the docket number exactly as written in the window.",
                    validation_fn=lambda ctx: _validate_docket(ctx, site),
                ),
                req(
                    "Report the court by choosing one of the candidates written nearby.",
                    validation_fn=lambda ctx: _validate_court(ctx, site),
                ),
            ],
        ),
        strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
        model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
    )
    try:
        proposal = _parse(result.result.value)
    except ValidationError:
        return None
    if not proposal.is_docket_citation or not proposal.docket:
        return None

    # Ground the docket against the site rather than the whole document: the same
    # number is often cited repeatedly, and a document-wide search would return
    # the first occurrence instead of this one.
    offset = max(0, site.span_start - 170)
    found = site.window.find(proposal.docket)
    if found < 0:
        return None
    docket_span = Span(start=offset + found, end=offset + found + len(proposal.docket))

    chosen = next((c for c in site.courts if c.text == proposal.court), None)
    return AdjudicatedDocket(
        docket_span=docket_span,
        docket_text=proposal.docket,
        court_span=Span(start=chosen.span_start, end=chosen.span_end) if chosen else None,
        court_text=chosen.text if chosen else None,
        court_id=chosen.court_id if chosen else None,
        court_name=chosen.court_name if chosen else None,
    )
