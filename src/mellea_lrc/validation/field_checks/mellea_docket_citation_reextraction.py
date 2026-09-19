"""One grounded, combined re-extraction of a docket citation after lookup misses.

This is source reading, not an identity decision. The model re-reads the target
citation as a whole—case name, docket number, court, date, and pin cite—while
only the docket value is currently eligible to re-enqueue a CourtListener
search. The other re-extracted fields are first-class source evidence for the
later candidate-identity checks; they are never silently normalized or copied
from a retrieved record.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Annotated

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.core.fuzziness import FuzzinessOption, FuzzinessType
from mellea_lrc.extraction.reading.dockets import DOCKET_PREFIX
from mellea_lrc.llm import (
    EvidenceCandidate,
    GroundingEvidence,
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.root_context import masked_root_context
from mellea_lrc.validation.types import (
    MelleaDocketCitationReextractionNode,
    MelleaDocketCitationReextractionOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.types import Document

MAX_TOKENS = 768
MAX_REPAIR_TURNS = 2
# A source locator or a model reproduction can carry isolated OCR/converter
# damage. The evidence still supplies the canonical value admitted to the
# record; 90% similarity permits a small number of such differences, while
# the grounding set refuses a proposal that could name two source identifiers.
DOCKET_NUMBER_GROUNDING = FuzzinessOption.edit_distance(
    similarity_percent=90,
    whitespace_relaxation=True,
)
_LEADING_DOCKET_LABEL = re.compile(DOCKET_PREFIX, re.IGNORECASE)
_PLURAL_CASE_LABEL = re.compile(r"\bCase\s+Nos?\.\s*", re.IGNORECASE)
_SOURCE_DOCKET_SEQUENCE = re.compile(r"[A-Za-z0-9:./\\-]+(?:\s+[A-Za-z0-9:./\\-]+)*")
_SOURCE_DOCKET_SEPARATOR = re.compile(r"\s+\band\b\s+", re.IGNORECASE)

# This prefix is intentionally general. It supplies the citation-reading
# contract rather than a jurisdictional docket grammar or corpus examples.
DOCKET_CITATION_REEXTRACTION_PREFIX = """
Read one target docket citation from a legal filing as one combined citation.
Re-extract every field that the filing actually states: case_name,
docket_number, court, date, and pin_cite. A docket number is the
court-assigned identifier for a case or proceeding; it is not an ECF entry
number, exhibit number, statute, or page number.

Copy each present field exactly from source_citation. Return null for a field
that source_citation does not state. Do not normalize punctuation, expand an
abbreviation, add digits, infer a value from outside knowledge, or borrow from
another citation. Do not include a leading label such as ``No.`` or ``Case
No.`` in docket_number. Preserve whitespace damage when it is written.

The prior extracted docket number is context only. Compare it to the source and
correct a copied character or boundary when necessary. Give one short prose
reason describing the reading.
""".strip()

INSTRUCTION = """
source_citation:
{{source_citation}}

source_locator:
{{source_locator}}

extracted_docket_number:
{{extracted_docket_number}}

The surrounding target-only filing context is below. Other citation locators
are blanked. Use it only to understand the target citation; every returned
field must still be copied from source_citation.

local_context:
{{local_context}}
""".strip()


class _DocketCitationReextractionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_name: str | None
    docket_number: str | None
    court: str | None
    date: str | None
    pin_cite: str | None
    reason: Annotated[str, StringConstraints(min_length=1)]


def _parse(value: object) -> _DocketCitationReextractionProposal:
    return _DocketCitationReextractionProposal.model_validate_json(str(value))


def _grounded_candidates(source_locator: str) -> GroundingEvidence[str]:
    """Expose complete digit-bearing identifier sequences as source evidence.

    This is only a grounding boundary: it takes no position on what docket
    syntax is valid. A standalone ``and`` separates complete cited locators,
    while a parenthetical court or date never becomes identifier evidence.
    """
    source_number = _source_number_region(source_locator)
    candidates = []
    for portion in _SOURCE_DOCKET_SEPARATOR.split(source_number):
        match = _SOURCE_DOCKET_SEQUENCE.match(portion.strip())
        if match is not None and any(character.isdigit() for character in match.group()):
            candidates.append(EvidenceCandidate(text=match.group(), value=match.group()))
    return GroundingEvidence(candidates)


def _source_number_region(source_locator: str) -> str:
    """Exclude a recognized leading docket label from identifier evidence."""
    match = _LEADING_DOCKET_LABEL.match(source_locator) or _PLURAL_CASE_LABEL.match(source_locator)
    return source_locator[match.end() :] if match is not None else source_locator


def _grounded_number(source_locator: str, proposed: str | None) -> str | None:
    """Return one source spelling or refuse an ambiguous/non-source proposal."""
    if proposed is None:
        return None
    matches = _grounded_candidates(source_locator).fuzzy_match(proposed, DOCKET_NUMBER_GROUNDING)
    exact = tuple(
        match
        for match in matches
        if match.match_type in {FuzzinessType.PERFECT_MATCH, FuzzinessType.WHITESPACE_RELAXATION}
    )
    if exact:
        return exact[0].candidate.value if len(exact) == 1 else None
    if not matches:
        return None
    best_edits = min(match.edits for match in matches)
    best = tuple(match for match in matches if match.edits == best_edits)
    return best[0].candidate.value if len(best) == 1 else None


def _copied_from(value: str | None, source_citation: str) -> bool:
    """Ground a non-docket field by its literal tokens and flexible whitespace."""
    if value is None:
        return True
    pattern = r"\s+".join(re.escape(token) for token in value.split())
    return bool(pattern) and re.search(pattern, source_citation) is not None


def _validate_reextraction(
    ctx: Context,
    *,
    source_locator: str,
    source_citation: str,
) -> ValidationResult:
    try:
        proposal = _parse(ctx.last_output().value)
    except ValidationError:
        # The shared IVR wrapper supplies the schema diagnostic before this
        # custom grounding condition runs.
        return ValidationResult(result=True)
    grounded_number = _grounded_number(source_locator, proposal.docket_number)
    if proposal.docket_number is not None and grounded_number is None:
        return ValidationResult(
            result=False,
            reason=(
                "docket_number must resolve to one unambiguous source_locator substring at 90% similarity "
                "after whitespace normalization."
            ),
        )
    missing = [
        label
        for label, value in (
            ("case_name", proposal.case_name),
            ("court", proposal.court),
            ("date", proposal.date),
            ("pin_cite", proposal.pin_cite),
        )
        if not _copied_from(value, source_citation)
    ]
    return ValidationResult(
        result=not missing,
        reason=None if not missing else f"fields must be copied from source_citation: {', '.join(missing)}",
    )


def _node(
    *,
    record: CitationRecord,
    source_citation: str,
    source_locator: str,
    trigger_node_id: str,
    status: ValidationNodeStatus,
    outcome: MelleaDocketCitationReextractionOutcome,
    reparsed_case_name: str | None = None,
    reparsed_docket_number: str | None = None,
    reparsed_court: str | None = None,
    reparsed_date: str | None = None,
    reparsed_pin_cite: str | None = None,
    grounded_docket_number: str | None = None,
    reason: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run=None,
) -> MelleaDocketCitationReextractionNode:
    citation = record.stated
    extracted = citation.docket_number if isinstance(citation, DocketCitation) else None
    return MelleaDocketCitationReextractionNode(
        node_id=f"{record.citation_id}:mellea_docket_citation_reextraction",
        status=status,
        outcome=outcome,
        source_citation=source_citation,
        source_locator=source_locator,
        extracted_docket_number=extracted,
        reparsed_case_name=reparsed_case_name,
        reparsed_docket_number=reparsed_docket_number,
        reparsed_court=reparsed_court,
        reparsed_date=reparsed_date,
        reparsed_pin_cite=reparsed_pin_cite,
        grounded_docket_number=grounded_docket_number,
        reason=reason,
        depends_on=(trigger_node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


async def run_mellea_docket_citation_reextraction(
    record: CitationRecord,
    *,
    document: Document,
    trigger_node_id: str,
    session: MelleaSession | None = None,
) -> MelleaDocketCitationReextractionNode:
    """Re-read one unresolved docket citation and ground every returned field."""
    source_locator = document.text[record.locator_span.start : record.locator_span.end]
    source_citation = document.text[record.full_span.start : record.full_span.end]
    context = masked_root_context(document, record, before=640, after=320)
    citation = record.stated
    extracted = citation.docket_number if isinstance(citation, DocketCitation) else None
    try:
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                description=INSTRUCTION,
                prefix=DOCKET_CITATION_REEXTRACTION_PREFIX,
                grounding_context={"local_context": context.text},
                user_variables={
                    "source_citation": source_citation,
                    "source_locator": source_locator,
                    "extracted_docket_number": extracted or "(not extracted)",
                },
                output_format=_DocketCitationReextractionProposal,
                requirements=[
                    req(
                        "every re-extracted field must be grounded in the target citation",
                        validation_fn=lambda ctx: _validate_reextraction(
                            ctx,
                            source_locator=source_locator,
                            source_citation=source_citation,
                        ),
                    )
                ],
            ),
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
    except Exception as exc:
        return _node(
            record=record,
            source_citation=source_citation,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketCitationReextractionOutcome.FAILED,
            status_message="Model docket-citation re-extraction failed during execution.",
            outcome_message="No docket-citation re-extraction was admitted.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not result.success:
        return _node(
            record=record,
            source_citation=source_citation,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketCitationReextractionOutcome.FAILED,
            status_message="Model docket-citation re-extraction exhausted its repair attempts.",
            outcome_message="No docket-citation re-extraction was admitted.",
            error=result.failure_reason or "Model docket-citation re-extraction exhausted its repair budget",
            run=result,
        )
    try:
        proposal = _parse(result.output)
    except ValidationError as exc:
        return _node(
            record=record,
            source_citation=source_citation,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaDocketCitationReextractionOutcome.FAILED,
            status_message="Model docket-citation re-extraction returned invalid structured output.",
            outcome_message="No docket-citation re-extraction was admitted.",
            error=str(exc),
            run=result,
        )

    grounded = _grounded_number(source_locator, proposal.docket_number)
    fields = {
        "reparsed_case_name": proposal.case_name,
        "reparsed_docket_number": proposal.docket_number,
        "reparsed_court": proposal.court,
        "reparsed_date": proposal.date,
        "reparsed_pin_cite": proposal.pin_cite,
        "grounded_docket_number": grounded,
        "reason": proposal.reason,
        "run": result,
    }
    if grounded is None:
        return _node(
            record=record,
            source_citation=source_citation,
            source_locator=source_locator,
            trigger_node_id=trigger_node_id,
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=MelleaDocketCitationReextractionOutcome.NO_DOCKET_NUMBER,
            status_message="Model docket-citation re-extraction completed.",
            outcome_message="The model found no supportable docket number inside the target citation.",
            **fields,
        )
    outcome = (
        MelleaDocketCitationReextractionOutcome.UNCHANGED
        if grounded == extracted
        else MelleaDocketCitationReextractionOutcome.CORRECTED
    )
    return _node(
        record=record,
        source_citation=source_citation,
        source_locator=source_locator,
        trigger_node_id=trigger_node_id,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        status_message="Model docket-citation re-extraction completed.",
        outcome_message=(
            "The model confirmed the stated docket number while re-reading the full citation."
            if outcome is MelleaDocketCitationReextractionOutcome.UNCHANGED
            else "The model recovered a different docket number from the target citation."
        ),
        **fields,
    )
