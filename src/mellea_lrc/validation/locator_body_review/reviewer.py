"""One source-grounded review across the saved third-party citation excerpts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from dotenv import load_dotenv
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.config import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundedFragment, GroundingEvidence
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullCitationVariant, FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.body_evidence import (
    BodyCitationTreatment,
    BodyCorroborationDecision,
    BodyEvidence,
    BodySource,
    compare_presence,
)
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.common import locator_text
from mellea_lrc.validation.reporter_review.court_context import inferred_reporter_court_note

if TYPE_CHECKING:
    from mellea import MelleaSession


MAX_TOKENS = 5500
MAX_MODEL_ATTEMPTS = 3
MAX_EVIDENCE_PER_SOURCE = 6
SESSION_ID = "mellea-lrc-locator-body-review-v1"
_GROUNDING = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)


def _diverse_evidence(
    source: BodySource, items: tuple[BodyEvidence, ...]
) -> tuple[tuple[BodySource, int, BodyEvidence], ...]:
    """Show distinct citing documents before second occurrences of one body."""
    groups: dict[str, list[tuple[int, BodyEvidence]]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item.body_id, []).append((index, item))
    selected: list[tuple[BodySource, int, BodyEvidence]] = []
    depth = 0
    while len(selected) < MAX_EVIDENCE_PER_SOURCE:
        next_round = [
            (index, item)
            for group in groups.values()
            if len(group) > depth
            for index, item in group[depth : depth + 1]
        ]
        if not next_round:
            break
        selected.extend(
            (source, index, item) for index, item in next_round[: MAX_EVIDENCE_PER_SOURCE - len(selected)]
        )
        depth += 1
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class BodyCorroborationContext:
    """One citation and reviewable excerpts from all completed body providers."""

    source: str
    root: FullCitationVariant
    before_text: str
    before_offset: int
    after_text: str
    after_offset: int
    locator_quote: str
    current: dict[str, str | None]
    evidence: tuple[tuple[BodySource, int, BodyEvidence], ...]
    inferred_court_note: str | None

    @classmethod
    def from_document(cls, document: Document, root: FullCitationVariant) -> BodyCorroborationContext:
        before_text, before_offset = before(document, root, 400)
        after_text, after_offset = after(document, root, 220)
        if root.court:
            court = root.court[-1]
            current_court = court.quote
            if current_court is None and court.normalizable:
                normalized = court.get_normalized()
                current_court = f"inferred court {normalized.id} ({normalized.name})"
        else:
            current_court = None
        evidence = tuple(
            selected
            for search in root.body_searches
            for selected in _diverse_evidence(search.source, search.evidence)
            if selected[2].anchor_kind == "locator"
        )
        return cls(
            source=document.text,
            root=root,
            before_text=before_text,
            before_offset=before_offset,
            after_text=after_text,
            after_offset=after_offset,
            locator_quote=locator_text(root),
            current={
                "locator": locator_text(root),
                "case_name": root.case_name[-1].quote if root.case_name else None,
                "court": current_court,
                "date": root.date[-1].quote if root.date else None,
            },
            evidence=evidence,
            inferred_court_note=(
                inferred_reporter_court_note(root) if isinstance(root, FullReporterCitation) else None
            ),
        )

    def selected(self, decision: BodyCorroborationDecision) -> BodyEvidence | None:
        return next(
            (
                item
                for source, index, item in self.evidence
                if source is decision.source and index == decision.evidence_index
            ),
            None,
        )

    def grounded_quote(self, decision: BodyCorroborationDecision) -> GroundedFragment[None] | None:
        evidence = self.selected(decision)
        if evidence is None or decision.citation_quote is None:
            return None
        return GroundingEvidence((EvidenceCandidate(evidence.excerpt, None),)).find_fragment(
            decision.citation_quote, _GROUNDING
        )

    def grounded_context(self, decision: BodyCorroborationDecision) -> GroundedFragment[None] | None:
        evidence = self.selected(decision)
        if evidence is None or decision.context_quote is None:
            return None
        return GroundingEvidence((EvidenceCandidate(evidence.excerpt, None),)).find_fragment(
            decision.context_quote, _GROUNDING
        )

    def field_span(self, field: str, proposed: str) -> Span | None:
        if field == "locator":
            if not isinstance(self.root, FullDocketCitation):
                raise ValueError("Reporter locator cannot be revised in body review")
            window = self.source[self.root.locator_span.start : self.root.locator_span.end]
            offset = self.root.locator_span.start
        elif field == "case_name":
            window, offset = self.before_text, self.before_offset
        elif field in {"court", "date"}:
            window, offset = self.after_text, self.after_offset
        else:
            raise ValueError(f"Unknown citation field: {field}")
        found = GroundingEvidence((EvidenceCandidate(window, offset),)).find_fragment(proposed, _GROUNDING)
        return Span(offset + found.start, offset + found.end) if found is not None else None

    def corrected_spans(self, decision: BodyCorroborationDecision) -> dict[str, Span] | None:
        """Ground every changed filing quote before any field log is appended."""
        if decision.filing is None:
            return {}
        spans: dict[str, Span] = {}
        for field in ("locator", "case_name", "court", "date"):
            proposed = getattr(decision.filing, field)
            current = self.current[field]
            if proposed is None or proposed == current:
                continue
            if field == "locator" and not isinstance(self.root, FullDocketCitation):
                return None
            span = self.field_span(field, proposed)
            if span is None:
                return None
            spans[field] = span
        return spans

    def validation_error(self, decision: BodyCorroborationDecision) -> str | None:
        if decision.source is None:
            return None
        evidence = self.selected(decision)
        if evidence is None:
            return "Selected source/index does not identify a supplied third-party excerpt"
        if evidence.anchor_kind != "locator":
            return "Locator review requires an identifier grounded in the fetched body"
        if (problem := compare_presence(decision)) is not None:
            return problem
        quote = self.grounded_quote(decision)
        if quote is None:
            return "Quote the whole citation from the selected excerpt; it must ground at 90% similarity"
        anchor = evidence.anchor_span
        if quote.end <= anchor.start or quote.start >= anchor.end:
            return "The whole-citation quote must overlap the grounded discovery anchor"
        if quote.start > anchor.start or quote.end < anchor.end:
            return "The whole-citation quote must include the located identifier in that excerpt"
        context = self.grounded_context(decision)
        if decision.context_quote is not None and context is None:
            return "The surrounding-context quote must ground in the selected excerpt"
        if decision.treatment is BodyCitationTreatment.EXPLICITLY_DISPUTES:
            if context is None:
                return "An explicit challenge needs a grounded quote of its surrounding discussion"
            if quote.start <= context.start and context.end <= quote.end:
                return "Quote the words challenging the citation, not just the citation itself"
        quoted_citation = GroundingEvidence((EvidenceCandidate(quote.text, None),))
        for field in ("locator", "case_name", "court", "date"):
            value = getattr(decision.third_party, field)
            if value is None:
                continue
            matched_field = quoted_citation.find_fragment(value, _GROUNDING)
            if matched_field is None:
                return f"The third-party {field} reading must occur inside its quoted citation"
            if re.findall(r"\d+", value) != re.findall(r"\d+", matched_field.text):
                return f"The third-party {field} reading's numeric parts must match its grounded quote"
        if decision.comparisons.locator.result is MatchResult.UNAVAILABLE:
            return "A selected third-party citation must compare both locator values"
        if self.corrected_spans(decision) is None:
            return "Changed filing fields must copy text from their allowed source windows"
        return None

    def prompt_evidence(self) -> str:
        return json.dumps(
            [
                {
                    "source": source.value,
                    "evidence_index": index,
                    "body_id": item.body_id,
                    "parent_id": item.parent_id,
                    "issued_on": item.issued_on.isoformat() if item.issued_on else None,
                    "excerpt": item.excerpt,
                    "anchor_kind": item.anchor_kind,
                    "anchor_span": {"start": item.anchor_span.start, "end": item.anchor_span.end},
                }
                for source, index, item in self.evidence
            ],
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class BodyCorroborationOutcome:
    decision: BodyCorroborationDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class BodyCorroborationReviewer(Protocol):
    def __call__(
        self, context: BodyCorroborationContext
    ) -> Awaitable[BodyCorroborationDecision | BodyCorroborationOutcome]: ...


_PREFIX = """Review whether an independent opinion or filing uses the supplied citation's reporter or docket locator. Every excerpt was found by that locator in fetched body text; a search hit or title alone is not evidence. The source IDs identify providers, not which answer is preferred. Examine all supplied excerpts and choose the most informative grounded occurrence, or decline with a specific reason. When one document explicitly challenges the cited locator and another merely repeats it, choose the explicit challenge. Do not choose a copy of the source filing as independent evidence.

Read how the independent document treats that occurrence. It may cite the authority, explicitly identify the citation as nonexistent or incorrect, or merely quote/mention it without taking a position. A citation repeated in a list of errors is not corroboration. Set treatment accordingly and explain your choice. Copy the whole citation itself into citation_quote, including its written case name, locator, court and date where present. Keep surrounding discussion out of citation_quote. If the surrounding words explain its treatment, copy a short, exact fragment into context_quote; this is required when treatment is explicitly_disputes. A table heading can supply that context when it clearly governs the selected row. Do not apply a heading to an unrelated row. Read the four fields literally from citation_quote into third_party; use null for a field absent there.

Reread the source filing independently into filing. Keep its current extracted value when correct. To replace or supply a field, copy its source quote exactly from the filing window: case name before the locator, court and date after the colocation group, docket number within its locator. A reporter locator is fixed. A case name must be complete enough to normalize: an adversarial name needs both printed parties joined by "v.", while "In re" and "Ex parte" need a printed subject. A lone party name or a fragment belonging to a neighboring citation is not a complete case name; set both filing.case_name and filing.normalized_case_name to null in that situation. Never fill a missing party from the third-party citation. If a complete grounded case name is present, supply its structured normalized form from the filing. For an inferred court, the supplied inference note explains why the filing may have no printed court label.

Compare the two *printed citations* field by field, separately from treatment. Identical written names may match even when the surrounding discussion says that their citation is fictitious; treatment expresses that negative evidence. Use match for equivalent values, mismatch for real conflicts, unavailable only when one side lacks that field. Consider conventional abbreviations and equivalent docket-number forms, but treat a misspelling as a mismatch rather than an abbreviation. A third-party document's issue date merely identifies when that evidence existed; it is not the date written in its citation. Explain each assessment. Report a field conflict even when other fields match. Use only the supplied texts for case-specific facts.

Return all required structured fields. If no excerpt contains a reviewable citation at the locator, set source, evidence_index, citation_quote, context_quote, treatment, filing, third_party, and comparisons to null and explain why."""

_INSTRUCTION = """Source citation kind: {{kind}}
Source locator: {{locator}}
Filing text before locator: {{before}}
Filing text after locator: {{after}}
Current filing readings: {{current}}
Reporter-based court inference, if any: {{inferred_court_note}}
Fetched third-party citation excerpts with source and evidence_index: {{evidence}}"""


def _validate_grounding(ctx: object, context: BodyCorroborationContext) -> ValidationResult:
    try:
        decision = BodyCorroborationDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    error = context.validation_error(decision)
    return ValidationResult(result=error is None, reason=error)


@dataclass(frozen=True, slots=True)
class IvrBodyCorroborationReviewer:
    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrBodyCorroborationReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, context: BodyCorroborationContext) -> BodyCorroborationOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={
                    "kind": context.root.kind.value,
                    "locator": context.locator_quote,
                    "before": context.before_text,
                    "after": context.after_text,
                    "current": json.dumps(context.current, ensure_ascii=False),
                    "inferred_court_note": context.inferred_court_note or "none",
                    "evidence": context.prompt_evidence(),
                },
                output_format=BodyCorroborationDecision,
                requirements=(
                    req(
                        "Ground the selected citation and any changed filing quotes in supplied text.",
                        validation_fn=lambda ctx: _validate_grounding(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return BodyCorroborationOutcome(
                decision=None,
                run=run,
                failure_reason=run.failure_reason or "Body corroboration IVR failed",
            )
        return BodyCorroborationOutcome(
            decision=BodyCorroborationDecision.model_validate_json(run.output), run=run
        )
