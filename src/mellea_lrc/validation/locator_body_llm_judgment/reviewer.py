"""One source-grounded review across the saved third-party citation excerpts."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
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
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.common import (
    diverse_evidence as _diverse_evidence,
)
from mellea_lrc.validation.body_search.common import (
    locator_text,
)
from mellea_lrc.validation.body_search.common import (
    source_copy_bodies as _source_copy_bodies,
)
from mellea_lrc.validation.body_search.grounding import ground_body_fragment
from mellea_lrc.validation.reporter_review.court_context import inferred_reporter_court_note

_WRITTEN_YEAR = re.compile(r"(?<!\d)(?:1[6-9]|20|21)\d{2}(?!\d)")


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
        copied_bodies = _source_copy_bodies(
            document.text,
            tuple((search.source, item) for search in root.body_searches for item in search.evidence),
        )
        evidence = tuple(
            selected
            for search in root.body_searches
            for selected in _diverse_evidence(search.source, search.evidence)
            if selected[2].anchor_kind == "locator"
            and (search.source, selected[2].body_id) not in copied_bodies
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
        return ground_body_fragment(
            GroundingEvidence((EvidenceCandidate(evidence.excerpt, None),)), decision.citation_quote
        )

    def grounded_context(self, decision: BodyCorroborationDecision) -> GroundedFragment[None] | None:
        evidence = self.selected(decision)
        if evidence is None or decision.context_quote is None:
            return None
        return ground_body_fragment(
            GroundingEvidence((EvidenceCandidate(evidence.excerpt, None),)), decision.context_quote
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
        found = ground_body_fragment(GroundingEvidence((EvidenceCandidate(window, offset),)), proposed)
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
        if decision.identity_verdict is IdentityVerdict.PARTIALLY_CORROBORATED and not isinstance(
            self.root, FullDocketCitation
        ):
            return "Case identity without decision identity is only available for docket citations"
        if (
            decision.identity_verdict is IdentityVerdict.CORRECT_IDENTITY
            and isinstance(self.root, FullDocketCitation)
            and decision.comparisons.date.result is not MatchResult.MATCH
        ):
            return "The cited docket decision date is not corroborated; qualify or defer the judgment"
        evidence = self.selected(decision)
        if evidence is None:
            return "Selected source/index does not identify a supplied third-party excerpt"
        if evidence.anchor_kind != "locator":
            return "Locator review requires an identifier grounded in the fetched body"
        if (problem := compare_presence(decision)) is not None:
            return problem
        quote = self.grounded_quote(decision)
        if quote is None:
            return "Quote the whole citation from the selected excerpt; it must ground at 98% similarity"
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
            matched_field = ground_body_fragment(quoted_citation, value)
            if matched_field is None:
                return f"The third-party {field} reading must occur inside its quoted citation"
        if decision.comparisons.locator.result is MatchResult.UNAVAILABLE:
            return "A selected third-party citation must compare both locator values"
        if self.corrected_spans(decision) is None:
            return "Changed filing fields must copy text from their allowed source windows"
        if decision.identity_verdict is IdentityVerdict.CORRECT_IDENTITY:
            filing_years = set(_WRITTEN_YEAR.findall(decision.filing.date or ""))
            cited_years = set(_WRITTEN_YEAR.findall(decision.third_party.date or ""))
            if len(filing_years) == len(cited_years) == 1 and filing_years != cited_years:
                return "The written decision years differ; an unqualified admission is unsupported"
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


_PREFIX = """Review whether an independent opinion or filing uses the supplied citation's reporter or docket locator. Every excerpt was found by that locator in fetched body text; a search hit or title alone is not evidence. The source IDs identify providers, not which answer is preferred. Examine all supplied excerpts and choose the most informative grounded occurrence, or decline with a specific reason. A challenge can be more informative than a routine citation, but first check that it challenges this citation's identity rather than a proposition, pinpoint, or another occurrence. A copy of the source filing is not independent evidence.

Read how the independent document treats that occurrence. It may cite the authority, explicitly identify the citation as nonexistent or incorrect, or merely quote/mention it without taking a position. A citation repeated in a list of errors is not corroboration. Set treatment accordingly and explain your choice. Copy the whole citation itself into citation_quote, including its written case name, locator, court and date where present. Keep surrounding discussion out of citation_quote. If the surrounding words explain its treatment, copy a short, exact fragment into context_quote; this is required when treatment is explicitly_disputes. A table heading can supply that context when it clearly governs the selected row. Do not apply a heading to an unrelated row. Read the four fields literally from citation_quote into third_party; use null for a field absent there. A locator is the base reporter or docket identifier, not an attached pinpoint page or star page; compare pinpoints separately from root identity.

Reread the source filing independently into filing. Keep its current extracted value when correct. To replace or supply a field, copy its source quote exactly from the filing window: case name before the locator, court and date after the colocation group, docket number within its locator. A reporter locator is fixed. For a complete adversarial name, use kind=adversarial with both printed parties joined by "v."; for "In re" or "Ex parte", use its printed subject. A credible lone-party or other incomplete name fragment belonging to this citation remains a filing.case_name reading and takes normalized_case_name with kind=partial and only its partial text. Leave plaintiff, defendant, and subject null for kind=partial. A fragment belonging to a neighboring citation is not this citation's name. Set both filing.case_name and filing.normalized_case_name to null only when no name or fragment is grounded for this citation. Never fill a missing party from the third-party citation. For an inferred court, the supplied inference note explains why the filing may have no printed court label.

Compare the two *printed citations* field by field, separately from treatment. Identical written names may match even when the surrounding discussion says that their citation is fictitious; treatment expresses that negative evidence. Use match for equivalent values, mismatch for real conflicts, unavailable only when one side lacks that field. Consider conventional abbreviations and equivalent docket-number forms, but treat a misspelling as a mismatch rather than an abbreviation. A third-party document's issue date merely identifies when that evidence existed; it is not the date written in its citation. Explain each assessment. Report a field conflict even when other fields match. Use only the supplied texts for case-specific facts.

Then make an independent final identity judgment in identity_verdict. The field comparisons are evidence, not a formula for this judgment. Use correct_identity only when the evidence supports the cited authority and decision without a material unresolved conflict; avoid admitting a false citation because a document merely repeats it. Use wrong_identity when the evidence clearly establishes a materially wrong cited identity or field. A docket can identify one case with multiple orders or opinions on different dates: if the case is corroborated but the particular cited decision or date is not, use partially_corroborated. This is a qualified finding, not a declaration that the full citation is correct. A different date alone does not establish a different case, and a matching docket alone does not prove the specific decision. If you select a grounded citation but cannot form a reliable positive, negative, or qualified opinion, use undetermined and explain the uncertainty. Explain the particular basis and any unresolved conflict in reason; do not treat the surrounding document's publication or filing date as the cited decision date.

Return all required structured fields. If no excerpt contains a reviewable citation at the locator, set source, evidence_index, citation_quote, context_quote, treatment, filing, third_party, comparisons, and identity_verdict to null, and explain why. That is a routing outcome, not an identity opinion."""

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
class IvrBodyCorroborationReviewer(IvrReviewer):
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
