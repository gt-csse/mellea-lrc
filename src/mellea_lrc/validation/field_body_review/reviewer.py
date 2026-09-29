"""Review case-name body hits as possible intended authorities, not locator proof."""

from __future__ import annotations

import json
import os
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
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundedFragment, GroundingEvidence
from mellea_lrc.model.citation_windows import after, before
from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.body_evidence import BodyEvidence, BodySource
from mellea_lrc.model.citations.field_body_evidence import IntendedCaseDecision
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.validation.body_search.common import diverse_evidence, locator_text, source_copy_bodies
from mellea_lrc.validation.body_search.grounding import ground_body_fragment

if TYPE_CHECKING:
    from mellea import MelleaSession


MAX_TOKENS = 4500
MAX_MODEL_ATTEMPTS = 3
SESSION_ID = "mellea-lrc-field-body-review-v1"


@dataclass(frozen=True, slots=True)
class IntendedCaseContext:
    """A routed root and independent, name-anchored excerpts from all providers."""

    source: str
    root: FullCitationVariant
    before_text: str
    after_text: str
    current: dict[str, str | None]
    evidence: tuple[tuple[BodySource, int, BodyEvidence], ...]

    @classmethod
    def from_document(cls, document: Document, root: FullCitationVariant) -> IntendedCaseContext:
        before_text, _ = before(document, root, 400)
        after_text, _ = after(document, root, 220)
        copies = source_copy_bodies(
            document.text,
            tuple((search.source, item) for search in root.field_body_searches for item in search.evidence),
        )
        evidence = tuple(
            selected
            for search in root.field_body_searches
            for selected in diverse_evidence(search.source, search.evidence)
            if selected[2].anchor_kind == "case_name" and (search.source, selected[2].body_id) not in copies
        )
        court = root.court[-1].quote if root.court else None
        if root.court and court is None and root.court[-1].normalizable:
            court = f"inferred {root.court[-1].get_normalized().name}"
        return cls(
            source=document.text,
            root=root,
            before_text=before_text,
            after_text=after_text,
            current={
                "locator": locator_text(root),
                "case_name": root.case_name[-1].quote if root.case_name else None,
                "court": court,
                "date": root.date[-1].quote if root.date else None,
            },
            evidence=evidence,
        )

    def selected(self, decision: IntendedCaseDecision) -> BodyEvidence | None:
        return next(
            (
                item
                for source, index, item in self.evidence
                if source is decision.source and index == decision.evidence_index
            ),
            None,
        )

    def grounded_quote(self, decision: IntendedCaseDecision) -> GroundedFragment[None] | None:
        item = self.selected(decision)
        if item is None or decision.citation_quote is None:
            return None
        return ground_body_fragment(
            GroundingEvidence((EvidenceCandidate(item.excerpt, None),)), decision.citation_quote
        )

    def validation_error(self, decision: IntendedCaseDecision) -> str | None:
        if decision.source is None:
            return None
        item = self.selected(decision)
        if item is None:
            return "Selected source and index do not identify a supplied case-name excerpt"
        grounded = self.grounded_quote(decision)
        if grounded is None:
            return "Quote the possible case citation from the fetched excerpt at 98% similarity"
        if grounded.start > item.anchor_span.start or grounded.end < item.anchor_span.end:
            return "The possible citation quote must include the discovered case-name anchor"
        quoted = GroundingEvidence((EvidenceCandidate(grounded.text, None),))
        for field in ("case_name", "locator", "court", "date"):
            value = getattr(decision, field)
            if value is not None and ground_body_fragment(quoted, value) is None:
                return f"The candidate {field} must be copied from its citation quote"
        return None

    def prompt_evidence(self) -> str:
        return json.dumps(
            [
                {
                    "source": source.value,
                    "evidence_index": index,
                    "body_id": item.body_id,
                    "issued_on": item.issued_on.isoformat() if item.issued_on else None,
                    "excerpt": item.excerpt,
                    "anchor_span": {"start": item.anchor_span.start, "end": item.anchor_span.end},
                }
                for source, index, item in self.evidence
            ],
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class IntendedCaseOutcome:
    decision: IntendedCaseDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class IntendedCaseReviewer(Protocol):
    def __call__(
        self, context: IntendedCaseContext
    ) -> Awaitable[IntendedCaseDecision | IntendedCaseOutcome]: ...


_PREFIX = """Find whether an independent opinion or filing suggests which case the source filing meant to cite. These excerpts were discovered by a case-name fragment, not by the source citation's reporter or docket locator. A name match, even with compatible court and date, does NOT establish that the source locator is correct. Do not issue a correct-identity verdict, correct the source locator, or silently treat different locators as equivalent. A case may have parallel citations, and different orders may share a docket; a different printed locator alone does not prove a wrong identity either.

Read the source citation and all supplied independent excerpts. If one excerpt plausibly identifies the intended case, select its source and evidence_index, quote only its case citation (including its own locator, court and date when printed), and copy its case_name, locator, court and date literally from that quote. Null is allowed for fields not printed there. Explain why it may be the intended case and any unresolved uncertainty about the source locator. Use confidence=likely only when the external citation and source context make the intended case clear; otherwise use possible. If the excerpts are unrelated or cannot distinguish plausible cases, decline: set source, evidence_index, citation_quote, case_name, locator, court, date and confidence to null, and explain why. A source document's issue date is not the cited decision date. Use only the supplied texts for case-specific facts."""

_INSTRUCTION = """Source citation kind: {{kind}}
Source filing locator: {{locator}}
Filing text before locator: {{before}}
Filing text after locator: {{after}}
Current source fields: {{current}}
Fetched independent excerpts: {{evidence}}"""


def _validate_grounding(ctx: object, context: IntendedCaseContext) -> ValidationResult:
    try:
        decision = IntendedCaseDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    error = context.validation_error(decision)
    return ValidationResult(result=error is None, reason=error)


@dataclass(frozen=True, slots=True)
class IvrIntendedCaseReviewer:
    session: MelleaSession
    model_options: dict[str, object]
    max_attempts: int = MAX_MODEL_ATTEMPTS

    @classmethod
    def from_env(cls) -> IvrIntendedCaseReviewer:
        load_dotenv(override=False)
        config = llm_api_config_from_env(os.environ)
        return cls(
            session=start_mellea_session_from_env(),
            model_options={
                **config.mellea_call_options(max_tokens=MAX_TOKENS),
                "extra_body": {"session_id": SESSION_ID},
            },
        )

    async def __call__(self, context: IntendedCaseContext) -> IntendedCaseOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=_PREFIX,
                user_variables={
                    "kind": context.root.kind.value,
                    "locator": context.current["locator"],
                    "before": context.before_text,
                    "after": context.after_text,
                    "current": json.dumps(context.current, ensure_ascii=False),
                    "evidence": context.prompt_evidence(),
                },
                output_format=IntendedCaseDecision,
                requirements=(
                    req(
                        "Ground the candidate citation and every copied field in the selected excerpt.",
                        validation_fn=lambda ctx: _validate_grounding(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return IntendedCaseOutcome(
                decision=None,
                run=run,
                failure_reason=run.failure_reason or "Intended-case review IVR failed",
            )
        return IntendedCaseOutcome(decision=IntendedCaseDecision.model_validate_json(run.output), run=run)
