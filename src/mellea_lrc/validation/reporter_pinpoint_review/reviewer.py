"""Shared IVR support review for a page bundle or a root's complete opinions.

Only the full-opinion route places whole root-owned writings in the reusable
system prefix. Dynamic propositions and citation context follow that prefix.
Page review supplies only selected pages, so it cannot silently use other pages.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields.pin_cite import PinCiteTarget
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_pages import OpinionPage
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionReviewScope,
    OpinionSupportResult,
    ReporterOpinionEvidence,
    ReporterSupportDecision,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span

GROUNDING = FuzzinessOption.edit_distance(similarity_percent=98, whitespace_relaxation=True)

_PREFIX = """Review how a legal citation uses the supplied judicial text. Use only this evidence, not memory of the case. Each occurrence has its own proposition, pinpoint, and intended writing; other occurrences of the same case do not decide this one.

Read the actual attributed use in its context, including introductory signals and explanatory parentheticals. A quotation represents the quoted words; a paraphrase may express equivalent meaning. Do not excuse invented quoted language merely because a related idea exists. An adverse or analogous signal can intentionally cite contrary or distinguishable reasoning. Assess that use rather than requiring every citation to affirm the filing's conclusion. Read qualifications, negations, and factual limits. Distinguish the court's own reasoning from quoted allegations, arguments it rejects, and editorial summaries. Honor an identified concurrence or dissent; matching words in another writing alone do not establish the intended attribution. Provider type/order and a combined opinion are clues, not proof of authorship.

Return supported when the attributed use is supported; contradicted when affirmative source evidence shows the attribution materially changes its meaning; not_found when the full supplied intended opinion was reviewed and the attributed words or idea are absent; unavailable when the supplied materials cannot settle the issue. Missing text or pagination alone never proves the attribution false. Never use not_found as a full-opinion conclusion from selected pages alone. Explain your reasoning as prose.

Distinguish no relevant discussion or quoted passage anywhere in the complete intended opinion from an existing passage whose meaning the filing misrepresents. An existing passage may fail to support the attributed use because of its meaning, qualifications, or relevance; that content assessment alone does not establish absence. Apply the result definitions to the attributed use and explain the distinction.

Quote short, exact judicial passages that establish support or contradiction, with their supplied opinion_id. Use the shortest sufficient contiguous phrase or sentence; surrounding source context is already retained. Copy evidence entirely from one supplied excerpt; do not invent or join separated text. Evidence is checked against source spans with shared whitespace and 98% edit-distance grounding. Supported and contradicted require evidence. Do not merely quote a matching phrase without considering its surrounding meaning.

Assess content and page placement independently. correct_page concerns whether the relevant discussion or quoted passage is at the written pinpoint, not whether the filing's interpretation is right. A correctly located discussion or quoted passage can be misrepresented, contradicted by its context, or irrelevant to the asserted proposition; those content problems do not by themselves make correct_page false. A shared topic or isolated matching word alone does not establish the relevant discussion or quoted passage.

pagination_available means this supplied source has usable pagination in the cited reporter/database namespace, not merely PDF/display pages or pagination of another reporter. If false, correct_page must be null and found_pages must be empty. If true, correct_page is true when the written page/paragraph range and any designated footnote contain the relevant discussion or quoted passage, false when placement is established to be wrong, and null when placement remains uncertain. A range needs the relevant passage somewhere within it, not independently on every page. Establish location or absence at the written target separately from the content result.

found_pages lists known locations of your quoted relevant discussions or quoted passages, whether they support or undermine the attributed use, as inclusive first/last ranges, kind (page, star, or paragraph), and footnote (null unless established). It is not an exhaustive list. Use source page markers, including explicit markers in the source text; do not invent page numbers or infer them from display order. Finding a passage elsewhere does not prove that equivalent relevant material is absent from the written target: review that target before returning correct_page=false. When returning correct_page=true, quote a relevant passage at that target. Pagination uncertainty never changes a supported attribution into a false one.

Shared saved opinion sources:
"""

_INSTRUCTION = """Review scope: {{scope}}
Complete text available for every retrieved subopinion: {{source_complete}}

Citing occurrence: {{citation_quote}}
Nearby filing text (absolute offset {{source_offset}}):
{{citing_context}}

Grounded proposition passages from this occurrence:
{{proposition}}

Written or immediately inherited pinpoint and requested page selections:
{{target}}

Treat the grounded proposition passages as the fixed attributed use for this occurrence. Assess every material assertion and qualification actually attributed to this citation, preserving its signal. A supported narrower parenthetical or related principle does not establish a broader asserted rule; do not approve only the matching part or replace the filing's attribution with a weaker claim.

Full-opinion fallback expands the source evidence, not the attributed use or the standard of support. Keep the same grounded proposition when moving from selected pages to the full opinion. If additional text changes the assessment, explain which new passage establishes support for that same attribution, including its material limits; do not justify a reversal by restating the attribution more generally.

Return separate content and page-location assessments. Decide the page fields from the location of the relevant discussion or quoted passage, independently of result. Explain absence, misrepresentation, or irrelevance without conflating it with placement.

If scope is cited_pages, evaluate only these supplied pages. A negative or uncertain page result will be reviewed against the full opinion later. If scope is full_opinion, search the complete supplied writings and identify the intended one using the context. Missing subopinions require caution about absence, though affirmative evidence may still settle a proposition. Return result, evidence, pagination_available, correct_page, found_pages, and reason."""


@dataclass(frozen=True, slots=True)
class OpinionExcerpt:
    opinion_id: str
    text: str
    offset: int
    pages: tuple[OpinionPage, ...] = ()


@dataclass(frozen=True, slots=True)
class ReporterPinpointReviewContext:
    citation_id: str
    root_id: str
    scope: OpinionReviewScope
    evidence_index: int
    prefix: str
    citing_context: str
    source_offset: int
    citation_quote: str
    proposition: str
    target: str
    source_complete: bool
    excerpts: tuple[OpinionExcerpt, ...]
    target_pages: tuple[PinCiteTarget, ...] = ()

    @classmethod
    def from_document(
        cls, document: Document, citation: Citation, scope: OpinionReviewScope
    ) -> ReporterPinpointReviewContext:
        evidence_index = len(citation.reporter_pinpoint_evidence) - 1
        evidence = citation.reporter_pinpoint_evidence[evidence_index]
        roots = {root.id: root for root in document.roots}
        root = roots.get(evidence.root_id)
        if not isinstance(root, FullReporterCitation):
            raise ValueError("Pinpoint review requires its attached reporter root")
        retrieval = root.reporter_root_opinion_retrieval
        index = root.reporter_root_opinion_page_index
        source = root.reporter_root_opinion_source
        if retrieval is None or index is None or source is None or evidence.proposition_index is None:
            raise ValueError("Pinpoint review needs the retained source, index, and proposition")
        proposition = citation.reporter_propositions[evidence.proposition_index]
        resolution = citation.reporter_page_resolutions[evidence.resolution_index]
        by_id = {item.id: item for item in document.citations}
        pin_source = by_id.get(resolution.pin_citation_id)
        if pin_source is None or pin_source.pin_cite is None or resolution.pin_reading_index is None:
            raise ValueError("Pinpoint review requires its original pinpoint reading")
        pin = pin_source.pin_cite[resolution.pin_reading_index]
        opinions = {item.opinion_id: item for item in index.opinions}
        metadata = {item.opinion_id: item.response or {} for item in retrieval.opinions}
        locator_source = by_id[resolution.locator_citation_id]
        readings = getattr(locator_source, "short_locator", None) or locator_source.locator
        locator = readings[resolution.locator_reading_index].get_normalized()

        def matching_pages(identifier: str) -> tuple[OpinionPage, ...]:
            import re

            from mellea_lrc.matching.literal_to_regex import fuzzy_literal

            return tuple(
                page
                for page in opinions[identifier].pages
                if page.volume == locator.volume
                and page.edition is not None
                and re.fullmatch(
                    fuzzy_literal(locator.edition, whitespace=True, newline=True), page.edition, re.I
                )
            )

        excerpts: list[OpinionExcerpt] = []
        if scope is OpinionReviewScope.CITED_PAGES:
            unique_pages = dict.fromkeys((page.opinion_id, page.page_index) for page in evidence.pages)
            for identifier, page_index in unique_pages:
                opinion = opinions[identifier]
                page = opinion.pages[page_index]
                excerpts.append(
                    OpinionExcerpt(
                        identifier,
                        opinion.text[page.span.start : page.span.end],
                        page.span.start,
                        tuple(
                            p
                            for p in matching_pages(identifier)
                            if p.span.start < page.span.end and p.span.end > page.span.start
                        ),
                    )
                )
        else:
            excerpts.extend(
                OpinionExcerpt(opinion.opinion_id, opinion.text, 0, matching_pages(opinion.opinion_id))
                for opinion in sorted(index.opinions, key=lambda item: int(item.opinion_id))
                if opinion.text.strip()
            )
        if not excerpts:
            raise ValueError("Cannot review an empty source bundle")
        shared = [
            {
                "opinion_id": excerpt.opinion_id,
                "type": metadata[excerpt.opinion_id].get("type"),
                "author": metadata[excerpt.opinion_id].get("author_str"),
                "source_offset": excerpt.offset,
                "text": excerpt.text,
                # All source namespaces belong in the shared prefix. Filtering
                # by this occurrence's reporter would break root-wide KV reuse
                # for parallel reporters; the requested namespace stays below.
                "page_markers": [
                    page.model_dump(mode="json")
                    for page in opinions[excerpt.opinion_id].pages
                    if page.span.start < excerpt.offset + len(excerpt.text) and page.span.end > excerpt.offset
                ],
            }
            for excerpt in excerpts
        ]
        span = citation.site_span
        start = max(0, span.start - 1400)
        stop = min(len(document.text), span.end + 700)
        return cls(
            citation_id=citation.id,
            root_id=root.id,
            scope=scope,
            evidence_index=evidence_index,
            prefix=_PREFIX
            + json.dumps(
                {
                    "cluster_id": source.cluster.id,
                    "parallel_reporter_citations": source.cluster.raw_json.get("citations", []),
                    "opinions": shared,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            citing_context=document.text[start:stop],
            source_offset=start,
            citation_quote=document.text[span.start : span.end],
            proposition=json.dumps([passage.model_dump(mode="json") for passage in proposition.passages]),
            target=json.dumps(
                {
                    "reporter": {"volume": locator.volume, "edition": locator.edition},
                    "quote": pin.quote,
                    "normalized": [item.model_dump(mode="json") for item in pin.get_normalized()],
                    "requested": [item.model_dump(mode="json") for item in resolution.pages],
                    "selected": [item.model_dump(mode="json") for item in evidence.pages],
                }
            ),
            source_complete=bool(retrieval.opinions)
            and all(
                item.text_field is not None and opinions[item.opinion_id].text.strip()
                for item in retrieval.opinions
            ),
            excerpts=tuple(excerpts),
            target_pages=pin.get_normalized(),
        )

    def ground(self, decision: ReporterSupportDecision, node_id: str) -> tuple[ReporterOpinionEvidence, ...]:
        if self.scope is OpinionReviewScope.CITED_PAGES and decision.result is OpinionSupportResult.NOT_FOUND:
            raise ValueError(
                "Selected pages cannot establish absence from the full opinion. "
                "Return unavailable for a page result that needs full-opinion review."
            )
        passages = []
        for proposed in decision.evidence:
            # Retain repeated source locations before choosing one consistent
            # with the page assessment. Quotes may span page boundaries.
            candidates = []
            for excerpt in self.excerpts:
                if excerpt.opinion_id != proposed.opinion_id:
                    continue
                windows = [(0, len(excerpt.text))]
                # A later exact match must not hide a valid fuzzy match on a
                # requested page. Keep the full excerpt for cross-page quotes.
                windows.extend(
                    (
                        max(0, page.span.start - excerpt.offset),
                        min(len(excerpt.text), page.span.end - excerpt.offset),
                    )
                    for page in excerpt.pages
                )
                for start, end in windows:
                    while start < end:
                        found = GroundingEvidence(
                            (EvidenceCandidate(excerpt.text[start:end], None),)
                        ).find_fragment(proposed.quote, GROUNDING)
                        if found is None:
                            break
                        passage = ReporterOpinionEvidence(
                            node_id=node_id,
                            root_id=self.root_id,
                            opinion_id=excerpt.opinion_id,
                            quote=found.text,
                            span=Span(
                                excerpt.offset + start + found.start, excerpt.offset + start + found.end
                            ),
                        )
                        candidates.append(EvidenceCandidate(found.text, passage))
                        start += found.end
            evidence = GroundingEvidence(candidates)

            def on_pages(
                candidate: EvidenceCandidate[ReporterOpinionEvidence], targets: tuple[PinCiteTarget, ...]
            ) -> bool:
                return any(
                    page.kind == target.kind
                    and target.first <= int(page.label.lstrip("*¶").strip()) <= target.last
                    for page in self._pages_at(candidate.value)
                    for target in targets
                )

            preferred = self.target_pages if decision.correct_page is True else decision.found_pages
            selected = (
                evidence.resolve_by_condition(
                    lambda candidate: (
                        on_pages(candidate, preferred) and on_pages(candidate, decision.found_pages)
                    )
                )
                or evidence.resolve_by_condition(lambda candidate: on_pages(candidate, decision.found_pages))
                or evidence.resolve_by_condition(lambda candidate: on_pages(candidate, preferred))
                or evidence.resolve_by_condition(lambda _candidate: True)
            )
            if selected is None:
                raise ValueError(
                    f"Quote for opinion_id {proposed.opinion_id} is not grounded inside a supplied excerpt"
                )
            passages.append(selected.value)
        self._validate_pages(decision, tuple(passages))
        return tuple(passages)

    def _pages_at(self, passage: ReporterOpinionEvidence) -> tuple[OpinionPage, ...]:
        return tuple(
            page
            for excerpt in self.excerpts
            if excerpt.opinion_id == passage.opinion_id
            for page in excerpt.pages
            if page.span.start < passage.span.end
            and page.span.end > passage.span.start
            and page.kind is not None
            and page.label.lstrip("*¶").strip().isdecimal()
        )

    def _validate_pages(
        self, decision: ReporterSupportDecision, passages: tuple[ReporterOpinionEvidence, ...]
    ) -> None:
        """Ground reported page numbers when source markers establish them.

        An unindexed marker remains a semantic reading task. In particular,
        an empty index is not proof that the source has no pagination. Footnote
        placement likewise requires reading the text, not page-number equality.
        """
        located: list[tuple[OpinionPage, ReporterOpinionEvidence]] = []
        unindexed = False
        for passage in passages:
            pages = self._pages_at(passage)
            unindexed |= not pages
            located.extend((page, passage) for page in pages)
        if (decision.found_pages or decision.correct_page is True) and not passages:
            raise ValueError(
                "A correct page or found_pages requires quoted relevant passages from the supplied source"
            )
        # Check only when every quoted passage has a confirmed namespace,
        # numeric label, and target kind. Unclassified source markers remain
        # a semantic reading task rather than contradicting a typed target.
        if not located or unindexed:
            return
        numbers = {
            (page.kind, int(page.label.lstrip("*¶").strip()))
            for page, _ in located
            if page.kind is not None and page.label.lstrip("*¶").strip().isdecimal()
        }
        locations = "; ".join(
            dict.fromkeys(
                f"opinion_id={passage.opinion_id}, source_span={passage.span.start}:{passage.span.end}, "
                f"{page.volume} {page.edition} {page.kind.value if page.kind else 'location'} {page.label}"
                for page, passage in located
            )
        )
        for target in decision.found_pages:
            if any((target.kind, number) not in numbers for number in range(target.first, target.last + 1)):
                raise ValueError(
                    "found_pages must locate quoted passages in the supplied cited-reporter page markers. "
                    f"The quoted passages resolve to: {locations}. "
                    "Revise found_pages using those source locations. Reassess correct_page against the written "
                    "target separately; finding a passage elsewhere alone does not prove the target is wrong."
                )
        if (
            decision.correct_page is True
            and self.target_pages
            and not any(
                (target.kind, number) in numbers
                for target in self.target_pages
                for number in range(target.first, target.last + 1)
            )
        ):
            raise ValueError(
                "correct_page=true needs a quoted relevant passage at the written target. "
                f"The quoted passages resolve to: {locations}. "
                "Quote a relevant passage at the written target or revise the page assessment; "
                "finding a passage elsewhere alone does not prove the target is wrong."
            )


@dataclass(frozen=True, slots=True)
class ReporterPinpointReviewOutcome:
    decision: ReporterSupportDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class ReporterPinpointReviewer(Protocol):
    def __call__(
        self, context: ReporterPinpointReviewContext
    ) -> Awaitable[ReporterSupportDecision | ReporterPinpointReviewOutcome]: ...


def _validate_evidence(ctx: object, context: ReporterPinpointReviewContext) -> ValidationResult:
    try:
        decision = ReporterSupportDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        return ValidationResult(result=True)
    try:
        context.ground(decision, "validation")
    except ValueError as error:
        return ValidationResult(result=False, reason=str(error))
    return ValidationResult(result=True)


@dataclass(frozen=True, slots=True)
class IvrReporterPinpointReviewer(IvrReviewer):
    async def __call__(self, context: ReporterPinpointReviewContext) -> ReporterPinpointReviewOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=context.prefix,
                user_variables={
                    "scope": context.scope.value,
                    "source_complete": str(context.source_complete),
                    "citation_quote": context.citation_quote,
                    "source_offset": str(context.source_offset),
                    "citing_context": context.citing_context,
                    "proposition": context.proposition,
                    "target": context.target,
                },
                output_format=ReporterSupportDecision,
                requirements=(
                    req(
                        "Ground each evidence quote inside a supplied opinion excerpt.",
                        validation_fn=lambda ctx: _validate_evidence(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return ReporterPinpointReviewOutcome(
                None, run=run, failure_reason=run.failure_reason or "IVR pinpoint review failed"
            )
        try:
            decision = ReporterSupportDecision.model_validate_json(run.output)
            context.ground(decision, "validation")
        except ValueError as error:
            return ReporterPinpointReviewOutcome(None, run=run, failure_reason=str(error))
        return ReporterPinpointReviewOutcome(decision, run=run)
