"""Local IVR service for choosing among saved opinion-page candidates."""

from __future__ import annotations

import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import ValidationError

from mellea_lrc.llm.ivr import InstructIvrSpec, run_instruct_ivr
from mellea_lrc.llm.reviewer import IvrReviewer
from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_page_resolution import (
    ReporterCitationOpinionDecision,
    ReporterCitationPageResolution,
)
from mellea_lrc.model.citations.short_reporter import ShortReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.ivr import IvrRun
from mellea_lrc.model.span import Span

_PREFIX = """Choose a representative opinion writing for each requested citation page using only the supplied filing context and saved sources. This is source selection, not a decision about whether a page supports the filing's argument. Do not judge case identity, change extracted fields, or invent source pages.

The shared sources below belong to the same cited case. They can include a combined opinion and separate majority, concurrence, or dissent writings, and several copies of the same writing. List order and type alone do not establish which writing the citation means. Use any supplied author, parenthetical, quotation, or nearby discussion to identify the intended writing. Quotation overlap alone does not justify selecting a dissent when the citation's context identifies another writing. When combined and separate sources represent the same intended writing, choose the best available representative and explain why. A page range can cross writings: choose independently for each requested page, without forcing every page into one opinion.

Honor the supplied reporter pagination and pagination_confirmed flag. A browser's display asterisk does not establish Westlaw star pagination. Do not treat an unconfirmed pagination namespace as confirmed merely because a printed number matches. Return null candidate_index when the supplied sources do not permit a defensible representative choice. Return exactly one choice for every requested page_index, using only its listed candidate_index values or null, followed by a prose reason.

Saved opinion sources (shared across citations to this root):
"""

_INSTRUCTION = """Citing occurrence: {{citation_quote}}
Its absolute source span: {{citation_span}}

Effective reporter locator and pinpoint (read through the saved source pointers, including inherited Id. readings):
{{target_readings}}

Nearby filing text (source offset {{source_offset}}):
{{citing_context}}

Requested pages and their saved candidates:
{{page_candidates}}

Return choices and reason. Each page_index identifies an item in the requested-pages list; candidate_index identifies an item in that page's candidates list. Null leaves that requested page unresolved."""


@dataclass(frozen=True, slots=True)
class ReporterCitationOpinionContext:
    citation_id: str
    root_id: str
    resolution: ReporterCitationPageResolution
    citing_context: str
    source_offset: int
    citation_quote: str
    citation_span: Span
    page_candidates: tuple[dict[str, Any], ...]
    target_readings: dict[str, Any]
    prefix: str

    @classmethod
    def from_document(cls, document: Document, citation: Citation) -> ReporterCitationOpinionContext:
        resolution = citation.reporter_page_resolutions[-1]
        root = next((item for item in document.roots if item.id == resolution.root_id), None)
        if not isinstance(root, FullReporterCitation):
            raise ValueError("Opinion selection needs its attached reporter root")
        index = root.reporter_root_opinion_page_index
        retrieval = root.reporter_root_opinion_retrieval
        source = root.reporter_root_opinion_source
        if index is None or retrieval is None or source is None:
            raise ValueError("Opinion selection needs the bound root source, saved opinions and page index")
        indexed = {item.opinion_id: item for item in index.opinions}
        raw = {item.opinion_id: item.response or {} for item in retrieval.opinions}
        cluster = source.cluster
        by_id = {item.id: item for item in document.citations}
        locator_source = by_id.get(resolution.locator_citation_id)
        pin_source = by_id.get(resolution.pin_citation_id)
        if (
            not isinstance(locator_source, FullReporterCitation | ShortReporterCitation)
            or resolution.locator_reading_index is None
            or pin_source is None
            or pin_source.pin_cite is None
            or resolution.pin_reading_index is None
        ):
            raise ValueError("Opinion selection needs the effective locator and pinpoint reading pointers")
        locator_log = (
            locator_source.short_locator
            if isinstance(locator_source, ShortReporterCitation)
            else locator_source.locator
        )
        locator = locator_log[resolution.locator_reading_index]
        pin = pin_source.pin_cite[resolution.pin_reading_index]
        normalized = locator.get_normalized()
        target_readings = {
            "locator": {
                "citation_id": locator_source.id,
                "reading_index": resolution.locator_reading_index,
                "quote": locator.quote,
                "normalized": {"volume": normalized.volume, "edition": normalized.edition},
            },
            "pin_cite": {
                "citation_id": pin_source.id,
                "reading_index": resolution.pin_reading_index,
                "quote": pin.quote,
                "normalized": [target.model_dump(mode="json") for target in pin.get_normalized()],
            },
        }
        shared = []
        for identifier, opinion in sorted(indexed.items(), key=lambda item: int(item[0])):
            if identifier not in raw:
                raise ValueError("Indexed opinion is missing its retained provider response")
            metadata = raw[identifier]
            shared.append(
                {
                    "opinion_id": identifier,
                    "type": metadata.get("type"),
                    "author": metadata.get("author"),
                    "author_str": metadata.get("author_str"),
                    "per_curiam": metadata.get("per_curiam"),
                    "text": opinion.text,
                    "pages": [page.model_dump(mode="json") for page in opinion.pages],
                }
            )
        requested = []
        for page_index, page in enumerate(resolution.pages):
            candidates = []
            for candidate_index, reference in enumerate(page.candidates):
                opinion = indexed.get(reference.opinion_id)
                if opinion is None or reference.page_index >= len(opinion.pages):
                    raise ValueError("Page candidate refers to an unavailable indexed opinion page")
                indexed_page = opinion.pages[reference.page_index]
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "opinion_id": reference.opinion_id,
                        "opinion_page_index": reference.page_index,
                        "pagination_confirmed": reference.pagination_confirmed,
                        "page": indexed_page.model_dump(mode="json"),
                        "text": opinion.text[indexed_page.span.start : indexed_page.span.end],
                    }
                )
            requested.append(
                {
                    "page_index": page_index,
                    "label": page.label,
                    "kind": page.kind.value,
                    "target_index": page.target_index,
                    "candidates": candidates,
                }
            )
        span = citation.site_span
        start = max(0, span.start - 500)
        stop = min(len(document.text), span.end + 500)
        return cls(
            citation_id=citation.id,
            root_id=root.id,
            resolution=resolution,
            citing_context=document.text[start:stop],
            source_offset=start,
            citation_quote=document.text[span.start : span.end],
            citation_span=span,
            page_candidates=tuple(requested),
            target_readings=target_readings,
            prefix=_PREFIX
            + json.dumps(
                {
                    "cluster_id": index.cluster_id,
                    "parallel_reporter_citations": cluster.raw_json.get("citations", []),
                    "opinions": shared,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

    def decision_error(self, decision: ReporterCitationOpinionDecision) -> str | None:
        expected = set(range(len(self.resolution.pages)))
        actual = {choice.page_index for choice in decision.choices}
        if expected != actual:
            return (
                f"Return exactly one choice for every requested page_index {sorted(expected)}; "
                f"missing {sorted(expected - actual)}, unknown {sorted(actual - expected)}."
            )
        for choice in decision.choices:
            count = len(self.resolution.pages[choice.page_index].candidates)
            if choice.candidate_index is not None and choice.candidate_index >= count:
                return (
                    f"For page_index {choice.page_index}, candidate_index must be one of "
                    f"{list(range(count))} or null; received {choice.candidate_index}."
                )
        return None


@dataclass(frozen=True, slots=True)
class ReporterCitationOpinionOutcome:
    decision: ReporterCitationOpinionDecision | None
    run: IvrRun | None = None
    failure_reason: str | None = None


class ReporterCitationOpinionReviewer(Protocol):
    def __call__(
        self, context: ReporterCitationOpinionContext
    ) -> Awaitable[ReporterCitationOpinionDecision | ReporterCitationOpinionOutcome]: ...


def _validate_choices(ctx: object, context: ReporterCitationOpinionContext) -> ValidationResult:
    try:
        decision = ReporterCitationOpinionDecision.model_validate_json(str(ctx.last_output().value))
    except ValidationError:
        # The IVR wrapper owns schema feedback; domain checks only inspect valid JSON.
        return ValidationResult(result=True)
    error = context.decision_error(decision)
    return ValidationResult(result=error is None, reason=error)


@dataclass(frozen=True, slots=True)
class IvrReporterCitationOpinionReviewer(IvrReviewer):
    async def __call__(self, context: ReporterCitationOpinionContext) -> ReporterCitationOpinionOutcome:
        run = await run_instruct_ivr(
            self.session,
            InstructIvrSpec(
                description=_INSTRUCTION,
                prefix=context.prefix,
                user_variables={
                    "citation_quote": context.citation_quote,
                    "citation_span": json.dumps(
                        {"start": context.citation_span.start, "end": context.citation_span.end}
                    ),
                    "source_offset": str(context.source_offset),
                    "citing_context": context.citing_context,
                    "page_candidates": json.dumps(context.page_candidates, ensure_ascii=False),
                    "target_readings": json.dumps(context.target_readings, ensure_ascii=False),
                },
                output_format=ReporterCitationOpinionDecision,
                requirements=(
                    req(
                        "Choose only supplied candidates and account for every requested page.",
                        validation_fn=lambda ctx: _validate_choices(ctx, context),
                    ),
                ),
            ),
            strategy=MultiTurnStrategy(loop_budget=self.max_attempts),
            model_options=dict(self.model_options),
        )
        if not run.success:
            return ReporterCitationOpinionOutcome(
                None, run=run, failure_reason=run.failure_reason or "IVR opinion selection failed"
            )
        try:
            decision = ReporterCitationOpinionDecision.model_validate_json(run.output)
        except ValidationError as error:
            return ReporterCitationOpinionOutcome(None, run=run, failure_reason=str(error))
        error = context.decision_error(decision)
        return ReporterCitationOpinionOutcome(
            decision if error is None else None, run=run, failure_reason=error
        )
