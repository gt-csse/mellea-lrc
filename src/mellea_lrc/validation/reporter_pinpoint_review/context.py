"""Source text, explicit pagination, and quote grounding for opinion review."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import pairwise

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import EvidenceCandidate, GroundingEvidence
from mellea_lrc.matching.literal_to_regex import fuzzy_literal
from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields.pin_cite import PinCiteTarget
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_pages import OpinionPage
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionEvidenceQuote,
    OpinionReviewScope,
    OpinionSupportResult,
    ReporterOpinionEvidence,
    ReporterSupportDecision,
)
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span

from .responses import (
    MAX_EVIDENCE_QUOTES,
    MAX_QUOTE_CHARACTERS,
    PaginatedSupportResponse,
    UnpaginatedSupportResponse,
    validate_evidence_quotes,
)

GROUNDING = FuzzinessOption.edit_distance(similarity_percent=95, whitespace_relaxation=True)

_PREFIX = f"""Review how a legal citation uses the supplied judicial text. Use only this evidence, not memory of the case. Each occurrence has its own proposition, pinpoint, and intended writing; other occurrences of the same case do not decide this one.

Read the actual attributed use in its context, including introductory signals and explanatory parentheticals. A quotation represents the quoted words; a paraphrase may express equivalent meaning. Do not excuse invented quoted language merely because a related idea exists. An adverse or analogous signal can intentionally cite contrary or distinguishable reasoning. Assess that use rather than requiring every citation to affirm the filing's conclusion. Read qualifications, negations, and factual limits. Distinguish the court's own reasoning from quoted allegations, arguments it rejects, and editorial summaries. Honor an identified concurrence or dissent; matching words in another writing alone do not establish the intended attribution. Provider type/order and a combined opinion are clues, not proof of authorship.

Return supported when the attributed use is supported; contradicted when affirmative source evidence shows the attribution materially changes its meaning; not_found when the full supplied intended opinion was reviewed and the attributed words or idea are absent; unavailable when the supplied materials cannot settle the issue. Missing text or pagination alone never proves the attribution false. Never use not_found as a full-opinion conclusion from selected pages alone. Explain your reasoning as prose.

Distinguish no relevant discussion or quoted passage anywhere in the complete intended opinion from an existing passage whose meaning the filing misrepresents. An existing passage may fail to support the attributed use because of its meaning, qualifications, or relevance; that content assessment alone does not establish absence. Apply the result definitions to the attributed use and explain the distinction.

Return at most {MAX_EVIDENCE_QUOTES} evidence items, usually one or two. Each item quotes one representative sentence or short fragment, with its supplied opinion_id, and must be no longer than {MAX_QUOTE_CHARACTERS} characters. Multiple sentences belong in separate evidence items. They may come from different locations; there is no requirement that the items form a continuous chunk. Each item must itself be copied from one continuous source passage: do not invent text, splice separated passages, or insert ellipses to join them. Put the detailed explanation and connections between passages in reason, not in quote. Surrounding source context is already retained. Each item is grounded independently with shared whitespace relaxation and 95% edit-distance matching. Supported and contradicted require evidence. For not_found or unavailable, evidence may be []; do not copy a whole opinion to demonstrate absence. If a short passage helps explain a partially supported attribution, quote only that passage. Do not merely quote a matching phrase without considering its surrounding meaning.

Assess content and page placement independently. correct_page concerns whether the relevant discussion or quoted passage is at the written pinpoint, not whether the filing's interpretation is right. A correctly located discussion or quoted passage can be misrepresented, contradicted by its context, or irrelevant to the asserted proposition; those content problems do not by themselves make correct_page false. A shared topic or isolated matching word alone does not establish the relevant discussion or quoted passage.

The supplied text is divided into source blocks. Each block retains its exact text and absolute source_offset. Its pages list identifies the indexed pagination namespaces that cover the block. A page_index is a reference within that opinion, not a printed page number. Different reporters can have different page labels over the same text. Use only the namespace requested by the citing occurrence. Unpaginated blocks have no usable page in that namespace.

When the response schema includes page fields, give each evidence quote its page_indices copied from the source blocks, and footnote (null unless established). A contiguous quote crossing a boundary must name every crossed page in the requested namespace. Use [] only when the quoted text has no usable pagination in that namespace. Each quote is grounded inside those named pages; selecting a page elsewhere in the opinion will fail validation. Printed found_pages are derived from the grounded source locations, not generated by you.

When requested, correct_page is true when the written page/paragraph range and any designated footnote contain the relevant discussion or quoted passage, false when placement is established to be wrong, and null when placement remains uncertain. A range needs the relevant passage somewhere within it, not independently on every page. When returning true, quote relevant evidence at the written target. Finding evidence elsewhere does not prove equivalent relevant material is absent at the target: review that target before returning false. Content and placement remain independent. When the schema has no page fields, assess content only; do not add page fields or invent pagination.

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

If scope is cited_pages, evaluate only these supplied pages. A negative or uncertain page result will be reviewed against the full opinion later. If scope is full_opinion, search the complete supplied writings and identify the intended one using the context. Missing subopinions require caution about absence, though affirmative evidence may still settle a proposition. Return result, evidence, and reason, plus the page fields only when included in the response schema."""


@dataclass(frozen=True, slots=True)
class OpinionExcerpt:
    opinion_id: str
    text: str
    offset: int
    pages: tuple[OpinionPage, ...] = ()
    page_indices: tuple[int, ...] = ()


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
            return tuple(
                page
                for page in opinions[identifier].pages
                if _page_number(page) is not None
                and page.volume == locator.volume
                and page.edition is not None
                and re.fullmatch(
                    fuzzy_literal(locator.edition, whitespace=True, newline=True), page.edition, re.I
                )
            )

        excerpts: list[OpinionExcerpt] = []
        if scope is OpinionReviewScope.CITED_PAGES:
            selected: dict[str, list[Span]] = {}
            for page in evidence.pages:
                selected.setdefault(page.opinion_id, []).append(
                    opinions[page.opinion_id].pages[page.page_index].span
                )
            for identifier, spans in selected.items():
                opinion = opinions[identifier]
                whole = OpinionExcerpt(identifier, opinion.text, 0)
                # Adjacent selected pages are one contiguous excerpt; separated
                # pages remain separate so evidence cannot join missing text.
                for start, end in _quote_windows(whole, spans, unpaginated=False):
                    indexed = tuple(
                        (i, page)
                        for i, page in enumerate(opinion.pages)
                        if page in matching_pages(identifier)
                        and page.span.start < end
                        and page.span.end > start
                    )
                    excerpts.append(
                        OpinionExcerpt(
                            identifier,
                            opinion.text[start:end],
                            start,
                            tuple(page for _, page in indexed),
                            tuple(i for i, _ in indexed),
                        )
                    )
        else:
            excerpts.extend(
                OpinionExcerpt(
                    opinion.opinion_id,
                    opinion.text,
                    0,
                    matching_pages(opinion.opinion_id),
                    tuple(i for i, p in enumerate(opinion.pages) if p in matching_pages(opinion.opinion_id)),
                )
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
                # Retain every namespace in the shared source prefix so full-opinion
                # KV reuse is independent of an occurrence's parallel reporter.
                "blocks": _source_blocks(excerpt, opinions[excerpt.opinion_id].pages),
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

    @property
    def response_model(self) -> type[PaginatedSupportResponse] | type[UnpaginatedSupportResponse]:
        return (
            PaginatedSupportResponse
            if any(excerpt.pages for excerpt in self.excerpts)
            else UnpaginatedSupportResponse
        )

    def parse_response(
        self, raw: str, node_id: str
    ) -> tuple[ReporterSupportDecision, tuple[ReporterOpinionEvidence, ...]]:
        """Validate the wire response and derive locations from page-bound quotes."""
        response = self.response_model.model_validate_json(raw)
        passages = []
        found_pages = []
        for proposed in response.evidence:
            windows = []
            for excerpt in self.excerpts:
                if excerpt.opinion_id != proposed.opinion_id:
                    continue
                if isinstance(response, PaginatedSupportResponse):
                    indexed = dict(zip(excerpt.page_indices, excerpt.pages, strict=True))
                    if any(index not in indexed for index in proposed.page_indices):
                        continue
                    spans = [indexed[index].span for index in proposed.page_indices]
                    windows.extend(
                        (excerpt, start, end)
                        for start, end in _quote_windows(excerpt, spans, unpaginated=not spans)
                    )
                else:
                    windows.append((excerpt, 0, len(excerpt.text)))
            passage = self._ground_quote(proposed, node_id, windows)
            pages = self._pages_at(passage)
            if isinstance(response, PaginatedSupportResponse):
                actual_indices = {
                    index
                    for excerpt in self.excerpts
                    if excerpt.opinion_id == passage.opinion_id
                    for index, page in zip(excerpt.page_indices, excerpt.pages, strict=True)
                    if _overlaps(page.span, passage.span)
                }
                if actual_indices != set(proposed.page_indices):
                    raise ValueError(
                        f"Evidence for opinion_id {proposed.opinion_id} crosses page_indices "
                        f"{sorted(actual_indices)}; name exactly these source pages."
                    )
                found_pages.extend(
                    PinCiteTarget(
                        first=_page_number(page),
                        last=_page_number(page),
                        kind=page.kind,
                        footnote=proposed.footnote,
                    )
                    for page in pages
                )
            passages.append(passage)
        decision = ReporterSupportDecision(
            result=response.result,
            evidence=tuple(
                OpinionEvidenceQuote(opinion_id=item.opinion_id, quote=item.quote)
                for item in response.evidence
            ),
            reason=response.reason,
            pagination_available=isinstance(response, PaginatedSupportResponse),
            correct_page=response.correct_page if isinstance(response, PaginatedSupportResponse) else None,
            found_pages=tuple(dict.fromkeys(found_pages)),
        )
        self._validate_pages(decision, tuple(passages))
        return decision, tuple(passages)

    def _ground_quote(
        self,
        proposed: OpinionEvidenceQuote,
        node_id: str,
        windows: list[tuple[OpinionExcerpt, int, int]],
    ) -> ReporterOpinionEvidence:
        for excerpt, start, end in windows:
            found = GroundingEvidence((EvidenceCandidate(excerpt.text[start:end], None),)).find_fragment(
                proposed.quote, GROUNDING
            )
            if found is not None:
                return ReporterOpinionEvidence(
                    node_id=node_id,
                    root_id=self.root_id,
                    opinion_id=excerpt.opinion_id,
                    quote=found.text,
                    span=Span(excerpt.offset + start + found.start, excerpt.offset + start + found.end),
                )
        raise ValueError(
            f"Quote for opinion_id {proposed.opinion_id} is not grounded inside its supplied "
            "pages/excerpt. Copy a contiguous quote from the supplied source text. "
            + (
                "Name its page_indices in the requested reporter namespace, not printed page numbers."
                if self.response_model is PaginatedSupportResponse
                else "This source has no usable pagination; do not add page fields."
            )
        )

    def ground(self, decision: ReporterSupportDecision, node_id: str) -> tuple[ReporterOpinionEvidence, ...]:
        """Ground a programmatic review decision against the supplied source."""
        validate_evidence_quotes(decision.evidence)
        passages = []
        for proposed in decision.evidence:
            excerpts = [item for item in self.excerpts if item.opinion_id == proposed.opinion_id]
            # A programmatic decision can identify one location among repeated
            # passages. Merge adjacent named pages before grounding a quote
            # that crosses their boundary, then try the written target.
            windows = []
            for targets in (decision.found_pages, self.target_pages if decision.correct_page is True else ()):
                for excerpt in excerpts:
                    spans = [
                        page.span
                        for page in excerpt.pages
                        if any(_on_target(page, target) for target in targets)
                    ]
                    if spans:
                        windows.extend(
                            (excerpt, start, end)
                            for start, end in _quote_windows(excerpt, spans, unpaginated=False)
                        )
            windows.extend((excerpt, 0, len(excerpt.text)) for excerpt in excerpts)
            passages.append(self._ground_quote(proposed, node_id, windows))
        self._validate_pages(decision, tuple(passages))
        return tuple(passages)

    def _pages_at(self, passage: ReporterOpinionEvidence) -> tuple[OpinionPage, ...]:
        return tuple(
            dict.fromkeys(
                page
                for excerpt in self.excerpts
                if excerpt.opinion_id == passage.opinion_id
                for page in excerpt.pages
                if _overlaps(page.span, passage.span) and _page_number(page) is not None
            )
        )

    def _validate_pages(
        self, decision: ReporterSupportDecision, passages: tuple[ReporterOpinionEvidence, ...]
    ) -> None:
        if self.scope is OpinionReviewScope.CITED_PAGES and decision.result is OpinionSupportResult.NOT_FOUND:
            raise ValueError(
                "Selected pages cannot establish absence from the full opinion; return unavailable."
            )
        located = [(page, passage) for passage in passages for page in self._pages_at(passage)]
        numbers = {(page.kind, _page_number(page)) for page, _ in located}
        for target in decision.found_pages:
            if any((target.kind, number) not in numbers for number in range(target.first, target.last + 1)):
                raise ValueError(
                    "found_pages must locate quoted passages in the supplied cited-reporter page markers."
                )
        if decision.correct_page is True and not any(
            _on_target(page, target)
            and (
                target.footnote is None
                or any(
                    found.kind == target.kind
                    and found.first <= _page_number(page) <= found.last
                    and found.footnote == target.footnote
                    for found in decision.found_pages
                )
            )
            for page, _ in located
            for target in self.target_pages
        ):
            raise ValueError("correct_page=true needs a quoted relevant passage at the written target.")


def _page_number(page: OpinionPage) -> int | None:
    label = page.label.lstrip("*¶").strip()
    return int(label) if page.kind is not None and label.isdecimal() and int(label) > 0 else None


def _on_target(page: OpinionPage, target: PinCiteTarget) -> bool:
    number = _page_number(page)
    return number is not None and page.kind == target.kind and target.first <= number <= target.last


def _overlaps(left: Span, right: Span) -> bool:
    return left.start < right.end and left.end > right.start


def _source_blocks(excerpt: OpinionExcerpt, pages: tuple[OpinionPage, ...]) -> list[dict]:
    """Partition exact source text once, retaining overlapping reporter namespaces."""
    start, end = excerpt.offset, excerpt.offset + len(excerpt.text)
    indexed = [
        (index, page) for index, page in enumerate(pages) if page.span.start < end and page.span.end > start
    ]
    boundaries = sorted(
        {
            start,
            end,
            *(max(start, page.span.start) for _, page in indexed),
            *(min(end, page.span.end) for _, page in indexed),
        }
    )
    return [
        {
            "source_offset": left,
            "text": excerpt.text[left - start : right - start],
            "pages": [
                {
                    "page_index": index,
                    "label": page.label,
                    "kind": page.kind,
                    "volume": page.volume,
                    "edition": page.edition,
                }
                for index, page in indexed
                if page.span.start <= left and page.span.end >= right
            ],
        }
        for left, right in pairwise(boundaries)
    ]


def _quote_windows(excerpt: OpinionExcerpt, spans: list[Span], *, unpaginated: bool) -> list[tuple[int, int]]:
    """Ground within named contiguous pages, or uncovered text when no page is named."""
    start, end = excerpt.offset, excerpt.offset + len(excerpt.text)
    merged = []
    for span in sorted(
        (page.span for page in excerpt.pages) if unpaginated else spans, key=lambda item: item.start
    ):
        left, right = max(start, span.start), min(end, span.end)
        if left >= right:
            continue
        # Indexers can leave layout whitespace between adjacent page spans.
        # Preserve that exact text when joining pages, never skipped prose.
        if merged and (left <= merged[-1][1] or excerpt.text[merged[-1][1] - start : left - start].isspace()):
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    if unpaginated:
        gaps = []
        cursor = start
        for left, right in merged:
            if cursor < left:
                gaps.append((cursor, left))
            cursor = right
        if cursor < end:
            gaps.append((cursor, end))
        merged = gaps
    return [(left - start, right - start) for left, right in merged]
