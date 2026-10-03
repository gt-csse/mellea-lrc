"""Index explicit pagination in saved reporter-root opinions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.citations.fields.reporter import normalize_reporter_locator
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_opinion import RetrievedReporterOpinion
from mellea_lrc.model.citations.reporter_pages import (
    IndexedReporterOpinion,
    OpinionPage,
    ReporterRootOpinionPageIndex,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "40_reporter_root_opinion_page_index"
SOURCE_STAGE = "39_reporter_root_opinion_retrieval"

_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_SKIP_TAGS = frozenset({"head", "script", "style"})
_LABEL_PATTERN = re.compile(r"^(?P<marker>\*|¶{1,2})?\s*(?P<number>\d+(?:[.-]\d+)*)$")


@dataclass(frozen=True, slots=True)
class _Marker:
    label: str
    kind: PinCiteKind | None
    raw_offset: int
    citation_index: str | None
    volume: int | None = None
    edition: str | None = None


@dataclass(frozen=True, slots=True)
class _ReporterCandidate:
    volume: int
    edition: str
    first_page: int


class _OpinionHTMLParser(HTMLParser):
    """Flatten HTML while retaining positions of known pagination markers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.markers: list[_Marker] = []
        self._raw_length = 0
        self._skip_depth = 0
        self._skip_tag: str | None = None
        self._active_marker: dict[str, Any] | None = None

    def _append(self, text: str) -> None:
        if not self._skip_depth and self._active_marker is None:
            self.output.append(text)
            self._raw_length += len(text)

    def _block_break(self) -> None:
        if self.output and self.output[-1] and not self.output[-1][-1].isspace():
            self.output.append("\n")
            self._raw_length += 1

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str | None]:
        return {key.lower(): value for key, value in attrs}

    @staticmethod
    def _marker_kind(label: str, fallback: PinCiteKind | None) -> PinCiteKind | None:
        match = _LABEL_PATTERN.fullmatch(label.strip())
        if match is None:
            return None
        marker = match.group("marker")
        if marker == "*":
            return fallback if fallback in {PinCiteKind.PAGE, PinCiteKind.STAR} else None
        if marker and marker.startswith("¶"):
            return PinCiteKind.PARAGRAPH
        return fallback

    def _begin_marker(
        self,
        *,
        label: str | None,
        kind: PinCiteKind | None,
        citation_index: str | None,
        volume: int | None,
        edition: str | None,
        tag: str,
    ) -> None:
        # A page break is a word boundary in the flattened rendering. Keep the
        # separator on the preceding page so the following page span starts at
        # its first actual source character.
        self._block_break()
        self._active_marker = {
            "label": label,
            "kind": kind,
            "citation_index": citation_index,
            "volume": volume,
            "edition": edition,
            "tag": tag,
            "text": [],
            "raw_offset": self._raw_length,
        }

    def _finish_marker(self) -> None:
        marker = self._active_marker
        if marker is None:
            return
        label = marker["label"] or "".join(marker["text"]).strip()
        kind = self._marker_kind(label, marker["kind"])
        label_match = _LABEL_PATTERN.fullmatch(label.strip())
        numeric_star_marker = (
            marker["kind"] is None and label_match is not None and label_match.group("marker") == "*"
        )
        if kind is not None or numeric_star_marker:
            self.markers.append(
                _Marker(
                    label=(label_match.group("number") if numeric_star_marker else label),
                    kind=kind,
                    raw_offset=marker["raw_offset"],
                    citation_index=marker["citation_index"],
                    volume=marker["volume"],
                    edition=marker["edition"],
                )
            )
        else:
            # Unknown marker formats stay visible in the rendered source. We
            # index only labels whose semantics are explicit and recognized.
            literal = "".join(marker["text"]) or marker["label"] or ""
            if literal:
                if not literal[-1].isspace():
                    literal += " "
                self.output.append(literal)
                self._raw_length += len(literal)
        self._active_marker = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if self._active_marker is not None:
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = 1
            self._skip_tag = tag
            return

        values = self._attrs(attrs)
        classes = set((values.get("class") or "").lower().split())
        raw_volume = values.get("volume")
        try:
            volume = int(raw_volume) if raw_volume is not None else None
        except ValueError:
            volume = None
        edition = values.get("edition") or values.get("reporter")
        explicit_kind = values.get("data-pagination-kind") or values.get("data-kind") or values.get("kind")
        try:
            explicit_marker_kind = PinCiteKind(explicit_kind) if explicit_kind else None
        except ValueError:
            explicit_marker_kind = None
        if tag == "page-number":
            self._begin_marker(
                label=values.get("label"),
                kind=PinCiteKind.PAGE,
                citation_index=values.get("citation-index"),
                volume=volume,
                edition=edition,
                tag=tag,
            )
            return
        if tag in {"paragraph-number", "paragraph-label"} or classes & {
            "paragraph-number",
            "paragraph-label",
        }:
            self._begin_marker(
                label=values.get("label"),
                kind=PinCiteKind.PARAGRAPH,
                citation_index=values.get("citation-index"),
                volume=volume,
                edition=edition,
                tag=tag,
            )
            return
        if classes & {"star-pagination", "page-label"}:
            # Lawbox uses this class for visible asterisks on ordinary reporter
            # pages too. Only source metadata can establish Westlaw-star semantics.
            marker_kind = (
                explicit_marker_kind
                if explicit_marker_kind is not None
                else PinCiteKind.PAGE
                if "page-label" in classes
                else None
            )
            self._begin_marker(
                label=values.get("label"),
                kind=marker_kind,
                citation_index=values.get("citation-index"),
                volume=volume,
                edition=edition,
                tag=tag,
            )
            return
        if tag == "br":
            self._append("\n")
        elif tag in _BLOCK_TAGS:
            self._block_break()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._active_marker is not None:
            if tag == self._active_marker["tag"]:
                self._finish_marker()
            return
        if self._skip_depth:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in _BLOCK_TAGS:
            self._block_break()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._active_marker is not None:
            self._active_marker["text"].append(data)
            return
        self._append(data)


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Return a readable single-space rendering and raw-boundary offsets."""
    output: list[str] = []
    offsets = [0] * (len(text) + 1)
    pending_space = False
    for index, character in enumerate(text):
        if character.isspace():
            if output:
                pending_space = True
        else:
            if pending_space:
                output.append(" ")
                pending_space = False
            output.append(character)
        offsets[index + 1] = len(output)
    return "".join(output), offsets


def _render_html(html_text: str) -> tuple[str, tuple[OpinionPage, ...]]:
    parser = _OpinionHTMLParser()
    parser.feed(html_text)
    parser.close()
    raw = "".join(parser.output)
    text, offsets = _collapse_whitespace(raw)
    # Markers are retained in source order; labels are metadata, not injected
    # into body text. Unknown formats remain plain text with an empty index.
    pages: list[OpinionPage] = []
    for index, marker in enumerate(parser.markers):
        start = min(offsets[marker.raw_offset], len(text))
        namespace = (marker.kind, marker.citation_index, marker.volume, marker.edition)
        next_marker = next(
            (
                candidate
                for candidate in parser.markers[index + 1 :]
                if (candidate.kind, candidate.citation_index, candidate.volume, candidate.edition)
                == namespace
            ),
            None,
        )
        raw_end = next_marker.raw_offset if next_marker is not None else len(raw)
        end = min(offsets[raw_end], len(text))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        pages.append(
            OpinionPage(
                label=marker.label,
                kind=marker.kind,
                volume=marker.volume,
                edition=marker.edition,
                span=Span(start, end),
                citation_index=marker.citation_index,
            )
        )
    return text, tuple(pages)


def _selected_cluster_candidates(citation: FullReporterCitation) -> tuple[_ReporterCandidate, ...]:
    """Read ordinary reporter citations from the root's bound original source."""
    source = citation.reporter_root_opinion_source
    if source is None:
        return ()
    raw_citations = source.cluster.raw_json.get("citations")
    if not isinstance(raw_citations, list):
        return ()

    candidates: set[_ReporterCandidate] = set()
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        cite_type = str(item.get("cite_type") or item.get("citation_type") or "").casefold()
        reporter = str(item.get("reporter") or "").strip()
        page = str(item.get("page") or "").strip()
        if cite_type in {"specialty_west", "specialty_lexis"} or "specialty" in cite_type:
            continue
        if re.search(r"\b(?:WL|LEXIS)\b", reporter, re.I):
            continue
        if not page.isdecimal():
            continue
        try:
            normalized = normalize_reporter_locator(f"{item.get('volume', '')} {reporter} {page}")
            first_page = int(normalized.page)
        except (TypeError, ValueError):
            continue
        candidates.add(
            _ReporterCandidate(
                volume=normalized.volume,
                edition=normalized.edition,
                first_page=first_page,
            )
        )
    return tuple(
        sorted(candidates, key=lambda candidate: (candidate.volume, candidate.edition, candidate.first_page))
    )


def _infer_page_namespaces(
    opinions: tuple[IndexedReporterOpinion, ...], candidates: tuple[_ReporterCandidate, ...]
) -> tuple[IndexedReporterOpinion, ...]:
    """Infer a reporter only when exactly one saved ordinary cite can own a namespace."""
    namespaces: dict[str, list[tuple[int, int, OpinionPage]]] = {}
    for opinion_index, opinion in enumerate(opinions):
        for page_index, page in enumerate(opinion.pages):
            if page.kind in {PinCiteKind.PAGE, None} and page.volume is None and page.edition is None:
                namespace = page.citation_index if page.citation_index is not None else "<unindexed>"
                namespaces.setdefault(namespace, []).append((opinion_index, page_index, page))

    updates: dict[tuple[int, int], tuple[int, str]] = {}
    for markers in namespaces.values():
        labels = [page.label for _, _, page in markers]
        if not labels or any(not label.isdecimal() for label in labels):
            continue
        observed_pages = tuple(int(label) for label in labels)
        compatible = [
            candidate
            for candidate in candidates
            if all(page >= candidate.first_page for page in observed_pages)
        ]
        if len(compatible) != 1:
            continue
        candidate = compatible[0]
        for opinion_index, page_index, _ in markers:
            updates[(opinion_index, page_index)] = (candidate.volume, candidate.edition)

    indexed: list[IndexedReporterOpinion] = []
    for opinion_index, opinion in enumerate(opinions):
        pages = tuple(
            OpinionPage.model_validate(
                {
                    **page.model_dump(mode="python"),
                    "kind": PinCiteKind.PAGE,
                    "volume": updates[(opinion_index, page_index)][0],
                    "edition": updates[(opinion_index, page_index)][1],
                    "pagination_inferred": True,
                }
            )
            if (opinion_index, page_index) in updates
            else page
            for page_index, page in enumerate(opinion.pages)
        )
        # Inference can unite a marker with an already explicit PAGE namespace.
        # Recompute its boundary after that union, rather than retaining spans
        # that overlapped only because their kinds were previously different.
        bounded: list[OpinionPage] = []
        for page_index, page in enumerate(pages):
            namespace = (page.kind, page.citation_index, page.volume, page.edition)
            following = next(
                (
                    other
                    for other in pages[page_index + 1 :]
                    if (other.kind, other.citation_index, other.volume, other.edition) == namespace
                ),
                None,
            )
            end = following.span.start if following is not None else len(opinion.text)
            while end > page.span.start and opinion.text[end - 1].isspace():
                end -= 1
            bounded.append(
                OpinionPage.model_validate(
                    {**page.model_dump(mode="python"), "span": Span(page.span.start, end)}
                )
            )
        indexed.append(
            IndexedReporterOpinion.model_validate(
                {**opinion.model_dump(mode="python"), "pages": tuple(bounded)}
            )
        )
    return tuple(indexed)


def _index_opinion(opinion: RetrievedReporterOpinion) -> IndexedReporterOpinion:
    response = opinion.response
    text_field = opinion.text_field
    if response is None or text_field is None:
        return IndexedReporterOpinion(opinion_id=opinion.opinion_id, text_field=None, text="", pages=())

    saved_text = response.get(text_field)
    if not isinstance(saved_text, str) or not saved_text:
        return IndexedReporterOpinion(opinion_id=opinion.opinion_id, text_field=text_field, text="", pages=())
    if text_field == "plain_text":
        return IndexedReporterOpinion(
            opinion_id=opinion.opinion_id, text_field=text_field, text=saved_text, pages=()
        )

    text, pages = _render_html(saved_text)
    if not pages:
        for alternative_field in (
            "html_with_citations",
            "html",
            "html_lawbox",
            "html_columbia",
            "html_anon_2020",
            "xml_harvard",
        ):
            if alternative_field == text_field:
                continue
            alternative_text = response.get(alternative_field)
            if not isinstance(alternative_text, str) or not alternative_text.strip():
                continue
            alternative_rendered, alternative_pages = _render_html(alternative_text)
            if alternative_pages:
                text_field = alternative_field
                text, pages = alternative_rendered, alternative_pages
                break
    return IndexedReporterOpinion(
        opinion_id=opinion.opinion_id,
        text_field=text_field,
        text=text,
        pages=pages,
    )


def index_reporter_root_opinion_pages(document: Document) -> Document:
    """Render stage-39 opinion payloads and index only explicit page markers."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if SOURCE_STAGE not in document.stage_runs:
        raise ValueError("Retrieve reporter-root opinions before indexing their pages")

    for citation in document.roots:
        if not isinstance(citation, FullReporterCitation):
            continue
        retrieval = citation.reporter_root_opinion_retrieval
        if retrieval is None:
            continue
        recorded = citation.record(STAGE)
        result = ReporterRootOpinionPageIndex(
            node_id=recorded.nodes[-1].id,
            cluster_id=retrieval.cluster_id,
            opinions=_infer_page_namespaces(
                tuple(_index_opinion(opinion) for opinion in retrieval.opinions),
                _selected_cluster_candidates(citation),
            ),
        )
        document = document.replace_citation(recorded.with_reporter_root_opinion_page_index(result))
    return document.complete(STAGE)
