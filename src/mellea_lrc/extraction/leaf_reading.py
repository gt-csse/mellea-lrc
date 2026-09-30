"""Shared source reading for leaf stages; no stage is executed here."""

from __future__ import annotations

import re
from functools import lru_cache

from eyecite import get_citations
from eyecite.models import CitationBase

from mellea_lrc.extraction.full_reporter_locator import _reporter_tokenizer
from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citations import FullReporterCitation, latest
from mellea_lrc.model.citations.fields.pin_cite import PIN_PREFIX
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span


@lru_cache(maxsize=32)
def events(source: str) -> tuple[CitationBase, ...]:
    """Keep noncase/unknown events: they are required for correct Id chronology."""
    return tuple(get_citations(source, tokenizer=_reporter_tokenizer()))


def require_leaves(document: Document, stage: str) -> None:
    if stage in document.stage_runs:
        raise ValueError(f"Stage already completed: {stage}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before growing leaves")


def pin_after(source: str, position: int, end: int) -> Span | None:
    """Read a directly adjacent pin with the same grammar as normalization."""
    match = PIN_PREFIX.match(source[position:end])
    if match is None:
        return None
    return Span(position + match.start("pin"), position + match.end("pin"))


def aliases(document: Document, *, before: int | None = None) -> dict[str, tuple[str, ...]]:
    """Current quoted party/subject names from every usable full occurrence."""
    roots = {root.id for root in document.roots}
    result: dict[str, set[str]] = {}
    for citation in document.full_locators:
        if before is not None and citation.site_span.start >= before:
            continue
        root_id = latest(citation.root_id)
        if root_id not in roots or not citation.case_name or not citation.case_name[-1].normalizable:
            continue
        name = citation.get_case_name()
        values = {name.as_citation(), name.plaintiff, name.defendant, name.subject, name.partial}

        # Printed short names can end at any whole-word boundary. These are
        # proposals, not case equivalence rules; ambiguous/prose uses require
        # semantic review. No entity suffix or jurisdiction-specific list.
        def prefixes(value: str | None) -> tuple[str, ...]:
            if not value:
                return ()
            return tuple(
                dict.fromkeys(
                    [
                        value,
                        *(value[: m.start()].rstrip(" ,") for m in re.finditer(r"\s+", value)),
                    ]
                )
            )

        for party in (name.plaintiff, name.defendant, name.subject):
            values.update(part for part in prefixes(party) if len(part.split()) >= 2)
        if name.plaintiff and name.defendant:
            values.update(
                f"{plaintiff} v. {defendant}"
                for plaintiff in prefixes(name.plaintiff)
                for defendant in prefixes(name.defendant)
            )
        for value in values:
            if value:
                result.setdefault(value, set()).add(root_id)
    return {name: tuple(sorted(ids)) for name, ids in result.items()}


def name_candidates(document: Document, name: str, before: int | None) -> tuple[str, ...]:
    """Whole written fragment, never any-word overlap."""
    expression = re.compile(rf"(?<!\w){fuzzy_literal(name, whitespace=True, newline=True)}(?!\w)", re.I)
    return tuple(
        sorted(
            {
                root
                for alias, roots in aliases(document, before=before).items()
                if expression.search(alias)
                for root in roots
            }
        )
    )


def reporter_candidates(document: Document, volume: int, edition: str, before: int) -> tuple[str, ...]:
    roots = {root.id for root in document.roots}
    candidates: set[str] = set()
    for citation in document.full_locators:
        if not isinstance(citation, FullReporterCitation) or citation.site_span.start >= before:
            continue
        root = latest(citation.root_id)
        if root not in roots or not citation.locator[-1].normalizable:
            continue
        value = citation.locator[-1].get_normalized()
        if value.volume == volume and value.edition == edition:
            candidates.add(root)
    return tuple(sorted(candidates))


def preceding_name(document: Document, span: Span) -> Span | None:
    """Read a name joined to a locator by a comma or opening parenthesis."""
    start = max(0, span.start - 160)
    for citation in document.citations:
        if citation.site_span.end <= span.start:
            start = max(start, citation.site_span.end)
    text = document.text[start : span.start]
    # Prefer an already introduced written name over a broad capitalized
    # phrase. Whole-name suffix matching prevents prose from being borrowed
    # into the name while preserving punctuation inside party names.
    known: list[Span] = []
    for name in aliases(document):
        expression = rf"(?<!\w){fuzzy_literal(name, whitespace=True, newline=True)}\s*[,(]\s*$"
        if match := re.search(expression, text, re.I):
            finish = len(text.rstrip()) - 1
            finish = len(text[:finish].rstrip())
            known.append(Span(start + match.start(), start + finish))
    if known:
        return min(known, key=lambda part: part.start)
    # A citation name is adjacent to the reporter, not the preceding sentence.
    match = re.search(r"(?P<name>[A-Z][\w.'’&/-]*(?:\s+[\w.'’&/-]+){0,12})\s*[,(]\s*$", text)
    if match is None:
        return None
    name_start = match.start("name")
    signal = re.match(r"(?:See(?:\s+also)?|But\s+see|Cf\.|Accord|Compare)\s+", match.group("name"), re.I)
    if signal:
        name_start += signal.end()
    return Span(start + name_start, start + match.end("name"))
