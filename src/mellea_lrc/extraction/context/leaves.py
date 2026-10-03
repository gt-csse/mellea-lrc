"""Shared source reading for leaf stages; no stage is executed here."""

from __future__ import annotations

import re

from eyecite.models import ReferenceCitation as EyeciteReferenceCitation

from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citations import FullReporterCitation, latest
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.events import events
from mellea_lrc.parsing.pin_cite import pin_after as read_pin_after


def require_leaves(document: Document, stage: str) -> None:
    if stage in document.stage_runs:
        raise ValueError(f"Stage already completed: {stage}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before growing leaves")


def pin_after(source: str, position: int, end: int) -> Span | None:
    """Read a directly adjacent pin with the same grammar as normalization."""
    reading = read_pin_after(source, position, end)
    return Span(*reading) if reading is not None else None


def reference_pin_after(source: str, position: int, end: int) -> Span | None:
    """Require an explicit pinpoint directly after a case-name reference.

    Unlike a reporter citation, a name followed by a bare number may be an
    address, date, or numbered paragraph. Only an ``at`` or page/paragraph
    marker establishes the supported reference shape. Commas and whitespace
    can separate the name and marker; intervening prose cannot.
    """
    at = fuzzy_literal("at ", whitespace=True, newline=True)
    marker = re.match(rf"\s*(?:,\s*)*(?:{at}|(?=[*¶]))", source[position:end], re.I)
    if marker is None:
        return None
    return pin_after(source, position + marker.end(), end)


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
    # Source recognition supplies boundaries even before other leaf types
    # have been created. Completing one type first must not borrow a name
    # across an intervening Id., supra or noncase citation. Name references
    # themselves are excluded: an adjacent name is what this reader seeks.
    for event in events(document.text):
        if not isinstance(event, EyeciteReferenceCitation):
            _, end = event.span_with_pincite()
            if end <= span.start:
                start = max(start, end)
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
