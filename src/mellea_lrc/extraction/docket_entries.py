"""Attach nearby docket-entry references to admitted full docket citations."""

from __future__ import annotations

import re
from collections import Counter

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.fields.docket import DOCKET_ENTRY_PATTERN
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "docket_entries"

# Entry references before a case docket often precede it by a comma or a
# bracket. Sentence/paragraph breaks and semicolons are not adjacency.
_BEFORE_JOIN = re.compile(r"[ \t,()\[\]]{1,12}\Z")

# A following entry is more easily confused with a reference to the citing
# filing. Require an immediate comma, or a bracketed entry optionally preceded
# by a short judge-initial parenthetical. Do not cross words or a line break.
_AFTER_COMMA = re.compile(r"[ \t]*,[ \t]*\Z")
_AFTER_BRACKET = re.compile(r"[ \t]*(?:\([A-Z]{1,5}\)[ \t]*)?,?[ \t]*\[[ \t]*\Z")
_BRACKET_CLOSE = re.compile(r"[ \t]*\]")


def _after_join(source: str, locator_end: int, entry_start: int, entry_end: int) -> bool:
    gap = source[locator_end:entry_start]
    if len(gap) > 12:
        return False
    if _AFTER_COMMA.fullmatch(gap):
        return True
    return bool(_AFTER_BRACKET.fullmatch(gap) and _BRACKET_CLOSE.match(source, entry_end))


def resolve_docket_entries(document: Document) -> Document:
    """Read optional entries after all locator discovery, before colocation.

    This only updates admitted docket citations. An entry with competing
    plausible owners stays unread for later review rather than being guessed.
    """
    if STAGE in document.stage_runs:
        return document
    if "docket_locators" not in document.stage_runs:
        raise ValueError("Find docket locators before reading docket entries")
    if "colocations" in document.stage_runs:
        raise ValueError("Read docket entries before resolving colocations")

    entries = tuple(DOCKET_ENTRY_PATTERN.finditer(document.text))
    proposals: dict[str, Span] = {}
    plausible_owners: Counter[tuple[int, int]] = Counter()
    for citation in document.full_locators:
        if not isinstance(citation, FullDocketCitation) or citation.docket_entry:
            continue
        locator = citation.locator_span
        before = next((entry for entry in reversed(entries) if entry.end() <= locator.start), None)
        after = next((entry for entry in entries if entry.start() >= locator.end), None)
        candidates: list[Span] = []
        if before and _BEFORE_JOIN.fullmatch(document.text[before.end() : locator.start]):
            candidates.append(Span(*before.span()))
        if after and _after_join(document.text, locator.end, after.start(), after.end()):
            candidates.append(Span(*after.span()))
        plausible_owners.update((span.start, span.end) for span in candidates)
        if len(candidates) == 1:
            proposals[citation.id] = candidates[0]

    for citation in document.full_locators:
        span = proposals.get(citation.id)
        if span is None or plausible_owners[(span.start, span.end)] != 1:
            continue
        updated = citation.record(STAGE).with_docket_entry(document.text, span)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
