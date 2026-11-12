"""Read docket entries that belong to a single discovered docket locator."""

import re
from collections import Counter

from mellea_lrc.extraction.find_docket_locators import SUBSTAGE as DOCKETS
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.fields.docket import DOCKET_ENTRY_PATTERN
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_docket_entries"

_BEFORE_JOIN = re.compile(r"[ \t,()\[\]]{1,12}\Z")
_AFTER_COMMA = re.compile(r"[ \t]*,[ \t]*\Z")
_AFTER_BRACKET = re.compile(r"[ \t]*(?:\([A-Z]{1,5}\)[ \t]*)?,?[ \t]*\[[ \t]*\Z")
_BRACKET_CLOSE = re.compile(r"[ \t]*\]")


def _after_join(source: str, locator_end: int, entry: Span) -> bool:
    gap = source[locator_end : entry.start]
    if len(gap) > 12:
        return False
    if _AFTER_COMMA.fullmatch(gap):
        return True
    return bool(_AFTER_BRACKET.fullmatch(gap) and _BRACKET_CLOSE.match(source, entry.end))


def _adjacent_entries(source: str, locator: Span, entries: tuple[Span, ...]) -> tuple[Span, ...]:
    before = next((entry for entry in reversed(entries) if entry.end <= locator.start), None)
    after = next((entry for entry in entries if entry.start >= locator.end), None)
    candidates = []
    if before is not None and _BEFORE_JOIN.fullmatch(source[before.end : locator.start]):
        candidates.append(before)
    if after is not None and _after_join(source, locator.end, after):
        candidates.append(after)
    return tuple(candidates)


def resolve_docket_entries(document: Document) -> Document:
    """Read entries with a unique adjacent docket and complete the substage."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if DOCKETS not in document.substage_runs:
        raise ValueError("Find docket locators before reading docket entries")

    entries = tuple(Span(*match.span()) for match in DOCKET_ENTRY_PATTERN.finditer(document.text))
    candidates = {}
    owners: Counter[Span] = Counter()
    dockets = [citation for citation in document.full_locators if isinstance(citation, FullDocketCitation)]
    for citation in dockets:
        nearby = (
            (citation.docket_entry[-1].span,)
            if citation.docket_entry
            else _adjacent_entries(document.text, citation.locator_span, entries)
        )
        owners.update(nearby)
        if len(nearby) == 1 and not citation.docket_entry:
            candidates[citation.id] = nearby[0]

    for citation in dockets:
        entry = candidates.get(citation.id)
        if entry is not None and owners[entry] == 1:
            document = document.replace_citation(
                citation.record(SUBSTAGE).with_docket_entry(document.text, entry)
            )
    return document.complete_substage(SUBSTAGE)
