"""Locator queries and grounded excerpts shared by body-text providers."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Literal

from pydantic import JsonValue
from rapidfuzz import fuzz

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import fuzzy_find
from mellea_lrc.model.citations import FullCitationVariant, FullDocketCitation
from mellea_lrc.model.citations.body_evidence import BodyEvidence, BodySource
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE as FIELD_IDENTITY_SUBSTAGE

_SPACE = re.compile(r"\s+")
_LOCATOR_MATCH = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
_BEFORE_CHARS = 2000
_AFTER_CHARS = 600
_SAME_DOCUMENT_SIMILARITY = 90
_SOURCE_COPY_SIMILARITY = 98.0
_SOURCE_COPY_CONTEXT_CHARS = 500
_SOURCE_COPY_MIN_CHARS = 300
_ALNUM_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_NAME_QUERY_NOISE = frozenset(
    {"and", "co", "corp", "corporation", "inc", "llc", "ltd", "of", "or", "not", "pc", "pllc", "the"}
)


def roots_for_body_search(document: Document) -> tuple[FullCitationVariant, ...]:
    """Search unresolved roots, except those awaiting selected-record aggregation."""
    return tuple(
        root
        for root in document.roots
        if (not root.identity_judgments and root.next_substage is None)
        or (root.next_substage is not None and root.next_substage != FIELD_IDENTITY_SUBSTAGE)
    )


def locator_text(root: FullCitationVariant) -> str:
    """Use the latest cited identifier without inferring a new case identity."""
    if isinstance(root, FullDocketCitation):
        return root.locator[-1].get_normalized().docket_number.strip()
    return _SPACE.sub(" ", root.locator[-1].quote).strip()


def field_query_name(root: FullCitationVariant) -> str | None:
    """Choose one concise *printed* name fragment for body grounding.

    A fragment only discovers possible authorities; it never verifies the
    citation's reporter or docket identifier. Court and date are left for
    candidate review, not used as search filters.
    """
    if not root.case_name:
        return None
    name = _SPACE.sub(" ", root.case_name[-1].quote).strip()
    if not name:
        return None
    sides = re.split(r"\s+v\.\s+", name, maxsplit=1, flags=re.IGNORECASE)
    if len(sides) == 2:
        eligible = [side.strip(" ,;:") for side in sides if len(re.sub(r"\W", "", side)) >= 4]
        if eligible:
            name = min(eligible, key=lambda side: (len(side), side.casefold()))
    elif name.casefold().startswith("in re "):
        name = name[6:]
    elif name.casefold().startswith("ex parte "):
        name = name[9:]
    # Keep a complete word sequence rather than clipping a long caption in
    # the middle of a token. Apart from whitespace folding, it is source text.
    return " ".join(name.split()[:3]) or None


def field_query_parties(root: FullCitationVariant) -> tuple[str, str] | None:
    """Take one printed, substantive word from each side of a case caption.

    The two words narrow discovery without requiring an exact caption or
    assuming that the parties are named identically in every body.
    """
    if not root.case_name:
        return None
    name = _SPACE.sub(" ", root.case_name[-1].quote).strip()
    sides = re.split(r"\s+v\.\s+", name, maxsplit=1, flags=re.IGNORECASE)
    if len(sides) != 2:
        return None

    def distinctive_word(side: str) -> str | None:
        words = [
            word
            for word in _ALNUM_WORDS.findall(side)
            if len(word) >= 2 and word.casefold() not in _NAME_QUERY_NOISE
        ]
        return max(words, key=len) if words else None

    left, right = (distinctive_word(side) for side in sides)
    if left is None or right is None or left.casefold() == right.casefold():
        return None
    return left, right


def evidence_date(value: object) -> date | None:
    """Accept only an exact calendar date; an imprecise date cannot satisfy a cutoff."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10]) if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|T)", value) else None
    except ValueError:
        return None


def eligible_on(issued_on: date | None, retrospective_date: date | None) -> bool:
    """Unrestricted research accepts undated text; retrospective runs do not."""
    return retrospective_date is None or (issued_on is not None and issued_on <= retrospective_date)


def make_body_evidences(
    *,
    body_id: str,
    parent_id: str | None,
    url: str | None,
    issued_on: date | None,
    date_basis: str | None,
    metadata: dict[str, JsonValue],
    body_text: str,
    locator: str,
    source_text: str,
    anchor_kind: Literal["locator", "case_name"] = "locator",
) -> tuple[BodyEvidence, ...]:
    """Retain up to three excerpts grounded at the searched text.

    An identical copy of the source filing cannot independently corroborate
    itself. Search snippets are not accepted here; callers supply fetched body
    text from an opinion, filing, or individual GovInfo granule.
    """
    # A provider may return the source filing with page furniture or OCR
    # differences. A near-copy of the entire filing is not independent
    # corroboration, even when its citation anchor is real.
    if (
        not body_text.strip()
        or fuzz.ratio(
            _SPACE.sub(" ", body_text).strip(),
            _SPACE.sub(" ", source_text).strip(),
            score_cutoff=_SAME_DOCUMENT_SIMILARITY,
        )
        >= _SAME_DOCUMENT_SIMILARITY
    ):
        return ()
    digest = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    results: list[BodyEvidence] = []
    scan_from = 0
    found_count = 0
    examined_count = 0
    while found_count < 3 and examined_count < 12:
        matched = fuzzy_find(locator, body_text[scan_from:], _LOCATOR_MATCH)
        if matched is None:
            break
        examined_count += 1
        local_start = body_text[scan_from:].find(matched)
        if local_start < 0 or not matched:
            raise ValueError("Grounded body anchor must occur in the fetched body")
        match_start = scan_from + local_start
        scan_from = match_start + len(matched)
        start = max(0, match_start - _BEFORE_CHARS)
        end = min(len(body_text), match_start + len(matched) + _AFTER_CHARS)
        results.append(
            BodyEvidence(
                body_id=body_id,
                parent_id=parent_id,
                url=url,
                issued_on=issued_on,
                date_basis=date_basis,
                metadata=metadata,
                body_sha256=digest,
                excerpt=body_text[start:end],
                source_offset=start,
                anchor_kind=anchor_kind,
                anchor_span=Span(match_start - start, match_start - start + len(matched)),
            )
        )
        found_count += 1
    return tuple(results)


def source_copy_bodies(
    source_text: str, items: tuple[tuple[BodySource, BodyEvidence], ...]
) -> set[tuple[BodySource, str]]:
    """Exclude another rendering of the source filing from independent evidence."""
    normalized_source = " ".join(_ALNUM_WORDS.findall(source_text.casefold()))
    copies: set[tuple[BodySource, str]] = set()
    for source, item in items:
        key = (source, item.body_id)
        if key in copies:
            continue
        center = (item.anchor_span.start + item.anchor_span.end) // 2
        start = max(0, center - _SOURCE_COPY_CONTEXT_CHARS)
        end = min(len(item.excerpt), center + _SOURCE_COPY_CONTEXT_CHARS)
        context = " ".join(_ALNUM_WORDS.findall(item.excerpt[start:end].casefold()))
        if len(context) < _SOURCE_COPY_MIN_CHARS:
            continue
        if (
            fuzz.partial_ratio(context, normalized_source, score_cutoff=_SOURCE_COPY_SIMILARITY)
            >= _SOURCE_COPY_SIMILARITY
        ):
            copies.add(key)
    return copies


def diverse_evidence(
    source: BodySource, items: tuple[BodyEvidence, ...], *, max_per_source: int = 6
) -> tuple[tuple[BodySource, int, BodyEvidence], ...]:
    """Show distinct source documents before second occurrences in one body."""
    groups: dict[str, list[tuple[int, BodyEvidence]]] = {}
    for index, item in enumerate(items):
        groups.setdefault(item.body_id, []).append((index, item))
    selected: list[tuple[BodySource, int, BodyEvidence]] = []
    depth = 0
    while len(selected) < max_per_source:
        next_round = [
            (index, item)
            for group in groups.values()
            if len(group) > depth
            for index, item in group[depth : depth + 1]
        ]
        if not next_round:
            break
        selected.extend((source, index, item) for index, item in next_round[: max_per_source - len(selected)])
        depth += 1
    return tuple(selected)
