"""Locator queries and grounded excerpts shared by body-text providers."""

from __future__ import annotations

import hashlib
import re
from datetime import date

from pydantic import JsonValue
from rapidfuzz import fuzz

from mellea_lrc.matching.fuzziness import FuzzinessOption
from mellea_lrc.matching.grounding import fuzzy_find
from mellea_lrc.model.citations import FullCitationVariant, FullDocketCitation
from mellea_lrc.model.citations.body_evidence import BodyEvidence
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

_SPACE = re.compile(r"\s+")
_LOCATOR_MATCH = FuzzinessOption.edit_distance(similarity_percent=90, whitespace_relaxation=True)
_NAME_MATCH = FuzzinessOption.edit_distance(similarity_percent=75, whitespace_relaxation=True)
_BEFORE_CHARS = 350
_AFTER_CHARS = 250
_SAME_DOCUMENT_SIMILARITY = 90


def roots_for_body_search(document: Document) -> tuple[FullCitationVariant, ...]:
    """Only roots without a final identity verdict need another evidence route."""
    return tuple(
        root
        for root in document.roots
        if not root.identity_judgments or root.identity_judgments[-1].verdict is IdentityVerdict.DEFERRED
    )


def locator_text(root: FullCitationVariant) -> str:
    """Use the latest cited identifier without inferring a new case identity."""
    if isinstance(root, FullDocketCitation):
        return root.locator[-1].get_normalized().docket_number.strip()
    return _SPACE.sub(" ", root.locator[-1].quote).strip()


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
    case_name: str | None = None,
) -> tuple[BodyEvidence, ...]:
    """Retain up to three exact excerpts per locator/name anchor.

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
    anchors = [("locator", locator, _LOCATOR_MATCH)]
    if case_name and len(case_name.strip()) >= 8:
        anchors.append(("case_name", case_name, _NAME_MATCH))
    for anchor_kind, anchor, policy in anchors:
        scan_from = 0
        found_count = 0
        examined_count = 0
        while found_count < 3 and examined_count < 12:
            matched = fuzzy_find(anchor, body_text[scan_from:], policy)
            if matched is None:
                break
            examined_count += 1
            local_start = body_text[scan_from:].find(matched)
            if local_start < 0 or not matched:
                raise ValueError("Grounded body anchor must occur in the fetched body")
            match_start = scan_from + local_start
            scan_from = match_start + len(matched)
            if any(item.source_offset + item.anchor_span.start == match_start for item in results):
                continue
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
