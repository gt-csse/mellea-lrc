"""Extract and label case law documents' citations."""

import uuid
from typing import cast
from collections.abc import Callable
from eyecite import get_citations, resolve_citations
from eyecite.models import (
    CitationBase,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    Resource,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
)

CITATION_TYPES = {
    FullCaseCitation,
    FullLawCitation,
    FullJournalCitation,
    ShortCaseCitation,
    SupraCitation,
    IdCitation,
    ReferenceCitation,
    UnknownCitation,
}

REFERENCE_TYPES = {ShortCaseCitation, SupraCitation, IdCitation, ReferenceCitation}


class PredictionError(Exception):
    def __init__(self, message: str, citation: CitationBase) -> None:
        self.message = message
        self.citation = citation
        super().__init__(message)


# ── Per-class field dicts ─────────────────────────────────────────────────────


def _full_case_fields(c: FullCaseCitation) -> dict:
    return {
        "plaintiff": c.metadata.plaintiff,
        "defendant": c.metadata.defendant,
        "volume": c.groups.get("volume"),
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("page"),
        "pin_cite": c.metadata.pin_cite,
        "extra": c.metadata.extra,
        "year": c.metadata.year,
        "court": c.metadata.court,
        "parenthetical": c.metadata.parenthetical,
        # omitted: month, day (rarely populated)
        # omitted: antecedent_guess (not applicable to full citations)
        # omitted: resolved_case_name, resolved_case_name_short (redundant with plaintiff/defendant)
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


def _full_law_fields(c: FullLawCitation) -> dict:
    return {
        "volume": c.groups.get("title"),  # law citations use "title" not "volume"
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("section"),  # law citations use "section" not "page"
        "pin_cite": c.metadata.pin_cite,
        "year": c.metadata.year,
        "publisher": c.metadata.publisher,
        "parenthetical": c.metadata.parenthetical,
        # omitted: month, day (rarely populated)
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


def _full_journal_fields(c: FullJournalCitation) -> dict:
    return {
        "volume": c.groups.get("volume"),
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("page"),
        "pin_cite": c.metadata.pin_cite,
        "year": c.metadata.year,
        "parenthetical": c.metadata.parenthetical,
        # omitted: month, day (rarely populated)
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


def _short_case_fields(c: ShortCaseCitation) -> dict:
    return {
        "volume": c.groups.get("volume"),
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("page"),
        "pin_cite": c.metadata.pin_cite,
        "court": c.metadata.court,
        "parenthetical": c.metadata.parenthetical,
        # omitted: antecedent_guess (raw heuristic; resolved antecedent injected separately)
        # omitted: month, day (rarely populated)
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


def _supra_fields(c: SupraCitation) -> dict:
    return {
        "pin_cite": c.metadata.pin_cite,
        "parenthetical": c.metadata.parenthetical,
        # omitted: antecedent_guess (raw heuristic; resolved antecedent injected separately)
        # omitted: volume (metadata field but rarely populated for supra citations)
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


def _reference_fields(c: ReferenceCitation) -> dict:
    return {
        "plaintiff": c.metadata.plaintiff,
        "defendant": c.metadata.defendant,
        # omitted: resolved_case_name, resolved_case_name_short (redundant with plaintiff/defendant)
    }


def _id_fields(c: IdCitation) -> dict:
    return {
        "pin_cite": c.metadata.pin_cite,
        "parenthetical": c.metadata.parenthetical,
        # omitted: pin_cite_span_start, pin_cite_span_end (not reliably parsed by eyecite)
    }


FIELD_HANDLERS: dict[type, Callable] = {
    FullCaseCitation: _full_case_fields,
    FullLawCitation: _full_law_fields,
    FullJournalCitation: _full_journal_fields,
    ShortCaseCitation: _short_case_fields,
    SupraCitation: _supra_fields,
    IdCitation: _id_fields,
    ReferenceCitation: _reference_fields,
    UnknownCitation: lambda _: {},  # no bibliographic fields; span is still labelled
}


def _get_citation_regionids(
    citations: list[CitationBase],
) -> list[tuple[CitationBase, str]]:
    citation_regionid: list[tuple[CitationBase, str]] = []
    for c in citations:
        cls = type(c)
        if cls not in CITATION_TYPES:
            raise ValueError(
                f"Unknown citation type: {cls.__name__}. All citation types must be handled explicitly."
            )
        citation_regionid.append((c, str(uuid.uuid4())[:8]))
    return citation_regionid


def _get_antecedent_map(
    resolutions: dict[Resource, list[CitationBase]],
    citation_regionid: list[tuple[CitationBase, str]],
) -> dict[str, str]:
    # Build antecedent map: ref region_id → full citation region_id.
    # resolutions[resource][0] is always the full citation (inserted first
    # by resolve_citations as citations are processed in document order).
    citation_to_region: dict[int, str] = {id(c): rid for c, rid in citation_regionid}
    antecedent_map: dict[str, str] = {}
    for _, refs in resolutions.items():
        # this should never fail per eyecite logic, we do not guard keyerror in case some edge cases appear
        full_region_id = citation_to_region[id(refs[0])]
        for ref in refs[1:]:
            ref_region_id = citation_to_region[id(ref)]
            antecedent_map[ref_region_id] = full_region_id
    return antecedent_map


# ── Main prediction builder ───────────────────────────────────────────────────


def citations_to_prediction(text: str) -> dict:
    """Run eyecite (with resolution) and return a single LS prediction dict.

    Args:
    ----
    text: str

    For each recognized citation:
    - A 'labels' result marks the span with the eyecite class name.
    - Per-region TextArea results carry bibliographic fields from eyecite metadata.
    - For reference types (ShortCaseCitation, SupraCitation, IdCitation) a
      Relation result links the region to its resolved full citation region.
    """
    citations = get_citations(text)

    resolutions = cast(
        dict[Resource, list[CitationBase]],
        resolve_citations(citations),
    )

    citation_regionid = _get_citation_regionids(citations)
    antecedent_map = _get_antecedent_map(resolutions, citation_regionid)

    results = []

    for c, region_id in citation_regionid:
        cls = type(c)
        span = c.full_span()
        matched = c.matched_text()

        results.append(
            {
                "id": region_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": span[0],
                    "end": span[1],
                    "text": matched,
                    "labels": [cls.__name__],
                },
            }
        )

        # Attach bibliographic fields. Always emit even when empty so Label
        # Studio clears stale values when switching between regions.
        for field_name, value in FIELD_HANDLERS[cls](c).items():
            results.append(
                {
                    "id": region_id,
                    "from_name": field_name,
                    "to_name": "text",
                    "type": "textarea",
                    "value": {"text": [value if value is not None else ""]},
                }
            )

    for ref_region_id, full_region_id in antecedent_map.items():
        results.append(
            {
                "from_id": ref_region_id,
                "to_id": full_region_id,
                "type": "relation",
                "direction": "right",
            }
        )

    return {
        "model_version": "eyecite-pre-annotation",
        "score": 1.0,
        "result": results,
    }
