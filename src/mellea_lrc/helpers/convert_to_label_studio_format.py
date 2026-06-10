"""Formats case citations for Label Studio."""

from collections.abc import Callable
import uuid

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

CITATION_TYPES: set = {
    FullCaseCitation,
    FullLawCitation,
    FullJournalCitation,
    ShortCaseCitation,
    SupraCitation,
    IdCitation,
    ReferenceCitation,
    UnknownCitation,
}

REFERENCE_TYPES: set = {ShortCaseCitation, SupraCitation, IdCitation, ReferenceCitation}


class UnknownCitationError(Exception):
    """Error for unknown citation types."""

    def __init__(self, message: str) -> None:
        """Initialize the exception class with message."""
        super().__init__(message)


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
    }


def _full_journal_fields(c: FullJournalCitation) -> dict:
    return {
        "volume": c.groups.get("volume"),
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("page"),
        "pin_cite": c.metadata.pin_cite,
        "year": c.metadata.year,
        "parenthetical": c.metadata.parenthetical,
    }


def _short_case_fields(c: ShortCaseCitation) -> dict:
    return {
        "volume": c.groups.get("volume"),
        "reporter": c.groups.get("reporter"),
        "page": c.groups.get("page"),
        "pin_cite": c.metadata.pin_cite,
        "court": c.metadata.court,
        "parenthetical": c.metadata.parenthetical,
    }


def _supra_fields(c: SupraCitation) -> dict:
    return {
        "pin_cite": c.metadata.pin_cite,
        "parenthetical": c.metadata.parenthetical,
    }


def _reference_fields(c: ReferenceCitation) -> dict:
    return {
        "plaintiff": c.metadata.plaintiff,
        "defendant": c.metadata.defendant,
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


def _assign_id_to_citation(citations: list[CitationBase]) -> list[tuple[CitationBase, str]]:
    """Assign unique IDs to citations.

    Args:
    ----
        citations: a list of citations

    Returns:
    -------
        a list of all citations with unique IDs

    """
    citations_with_id = []
    # Check if citation is onIe of eyecite's
    for cite in citations:
        # Get the type
        cls = type(cite)
        if cls not in CITATION_TYPES:
            raise UnknownCitationError("Ensure all citations types are")
        citations_with_id.append((cite, str(uuid.uuid4())[:8]))
    return citations_with_id


def _get_antecedent_mapping(
    resolutions: dict[Resource, list[CitationBase]], citations_with_id: list[tuple[CitationBase, str]]
) -> dict[str, str]:
    """Build mapping from reference id -> citations.

    Args:
    ----
        resolutions: a dict of resolved citations
        citations_with_id: a list of citions with unique IDs

    Returns:
    -------
        a dict that maps all citations with the same reference to a unique id

    """
    # Create a list of unique ID for each reference
    reference_ids = {id(cite): cite_id for cite, cite_id in citations_with_id}
    mapping: dict[str, str] = {}
    for references in resolutions.values():
        full_reference_id = reference_ids[id(references[0])]
        for reference in references[1:]:
            reference_id = reference_ids[id(reference)]
            mapping[reference_id] = full_reference_id
    return mapping


def create_bibliography(citations: list[CitationBase], resources: dict[Resource, list[CitationBase]]) -> dict:
    """Format citations for Label Studio.

    Args:
    ----
        citations: a list of citations
        resources: a dict of resolved citations

    Returns:
    -------
        A dict formatted for Label Studio

    """

    citations_with_id = _assign_id_to_citation(citations)
    antecedent_mapping = _get_antecedent_mapping(resources, citations_with_id)
    results: list = []
    for cite, resource_id in citations_with_id:
        cls = type(cite)
        span = cite.full_span()
        matched = cite.matched_text()
        results.append(
            {
                "id": resource_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {"start": span[0], "end": span[1], "text": matched, "labels": [cls.__name__]},
            }
        )
        result = FIELD_HANDLERS[cls](cite)
        for field_name, value in result.items():
            results.append(
                {
                    "id": resource_id,
                    "from_name": field_name,
                    "to_name": "text",
                    "type": "textarea",
                    "value": {"text": [value if value is not None else ""]},
                }
            )
    for resource_id, full_resource_id in antecedent_mapping.items():
        results.append(
            {"from_id": resource_id, "to_id": full_resource_id, "type": "relation", "direction": "right"}
        )
    return {"model_version": "0.0.0", "score": 1.0, "result": results}
