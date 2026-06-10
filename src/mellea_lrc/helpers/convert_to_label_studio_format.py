"""Formats case citations for Label Studio."""

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

def _get_antecedent_mapping(resolutions: dict[Resource, list[CitationBase]]):


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
    result: list = []

    antecedent_mapping = 

    return {"model_version": "TEMP:TODO", "score": 1.0, "result": result}
