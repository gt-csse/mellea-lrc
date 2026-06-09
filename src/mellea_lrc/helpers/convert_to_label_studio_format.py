"""Formats case citations for Label Studio."""

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


def create_bibliography(citations: list) -> dict:
    """Format citations for Label Studio."""
    result: list = []

    return {"model_version": "TEMP:TODO", "score": 1.0, "result": result}
