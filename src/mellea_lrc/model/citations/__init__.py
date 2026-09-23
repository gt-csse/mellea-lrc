"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from pydantic import Field

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.full_docket import FullDocketCitation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.history import (
    CitationField,
    FieldUpdate,
    Node,
    latest,
)
from mellea_lrc.model.citations.kind import FullCitationKind

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]


def full_citation_type(kind: FullCitationKind) -> type[FullReporterCitation] | type[FullDocketCitation]:
    """Choose the concrete citation type."""
    if kind is FullCitationKind.REPORTER:
        return FullReporterCitation
    if kind is FullCitationKind.DOCKET:
        return FullDocketCitation
    raise ValueError(f"Unsupported full citation kind: {kind}")


__all__ = [
    "CitationDate",
    "CitationField",
    "FieldUpdate",
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullReporterCitation",
    "Node",
    "full_citation_type",
    "latest",
]
