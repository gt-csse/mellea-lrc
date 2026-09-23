"""Typed full citation occurrences independent of processing stage."""

from typing import Annotated, TypeAlias

from pydantic import Field

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.citations.full import FullCitation, FullCitationKind
from mellea_lrc.model.citations.full_docket import FullDocketCitation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation

FullCitationVariant: TypeAlias = Annotated[
    FullReporterCitation | FullDocketCitation,
    Field(discriminator="kind"),
]


def full_citation_type(kind: FullCitationKind) -> type[FullReporterCitation] | type[FullDocketCitation]:
    """Choose the concrete type recorded by a create operation."""
    if kind is FullCitationKind.REPORTER:
        return FullReporterCitation
    if kind is FullCitationKind.DOCKET:
        return FullDocketCitation
    raise ValueError(f"Unsupported full citation kind: {kind}")


__all__ = [
    "CitationDate",
    "FullCitation",
    "FullCitationKind",
    "FullCitationVariant",
    "FullDocketCitation",
    "FullReporterCitation",
    "full_citation_type",
]
