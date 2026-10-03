"""Typed page indexes for saved reporter-root opinion text."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.pin_cite import PinCiteKind
from mellea_lrc.model.span import Span


class OpinionPage(BaseModel):
    """One source-marked page or paragraph in a rendered opinion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    kind: PinCiteKind | None
    volume: int | None = None
    edition: str | None = None
    span: Span
    citation_index: str | None = None
    pagination_inferred: bool = False

    @model_validator(mode="after")
    def _validate_metadata(self) -> OpinionPage:
        if not self.label.strip():
            raise ValueError("An indexed page needs its printed label")
        if self.volume is not None and self.volume < 1:
            raise ValueError("A reporter volume must be positive")
        if self.edition == "":
            raise ValueError("An edition cannot be empty")
        if self.citation_index == "":
            raise ValueError("A citation index cannot be empty")
        if self.pagination_inferred and (self.volume is None or self.edition is None):
            raise ValueError("An inferred pagination namespace needs both volume and edition")
        return self


class IndexedReporterOpinion(BaseModel):
    """A saved opinion rendering with spans for its explicit page markers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    opinion_id: str
    text_field: str | None
    text: str
    pages: tuple[OpinionPage, ...] = ()

    @model_validator(mode="after")
    def _validate_index(self) -> IndexedReporterOpinion:
        if not self.opinion_id.isdecimal():
            raise ValueError("An indexed opinion ID must contain decimal digits")
        if self.text_field is None and (self.text or self.pages):
            raise ValueError("An opinion without a saved text field cannot have rendered text or pages")
        previous_start = -1
        namespace_ends: dict[tuple[PinCiteKind | None, str | None, int | None, str | None], int] = {}
        for page in self.pages:
            if page.span.end > len(self.text):
                raise ValueError("A page span extends past its rendered opinion text")
            if page.span.start < previous_start:
                raise ValueError("Opinion page markers must be in source order")
            namespace = (page.kind, page.citation_index, page.volume, page.edition)
            if page.span.start < namespace_ends.get(namespace, 0):
                raise ValueError("Pages in one pagination namespace cannot overlap")
            # Different pagination namespaces can overlap because each span
            # ends at the next marker in its own namespace.
            namespace_ends[namespace] = page.span.end
            previous_start = page.span.start
        return self


class ReporterRootOpinionPageIndex(BaseModel):
    """Rendered text and explicit page markers for all opinions in a cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    cluster_id: str
    opinions: tuple[IndexedReporterOpinion, ...]

    @model_validator(mode="after")
    def _validate_opinions(self) -> ReporterRootOpinionPageIndex:
        if not self.cluster_id.isdecimal():
            raise ValueError("A reporter opinion page index needs a decimal cluster ID")
        identifiers = tuple(item.opinion_id for item in self.opinions)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("An opinion page index cannot repeat opinion IDs")
        return self
