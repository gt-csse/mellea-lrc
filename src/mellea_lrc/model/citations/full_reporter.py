"""Full citations identified by a reporter or legal database locator."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import LocatorField, PageField, ReporterField, VolumeField
from mellea_lrc.model.citations.full import FullCitation, _source_slice
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence; its locator text owns the source span."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator_text: tuple[LocatorField, ...]
    volume: tuple[VolumeField, ...] = ()
    reporter: tuple[ReporterField, ...] = ()
    page: tuple[PageField, ...] = ()

    @property
    def locator_span(self) -> Span:
        span = self.locator_text[-1].span
        assert span is not None
        return span

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
        volume: str | None,
        reporter: str | None,
        page: str | None,
    ) -> Self:
        """Create the citation and initial readings in one decision node."""
        written = _source_slice(source, span)
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        components: dict[str, tuple[VolumeField | ReporterField | PageField, ...]] = {}
        offset = 0
        for name, value, field_type in (
            ("volume", volume, VolumeField),
            ("reporter", reporter, ReporterField),
            ("page", page, PageField),
        ):
            if value is None:
                components[name] = ()
                continue
            start = written.find(value, offset)
            if start < 0:
                raise ValueError(f"Eyecite {name} is not in its locator quote")
            end = start + len(value)
            components[name] = (
                field_type.from_source(
                    source,
                    Span(span.start + start, span.start + end),
                    normalized=value,
                    node_id=node.id,
                ),
            )
            offset = end
        return cls(
            id=citation_id,
            nodes=(node,),
            locator_text=(LocatorField.from_source(source, span, normalized=written, node_id=node.id),),
            volume=components["volume"],
            reporter=components["reporter"],
            page=components["page"],
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if (
            not self.locator_text
            or self.locator_text[0].node_id != self.nodes[0].id
            or any(update.span is None for update in self.locator_text)
        ):
            raise ValueError("Reporter citation needs a source-spanned locator")
        return self
