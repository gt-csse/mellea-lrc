"""The document and cross-citation relationships across pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations import FullCitationKind, FullCitationVariant, full_citation_type, latest
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, CitationField
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.preprocessed_document import PreprocessedDocument


class Document(PreprocessedDocument):
    """Current citations and stage results; each citation owns its history."""

    citations: tuple[FullCitationVariant, ...] = ()
    colocations: tuple[Colocation, ...] = ()
    completed_stages: tuple[str, ...] = ()

    @classmethod
    def from_preprocessed(cls, source: PreprocessedDocument) -> Self:
        """Start citation work without changing source text or offsets."""
        return cls.model_validate(source.model_dump(mode="python"))

    @classmethod
    def from_source(cls, source: Path | str) -> Self:
        """Preprocess a file path or supplied text, then start citation work."""
        from mellea_lrc.preprocessing import preprocess

        return cls.from_preprocessed(preprocess(source))

    @property
    def full_locators(self) -> tuple[FullCitationVariant, ...]:
        """Every detected full locator occurrence, including repeated roots."""
        return tuple(citation for citation in self.citations if latest(citation.locator_span) is not None)

    @property
    def roots(self) -> tuple[FullCitationVariant, ...]:
        """Canonical first occurrences after root formation."""
        return tuple(citation for citation in self.citations if latest(citation.root_id) == citation.id)

    def create_citation(self, stage: str, citation_id: str, kind: FullCitationKind) -> Self:
        """Add an empty concrete citation with its own creation node."""
        if any(citation.id == citation_id for citation in self.citations):
            raise ValueError(f"Citation already exists: {citation_id}")
        citation = full_citation_type(kind).create(citation_id, stage)
        return self._replace_citation(citation)

    def update_fields(self, stage: str, citation_id: str, changes: dict[CitationField, object]) -> Self:
        """Ask one citation to record and apply a field-reading decision."""
        citation = self._citation(citation_id)
        return self._replace_citation(citation.update_fields(stage, changes))

    def withdraw_citation(self, stage: str, citation_id: str) -> Self:
        """Record a citation-local withdrawal to the dummy root head."""
        citation = self._citation(citation_id)
        return self._replace_citation(citation.withdraw(stage))

    def _citation(self, citation_id: str) -> FullCitationVariant:
        for citation in self.citations:
            if citation.id == citation_id:
                return citation
        raise KeyError(f"Unknown citation: {citation_id}")

    def _replace_citation(self, changed: FullCitationVariant) -> Self:
        remaining = [citation for citation in self.citations if citation.id != changed.id]

        def position(item: FullCitationVariant) -> tuple[int, str]:
            site = latest(item.locator_span)
            return (site.start if site is not None else len(self.text), item.id)

        ordered = tuple(sorted((*remaining, changed), key=position))
        return type(self).model_validate({**self.model_dump(mode="python"), "citations": ordered})

    def complete(self, stage: str, *, colocations: tuple[Colocation, ...] | None = None) -> Self:
        """Mark an inspectable stage complete, even if it found nothing."""
        if stage in self.completed_stages:
            return self
        changes = self.model_dump(mode="python")
        changes["completed_stages"] = (*self.completed_stages, stage)
        if colocations is not None:
            changes["colocations"] = colocations
        return type(self).model_validate(changes)

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        """Check relationships; each citation validates its own local updates."""
        ids = {citation.id for citation in self.citations}
        if len(ids) != len(self.citations):
            raise ValueError("Duplicate citation in document state")
        for citation in self.citations:
            root_id = latest(citation.root_id)
            if root_id not in {None, WITHDRAWN_ROOT_ID} and root_id not in ids:
                raise ValueError("Citation points to an unknown root")
        if "colocations" in self.completed_stages:
            grouped: dict[str, list[str]] = {}
            for citation in self.citations:
                colocation_id = latest(citation.colocation_id)
                if colocation_id is not None:
                    grouped.setdefault(colocation_id, []).append(citation.id)
            listed = {group.id: group.citation_ids for group in self.colocations}
            if {key: tuple(value) for key, value in grouped.items()} != listed:
                raise ValueError("Colocation groups disagree with citation state")
            if any(len(group.citation_ids) < 2 for group in self.colocations):
                raise ValueError("A colocation group needs at least two citations")
        return self
