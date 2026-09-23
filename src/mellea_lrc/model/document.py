"""The document and its typed citation collection across extraction stages."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations import FullCitationVariant, latest
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.preprocessed_document import PreprocessedDocument


class Document(PreprocessedDocument):
    """Source text, citations with their histories, and completed stages."""

    citations: tuple[FullCitationVariant, ...] = ()
    completed_stages: tuple[str, ...] = ()

    @classmethod
    def from_preprocessed(cls, source: PreprocessedDocument) -> Self:
        return cls.model_validate(source.model_dump(mode="python"))

    @classmethod
    def from_source(cls, source: Path | str) -> Self:
        from mellea_lrc.preprocessing import preprocess

        return cls.from_preprocessed(preprocess(source))

    @property
    def full_locators(self) -> tuple[FullCitationVariant, ...]:
        """Every full locator occurrence, including repeated roots."""
        return self.citations

    @property
    def roots(self) -> tuple[FullCitationVariant, ...]:
        return tuple(citation for citation in self.citations if latest(citation.root_id) == citation.id)

    @property
    def colocations(self) -> tuple[Colocation, ...]:
        """Rebuild parsing groups from citation-local assignment logs."""
        groups: dict[str, list[str]] = {}
        for citation in self.citations:
            group_id = latest(citation.colocation_id)
            if group_id is not None:
                groups.setdefault(group_id, []).append(citation.id)
        return tuple(Colocation(id=group_id, citation_ids=tuple(ids)) for group_id, ids in groups.items())

    def add_citation(self, citation: FullCitationVariant) -> Self:
        if any(existing.id == citation.id for existing in self.citations):
            raise ValueError(f"Citation already exists: {citation.id}")
        return self._with_citation(citation)

    def replace_citation(self, citation: FullCitationVariant) -> Self:
        original = next((item for item in self.citations if item.id == citation.id), None)
        if original is None:
            raise KeyError(f"Unknown citation: {citation.id}")
        if type(original) is not type(citation):
            raise ValueError("Citation type cannot change")
        if citation.nodes[: len(original.nodes)] != original.nodes:
            raise ValueError("Citation nodes must be append-only")
        new_nodes = {node.id for node in citation.nodes[len(original.nodes) :]}
        for name in type(citation).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            prior = getattr(original, name)
            current = getattr(citation, name)
            if current[: len(prior)] != prior:
                raise ValueError(f"Citation {name} must be append-only")
            if any(update.node_id not in new_nodes for update in current[len(prior) :]):
                raise ValueError("New field readings need a new decision node")
        return self._with_citation(citation)

    def _with_citation(self, citation: FullCitationVariant) -> Self:
        remaining = (item for item in self.citations if item.id != citation.id)
        ordered = tuple(sorted((*remaining, citation), key=lambda item: (item.locator_span.start, item.id)))
        return type(self).model_validate({**self.model_dump(mode="python"), "citations": ordered})

    def complete(self, stage: str) -> Self:
        """Save a completed stage even when it found no citations."""
        if stage in self.completed_stages:
            return self
        return type(self).model_validate(
            {**self.model_dump(mode="python"), "completed_stages": (*self.completed_stages, stage)}
        )

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        by_id = {citation.id: citation for citation in self.citations}
        if len(by_id) != len(self.citations):
            raise ValueError("Duplicate citation in document state")
        for citation in self.citations:
            citation.validate_source(self.text)
            root_id = latest(citation.root_id)
            if root_id not in {None, WITHDRAWN_ROOT_ID} and root_id not in by_id:
                raise ValueError("Citation points to an unknown root")
        if "colocations" in self.completed_stages and any(
            len(group.citation_ids) < 2 for group in self.colocations
        ):
            raise ValueError("A colocation group needs at least two citations")
        return self
