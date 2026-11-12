"""Native citation collections and replayable substage checkpoints."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Self

from pydantic import TypeAdapter, field_validator, model_validator

from mellea_lrc.model.citations import FullCitationVariant
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import latest
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.execution import CheckpointRun
from mellea_lrc.model.preprocessed_document import PreprocessedDocument

_FULL_CITATION = TypeAdapter(FullCitationVariant)


class Document(PreprocessedDocument):
    """Source text, immutable citation histories, and ordered completion events."""

    citations: tuple[FullCitationVariant, ...] = ()
    runs: tuple[CheckpointRun, ...] = ()

    @classmethod
    def from_preprocessed(cls, source: PreprocessedDocument) -> Self:
        """Keep the exact source text, provenance, and component coordinates."""
        return cls.model_validate(source.model_dump(mode="python"))

    @property
    def substage_runs(self) -> tuple[str, ...]:
        """Completed atomic runs in their persisted event order."""
        return tuple(run.name for run in self.runs)

    @property
    def full_locators(self) -> tuple[FullCitationVariant, ...]:
        """Full reporter and docket occurrences in source order."""
        return self.citations

    @property
    def colocations(self) -> tuple[Colocation, ...]:
        """Derive current parsing groups from citation assignment histories."""
        groups: dict[str, list[str]] = {}
        for citation in self.full_locators:
            group_id = latest(citation.colocation_id)
            if group_id is not None:
                groups.setdefault(group_id, []).append(citation.id)
        return tuple(Colocation(id=group_id, citation_ids=tuple(ids)) for group_id, ids in groups.items())

    def _ensure_substage_open(self, substage: str) -> None:
        if substage in self.substage_runs:
            raise ValueError(f"Substage already completed: {substage}")

    def _pending_substages(self) -> set[str]:
        completed = set(self.substage_runs)
        return {
            node.substage
            for citation in self.citations
            for node in citation.nodes
            if node.substage not in completed
        }

    def add_citation(self, citation: FullCitationVariant) -> Self:
        """Add one grounded occurrence in source order without changing this document."""
        citation = _FULL_CITATION.validate_python(citation.model_dump(mode="python"))
        if any(existing.id == citation.id for existing in self.citations):
            raise ValueError(f"Citation already exists: {citation.id}")
        for node in citation.nodes:
            self._ensure_substage_open(node.substage)
        return self._with_citation(citation)

    def replace_citation(self, citation: FullCitationVariant) -> Self:
        """Append decisions and readings while preserving the stored history exactly."""
        citation = _FULL_CITATION.validate_python(citation.model_dump(mode="python"))
        original = next((item for item in self.citations if item.id == citation.id), None)
        if original is None:
            raise KeyError(f"Unknown citation: {citation.id}")
        if type(original) is not type(citation):
            raise ValueError("Citation type cannot change")
        if citation.nodes[: len(original.nodes)] != original.nodes:
            raise ValueError("Citation nodes must be append-only")
        appended_nodes = citation.nodes[len(original.nodes) :]
        for node in appended_nodes:
            self._ensure_substage_open(node.substage)
        new_nodes = {node.id for node in appended_nodes}
        for name in type(citation).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            prior = getattr(original, name) or ()
            current = getattr(citation, name) or ()
            if current[: len(prior)] != prior:
                raise ValueError(f"Citation {name} must be append-only")
            if any(reading.node_id not in new_nodes for reading in current[len(prior) :]):
                raise ValueError("New citation readings need a new decision node")
        return self._with_citation(citation)

    def _with_citation(self, citation: FullCitationVariant) -> Self:
        remaining = (item for item in self.citations if item.id != citation.id)
        ordered = tuple(sorted((*remaining, citation), key=lambda item: (item.site_span.start, item.id)))
        return type(self).model_validate({**self.model_dump(mode="python"), "citations": ordered})

    def complete_substage(self, substage: str) -> Self:
        """Commit an atomic boundary, including a run with no citation changes."""
        self._ensure_substage_open(substage)
        if self._pending_substages() - {substage}:
            raise ValueError("Complete the pending substage before starting another substage run")
        run = CheckpointRun(name=substage)
        return type(self).model_validate({**self.model_dump(mode="python"), "runs": (*self.runs, run)})

    def get_substage(self, substage: str) -> Self:
        """Reconstruct exactly the document at a completed substage boundary."""
        try:
            cutoff = self.substage_runs.index(substage) + 1
        except ValueError as exc:
            raise KeyError(f"Substage has not run: {substage}") from exc
        runs = self.runs[:cutoff]
        included = {run.name for run in runs}
        citations: list[FullCitationVariant] = []
        for citation in self.citations:
            count = 0
            for node in citation.nodes:
                if node.substage not in included:
                    break
                count += 1
            if count:
                citations.append(citation._through_node_count(count))
        return type(self).model_validate(
            {**self.model_dump(mode="python"), "citations": tuple(citations), "runs": runs}
        )

    @field_validator("citations", mode="before")
    @classmethod
    def _revalidate_citation_instances(cls, value: object) -> object:
        """Recheck nested histories even for copied or constructed citation instances."""
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray, Mapping)):
            return tuple(
                citation.model_dump(mode="python") if isinstance(citation, FullCitation) else citation
                for citation in value
            )
        return value

    @model_validator(mode="after")
    def _validate_citations(self) -> Self:
        ids = tuple(citation.id for citation in self.citations)
        if len(set(ids)) != len(ids):
            raise ValueError("Citation IDs must be unique")
        ordered = tuple(sorted(self.citations, key=lambda item: (item.site_span.start, item.id)))
        if self.citations != ordered:
            raise ValueError("Citations must be ordered by source position")
        for citation in self.citations:
            citation.validate_source(self.text)
        return self

    @model_validator(mode="after")
    def _validate_runs(self) -> Self:
        names = self.substage_runs
        if len(set(names)) != len(names):
            raise ValueError("Substage runs must be unique")
        pending = self._pending_substages()
        if len(pending) > 1:
            raise ValueError("Only one substage may have pending citation changes")
        positions = {name: index for index, name in enumerate(names)}
        for citation in self.citations:
            previous = -1
            for node in citation.nodes:
                position = positions.get(node.substage, len(names))
                if position < previous:
                    raise ValueError("Citation nodes cannot move backward across substage runs")
                previous = position
        return self

    @model_validator(mode="after")
    def _validate_colocations(self) -> Self:
        """Every completed checkpoint must retain whole parsing groups."""
        positions = {name: index for index, name in enumerate(self.substage_runs)}
        for cutoff in range(len(self.substage_runs)):
            sizes: dict[str, int] = {}
            for citation in self.full_locators:
                node_positions = {
                    node.id: positions.get(node.substage, len(positions)) for node in citation.nodes
                }
                group_id = next(
                    (
                        update.value
                        for update in reversed(citation.colocation_id)
                        if node_positions[update.node_id] <= cutoff
                    ),
                    None,
                )
                if group_id is not None:
                    sizes[group_id] = sizes.get(group_id, 0) + 1
            if any(size < 2 for size in sizes.values()):
                raise ValueError("A colocation group needs at least two citations")
        return self
