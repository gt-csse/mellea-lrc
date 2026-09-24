"""The document and its typed citation collection across extraction stages."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations import (
    CitationVariant,
    FullCitation,
    FullCitationVariant,
    ShortReporterCitation,
    latest,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.preprocessed_document import PreprocessedDocument


class Document(PreprocessedDocument):
    """Source text, citation histories, and ordered atomic stage runs."""

    citations: tuple[CitationVariant, ...] = ()
    stage_runs: tuple[str, ...] = ()

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
        return tuple(citation for citation in self.citations if isinstance(citation, FullCitation))

    @property
    def short_reporters(self) -> tuple[ShortReporterCitation, ...]:
        """Short reporter occurrences, kept outside full-locator parsing."""
        return tuple(citation for citation in self.citations if isinstance(citation, ShortReporterCitation))

    @property
    def roots(self) -> tuple[FullCitationVariant, ...]:
        return tuple(citation for citation in self.full_locators if latest(citation.root_id) == citation.id)

    @property
    def colocations(self) -> tuple[Colocation, ...]:
        """Rebuild parsing groups from citation-local assignment logs."""
        groups: dict[str, list[str]] = {}
        for citation in self.full_locators:
            group_id = latest(citation.colocation_id)
            if group_id is not None:
                groups.setdefault(group_id, []).append(citation.id)
        return tuple(Colocation(id=group_id, citation_ids=tuple(ids)) for group_id, ids in groups.items())

    def add_citation(self, citation: CitationVariant) -> Self:
        if any(existing.id == citation.id for existing in self.citations):
            raise ValueError(f"Citation already exists: {citation.id}")
        if citation.nodes[0].stage in self.stage_runs:
            raise ValueError("Cannot create a citation in a completed stage")
        return self._with_citation(citation)

    def replace_citation(self, citation: CitationVariant) -> Self:
        original = next((item for item in self.citations if item.id == citation.id), None)
        if original is None:
            raise KeyError(f"Unknown citation: {citation.id}")
        if type(original) is not type(citation):
            raise ValueError("Citation type cannot change")
        if citation.nodes[: len(original.nodes)] != original.nodes:
            raise ValueError("Citation nodes must be append-only")
        appended_nodes = citation.nodes[len(original.nodes) :]
        if any(node.stage in self.stage_runs for node in appended_nodes):
            raise ValueError("Cannot add citation nodes to a completed stage")
        new_nodes = {node.id for node in appended_nodes}
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

    def _with_citation(self, citation: CitationVariant) -> Self:
        remaining = (item for item in self.citations if item.id != citation.id)
        ordered = tuple(sorted((*remaining, citation), key=lambda item: (item.site_span.start, item.id)))
        return type(self).model_validate({**self.model_dump(mode="python"), "citations": ordered})

    def complete(self, stage: str) -> Self:
        """Commit one atomic run, including runs with no citation changes."""
        if stage in self.stage_runs:
            return self
        pending = {
            node.stage
            for citation in self.citations
            for node in citation.nodes
            if node.stage not in self.stage_runs
        }
        if pending - {stage}:
            raise ValueError("Complete the pending stage before starting another stage run")
        return type(self).model_validate(
            {**self.model_dump(mode="python"), "stage_runs": (*self.stage_runs, stage)}
        )

    def get_stage(self, stage: str) -> Self:
        """Recover exactly the document returned by a completed stage run."""
        try:
            cutoff = self.stage_runs.index(stage) + 1
        except ValueError as exc:
            raise KeyError(f"Stage has not run: {stage}") from exc
        included = set(self.stage_runs[:cutoff])
        citations: list[CitationVariant] = []
        for citation in self.citations:
            count = 0
            for node in citation.nodes:
                if node.stage not in included:
                    break
                count += 1
            if count:
                citations.append(citation._through_node_count(count))
        citations.sort(key=lambda item: (item.site_span.start, item.id))
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "citations": tuple(citations),
                "stage_runs": self.stage_runs[:cutoff],
            }
        )

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if len(set(self.stage_runs)) != len(self.stage_runs):
            raise ValueError("Stage runs must be unique")
        stage_positions = {stage: index for index, stage in enumerate(self.stage_runs)}
        pending_stages: set[str] = set()
        by_id = {citation.id: citation for citation in self.citations}
        if len(by_id) != len(self.citations):
            raise ValueError("Duplicate citation in document state")
        if tuple(sorted(self.citations, key=lambda item: (item.site_span.start, item.id))) != self.citations:
            raise ValueError("Citations must be ordered by site span and ID")
        for citation in self.citations:
            previous_stage = -1
            for node in citation.nodes:
                position = stage_positions.get(node.stage)
                if position is None:
                    pending_stages.add(node.stage)
                    previous_stage = len(self.stage_runs)
                elif position < previous_stage:
                    raise ValueError("Citation nodes cannot move backward through stage runs")
                else:
                    previous_stage = position
            citation.validate_source(self.text)
            node_stages = {node.id: node.stage for node in citation.nodes}
            for update in citation.root_id:
                if update.value in {None, WITHDRAWN_ROOT_ID}:
                    continue
                target = by_id.get(update.value)
                if target is None:
                    raise ValueError("Citation points to an unknown root")
                if not isinstance(target, FullCitation):
                    raise ValueError("Citation root must be a full citation")
                target_stage = stage_positions.get(target.nodes[0].stage, len(self.stage_runs))
                update_stage = stage_positions.get(node_stages[update.node_id], len(self.stage_runs))
                if target_stage > update_stage:
                    raise ValueError("Citation root cannot be created after its assignment")
        if len(pending_stages) > 1:
            raise ValueError("Only one stage can have uncommitted citation nodes")
        # A later stage must not make an invalid earlier checkpoint look valid.
        for cutoff in range(len(self.stage_runs)):
            group_sizes: dict[str, int] = {}
            for citation in self.full_locators:
                if stage_positions.get(citation.nodes[0].stage, len(self.stage_runs)) > cutoff:
                    continue
                node_positions = {
                    node.id: stage_positions.get(node.stage, len(self.stage_runs)) for node in citation.nodes
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
                    group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
            if any(size < 2 for size in group_sizes.values()):
                raise ValueError("A colocation group needs at least two citations")
        return self
