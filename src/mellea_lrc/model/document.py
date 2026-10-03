"""The document and its typed citation collection across extraction stages."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations import (
    CitationVariant,
    FullCitation,
    FullCitationVariant,
    FullReporterCitation,
    LeafCitation,
    ShortReporterCitation,
    latest,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.preprocessed_document import PreprocessedDocument
from mellea_lrc.model.site_review import SiteReview


def _colocations(
    citations: tuple[FullCitationVariant | ShortReporterCitation, ...],
) -> tuple[Colocation, ...]:
    """Rebuild ordered parsing groups from one collection's assignment history."""
    groups: dict[str, list[str]] = {}
    for citation in citations:
        group_id = latest(citation.colocation_id)
        if group_id is not None:
            groups.setdefault(group_id, []).append(citation.id)
    return tuple(Colocation(id=group_id, citation_ids=tuple(ids)) for group_id, ids in groups.items())


class Document(PreprocessedDocument):
    """Source text, citation histories, and ordered atomic stage runs."""

    citations: tuple[CitationVariant, ...] = ()
    site_reviews: tuple[SiteReview, ...] = ()
    # TODO: Type stage names (including Node.stage and SiteReview.stage) once
    # the stage catalog settles. Strings currently allow development checkpoints.
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
    def short_citations(self) -> tuple[LeafCitation, ...]:
        """All short-form histories, including withdrawn or unresolved sites."""
        return tuple(citation for citation in self.citations if isinstance(citation, LeafCitation))

    @property
    def leaves(self) -> tuple[CitationVariant, ...]:
        """Attached occurrences, including repeated full citations."""
        return tuple(
            citation
            for citation in self.citations
            if latest(citation.root_id) not in {None, WITHDRAWN_ROOT_ID, citation.id}
        )

    @property
    def colocations(self) -> tuple[Colocation, ...]:
        """Rebuild parsing groups from citation-local assignment logs."""
        return _colocations(self.full_locators)

    @property
    def short_reporter_colocations(self) -> tuple[Colocation, ...]:
        """Rebuild short reporter parsing groups without changing full groups."""
        return _colocations(self.short_reporters)

    def add_citation(self, citation: CitationVariant) -> Self:
        if any(existing.id == citation.id for existing in self.citations):
            raise ValueError(f"Citation already exists: {citation.id}")
        if citation.nodes[0].stage in self.stage_runs:
            raise ValueError("Cannot create a citation in a completed stage")
        return self._with_citation(citation)

    def add_site_review(self, review: SiteReview) -> Self:
        """Append a review even when no citation was created at that site."""
        if review.stage in self.stage_runs:
            raise ValueError("Cannot review a site in a completed stage")
        return type(self).model_validate(
            {**self.model_dump(mode="python"), "site_reviews": (*self.site_reviews, review)}
        )

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
            if isinstance(prior, tuple) or isinstance(current, tuple):
                # Nullable histories begin as None and become tuples on their
                # first reading. Validate their entries as an append-only log.
                prior_entries = prior if prior is not None else ()
                current_entries = current if current is not None else ()
                if current_entries[: len(prior_entries)] != prior_entries:
                    raise ValueError(f"Citation {name} must be append-only")
                added = current_entries[len(prior_entries) :]
            else:
                if prior is not None and current != prior:
                    raise ValueError(f"Citation {name} must be append-only")
                added = (current,) if prior is None and current is not None else ()
            if any(update.node_id not in new_nodes for update in added):
                raise ValueError("New citation records need a new decision node")
        return self._with_citation(citation)

    def _with_citation(self, citation: CitationVariant) -> Self:
        remaining = (item for item in self.citations if item.id != citation.id)
        ordered = tuple(sorted((*remaining, citation), key=lambda item: (item.site_span.start, item.id)))
        return type(self).model_validate({**self.model_dump(mode="python"), "citations": ordered})

    def complete(self, stage: str) -> Self:
        """Commit one atomic run, including runs with no citation changes."""
        if stage in self.stage_runs:
            raise ValueError(f"Stage already completed: {stage}")
        pending = {
            node.stage
            for citation in self.citations
            for node in citation.nodes
            if node.stage not in self.stage_runs
        }
        pending.update(review.stage for review in self.site_reviews if review.stage not in self.stage_runs)
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
                "site_reviews": tuple(review for review in self.site_reviews if review.stage in included),
                "stage_runs": self.stage_runs[:cutoff],
            }
        )

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if len(set(self.stage_runs)) != len(self.stage_runs):
            raise ValueError("Stage runs must be unique")
        stage_positions = {stage: index for index, stage in enumerate(self.stage_runs)}
        pending_stages: set[str] = set()
        for review in self.site_reviews:
            if review.candidate_span.start == review.candidate_span.end or not review.candidate_text.strip():
                raise ValueError("Site review needs a nonempty source span")
            if self.text[review.candidate_span.start : review.candidate_span.end] != review.candidate_text:
                raise ValueError("Site review quote does not match its source span")
            if review.stage not in stage_positions:
                pending_stages.add(review.stage)
            if review.outcome == "accepted" and review.citation_id is None:
                raise ValueError("Accepted site review needs a citation ID")
            if review.outcome != "accepted" and review.citation_id is not None:
                raise ValueError("Unaccepted site review cannot point to a citation")
        by_id = {citation.id: citation for citation in self.citations}
        if len(by_id) != len(self.citations):
            raise ValueError("Duplicate citation in document state")
        for review in self.site_reviews:
            if review.citation_id is None:
                continue
            citation = by_id.get(review.citation_id)
            if (
                citation is None
                or citation.nodes[0].stage != review.stage
                or citation.site_span != review.candidate_span
            ):
                raise ValueError("Accepted site review must point to its created citation")
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
            for evidence in citation.reporter_opinion_evidence:
                root = by_id.get(evidence.root_id)
                if (
                    not isinstance(root, FullReporterCitation)
                    or root.reporter_root_opinion_page_index is None
                ):
                    raise ValueError("Opinion evidence requires its root's saved source text")
                opinion = next(
                    (
                        item
                        for item in root.reporter_root_opinion_page_index.opinions
                        if item.opinion_id == evidence.opinion_id
                    ),
                    None,
                )
                if opinion is None:
                    raise ValueError("Opinion evidence references an unknown opinion")
                evidence.validate_source(opinion.text)
                root_stages = {node.id: node.stage for node in root.nodes}
                if stage_positions.get(
                    root_stages[root.reporter_root_opinion_page_index.node_id], len(self.stage_runs)
                ) > stage_positions.get(node_stages[evidence.node_id], len(self.stage_runs)):
                    raise ValueError("Opinion evidence cannot reference a future source index")
            for review in citation.reporter_support_reviews:
                accepted = citation.reporter_pinpoint_evidence[review.evidence_index]
                for offset, index in enumerate(review.opinion_evidence_indices):
                    evidence = citation.reporter_opinion_evidence[index]
                    if (
                        evidence.root_id != accepted.root_id
                        or evidence.opinion_id != review.decision.evidence[offset].opinion_id
                    ):
                        raise ValueError(
                            "Support evidence must belong to its source root and declared opinion"
                        )
            for resolution in citation.reporter_page_resolutions:
                resolution_stage = stage_positions.get(node_stages[resolution.node_id], len(self.stage_runs))
                root = by_id.get(resolution.root_id)
                if (
                    not isinstance(root, FullReporterCitation)
                    or root.reporter_root_opinion_page_index is None
                ):
                    raise ValueError("Page resolution requires its reporter root's saved page index")
                root_stages = {node.id: node.stage for node in root.nodes}
                page_index = root.reporter_root_opinion_page_index
                if (
                    stage_positions.get(root_stages[page_index.node_id], len(self.stage_runs))
                    > resolution_stage
                ):
                    raise ValueError("Page resolution cannot reference a later page index")
                assigned_root = next(
                    (
                        update.value
                        for update in reversed(citation.root_id)
                        if stage_positions.get(node_stages[update.node_id], len(self.stage_runs))
                        <= resolution_stage
                    ),
                    None,
                )
                if assigned_root != root.id:
                    raise ValueError("Page resolution must belong to the citation's attached root")
                for identifier, reading_index, field in (
                    (resolution.locator_citation_id, resolution.locator_reading_index, "locator"),
                    (resolution.pin_citation_id, resolution.pin_reading_index, "pin_cite"),
                ):
                    if identifier is None:
                        continue
                    source = by_id.get(identifier)
                    if source is None:
                        raise ValueError("Page resolution references an unknown source citation")
                    source_stages = {node.id: node.stage for node in source.nodes}
                    source_root = next(
                        (
                            update.value
                            for update in reversed(source.root_id)
                            if stage_positions.get(source_stages[update.node_id], len(self.stage_runs))
                            <= resolution_stage
                        ),
                        None,
                    )
                    if source_root != root.id:
                        raise ValueError("Page resolution source belongs to another root")
                    if field == "locator" and isinstance(source, ShortReporterCitation):
                        field = "short_locator"
                    readings = getattr(source, field, None)
                    if readings is None or reading_index >= len(readings):
                        raise ValueError("Page resolution references an unavailable field reading")
                    if (
                        stage_positions.get(
                            source_stages[readings[reading_index].node_id], len(self.stage_runs)
                        )
                        > resolution_stage
                    ):
                        raise ValueError("Page resolution cannot reference a later field reading")
                opinions = {opinion.opinion_id: opinion for opinion in page_index.opinions}
                for requested_page in resolution.pages:
                    for reference in requested_page.candidates:
                        opinion = opinions.get(reference.opinion_id)
                        if opinion is None or reference.page_index >= len(opinion.pages):
                            raise ValueError("Page resolution references an unavailable opinion page")
                        page = opinion.pages[reference.page_index]
                        if reference.pagination_confirmed and (
                            page.kind != requested_page.kind or page.volume is None or page.edition is None
                        ):
                            raise ValueError("Confirmed pagination requires a known reporter namespace")
            if isinstance(citation, LeafCitation):
                for record in (*citation.attributions, *citation.reviews):
                    record_stage = stage_positions.get(node_stages[record.node_id], len(self.stage_runs))
                    for root_id in record.candidate_root_ids:
                        candidate = by_id.get(root_id)
                        if not isinstance(candidate, FullCitation):
                            raise ValueError("Leaf candidate must refer to an existing full citation")
                        creation_stage = stage_positions.get(candidate.nodes[0].stage, len(self.stage_runs))
                        if creation_stage > record_stage:
                            raise ValueError("Leaf candidate cannot be created after its assessment")
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
            for citations in (self.full_locators, self.short_reporters):
                group_sizes: dict[str, int] = {}
                for citation in citations:
                    if stage_positions.get(citation.nodes[0].stage, len(self.stage_runs)) > cutoff:
                        continue
                    node_positions = {
                        node.id: stage_positions.get(node.stage, len(self.stage_runs))
                        for node in citation.nodes
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
