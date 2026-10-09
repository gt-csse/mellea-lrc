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
from mellea_lrc.model.citations.tags import CitationTag, CitationTagKind
from mellea_lrc.model.colocation import Colocation
from mellea_lrc.model.execution import CheckpointRun, get_stage_definition, stage_for_substage
from mellea_lrc.model.preprocessed_document import PreprocessedDocument
from mellea_lrc.model.site_review import SiteReview
from mellea_lrc.model.span import is_within


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
    """Source text, citation histories, and ordered typed checkpoint events."""

    citations: tuple[CitationVariant, ...] = ()
    site_reviews: tuple[SiteReview, ...] = ()
    runs: tuple[CheckpointRun, ...] = ()

    @property
    def substage_runs(self) -> tuple[str, ...]:
        """Atomic commits in event order; persisted only in ``runs``."""
        return tuple(run.name for run in self.runs if run.kind == "substage")

    @property
    def stage_runs(self) -> tuple[str, ...]:
        """Completed group boundaries in event order."""
        return tuple(run.name for run in self.runs if run.kind == "stage")

    def _ensure_substage_open(self, substage: str) -> None:
        if substage in self.substage_runs:
            raise ValueError(f"Substage already completed: {substage}")
        if stage_for_substage(substage) in self.stage_runs:
            raise ValueError(f"Cannot change a completed stage: {stage_for_substage(substage)}")

    def _pending_substages(self) -> set[str]:
        completed = set(self.substage_runs)
        return {
            node.substage
            for citation in self.citations
            for node in citation.nodes
            if node.substage not in completed
        } | {review.substage for review in self.site_reviews if review.substage not in completed}

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
        self._ensure_substage_open(citation.nodes[0].substage)
        creation_span = citation._through_node_count(1).site_span
        tags = tuple(
            CitationTag(
                node_id=citation.nodes[0].id,
                kind=CitationTagKind.TABLE_OF_AUTHORITIES,
                component_index=index,
            )
            for index, component in enumerate(self.index_spans)
            if is_within(creation_span, (component,))
        )
        if tags:
            if citation.tags and citation.tags != tags:
                raise ValueError("Citation component tags differ from its source components")
            if not citation.tags:
                citation = type(citation).model_validate({**citation.model_dump(mode="python"), "tags": tags})
        return self._with_citation(citation)

    def add_site_review(self, review: SiteReview) -> Self:
        """Append a review even when no citation was created at that site."""
        self._ensure_substage_open(review.substage)
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
        for node in appended_nodes:
            self._ensure_substage_open(node.substage)
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

    def complete_substage(self, substage: str) -> Self:
        """Commit one atomic run, including runs with no citation changes."""
        self._ensure_substage_open(substage)
        if self._pending_substages() - {substage}:
            raise ValueError("Complete the pending substage before starting another substage run")
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "runs": (*self.runs, CheckpointRun(kind="substage", name=substage)),
            }
        )

    def complete_stage(self, stage: str) -> Self:
        """Commit a semantic group after its required atomic runs have completed."""
        get_stage_definition(stage)
        if stage in self.stage_runs:
            raise ValueError(f"Stage already completed: {stage}")
        if self._pending_substages():
            raise ValueError("Complete the pending substage before completing a stage")
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "runs": (*self.runs, CheckpointRun(kind="stage", name=stage)),
            }
        )

    def get_substage(self, substage: str) -> Self:
        """Recover exactly the document returned by a completed substage run."""
        return self._get_checkpoint("substage", substage)

    def get_stage(self, stage: str) -> Self:
        """Recover exactly the document at the semantic group's completion event."""
        get_stage_definition(stage)
        return self._get_checkpoint("stage", stage)

    def _get_checkpoint(self, kind: str, name: str) -> Self:
        try:
            cutoff = next(
                index + 1 for index, run in enumerate(self.runs) if run.kind == kind and run.name == name
            )
        except StopIteration as exc:
            raise KeyError(f"{kind.capitalize()} has not run: {name}") from exc
        runs = self.runs[:cutoff]
        included = {run.name for run in runs if run.kind == "substage"}
        citations: list[CitationVariant] = []
        for citation in self.citations:
            count = 0
            for node in citation.nodes:
                if node.substage not in included:
                    break
                count += 1
            if count:
                citations.append(citation._through_node_count(count))
        citations.sort(key=lambda item: (item.site_span.start, item.id))
        return type(self).model_validate(
            {
                **self.model_dump(mode="python"),
                "citations": tuple(citations),
                "site_reviews": tuple(review for review in self.site_reviews if review.substage in included),
                "runs": runs,
            }
        )

    @model_validator(mode="after")
    def _validate_runs(self) -> Self:
        """Validate durable group markers and prevent reopening a committed group."""
        substages: set[str] = set()
        stages: set[str] = set()
        latest_substage: str | None = None
        for run_index, run in enumerate(self.runs):
            if run.kind == "substage":
                if run.name in substages:
                    raise ValueError("Substage runs must be unique")
                if stage_for_substage(run.name) in stages:
                    raise ValueError("Cannot run a substage after its completed stage")
                substages.add(run.name)
                latest_substage = run.name
                continue
            if run.name in stages:
                raise ValueError("Stage runs must be unique")
            try:
                definition = get_stage_definition(run.name)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            missing = set(definition.required_substages) - substages
            if missing:
                raise ValueError(f"Stage has unfinished required substages: {', '.join(sorted(missing))}")
            if latest_substage is None or stage_for_substage(latest_substage) != run.name:
                raise ValueError("Stage must complete immediately after its own final substage")
            members = {member.name: index for index, member in enumerate(definition.substages)}
            positions = [
                members[previous.name]
                for previous in self.runs[:run_index]
                if previous.kind == "substage" and previous.name in members
            ]
            if positions != sorted(positions):
                raise ValueError("Stage substages must complete in catalog order")
            stages.add(run.name)
        if any(stage_for_substage(substage) in stages for substage in self._pending_substages()):
            raise ValueError("Completed stage cannot contain pending substage changes")
        return self

    @model_validator(mode="after")
    def _validate_leaf_correction_evidence(self) -> Self:
        """Recheck leaf-to-root validation pointers after native JSON loading."""
        by_id = {citation.id: citation for citation in self.citations}
        substage_positions = {substage: index for index, substage in enumerate(self.substage_runs)}
        for citation in self.citations:
            nodes = {node.id: node.substage for node in citation.nodes}
            for review in citation.leaf_field_correction_reviews:
                position = substage_positions.get(nodes[review.node_id], len(self.substage_runs))
                attached = next(
                    (
                        item.value
                        for item in reversed(citation.root_id)
                        if substage_positions.get(nodes[item.node_id], len(self.substage_runs)) <= position
                    ),
                    None,
                )
                root = by_id.get(review.root_id)
                if attached != review.root_id or not isinstance(root, FullCitation) or root.id == citation.id:
                    raise ValueError("Leaf correction must reference its attached full root")
                root_nodes = {node.id: node.substage for node in root.nodes}
                for reference in review.evidence_refs:
                    history = getattr(root, reference.history, None)
                    if isinstance(history, tuple):
                        if reference.record_index is None or reference.record_index >= len(history):
                            raise ValueError("Leaf correction references a missing root history entry")
                        record = history[reference.record_index]
                    else:
                        if reference.record_index is not None or history is None:
                            raise ValueError("Leaf correction references a missing root evidence record")
                        record = history
                    if record.node_id != reference.node_id:
                        raise ValueError("Leaf correction evidence must point to its root decision node")
                    if substage_positions.get(root_nodes[record.node_id], len(self.substage_runs)) > position:
                        raise ValueError("Leaf correction cannot reference future root validation")
                    selected = reference.selected_candidate_index
                    if selected is not None:
                        if reference.history == "reporter_exact_lookup":
                            candidates = record.response.clusters if record.response is not None else ()
                        elif reference.history in {"docket_lookup", "govinfo_docket_lookup"}:
                            candidates = record.candidates
                        else:
                            raise ValueError("Only a saved lookup can reference a selected candidate")
                        if selected >= len(candidates):
                            raise ValueError(
                                "Leaf correction evidence references a missing selected candidate"
                            )
                if any(window.span.end > len(self.text) for window in review.windows):
                    raise ValueError("Leaf correction source window is outside the filing")
        return self

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if len(set(self.substage_runs)) != len(self.substage_runs):
            raise ValueError("Substage runs must be unique")
        substage_positions = {substage: index for index, substage in enumerate(self.substage_runs)}
        pending_substages: set[str] = set()
        for review in self.site_reviews:
            if review.candidate_span.start == review.candidate_span.end or not review.candidate_text.strip():
                raise ValueError("Site review needs a nonempty source span")
            if self.text[review.candidate_span.start : review.candidate_span.end] != review.candidate_text:
                raise ValueError("Site review quote does not match its source span")
            if review.substage not in substage_positions:
                pending_substages.add(review.substage)
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
                or citation.nodes[0].substage != review.substage
                or citation.site_span != review.candidate_span
            ):
                raise ValueError("Accepted site review must point to its created citation")
        if tuple(sorted(self.citations, key=lambda item: (item.site_span.start, item.id))) != self.citations:
            raise ValueError("Citations must be ordered by site span and ID")
        for citation in self.citations:
            previous_stage = -1
            for node in citation.nodes:
                position = substage_positions.get(node.substage)
                if position is None:
                    pending_substages.add(node.substage)
                    previous_stage = len(self.substage_runs)
                elif position < previous_stage:
                    raise ValueError("Citation nodes cannot move backward through substage runs")
                else:
                    previous_stage = position
            citation.validate_source(self.text)
            node_substages = {node.id: node.substage for node in citation.nodes}
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
                root_substages = {node.id: node.substage for node in root.nodes}
                if substage_positions.get(
                    root_substages[root.reporter_root_opinion_page_index.node_id], len(self.substage_runs)
                ) > substage_positions.get(node_substages[evidence.node_id], len(self.substage_runs)):
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
                resolution_stage = substage_positions.get(
                    node_substages[resolution.node_id], len(self.substage_runs)
                )
                root = by_id.get(resolution.root_id)
                if (
                    not isinstance(root, FullReporterCitation)
                    or root.reporter_root_opinion_page_index is None
                ):
                    raise ValueError("Page resolution requires its reporter root's saved page index")
                root_substages = {node.id: node.substage for node in root.nodes}
                page_index = root.reporter_root_opinion_page_index
                if (
                    substage_positions.get(root_substages[page_index.node_id], len(self.substage_runs))
                    > resolution_stage
                ):
                    raise ValueError("Page resolution cannot reference a later page index")
                assigned_root = next(
                    (
                        update.value
                        for update in reversed(citation.root_id)
                        if substage_positions.get(node_substages[update.node_id], len(self.substage_runs))
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
                    source_substages = {node.id: node.substage for node in source.nodes}
                    source_root = next(
                        (
                            update.value
                            for update in reversed(source.root_id)
                            if substage_positions.get(
                                source_substages[update.node_id], len(self.substage_runs)
                            )
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
                        substage_positions.get(
                            source_substages[readings[reading_index].node_id], len(self.substage_runs)
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
                    record_position = substage_positions.get(
                        node_substages[record.node_id], len(self.substage_runs)
                    )
                    for root_id in record.candidate_root_ids:
                        candidate = by_id.get(root_id)
                        if not isinstance(candidate, FullCitation):
                            raise ValueError("Leaf candidate must refer to an existing full citation")
                        creation_stage = substage_positions.get(
                            candidate.nodes[0].substage, len(self.substage_runs)
                        )
                        if creation_stage > record_position:
                            raise ValueError("Leaf candidate cannot be created after its assessment")
            for update in citation.root_id:
                if update.value in {None, WITHDRAWN_ROOT_ID}:
                    continue
                target = by_id.get(update.value)
                if target is None:
                    raise ValueError("Citation points to an unknown root")
                if not isinstance(target, FullCitation):
                    raise ValueError("Citation root must be a full citation")
                target_substage = substage_positions.get(target.nodes[0].substage, len(self.substage_runs))
                update_position = substage_positions.get(
                    node_substages[update.node_id], len(self.substage_runs)
                )
                if target_substage > update_position:
                    raise ValueError("Citation root cannot be created after its assignment")
        if len(pending_substages) > 1:
            raise ValueError("Only one substage can have uncommitted citation nodes")
        # A later substage must not make an invalid earlier checkpoint look valid.
        for cutoff in range(len(self.substage_runs)):
            root_ids: dict[str, str | None] = {}
            for citation in self.citations:
                node_positions = {
                    node.id: substage_positions.get(node.substage, len(self.substage_runs))
                    for node in citation.nodes
                }
                root_ids[citation.id] = next(
                    (
                        update.value
                        for update in reversed(citation.root_id)
                        if node_positions[update.node_id] <= cutoff
                    ),
                    None,
                )
            for root_id in root_ids.values():
                if root_id not in {None, WITHDRAWN_ROOT_ID} and root_ids.get(root_id) != root_id:
                    raise ValueError(
                        "Citation attachment must point to a self-root at every completed substage"
                    )
            for citations in (self.full_locators, self.short_reporters):
                group_sizes: dict[str, int] = {}
                for citation in citations:
                    if substage_positions.get(citation.nodes[0].substage, len(self.substage_runs)) > cutoff:
                        continue
                    node_positions = {
                        node.id: substage_positions.get(node.substage, len(self.substage_runs))
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

    @model_validator(mode="after")
    def _validate_component_tags(self) -> Self:
        for citation in self.citations:
            if not citation.tags:
                continue
            if len({(tag.kind, tag.component_index) for tag in citation.tags}) != len(citation.tags):
                raise ValueError("Citation component tags must be unique")
            creation_span = citation._through_node_count(1).site_span
            for tag in citation.tags:
                if tag.node_id != citation.nodes[0].id:
                    raise ValueError("Citation component tags must belong to the creation node")
                if tag.component_index >= len(self.index_spans):
                    raise ValueError("Citation tag references an unknown TOA component")
                if not is_within(creation_span, (self.index_spans[tag.component_index],)):
                    raise ValueError("Citation tag must contain its occurrence inside the TOA component")
        return self
