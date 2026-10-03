"""Citation-local history shared by full citations and short forms."""

from __future__ import annotations

import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.fields.case_name import CaseName, CaseNameField, CaseNameKind
from mellea_lrc.model.citations.fields.pin_cite import PinCiteField, PinCiteValue
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID, Node, RelationshipUpdate
from mellea_lrc.model.citations.reporter_page_resolution import (
    OpinionPageReference,
    ReporterCitationOpinionReview,
    ReporterCitationPageResolution,
)
from mellea_lrc.model.citations.reporter_pinpoint import (
    ReporterCitationPinpointEvidence,
    ReporterCitationProposition,
    ReporterCitationSupportReview,
    ReporterOpinionEvidence,
    ReporterPinpointJudgment,
)
from mellea_lrc.model.span import Span


class Citation(BaseModel):
    """An immutable citation whose field readings point to decision nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: str
    nodes: tuple[Node, ...]
    case_name: tuple[CaseNameField, ...] = ()
    root_id: tuple[RelationshipUpdate[str | None], ...] = ()
    routes: tuple[RelationshipUpdate[str | None], ...] = ()
    # A pinpoint belongs to this occurrence, regardless of its citation kind.
    # None is the only representation of no reading; a populated history is
    # nonempty and preserves every source-grounded reading in node order.
    pin_cite: Annotated[tuple[PinCiteField, ...], Field(min_length=1)] | None = None
    reporter_page_resolutions: tuple[ReporterCitationPageResolution, ...] = ()
    reporter_opinion_reviews: tuple[ReporterCitationOpinionReview, ...] = ()
    reporter_propositions: tuple[ReporterCitationProposition, ...] = ()
    reporter_pinpoint_evidence: tuple[ReporterCitationPinpointEvidence, ...] = ()
    reporter_opinion_evidence: tuple[ReporterOpinionEvidence, ...] = ()
    reporter_support_reviews: tuple[ReporterCitationSupportReview, ...] = ()
    reporter_pinpoint_judgments: tuple[ReporterPinpointJudgment, ...] = ()

    def get_case_name(self) -> CaseName:
        """Read the current written name, including the explicit unstated outcome."""
        if not self.case_name:
            return CaseName(kind=CaseNameKind.NOT_STATED)
        return self.case_name[-1].get_normalized()

    def with_case_name(self, source: str, span: Span, *, normalized: CaseName | None = None) -> Self:
        """Append a grounded name, using a model reading when supplied."""
        reading = (
            CaseNameField.from_source(source, span, node_id=self._decision_node_id())
            if normalized is None
            else CaseNameField.from_model(source, span, normalized, node_id=self._decision_node_id())
        )
        return self._with_log(case_name=(*self.case_name, reading))

    def with_reporter_proposition(self, result: ReporterCitationProposition) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Proposition must point to the current decision node")
        return self._with_log(reporter_propositions=(*self.reporter_propositions, result))

    def with_reporter_pinpoint_evidence(self, result: ReporterCitationPinpointEvidence) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Pinpoint evidence must point to the current decision node")
        return self._with_log(reporter_pinpoint_evidence=(*self.reporter_pinpoint_evidence, result))

    def with_reporter_opinion_evidence(self, result: ReporterOpinionEvidence) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Opinion evidence must point to the current decision node")
        return self._with_log(reporter_opinion_evidence=(*self.reporter_opinion_evidence, result))

    def with_reporter_support_review(self, result: ReporterCitationSupportReview) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Support review must point to the current decision node")
        return self._with_log(reporter_support_reviews=(*self.reporter_support_reviews, result))

    def with_reporter_pinpoint_judgment(self, result: ReporterPinpointJudgment) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Pinpoint judgment must point to the current decision node")
        return self._with_log(reporter_pinpoint_judgments=(*self.reporter_pinpoint_judgments, result))

    def get_reporter_page_selection(self) -> tuple[OpinionPageReference | None, ...]:
        """Read the latest selection without hiding unresolved requested pages."""
        if not self.reporter_page_resolutions:
            raise ValueError("Reporter citation pages have not been resolved")
        resolution_index = len(self.reporter_page_resolutions) - 1
        resolution = self.reporter_page_resolutions[resolution_index]
        review = next(
            (
                item
                for item in reversed(self.reporter_opinion_reviews)
                if item.resolution_index == resolution_index
            ),
            None,
        )
        if review is not None and review.decision is not None:
            choices = {choice.page_index: choice.candidate_index for choice in review.decision.choices}
            return tuple(
                page.candidates[choices[index]] if choices[index] is not None else None
                for index, page in enumerate(resolution.pages)
            )
        return tuple(
            page.candidates[0]
            if len(page.candidates) == 1 and page.candidates[0].pagination_confirmed
            else None
            for page in resolution.pages
        )

    @property
    def next_stage(self) -> str | None:
        """The currently queued stage, independent of any identity opinion."""
        return self.routes[-1].value if self.routes else None

    @property
    def site_span(self) -> Span:
        """The source site used to order citation occurrences."""
        raise NotImplementedError

    def record(self, stage: str) -> Self:
        """Record one decision; named field methods may share its node."""
        node = Node(id=f"{self.id}:node:{len(self.nodes)}", stage=stage)
        return type(self).model_validate({**self.model_dump(mode="python"), "nodes": (*self.nodes, node)})

    def _decision_node_id(self) -> str:
        if len(self.nodes) == 1:
            raise ValueError("Record a decision node before changing citation fields")
        return self.nodes[-1].id

    def _with_log(self, **logs: object) -> Self:
        """Validate the immutable citation after a named field method changes it."""
        return type(self).model_validate({**self.model_dump(mode="python"), **logs})

    def _through_node_count(self, count: int) -> Self:
        """Recover the citation and field readings through one node boundary."""
        if not 1 <= count <= len(self.nodes):
            raise ValueError("Citation cutoff must retain its creation node")
        nodes = self.nodes[:count]
        node_ids = {node.id for node in nodes}
        data = {**self.model_dump(mode="python"), "nodes": nodes}
        for name in type(self).model_fields:
            if name not in {"id", "kind", "nodes"}:
                value = getattr(self, name)
                if isinstance(value, tuple):
                    retained = tuple(update for update in value if update.node_id in node_ids)
                    data[name] = retained if retained else type(self).model_fields[name].get_default()
                else:
                    data[name] = value if value is not None and value.node_id in node_ids else None
        return type(self).model_validate(data)

    def with_root(self, root_id: str | None) -> Self:
        """Append a root attachment without erasing earlier assignments."""
        return self._with_log(
            root_id=(*self.root_id, RelationshipUpdate(value=root_id, node_id=self._decision_node_id())),
        )

    def with_reporter_page_resolution(self, result: ReporterCitationPageResolution) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Page resolution must point to the current decision node")
        if any(item.node_id == result.node_id for item in self.reporter_page_resolutions):
            raise ValueError("Page resolution is already recorded at this node")
        return self._with_log(reporter_page_resolutions=(*self.reporter_page_resolutions, result))

    def with_reporter_opinion_review(self, result: ReporterCitationOpinionReview) -> Self:
        if result.node_id != self._decision_node_id():
            raise ValueError("Opinion review must point to the current decision node")
        if any(item.node_id == result.node_id for item in self.reporter_opinion_reviews):
            raise ValueError("Opinion review is already recorded at this node")
        return self._with_log(reporter_opinion_reviews=(*self.reporter_opinion_reviews, result))

    def with_pin_cite(self, source: str, span: Span) -> Self:
        """Append a grounded pinpoint using the shared typed normalization."""
        reading = PinCiteField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(pin_cite=(*(self.pin_cite or ()), reading))

    def get_pin_cite(self) -> PinCiteValue | None:
        """Read the latest targets, or None when no pinpoint has been read.

        A quoted reading that cannot normalize raises through its field getter,
        rather than being confused with absence.
        """
        return None if self.pin_cite is None else self.pin_cite[-1].get_normalized()

    def with_route(self, next_stage: str | None) -> Self:
        """Append a routing decision; None clears an earlier route."""
        if (
            next_stage is not None
            and re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?_)?[a-z][a-z0-9_]*", next_stage) is None
        ):
            raise ValueError("A route must name a lowercase stage ID")
        return self._with_log(
            routes=(*self.routes, RelationshipUpdate(value=next_stage, node_id=self._decision_node_id()))
        )

    def withdraw(self) -> Self:
        """Attach to the dummy head when no real root applies, retaining history."""
        return self.with_root(WITHDRAWN_ROOT_ID)

    def validate_source(self, source: str) -> None:
        """Check all stored quotes against the source, including after JSON loading."""
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            value = getattr(self, name)
            entries = value if isinstance(value, tuple) else (value,) if value is not None else ()
            for entry in entries:
                if isinstance(entry, CitationField):
                    entry.validate_source(source)
        for proposition in self.reporter_propositions:
            for passage in proposition.passages:
                passage.validate_source(source)

    @model_validator(mode="after")
    def _validate_pinpoint_references(self) -> Self:
        positions = {node.id: index for index, node in enumerate(self.nodes)}

        def preceding(log: tuple, index: int, record: object) -> object:
            if not 0 <= index < len(log):
                raise ValueError("Pinpoint record references a missing history entry")
            target = log[index]
            if positions[target.node_id] > positions[record.node_id]:
                raise ValueError("Pinpoint record cannot reference a future decision")
            return target

        for proposition in self.reporter_propositions:
            preceding(self.reporter_page_resolutions, proposition.resolution_index, proposition)
        for log in (
            self.reporter_propositions,
            self.reporter_pinpoint_evidence,
            self.reporter_support_reviews,
            self.reporter_pinpoint_judgments,
        ):
            if len({record.node_id for record in log}) != len(log):
                raise ValueError("A decision node can append only one reading or judgment to each history")
        for evidence in self.reporter_pinpoint_evidence:
            resolution = preceding(self.reporter_page_resolutions, evidence.resolution_index, evidence)
            if resolution.root_id != evidence.root_id:
                raise ValueError("Pinpoint evidence must reference its resolved root")
            if evidence.proposition_index is not None:
                proposition = preceding(self.reporter_propositions, evidence.proposition_index, evidence)
                if proposition.resolution_index != evidence.resolution_index:
                    raise ValueError("Pinpoint evidence must use its resolution's proposition")
            allowed = tuple(reference for page in resolution.pages for reference in page.candidates)
            if any(reference not in allowed for reference in evidence.pages):
                raise ValueError("Pinpoint evidence must use a resolved page candidate")
        for review in self.reporter_support_reviews:
            preceding(self.reporter_pinpoint_evidence, review.evidence_index, review)
            for index in review.opinion_evidence_indices:
                preceding(self.reporter_opinion_evidence, index, review)
        for judgment in self.reporter_pinpoint_judgments:
            preceding(self.reporter_pinpoint_evidence, judgment.evidence_index, judgment)
            if judgment.review_index is not None:
                review = preceding(self.reporter_support_reviews, judgment.review_index, judgment)
                if review.evidence_index != judgment.evidence_index:
                    raise ValueError("Pinpoint judgment and review must assess the same evidence")
        return self

    @model_validator(mode="after")
    def _validate_history(self) -> Self:
        if not self.nodes:
            raise ValueError("Citation must have a creation node")
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        if len(positions) != len(self.nodes):
            raise ValueError("Duplicate citation node")
        for name in type(self).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            previous = -1
            value = getattr(self, name)
            updates = value if isinstance(value, tuple) else (value,) if value is not None else ()
            for update in updates:
                position = positions.get(update.node_id)
                if position is None:
                    raise ValueError(f"{name} refers to a missing node")
                # Field readings and relationships are successive updates.
                # One decision may, however, assess several lookup candidates
                # and append several evidence records to the same log.
                if position < previous or (
                    position == previous and isinstance(update, (CitationField, RelationshipUpdate))
                ):
                    raise ValueError(f"{name} updates are out of order")
                previous = position
        if any(
            route.value is not None
            and re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?_)?[a-z][a-z0-9_]*", route.value) is None
            for route in self.routes
        ):
            raise ValueError("A route must name a lowercase stage ID")
        for review in self.reporter_opinion_reviews:
            if review.resolution_index >= len(self.reporter_page_resolutions):
                raise ValueError("Opinion review points to a missing page resolution")
            resolution = self.reporter_page_resolutions[review.resolution_index]
            if positions[resolution.node_id] >= positions[review.node_id]:
                raise ValueError("Opinion review must follow its page resolution")
            if review.decision is not None:
                choices = review.decision.choices
                if {choice.page_index for choice in choices} != set(range(len(resolution.pages))):
                    raise ValueError("Opinion review must account for every requested page")
                for choice in choices:
                    candidates = resolution.pages[choice.page_index].candidates
                    if choice.candidate_index is not None and choice.candidate_index >= len(candidates):
                        raise ValueError("Opinion review selected a missing page candidate")
        return self
