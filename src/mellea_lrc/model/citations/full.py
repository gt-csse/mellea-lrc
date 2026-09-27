"""Fields and updates belonging specifically to full citations."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from mellea_lrc.model.citations.body_evidence import BodyCorroborationReview, BodySearch
from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields import CaseName, CaseNameField, CourtField, DateField, PinCiteField
from mellea_lrc.model.citations.history import RelationshipUpdate
from mellea_lrc.model.citations.judgments import (
    IdentityBasis,
    IdentityJudgment,
    IdentityVerdict,
    ReporterExactCaseNameJudgment,
    ReporterExactCourtJudgment,
    ReporterExactDateJudgment,
)
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullCitation(Citation):
    """A full reporter or docket citation with contextual field histories."""

    kind: FullCitationKind
    case_name: tuple[CaseNameField, ...] = ()
    court: tuple[CourtField, ...] = ()
    date: tuple[DateField, ...] = ()
    pin_cite: tuple[PinCiteField, ...] = ()
    colocation_id: tuple[RelationshipUpdate[str | None], ...] = ()
    case_name_judgments: tuple[ReporterExactCaseNameJudgment, ...] = ()
    court_judgments: tuple[ReporterExactCourtJudgment, ...] = ()
    date_judgments: tuple[ReporterExactDateJudgment, ...] = ()
    identity_judgments: tuple[IdentityJudgment, ...] = ()
    body_searches: tuple[BodySearch, ...] = ()
    body_reviews: tuple[BodyCorroborationReview, ...] = ()

    @property
    def locator_span(self) -> Span:
        """The written identifier span; concrete full types supply it."""
        raise NotImplementedError

    @property
    def site_span(self) -> Span:
        return self.locator_span

    def with_case_name(self, source: str, span: Span, *, normalized: CaseName | None = None) -> Self:
        """Append a grounded case name, using a model reading when supplied."""
        reading = (
            CaseNameField.from_source(source, span, node_id=self._decision_node_id())
            if normalized is None
            else CaseNameField.from_model(source, span, normalized, node_id=self._decision_node_id())
        )
        return self._with_log(
            case_name=(*self.case_name, reading),
        )

    def with_court(self, source: str, span: Span) -> Self:
        """Append a grounded court and normalize its source quote."""
        reading = CourtField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_inferred_court(self, court_id: str) -> Self:
        """Record a court inferred from the reporter, without a court quote."""
        reading = CourtField.inferred(court_id, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_date(self, source: str, span: Span) -> Self:
        """Append a grounded date and normalize its source quote."""
        reading = DateField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(
            date=(*self.date, reading),
        )

    def with_pin_cite(self, source: str, span: Span) -> Self:
        """Quote and normalize a pinpoint reference."""
        return self._with_log(
            pin_cite=(
                *self.pin_cite,
                PinCiteField.from_source(source, span, node_id=self._decision_node_id()),
            ),
        )

    def with_colocation(self, group_id: str) -> Self:
        """Append a parsing-group assignment."""
        return self._with_log(
            colocation_id=(
                *self.colocation_id,
                RelationshipUpdate(value=group_id, node_id=self._decision_node_id()),
            ),
        )

    def with_identity_judgment(
        self,
        verdict: IdentityVerdict,
        next_stage: str | None = None,
        *,
        basis: IdentityBasis | None = None,
    ) -> Self:
        """Append a verdict without changing any earlier decision."""
        judgment = IdentityJudgment(
            node_id=self._decision_node_id(), verdict=verdict, next_stage=next_stage, basis=basis
        )
        return self._with_log(identity_judgments=(*self.identity_judgments, judgment))

    def with_body_search(self, result: BodySearch) -> Self:
        """Append one provider's complete body-search checkpoint."""
        if result.node_id != self._decision_node_id():
            raise ValueError("Body search must point to the current decision node")
        if any(item.source is result.source for item in self.body_searches):
            raise ValueError("Body source has already been searched")
        if self.body_searches and result.retrospective_date != self.body_searches[0].retrospective_date:
            raise ValueError("Body searches for one citation must use the same retrospective date")
        return self._with_log(body_searches=(*self.body_searches, result))

    def with_body_review(self, result: BodyCorroborationReview) -> Self:
        """Append one cross-provider body-citation comparison."""
        if result.node_id != self._decision_node_id():
            raise ValueError("Body review must point to the current decision node")
        if self.body_reviews:
            raise ValueError("Body corroboration has already been reviewed")
        return self._with_log(body_reviews=(*self.body_reviews, result))

    @model_validator(mode="after")
    def _validate_body_history(self) -> Self:
        if len({search.source for search in self.body_searches}) != len(self.body_searches):
            raise ValueError("A body provider cannot be searched twice")
        if len({search.retrospective_date for search in self.body_searches}) > 1:
            raise ValueError("Body searches must share one retrospective date")
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        for review in self.body_reviews:
            decision = review.decision
            if decision is None or decision.source is None:
                continue
            search = next((item for item in self.body_searches if item.source is decision.source), None)
            if search is None or decision.evidence_index >= len(search.evidence):
                raise ValueError("Body review points to missing provider evidence")
            if positions[search.node_id] >= positions[review.node_id]:
                raise ValueError("Body review must follow its provider search")
            excerpt = search.evidence[decision.evidence_index].excerpt
            span = review.quote_span
            if (
                span is None
                or span.end > len(excerpt)
                or excerpt[span.start : span.end] != review.grounded_quote
            ):
                raise ValueError("Body review quote must match the saved evidence excerpt")
        return self
