"""Full citations identified by a case docket number."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.docket_lookup import DocketLookup, DocketLookupReview
from mellea_lrc.model.citations.docket_root_equivalence import DocketRootReview
from mellea_lrc.model.citations.fields import DocketEntryField, FullDocketLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.span import Span


class FullDocketCitation(FullCitation):
    """A docket occurrence with an optional adjacent entry reference."""

    kind: Literal[FullCitationKind.DOCKET] = FullCitationKind.DOCKET
    locator: tuple[FullDocketLocator, ...]
    docket_entry: tuple[DocketEntryField, ...] = ()
    docket_root_reviews: tuple[DocketRootReview, ...] = ()
    docket_lookup: DocketLookup | None = None
    docket_lookup_review: DocketLookupReview | None = None

    @property
    def locator_span(self) -> Span:
        return self.locator[-1].span

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
        number_span: Span,
    ) -> Self:
        """Create the citation with only its grounded docket locator."""
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullDocketLocator.from_source(source, span, number_span, node_id=node.id),),
        )

    def with_docket_entry(self, source: str, span: Span) -> Self:
        """Quote an adjacent entry under the current decision node."""
        return self._with_log(
            docket_entry=(
                *self.docket_entry,
                DocketEntryField.from_source(source, span, node_id=self._decision_node_id()),
            ),
        )

    def with_docket_root_review(self, review: DocketRootReview) -> Self:
        """Append one decision without changing any root assignment itself."""
        if review.node_id != self._decision_node_id():
            raise ValueError("Docket review must belong to the current decision node")
        return self._with_log(docket_root_reviews=(*self.docket_root_reviews, review))

    def with_docket_lookup(self, result: DocketLookup) -> Self:
        """Save one complete search trace at the current root decision node."""
        if self.docket_lookup is not None:
            raise ValueError("Docket lookup is already recorded")
        if result.node_id != self._decision_node_id():
            raise ValueError("Docket lookup must point to the current decision node")
        return self._with_log(docket_lookup=result)

    def with_docket_lookup_review(self, result: DocketLookupReview) -> Self:
        """Save one later model review of the lookup shortlist."""
        if self.docket_lookup_review is not None:
            raise ValueError("Docket lookup is already reviewed")
        if result.node_id != self._decision_node_id():
            raise ValueError("Docket review must point to the current decision node")
        return self._with_log(docket_lookup_review=result)

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if not self.locator or self.locator[0].node_id != self.nodes[0].id:
            raise ValueError("Docket citation needs a source-spanned locator")
        if self.case_name_judgments or self.court_judgments or self.date_judgments:
            raise ValueError("Reporter exact judgments cannot belong to a docket citation")
        if self.docket_lookup_review is not None:
            if self.docket_lookup is None:
                raise ValueError("Docket review requires a saved lookup")
            positions = {node.id: index for index, node in enumerate(self.nodes)}
            if positions.get(self.docket_lookup_review.node_id, -1) <= positions.get(
                self.docket_lookup.node_id, -1
            ):
                raise ValueError("Docket review must follow the lookup at a later node")
            decision = self.docket_lookup_review.decision
            if (
                decision is not None
                and decision.selected_candidate_index is not None
                and decision.selected_candidate_index not in self.docket_lookup.shortlisted_candidate_indices
            ):
                raise ValueError("Docket review must select a shortlisted candidate")
        return self
