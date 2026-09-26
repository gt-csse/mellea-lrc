"""Full citations identified by a reporter or legal database locator."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields import FullReporterLocator
from mellea_lrc.model.citations.full import FullCitation
from mellea_lrc.model.citations.history import Node
from mellea_lrc.model.citations.judgments import (
    MatchResult,
    ReporterExactCaseNameJudgment,
    ReporterExactCourtJudgment,
    ReporterExactDateJudgment,
)
from mellea_lrc.model.citations.kind import FullCitationKind
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactAmbiguityOutcome,
    ReporterExactAmbiguityResolution,
    ReporterExactCandidateDocket,
    ReporterExactDocket,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterUniqueReview,
)
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence whose one locator field owns all locator parts."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator: tuple[FullReporterLocator, ...]
    reporter_exact_lookup: ReporterExactLookup | None = None
    reporter_exact_docket: ReporterExactDocket | None = None
    reporter_exact_candidate_dockets: tuple[ReporterExactCandidateDocket, ...] = ()
    reporter_exact_ambiguity_resolution: ReporterExactAmbiguityResolution | None = None
    reporter_unique_review: ReporterUniqueReview | None = None

    @property
    def locator_span(self) -> Span:
        return self.locator[-1].span

    def with_reporter_exact_lookup(self, result: ReporterExactLookup) -> Self:
        """Store the one exact-lookup response at the current decision node."""
        if self.reporter_exact_lookup is not None:
            raise ValueError("Reporter exact lookup is already recorded")
        if result.node_id != self._decision_node_id():
            raise ValueError("Reporter lookup must point to the current decision node")
        return self._with_log(reporter_exact_lookup=result)

    def with_reporter_exact_docket(self, result: ReporterExactDocket) -> Self:
        """Save the linked docket before using its court in a judgment."""
        if self.reporter_exact_docket is not None:
            raise ValueError("Reporter exact docket is already recorded")
        if result.node_id != self._decision_node_id():
            raise ValueError("Reporter docket must point to the current decision node")
        return self._with_log(reporter_exact_docket=result)

    def with_reporter_exact_candidate_docket(self, result: ReporterExactCandidateDocket) -> Self:
        """Save a candidate's linked docket before judging its court."""
        if result.node_id != self._decision_node_id():
            raise ValueError("Candidate docket must point to the current decision node")
        if any(
            item.candidate_index == result.candidate_index for item in self.reporter_exact_candidate_dockets
        ):
            raise ValueError("Candidate docket is already recorded")
        return self._with_log(
            reporter_exact_candidate_dockets=(*self.reporter_exact_candidate_dockets, result)
        )

    def with_reporter_exact_ambiguity_resolution(self, result: ReporterExactAmbiguityResolution) -> Self:
        """Append one explicit rule-only selection result."""
        if self.reporter_exact_ambiguity_resolution is not None:
            raise ValueError("Reporter exact ambiguity is already resolved")
        if result.node_id != self._decision_node_id():
            raise ValueError("Ambiguity resolution must point to the current decision node")
        return self._with_log(reporter_exact_ambiguity_resolution=result)

    def with_reporter_unique_review(self, result: ReporterUniqueReview) -> Self:
        """Save one combined model review and its repair trace."""
        if self.reporter_unique_review is not None:
            raise ValueError("Reporter unique lookup has already been reviewed")
        if result.node_id != self._decision_node_id():
            raise ValueError("Reporter review must point to the current decision node")
        return self._with_log(reporter_unique_review=result)

    def with_case_name_judgment(
        self, reading_index: int | None, candidate_index: int, result: MatchResult
    ) -> Self:
        if any(
            item.node_id == self._decision_node_id() and item.candidate_index == candidate_index
            for item in self.case_name_judgments
        ):
            raise ValueError("Case-name judgment is already recorded for this candidate at this node")
        judgment = ReporterExactCaseNameJudgment(
            node_id=self._decision_node_id(),
            reading_index=reading_index,
            candidate_index=candidate_index,
            result=result,
        )
        return self._with_log(case_name_judgments=(*self.case_name_judgments, judgment))

    def with_court_judgment(
        self, reading_index: int | None, candidate_index: int, result: MatchResult
    ) -> Self:
        if any(
            item.node_id == self._decision_node_id() and item.candidate_index == candidate_index
            for item in self.court_judgments
        ):
            raise ValueError("Court judgment is already recorded for this candidate at this node")
        judgment = ReporterExactCourtJudgment(
            node_id=self._decision_node_id(),
            reading_index=reading_index,
            candidate_index=candidate_index,
            result=result,
        )
        return self._with_log(court_judgments=(*self.court_judgments, judgment))

    def with_date_judgment(
        self, reading_index: int | None, candidate_index: int, result: MatchResult
    ) -> Self:
        if any(
            item.node_id == self._decision_node_id() and item.candidate_index == candidate_index
            for item in self.date_judgments
        ):
            raise ValueError("Date judgment is already recorded for this candidate at this node")
        judgment = ReporterExactDateJudgment(
            node_id=self._decision_node_id(),
            reading_index=reading_index,
            candidate_index=candidate_index,
            result=result,
        )
        return self._with_log(date_judgments=(*self.date_judgments, judgment))

    @classmethod
    def from_locator(
        cls,
        *,
        citation_id: str,
        stage: str,
        source: str,
        span: Span,
    ) -> Self:
        """Create one source-grounded locator and its normalized Reporter."""
        node = Node(id=f"{citation_id}:node:0", stage=stage)
        return cls(
            id=citation_id,
            nodes=(node,),
            locator=(FullReporterLocator.from_source(source, span, node_id=node.id),),
        )

    @model_validator(mode="after")
    def _validate_locator(self) -> Self:
        if not self.locator or self.locator[0].node_id != self.nodes[0].id:
            raise ValueError("Reporter citation needs a source-spanned locator")
        lookup = self.reporter_exact_lookup
        docket = self.reporter_exact_docket
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        if docket is not None:
            if (
                lookup is None
                or lookup.outcome is not ReporterExactLookupOutcome.UNIQUE
                or lookup.response is None
                or lookup.response.clusters[0].docket_id != docket.docket_id
            ):
                raise ValueError("Reporter docket must belong to the unique lookup cluster")
            if positions[lookup.node_id] > positions[docket.node_id]:
                raise ValueError("Reporter docket cannot precede its lookup response")
        for candidate_docket in self.reporter_exact_candidate_dockets:
            if (
                lookup is None
                or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
                or lookup.response is None
                or candidate_docket.candidate_index >= len(lookup.response.clusters)
                or lookup.response.clusters[candidate_docket.candidate_index].docket_id
                != candidate_docket.docket_id
            ):
                raise ValueError("Candidate docket must belong to the indexed ambiguous cluster")
            if positions[lookup.node_id] > positions[candidate_docket.node_id]:
                raise ValueError("Candidate docket cannot precede its lookup response")
        if len({item.candidate_index for item in self.reporter_exact_candidate_dockets}) != len(
            self.reporter_exact_candidate_dockets
        ):
            raise ValueError("Candidate dockets must have distinct indices")
        resolution = self.reporter_exact_ambiguity_resolution
        if resolution is not None:
            if lookup is None or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS:
                raise ValueError("Ambiguity resolution requires an ambiguous exact lookup")
            if any(index >= len(lookup.response.clusters) for index in resolution.passing_candidate_indices):
                raise ValueError("Ambiguity resolution points beyond the lookup candidates")
            if positions[lookup.node_id] > positions[resolution.node_id]:
                raise ValueError("Ambiguity resolution cannot precede its lookup response")
            if (resolution.outcome is ReporterExactAmbiguityOutcome.CANDIDATE_LIMIT_EXCEEDED) != (
                len(lookup.response.clusters) >= 20
            ):
                raise ValueError("Large candidate deferral must match the lookup candidate count")
        review = self.reporter_unique_review
        if review is not None:
            if lookup is None or lookup.outcome is not ReporterExactLookupOutcome.UNIQUE:
                raise ValueError("Unique reporter review requires one saved lookup candidate")
            if positions[lookup.node_id] >= positions[review.node_id]:
                raise ValueError("Unique reporter review must follow its lookup")
        for log, readings in (
            (self.case_name_judgments, self.case_name),
            (self.court_judgments, self.court),
            (self.date_judgments, self.date),
        ):
            for judgment in log:
                if judgment.reading_index is None and readings:
                    raise ValueError("Judgment must point to an available field reading")
                if judgment.reading_index is not None and judgment.reading_index >= len(readings):
                    raise ValueError("Judgment points beyond its field-reading log")
                if (
                    lookup is None
                    or lookup.response is None
                    or judgment.candidate_index >= len(lookup.response.clusters)
                ):
                    raise ValueError("Judgment points beyond the reporter lookup candidates")
                if (
                    judgment.reading_index is not None
                    and positions[readings[judgment.reading_index].node_id] > positions[judgment.node_id]
                ):
                    raise ValueError("Judgment cannot assess a later field reading")
                if positions[lookup.node_id] > positions[judgment.node_id]:
                    raise ValueError("Judgment cannot precede its lookup response")
            if len({(item.node_id, item.candidate_index) for item in log}) != len(log):
                raise ValueError("Reporter field judgments must have distinct candidates within a node")
        return self
