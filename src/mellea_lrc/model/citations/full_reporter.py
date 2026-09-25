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
    ReporterExactDocket,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.span import Span


class FullReporterCitation(FullCitation):
    """A reporter occurrence whose one locator field owns all locator parts."""

    kind: Literal[FullCitationKind.REPORTER] = FullCitationKind.REPORTER
    locator: tuple[FullReporterLocator, ...]
    reporter_exact_lookup: ReporterExactLookup | None = None
    reporter_exact_docket: ReporterExactDocket | None = None

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

    def with_case_name_judgment(self, reading_index: int, candidate_index: int, result: MatchResult) -> Self:
        judgment = ReporterExactCaseNameJudgment(
            node_id=self._decision_node_id(),
            reading_index=reading_index,
            candidate_index=candidate_index,
            result=result,
        )
        return self._with_log(case_name_judgments=(*self.case_name_judgments, judgment))

    def with_court_judgment(self, reading_index: int, candidate_index: int, result: MatchResult) -> Self:
        judgment = ReporterExactCourtJudgment(
            node_id=self._decision_node_id(),
            reading_index=reading_index,
            candidate_index=candidate_index,
            result=result,
        )
        return self._with_log(court_judgments=(*self.court_judgments, judgment))

    def with_date_judgment(self, reading_index: int, candidate_index: int, result: MatchResult) -> Self:
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
        for log, readings in (
            (self.case_name_judgments, self.case_name),
            (self.court_judgments, self.court),
            (self.date_judgments, self.date),
        ):
            for judgment in log:
                if judgment.reading_index >= len(readings):
                    raise ValueError("Judgment points beyond its field-reading log")
                if (
                    lookup is None
                    or lookup.response is None
                    or judgment.candidate_index >= len(lookup.response.clusters)
                ):
                    raise ValueError("Judgment points beyond the reporter lookup candidates")
                if positions[readings[judgment.reading_index].node_id] > positions[judgment.node_id]:
                    raise ValueError("Judgment cannot assess a later field reading")
                if positions[lookup.node_id] > positions[judgment.node_id]:
                    raise ValueError("Judgment cannot precede its lookup response")
        return self
