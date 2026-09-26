"""Fields and updates belonging specifically to full citations."""

from __future__ import annotations

from typing import Self

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.fields import CaseNameField, CourtField, DateField, PinCiteField
from mellea_lrc.model.citations.history import RelationshipUpdate
from mellea_lrc.model.citations.judgments import (
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

    @property
    def locator_span(self) -> Span:
        """The written identifier span; concrete full types supply it."""
        raise NotImplementedError

    @property
    def site_span(self) -> Span:
        return self.locator_span

    def with_case_name(self, source: str, span: Span) -> Self:
        """Quote a case name and parse its parties or subject."""
        return self._with_log(
            case_name=(
                *self.case_name,
                CaseNameField.from_source(source, span, node_id=self._decision_node_id()),
            ),
        )

    def with_court(self, source: str, span: Span) -> Self:
        """Quote and normalize an explicit court."""
        reading = CourtField.from_source(source, span, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_inferred_court(self, court_id: str) -> Self:
        """Record a court inferred from the reporter, without a court quote."""
        reading = CourtField.inferred(court_id, node_id=self._decision_node_id())
        return self._with_log(court=(*self.court, reading))

    def with_date(self, source: str, span: Span) -> Self:
        """Quote and normalize a written calendar date."""
        return self._with_log(
            date=(*self.date, DateField.from_source(source, span, node_id=self._decision_node_id())),
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
        self, verdict: IdentityVerdict, next_stage: str | None = None
    ) -> Self:
        """Append a verdict without changing any earlier decision."""
        judgment = IdentityJudgment(node_id=self._decision_node_id(), verdict=verdict, next_stage=next_stage)
        return self._with_log(identity_judgments=(*self.identity_judgments, judgment))
