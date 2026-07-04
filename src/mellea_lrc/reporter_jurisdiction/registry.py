"""Curated reporter publication-scope registry.

Only mappings already documented by the project, plus narrowly established
modern federal reporter families, belong here. Observed sample courts are not
enough to add an exhaustive mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

from mellea_lrc.reporter_jurisdiction.types import CourtClass, ReporterCoverage


@dataclass(frozen=True, slots=True)
class ReporterScope:
    """Internal registry value used to construct public inference results."""

    court_ids: tuple[str, ...] = ()
    court_classes: tuple[CourtClass, ...] = ()
    jurisdiction_ids: tuple[str, ...] = ()
    coverage: ReporterCoverage = ReporterCoverage.UNKNOWN
    statement: str = ""


def _exact(court_id: str, court_class: CourtClass, statement: str) -> ReporterScope:
    return ReporterScope(
        court_ids=(court_id,),
        court_classes=(court_class,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.EXHAUSTIVE,
        statement=statement,
    )


REPORTER_SCOPES: dict[str, ReporterScope] = {
    "U.S.": _exact("scotus", CourtClass.FEDERAL_SUPREME, "Publishes Supreme Court decisions."),
    "S. Ct.": _exact("scotus", CourtClass.FEDERAL_SUPREME, "Publishes Supreme Court decisions."),
    "L. Ed.": _exact("scotus", CourtClass.FEDERAL_SUPREME, "Publishes Supreme Court decisions."),
    "L. Ed. 2d": _exact("scotus", CourtClass.FEDERAL_SUPREME, "Publishes Supreme Court decisions."),
    "U.S. LEXIS": _exact("scotus", CourtClass.FEDERAL_SUPREME, "Identifies Supreme Court decisions."),
    "T.C.": _exact("tax", CourtClass.SPECIALIZED_FEDERAL, "Publishes United States Tax Court decisions."),
    "B.T.A.": _exact("bta", CourtClass.SPECIALIZED_FEDERAL, "Publishes Board of Tax Appeals decisions."),
    "Fed. Cl.": _exact("uscfc", CourtClass.SPECIALIZED_FEDERAL, "Publishes Court of Federal Claims decisions."),
    "Cl. Ct.": _exact("uscfc", CourtClass.SPECIALIZED_FEDERAL, "Publishes United States Claims Court decisions."),
    "Ct. Int'l Trade": _exact("cit", CourtClass.SPECIALIZED_FEDERAL, "Publishes Court of International Trade decisions."),
    "Cust. Ct.": _exact("cusc", CourtClass.SPECIALIZED_FEDERAL, "Publishes United States Customs Court decisions."),
    "C.C.P.A.": _exact("ccpa", CourtClass.SPECIALIZED_FEDERAL, "Publishes Court of Customs and Patent Appeals decisions."),
    "Vet. App.": _exact("cavc", CourtClass.SPECIALIZED_FEDERAL, "Publishes Court of Appeals for Veterans Claims decisions."),
    "M.S.P.R.": _exact("mspb", CourtClass.SPECIALIZED_FEDERAL, "Publishes Merit Systems Protection Board decisions."),
    "C.M.A.": _exact("cma", CourtClass.SPECIALIZED_FEDERAL, "Publishes Court of Military Appeals decisions."),
    "F.3d": ReporterScope(
        court_classes=(CourtClass.FEDERAL_APPELLATE,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.EXHAUSTIVE,
        statement="Publishes decisions from the federal courts of appeals, not one circuit.",
    ),
    "F.4th": ReporterScope(
        court_classes=(CourtClass.FEDERAL_APPELLATE,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.EXHAUSTIVE,
        statement="Publishes decisions from the federal courts of appeals, not one circuit.",
    ),
    "F. Supp. 2d": ReporterScope(
        court_classes=(CourtClass.FEDERAL_DISTRICT,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.EXHAUSTIVE,
        statement="Publishes decisions from federal district courts, not one district.",
    ),
    "F. Supp. 3d": ReporterScope(
        court_classes=(CourtClass.FEDERAL_DISTRICT,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.EXHAUSTIVE,
        statement="Publishes decisions from federal district courts, not one district.",
    ),
    "B.R.": ReporterScope(
        court_classes=(CourtClass.FEDERAL_BANKRUPTCY,),
        jurisdiction_ids=("us-federal",),
        coverage=ReporterCoverage.PARTIAL,
        statement="Provides a federal bankruptcy direction but covers multiple tribunals.",
    ),
}


RECOGNIZED_WITHOUT_CONSTRAINT = frozenset({"WL", "LEXIS"})

