"""Curated reporter publication-scope registry.

Two structures are exported:

``EXHAUSTIVE_REPORTERS``
    Reporters that publish decisions from exactly one CourtListener court.
    Each entry is a ``ReporterScope`` carrying the singleton court slug and a
    provenance statement.  Every key in this dict is also in ``VALID_REPORTERS``.

``VALID_REPORTERS``
    All reporter strings the project recognises.  Includes every key from
    ``EXHAUSTIVE_REPORTERS`` plus reporters that are known but do not yet
    resolve to a single court (e.g. ``F.3d``, ``WL``).  A reporter absent
    from this set is ``UNRECOGNIZED`` and terminates inference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReporterScope:
    """Internal registry value for an exhaustive single-court reporter."""

    court_id: str
    statement: str


def _scope(court_id: str, statement: str) -> ReporterScope:
    return ReporterScope(court_id=court_id, statement=statement)


EXHAUSTIVE_REPORTERS: dict[str, ReporterScope] = {
    "U.S.": _scope("scotus", "Publishes Supreme Court decisions."),
    "S. Ct.": _scope("scotus", "Publishes Supreme Court decisions."),
    "L. Ed.": _scope("scotus", "Publishes Supreme Court decisions."),
    "L. Ed. 2d": _scope("scotus", "Publishes Supreme Court decisions."),
    "U.S. LEXIS": _scope("scotus", "Identifies Supreme Court decisions."),
    "T.C.": _scope("tax", "Publishes United States Tax Court decisions."),
    "B.T.A.": _scope("bta", "Publishes Board of Tax Appeals decisions."),
    "Fed. Cl.": _scope("uscfc", "Publishes Court of Federal Claims decisions."),
    "Cl. Ct.": _scope("uscfc", "Publishes United States Claims Court decisions."),
    "Ct. Int'l Trade": _scope("cit", "Publishes Court of International Trade decisions."),
    "Cust. Ct.": _scope("cusc", "Publishes United States Customs Court decisions."),
    "C.C.P.A.": _scope("ccpa", "Publishes Court of Customs and Patent Appeals decisions."),
    "Vet. App.": _scope("cavc", "Publishes Court of Appeals for Veterans Claims decisions."),
    "C.M.A.": _scope("cma", "Publishes Court of Military Appeals decisions."),
}

# Reporters recognised by the project but not resolving to a single court.
# Multi-court federal reporters (F.3d, F.4th, F. Supp. 2d, F. Supp. 3d, B.R.)
# are listed here pending jurisdiction-category wiring (point 4 hook).
_VALID_WITHOUT_EXACT_COURT: frozenset[str] = frozenset({
    "WL",
    "LEXIS",
    "F.3d",
    "F.4th",
    "F. Supp. 2d",
    "F. Supp. 3d",
    "B.R.",
    "M.S.P.R.",
})

# Full set of recognised reporters — guard used by inference before status is assigned.
VALID_REPORTERS: frozenset[str] = frozenset(EXHAUSTIVE_REPORTERS) | _VALID_WITHOUT_EXACT_COURT
