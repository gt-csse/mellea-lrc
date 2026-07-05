"""Curated reporter publication-scope registry.

Two structures are exported:

``EXHAUSTIVE_REPORTERS``
    (Court Guard / Translation Helper)
    Reporters that publish decisions from exactly one CourtListener court.
    Each entry is a ``ReporterScope`` carrying the singleton court slug and a
    provenance statement. We currently manually curate this dictionary to enforce
    a strict 1:1 translation from reporter to CourtListener court_id.

``VALID_REPORTERS``
    (Reporter Guard)
    All reporter strings the project recognises. This is powered dynamically by
    the comprehensive CourtListener `cl_reporters.json` dataset, plus any explicitly
    known multi-court reporters. A reporter absent from this set is ``UNRECOGNIZED``
    and terminates inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReporterScope:
    """Internal registry value for an exhaustive single-court reporter."""

    court_id: str
    statement: str


def _scope(court_id: str, statement: str) -> ReporterScope:
    return ReporterScope(court_id=court_id, statement=statement)


# EXHAUSTIVE_REPORTERS: dict[str, ReporterScope] = {
#     "U.S.": _scope("scotus", "Publishes Supreme Court decisions."),
#     "S. Ct.": _scope("scotus", "Publishes Supreme Court decisions."),
#     "L. Ed.": _scope("scotus", "Publishes Supreme Court decisions."),
#     "L. Ed. 2d": _scope("scotus", "Publishes Supreme Court decisions."),
#     "U.S. LEXIS": _scope("scotus", "Identifies Supreme Court decisions."),
#     "T.C.": _scope("tax", "Publishes United States Tax Court decisions."),
#     "B.T.A.": _scope("bta", "Publishes Board of Tax Appeals decisions."),
#     "Fed. Cl.": _scope("uscfc", "Publishes Court of Federal Claims decisions."),
#     "Cl. Ct.": _scope("uscfc", "Publishes United States Claims Court decisions."),
#     "Ct. Int'l Trade": _scope("cit", "Publishes Court of International Trade decisions."),
#     "Cust. Ct.": _scope("cusc", "Publishes United States Customs Court decisions."),
#     "C.C.P.A.": _scope("ccpa", "Publishes Court of Customs and Patent Appeals decisions."),
#     "Vet. App.": _scope("cavc", "Publishes Court of Appeals for Veterans Claims decisions."),
#     "C.M.A.": _scope("cma", "Publishes Court of Military Appeals decisions."),
# }

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

def _load_cl_reporters() -> tuple[frozenset[str], dict[str, list[str]]]:
    json_path = Path(__file__).parent / "cl_reporters.json"
    if not json_path.exists():
        return frozenset(), {}
    
    with open(json_path) as f:
        data = json.load(f)
        
    editions = set()
    mlz_map = {}
    for root_key, entries in data.items():
        for entry in entries:
            mlz = entry.get("mlz_jurisdiction", [])
            for edition in entry.get("editions", {}):
                editions.add(edition)
                if edition not in mlz_map:
                    mlz_map[edition] = []
                mlz_map[edition].extend(mlz)
    return frozenset(editions), mlz_map

_CL_REPORTERS, REPORTER_MLZ_JURISDICTIONS = _load_cl_reporters()

# Full set of recognised reporters (Reporter Guard)
VALID_REPORTERS: frozenset[str] = _VALID_WITHOUT_EXACT_COURT | _CL_REPORTERS

def _load_mlz_to_cl_map() -> dict[str, list[str]]:
    json_path = Path(__file__).parent / "mlz_to_cl_map.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        return json.load(f)

MLZ_TO_CL_MAP: dict[str, list[str]] = _load_mlz_to_cl_map()
