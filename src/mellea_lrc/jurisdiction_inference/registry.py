"""Curated reporter publication-scope registry.

Two structures are exported:

``VALID_REPORTERS``
    (Reporter Guard)
    All reporter strings the project recognises. This is powered dynamically by
    the comprehensive CourtListener `cl_reporters.json` dataset, plus any explicitly
    known multi-court reporters. A reporter absent from this set is ``UNRECOGNIZED``
    and terminates inference.
"""

from __future__ import annotations

import json
from pathlib import Path


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

    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)

    editions = set()
    mlz_map: dict[str, list[str]] = {}
    for entries in data.values():
        for entry in entries:
            mlz = entry.get("mlz_jurisdiction", [])
            for edition in entry.get("editions", {}):
                editions.add(edition)
                mlz_map.setdefault(edition, []).extend(mlz)
    return frozenset(editions), mlz_map

_CL_REPORTERS, REPORTER_MLZ_JURISDICTIONS = _load_cl_reporters()

# Full set of recognised reporters (Reporter Guard)
VALID_REPORTERS: frozenset[str] = _VALID_WITHOUT_EXACT_COURT | _CL_REPORTERS

def _load_mlz_to_cl_map() -> dict[str, list[str]]:
    json_path = Path(__file__).parent / "mlz_to_cl_map.json"
    if not json_path.exists():
        return {}
    with json_path.open(encoding="utf-8") as file:
        return json.load(file)

MLZ_TO_CL_MAP: dict[str, list[str]] = _load_mlz_to_cl_map()
