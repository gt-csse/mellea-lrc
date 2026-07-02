"""Reporter→court inference during court assessment.

Applied in ``assess_court`` when the initial comparison is ``missing`` and the
reporter publishes decisions from exactly one court. Extraction keeps raw
eyecite output; assessment records inference as a field-local follow-up.

"""

from __future__ import annotations

# Reporters that publish decisions from exactly one court.
_REPORTER_TO_COURT: dict[str, str] = {
    "U.S.": "scotus",
    "S. Ct.": "scotus",
    "L. Ed.": "scotus",
    "L. Ed. 2d": "scotus",
    "U.S. LEXIS": "scotus",
    "T.C.": "tax",
    "B.T.A.": "bta",
    "Fed. Cl.": "uscfc",
    "Cl. Ct.": "uscfc",
    "Ct. Int'l Trade": "cit",
    "Cust. Ct.": "cusc",
    "C.C.P.A.": "ccpa",
    "Vet. App.": "cavc",
    "M.S.P.R.": "mspb",
    "C.M.A.": "cma",
}


def infer_court_from_reporter(reporter: str | None) -> str | None:
    """Return the CourtListener slug for a reporter exclusive to one court."""
    if not reporter:
        return None
    return _REPORTER_TO_COURT.get(reporter)
