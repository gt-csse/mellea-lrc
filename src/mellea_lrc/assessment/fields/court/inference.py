"""Reporter→court inference during court assessment.

Applied in ``assess_court`` when the initial comparison is ``missing`` and the
reporter unambiguously identifies SCOTUS. Extraction keeps raw eyecite output;
validation compares that raw value against CourtListener without inference.

See ``docs/Validation Model Development.md`` — Court field assessment.
"""

from __future__ import annotations

# Reporters for which the publishing court is unambiguous.
_REPORTER_TO_COURT: dict[str, str] = {
    "U.S.": "scotus",
    "S. Ct.": "scotus",
    "L. Ed.": "scotus",
    "L. Ed. 2d": "scotus",
}


def infer_court_from_reporter(reporter: str | None) -> str | None:
    """Return the canonical CourtListener court slug for a SCOTUS-only reporter."""
    if not reporter:
        return None
    return _REPORTER_TO_COURT.get(reporter)
