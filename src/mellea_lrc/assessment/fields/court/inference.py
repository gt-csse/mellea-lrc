"""Reporter→court inference during court assessment.

Applied in ``assess_court`` when the initial comparison is ``missing`` and the
reporter is an exhaustive singleton. Returns the full inference object so the
follow-up trace is self-describing.

"""

from __future__ import annotations

from mellea_lrc.reporter_jurisdiction import infer_reporter_jurisdiction
from mellea_lrc.reporter_jurisdiction.types import ReporterJurisdictionInference


def get_reporter_inference(reporter: str | None) -> ReporterJurisdictionInference:
    """Return the full reporter jurisdiction inference for use in court assessment."""
    return infer_reporter_jurisdiction(reporter)
