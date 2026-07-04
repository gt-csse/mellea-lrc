"""Reporter→court inference during court assessment.

Applied in ``assess_court`` when the initial comparison is ``missing`` and the
reporter publishes decisions from exactly one court. Extraction keeps raw
eyecite output; assessment records inference as a field-local follow-up.

"""

from __future__ import annotations

from mellea_lrc.reporter_jurisdiction import infer_reporter_jurisdiction


def infer_court_from_reporter(reporter: str | None) -> str | None:
    """Project broader reporter evidence to an exhaustive singleton court."""
    return infer_reporter_jurisdiction(reporter).exact_court_id
