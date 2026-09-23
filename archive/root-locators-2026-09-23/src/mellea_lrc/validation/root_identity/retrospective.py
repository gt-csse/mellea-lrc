"""Conservative as-of eligibility for reporter-root retrieval evidence."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.courtlistener import CourtListenerOpinionCluster


def parse_evidence_date(value: object) -> date | None:
    """Accept an ISO calendar day (or ISO timestamp); unknown dates stay unknown."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    day = value[:10]
    if len(value) > 10 and value[10] not in {"T", " ", "Z"}:
        return None
    try:
        return date.fromisoformat(day)
    except ValueError:
        return None


def eligible_on_or_before(value: object, retrospective_date: date | None) -> bool:
    """With a cutoff, do not use later or undated evidence."""
    if retrospective_date is None:
        return True
    evidence_date = parse_evidence_date(value)
    return evidence_date is not None and evidence_date <= retrospective_date


def eligible_cluster(
    cluster: CourtListenerOpinionCluster,
    retrospective_date: date | None,
) -> bool:
    return eligible_on_or_before(cluster.date_filed, retrospective_date)


def eligible_metadata_candidate(
    candidate: Mapping[str, object],
    *,
    source: str,
    retrospective_date: date | None,
) -> bool:
    if source == "courtlistener":
        # An explicit decision date is the strongest date on a docket search
        # result. Otherwise the docket filing date dates only the metadata.
        value = candidate.get("decisionDate") or candidate.get("dateFiled")
    elif source == "govinfo":
        # A USCOURTS package can contain several opinions. Its issue date
        # cannot establish when each opinion existed, and package metadata
        # may have been revised. Granule-dated retrieval is a separate route.
        return retrospective_date is None
    else:
        msg = f"Unknown reporter metadata source: {source}"
        raise ValueError(msg)
    return eligible_on_or_before(value, retrospective_date)


def consistent_cutoff(saved: str | None, requested: date | None) -> date | None:
    """Resume a dated checkpoint without silently weakening its cutoff."""
    if saved is None:
        return requested
    prior = parse_evidence_date(saved)
    if prior is None:
        msg = f"Saved retrospective date is malformed: {saved!r}"
        raise ValueError(msg)
    if requested is not None and requested != prior:
        msg = f"Retrospective date {requested} differs from saved cutoff {prior}."
        raise ValueError(msg)
    return prior


def require_replay_cutoff(
    saved: str | None,
    requested: date | None,
    *,
    stage: str,
) -> None:
    """A completed undated stage cannot be retroactively treated as dated."""
    if requested is not None and saved != requested.isoformat():
        msg = (
            f"Completed {stage} used retrospective date {saved!r}; "
            f"restart from the preceding checkpoint to use {requested}."
        )
        raise ValueError(msg)
