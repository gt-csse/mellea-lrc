"""Expand retrieved court codes for reporter-citation model review."""

from __future__ import annotations

from mellea_lrc.courtlistener import CourtListenerCluster, CourtListenerDocket
from mellea_lrc.model.citations.fields.court import Court, court_id_if_unique


def _identified_court(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    court_id = value.rstrip("/").rsplit("/", maxsplit=1)[-1] if "/courts/" in value else value
    try:
        court = Court.from_id(court_id)
    except ValueError:
        mapped_id = court_id_if_unique(value)
        if mapped_id is None:
            return None
        court = Court.from_id(mapped_id)
    return {"id": court.id, "full_name": court.name}


def _names(record: CourtListenerCluster | CourtListenerDocket | None) -> dict[str, dict[str, str]]:
    if record is None:
        return {}
    return {
        field: identified
        for field in ("court_id", "court")
        if (identified := _identified_court(getattr(record, field))) is not None
    }


def reporter_court_context(
    candidate: CourtListenerCluster, docket: CourtListenerDocket | None
) -> dict[str, dict[str, dict[str, str]]]:
    """Show known names without merging opinion and linked-docket evidence.

    Raw codes remain in the separately supplied CourtListener records. A missing
    expansion means the database has no unique name for that raw value.
    """
    return {"opinion_cluster": _names(candidate), "linked_docket": _names(docket)}
