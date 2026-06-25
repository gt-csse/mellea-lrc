"""CourtListener cluster field readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

Cluster = Mapping[str, object]


def first_cluster_case_name(clusters: Sequence[Cluster]) -> str | None:
    """Return the canonical case name from the first CourtListener cluster."""
    if not clusters:
        return None
    case_name = clusters[0].get("case_name") or clusters[0].get("caseName")
    return str(case_name) if isinstance(case_name, str) and case_name else None


def first_cluster_year(clusters: Sequence[Cluster]) -> str | None:
    """Return the filing year from the first CourtListener cluster."""
    if not clusters:
        return None
    date_filed = clusters[0].get("date_filed") or clusters[0].get("dateFiled")
    return str(date_filed)[:4] if isinstance(date_filed, str) and date_filed else None
