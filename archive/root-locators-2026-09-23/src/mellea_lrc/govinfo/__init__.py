"""GovInfo retrieval support for Federal court-opinion packages."""

from mellea_lrc.govinfo.client import (
    GovInfoClient,
    GovInfoConfig,
    GovInfoError,
    GovInfoSearchResult,
    govinfo_package_candidate,
    govinfo_package_url,
    govinfo_uscourts_body_query,
    govinfo_uscourts_case_name_query,
    govinfo_uscourts_docket_query,
)

__all__ = [
    "GovInfoClient",
    "GovInfoConfig",
    "GovInfoError",
    "GovInfoSearchResult",
    "govinfo_package_candidate",
    "govinfo_package_url",
    "govinfo_uscourts_body_query",
    "govinfo_uscourts_case_name_query",
    "govinfo_uscourts_docket_query",
]
