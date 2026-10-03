"""Materialize provider failures as source-preserving docket lookup records."""

from __future__ import annotations

import json

from mellea_lrc.model.citations.docket_lookup import DocketLookupFailure
from mellea_lrc.providers.courtlistener import CourtListenerError
from mellea_lrc.providers.govinfo import GovInfoError


def docket_lookup_failure(error: CourtListenerError | GovInfoError) -> DocketLookupFailure:
    """Retain provider details while converting non-JSON values to strings."""
    detail = json.loads(json.dumps(error.upstream_detail, default=str))
    return DocketLookupFailure(
        failure_type=error.failure_type,
        message=str(error) or type(error).__name__,
        upstream_status_code=error.upstream_status_code,
        url=error.url,
        upstream_detail=detail,
    )
