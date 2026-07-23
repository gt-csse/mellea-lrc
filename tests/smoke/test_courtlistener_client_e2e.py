"""Opt-in end-to-end CourtListener client check for a known reported case."""

from __future__ import annotations

from dotenv import load_dotenv
import pytest

from mellea_lrc.courtlistener import CourtListenerClient

pytestmark = pytest.mark.remote_smoke


def test_brown_v_board_citation_resolves_to_supreme_court_docket() -> None:
    """Exercise citation lookup followed by docket-court retrieval for Brown."""
    load_dotenv(".env")
    client = CourtListenerClient()

    lookup = client.lookup_citation("347", "U.S.", "483")

    assert lookup.status == 200
    assert len(lookup.records) == 1
    record = lookup.records[0]
    assert record.case_name == "Brown v. Board of Education"
    assert record.docket_id == "84657"

    docket = client.get_docket(record.docket_id)

    assert docket.docket_id == "84657"
    assert docket.case_name == "Brown v. Board of Education"
    assert docket.court_id == "scotus"
