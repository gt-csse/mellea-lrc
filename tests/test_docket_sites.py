"""The unfinished docket hunter fails explicitly instead of using stale rules."""

from __future__ import annotations

from typing import cast

import pytest

from mellea_lrc.extraction import Document
from mellea_lrc.extraction.adjudication.candidates.docket_sites import suspected_dockets


def test_docket_site_hunting_is_explicitly_unavailable() -> None:
    with pytest.raises(NotImplementedError, match="Docket site hunting is not implemented"):
        suspected_dockets(cast(Document, object()))
