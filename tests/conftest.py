"""Keep unit-test extraction policy independent of the developer's local .env."""

from pathlib import Path

import pytest
from dotenv import find_dotenv

from mellea_lrc.config import field_extraction


@pytest.fixture(autouse=True)
def rule_field_policy(monkeypatch):
    """Rule tests are explicit; ML tests supply their own configuration or mock."""
    read_env = field_extraction.read_env
    local_env = Path(__file__).resolve().parents[1] / ".env"

    def settings():
        values = read_env()
        if Path(find_dotenv(usecwd=True)).resolve() != local_env:
            return values
        return {
            **values,
            "MELLEA_LRC_FIELD_EXTRACTION_ML_ENABLED": "false",
            "MELLEA_LRC_FIELD_EXTRACTION_CASE_NAME": "RULE",
            "MELLEA_LRC_FIELD_EXTRACTION_COURT": "RULE",
            "MELLEA_LRC_FIELD_EXTRACTION_DATE": "RULE",
            "MELLEA_LRC_FIELD_EXTRACTION_PIN_CITE": "RULE",
        }

    monkeypatch.setattr(field_extraction, "read_env", settings)
