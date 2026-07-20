"""LLM remote sanity tests for cited-document identity inference."""

# ruff: noqa: INP001

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from mellea_lrc.cited_document_inference import (
    CitedDocumentIdentity,
    CitedDocumentInferenceStatus,
    CitedDocumentReference,
    infer_cited_document,
)
from mellea_lrc.core.env import load_env_file
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

pytestmark = [pytest.mark.remote_smoke, pytest.mark.llm_remote_sanity]

CANDIDATE_DOCUMENT = """
UNITED STATES COURT OF APPEALS FOR THE TENTH CIRCUIT

WELL, Plaintiff-Appellant,
v.
PELHAM, Defendant-Appellee.

No. 20-1234
Filed March 19, 2021

OPINION
The district court's judgment is affirmed.
""".strip()


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (
            CitedDocumentReference(
                citation_text="Well v. Pelham, 2021 WL 999999 (10th Cir. Mar. 19, 2021)",
                case_name="Well v. Pelham",
                locator="2021 WL 999999",
                court="ca10",
                decision_date="2021-03-19",
                docket_number="20-1234",
            ),
            CitedDocumentIdentity.SAME_DOCUMENT,
        ),
        (
            CitedDocumentReference(
                citation_text="Alpha v. Beta, 2021 WL 999999 (9th Cir. Apr. 2, 2021)",
                case_name="Alpha v. Beta",
                locator="2021 WL 999999",
                court="ca9",
                decision_date="2021-04-02",
                docket_number="19-9876",
            ),
            CitedDocumentIdentity.DIFFERENT_DOCUMENT,
        ),
    ],
)
def test_cited_document_inference_live_identity_cases(
    reference: CitedDocumentReference,
    expected: CitedDocumentIdentity,
) -> None:
    """Exercise matching and conflicting identity judgments on one document."""
    _load_llm_env_or_skip()
    result = asyncio.run(
        infer_cited_document(
            start_mellea_session_from_env(),
            candidate_document=CANDIDATE_DOCUMENT,
            reference=reference,
        )
    )

    assert result.status is CitedDocumentInferenceStatus.COMPLETED, result.error_message
    assert result.identity is expected
    assert result.evidence
    assert all(excerpt in CANDIDATE_DOCUMENT for excerpt in result.evidence)


def _load_llm_env_or_skip() -> None:
    try:
        load_env_file(Path(".env"), override=False)
    except FileNotFoundError:
        pytest.skip("Create .env to run the cited-document inference LLM sanity test.")
    try:
        llm_api_config_from_env(os.environ)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run the cited-document inference LLM sanity test.")
