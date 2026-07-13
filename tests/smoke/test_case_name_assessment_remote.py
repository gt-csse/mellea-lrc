"""LLM remote sanity tests for live case-name assessment calls."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from mellea_lrc.assessment.fields.case_name.classify import (
    CASE_NAME_VERDICT_MAX_TOKENS,
    semantic_match_case_name,
)
from mellea_lrc.assessment.fields.case_name.reextract_after_retrieval import ReextractionStatus, reextract_case_name_after_retrieval
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.core.env import load_env_file
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

pytestmark = [pytest.mark.remote_smoke, pytest.mark.llm_remote_sanity]


def test_semantic_match_case_name_live_example() -> None:
    _load_llm_env_or_skip()
    session = start_mellea_session_from_env()

    verdict = asyncio.run(
        semantic_match_case_name(
            session,
            local_context="The opinion cited Brown v. Bd. of Educ., 347 U.S. 483.",
            extracted_case_name="Brown v. Bd. of Educ.",
            retrieved_case_name="Brown v. Board of Education",
            model_options=structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
        )
    )

    assert verdict == "semantic_match"


def test_reextract_case_name_after_retrieval_live_example() -> None:
    _load_llm_env_or_skip()
    session = start_mellea_session_from_env()

    result = asyncio.run(
        reextract_case_name_after_retrieval(
            session,
            document_context="The court relied on Smith v. Jones, 999 U.S. 999, for the rule.",
            extracted_case_name=None,
            courtlistener_case_name="Smith v. Jones",
            citation_locator="999 U.S. 999",
        )
    )

    assert result.status == ReextractionStatus.ACCEPTED
    assert result.proposal is not None
    assert result.proposal.case_name == "Smith v. Jones"


def _load_llm_env_or_skip() -> None:
    try:
        load_env_file(Path(".env"), override=False)
    except FileNotFoundError:
        pytest.skip("Create .env to run the case-name assessment LLM remote sanity test.")
    try:
        llm_api_config_from_env(os.environ)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run the case-name assessment LLM remote sanity test.")
