"""LLM remote sanity tests for live case-name search preparation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from mellea.core import ValidationResult
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.requirements import check
from mellea.stdlib.sampling import MultiTurnStrategy

from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.core.env import load_env_file
from mellea_lrc.core.spans import Span
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.retrieval.case_name_prepare import (
    PREPARATION_MAX_TOKENS,
    _case_name_preparation_requirements,
    _prepare_case_name,
)

pytestmark = [pytest.mark.remote_smoke, pytest.mark.llm_remote_sanity]


@pytest.mark.parametrize(
    ("name", "text", "locator", "expected_plaintiff", "expected_defendant"),
    [
        (
            "simple",
            "The court discussed Smith v. Jones, 999 U.S. 999, before turning to damages.",
            "999 U.S. 999",
            "Smith",
            "Jones",
        ),
        (
            "newline",
            "The authority was Long Plaintiff\nv.\nComplex Defendant, 999 U.S. 999, and it controlled.",
            "999 U.S. 999",
            "Long Plaintiff",
            "Complex Defendant",
        ),
        (
            "multiple",
            "First see Alpha v. Beta, 111 F.3d 222. But the cited case is Gamma v. Delta, 999 U.S. 999.",
            "999 U.S. 999",
            "Gamma",
            "Delta",
        ),
    ],
)
def test_prepare_case_name_live_examples(
    name: str,
    text: str,
    locator: str,
    expected_plaintiff: str,
    expected_defendant: str,
) -> None:
    """Exercise the live instruct preparation prompt against representative windows."""
    proposal, _final_ctx = asyncio.run(
        _call_prepare_case_name(
            text=text,
            locator=locator,
            extracted_plaintiff="Alpha" if name == "multiple" else "",
            extracted_defendant="Beta" if name == "multiple" else "",
        )
    )

    assert proposal.classification == "complete_case_name"
    assert proposal.plaintiff == expected_plaintiff
    assert proposal.defendant == expected_defendant


def test_prepare_case_name_live_context_records_retry_turns() -> None:
    """Forced requirement failures should leave model outputs and retry feedback in context."""
    proposal, final_ctx = asyncio.run(
        _call_prepare_case_name(
            text="The court discussed Smith v. Jones, 999 U.S. 999, before turning to damages.",
            locator="999 U.S. 999",
            extra_requirements=[
                check(
                    "forced observability failure",
                    validation_fn=lambda _ctx: ValidationResult(
                        result=False,
                        reason="forced observability failure",
                    ),
                )
            ],
        )
    )

    outputs = [item for item in final_ctx.as_list() if isinstance(item, ModelOutputThunk)]
    assert proposal.plaintiff == "Smith"
    assert proposal.defendant == "Jones"
    assert len(outputs) >= 2
    assert all('"case_name"' not in str(output.value) for output in outputs)


async def _call_prepare_case_name(
    *,
    text: str,
    locator: str,
    extracted_plaintiff: str = "",
    extracted_defendant: str = "",
    extra_requirements: list[Any] | None = None,
) -> tuple[Any, Any]:
    _load_llm_env_or_skip()
    session = start_mellea_session_from_env()
    window = _window_for(text, locator)
    requirements = [*_case_name_preparation_requirements(window, locator), *(extra_requirements or [])]
    return await _prepare_case_name(
        session,
        local_context=window.text,
        locator=locator,
        extracted_plaintiff=extracted_plaintiff,
        extracted_defendant=extracted_defendant,
        requirements=requirements,
        strategy=MultiTurnStrategy(loop_budget=3),
        model_options=structured_model_options(max_tokens=PREPARATION_MAX_TOKENS),
    )


def _window_for(text: str, locator: str) -> DocumentTextWindow:
    start = text.index(locator)
    return DocumentTextWindow.around(
        text,
        Span(start, start + len(locator)),
        before_chars=320,
        after_chars=0,
    )


def _load_llm_env_or_skip() -> None:
    try:
        load_env_file(Path(".env"), override=False)
    except FileNotFoundError:
        pytest.skip("Create .env to run the case-name preparation LLM remote sanity test.")
    try:
        llm_api_config_from_env(os.environ)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run the case-name preparation LLM remote sanity test.")
