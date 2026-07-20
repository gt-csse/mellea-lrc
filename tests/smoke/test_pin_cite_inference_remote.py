"""LLM remote sanity tests for locator-found pin-cite content assessment."""

# ruff: noqa: FBT001, INP001, T201

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.env import load_env_file
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.pin_cite_inference import (
    PinCiteInferenceStatus,
    assess_found_locator_pin_cite,
)

pytestmark = [pytest.mark.remote_smoke, pytest.mark.llm_remote_sanity]


@pytest.mark.parametrize(
    ("name", "citing_document", "matched_citation", "locator", "pin_cite", "cited_document", "expected"),
    [
        (
            "westlaw_star_page",
            "The cited court granted a protective order. See Example v. Case, 2021 WL 10, at *3.",
            "Example v. Case, 2021 WL 10, at *3",
            "2021 WL 10",
            "*3",
            "*2\nBackground facts.\n\n*3\nThe court grants the motion for protective order.\n\n*4\nConclusion.",
            True,
        ),
        (
            "reporter_page",
            "Social-media evidence may be admitted. See Example v. Case, 416 F. Supp. 3d 1131, 1142.",
            "Example v. Case, 416 F. Supp. 3d 1131, 1142",
            "416 F. Supp. 3d 1131",
            "1142",
            "[1141]\nBackground.\n\n[1142]\nThe court holds that the social-media evidence is admissible.\n\n[1143]\nConclusion.",
            True,
        ),
        (
            "pinpoint_unavailable",
            "The evidence is admissible. See Example v. Case, 2021 WL 10, at *3.",
            "Example v. Case, 2021 WL 10, at *3",
            "2021 WL 10",
            "*3",
            "The court holds that the evidence is admissible and grants the motion.",
            None,
        ),
        (
            "pinpoint_content_mismatch",
            "Social-media evidence is admissible. See Example v. Case, 2021 WL 10, at *3.",
            "Example v. Case, 2021 WL 10, at *3",
            "2021 WL 10",
            "*3",
            "*2\nBackground facts.\n\n*3\nThe court describes the filing deadline and says nothing about admissibility.\n\n*4\nConclusion.",
            False,
        ),
    ],
)
def test_pin_cite_inference_live_examples(
    name: str,
    citing_document: str,
    matched_citation: str,
    locator: str,
    pin_cite: str,
    cited_document: str,
    expected: bool | None,
) -> None:
    """Exercise content assessment using the locator-found upstream citation object."""
    _load_llm_env_or_skip()
    citation_start = citing_document.index(matched_citation)
    citation = ExtractedCitation(
        citation_id="citation-1",
        citation_span=Span(citation_start, citation_start + len(matched_citation)),
        matched_locator_text=locator,
        matched_citation_text=matched_citation,
        citation=FullCaseCitation(
            plaintiff="Example",
            defendant="Case",
            pin_cite=pin_cite,
            reporter=Reporter(edition_short_name="WL"),
        ),
    )
    result = asyncio.run(
        assess_found_locator_pin_cite(
            start_mellea_session_from_env(),
            citing_document=citing_document,
            citation=citation,
            cited_document=cited_document,
        )
    )
    print(f"[{name}] {result.validation or result.error_message}")

    assert result.status is PinCiteInferenceStatus.COMPLETED, (name, result.error_message)
    assert result.validation is not None
    assert result.validation.reason
    assert result.validation.pin_cite_match is expected, (name, result.validation)


def _load_llm_env_or_skip() -> None:
    try:
        load_env_file(Path(".env"), override=False)
    except FileNotFoundError:
        pytest.skip("Create .env to run the pin-cite inference LLM sanity test.")
    try:
        llm_api_config_from_env(os.environ)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run the pin-cite inference LLM sanity test.")
