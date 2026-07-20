"""Tests for locator-found pin-cite content assessment."""

# ruff: noqa: D103, FBT001, INP001

from __future__ import annotations

import asyncio

import pytest
from mellea.core.base import ModelOutputThunk
from mellea.stdlib.context import ChatContext

from mellea_lrc.core.citations import FullCaseCitation, Reporter
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.pin_cite_inference import (
    PinCiteInferenceStatus,
    _proposal_from_output,
    _validate_output_schema_and_order,
    _validate_pinpoint_grounding,
    assess_found_locator_pin_cite,
    render_pin_cite_prompt,
)


def _context(output: str) -> ChatContext:
    return ChatContext().add(ModelOutputThunk(value=output))


def _citation(
    citing_document: str,
    *,
    matched_citation_text: str = "Example v. Case, 2021 WL 10, at *3",
    matched_locator_text: str = "2021 WL 10",
    pin_cite: str | None = "*3",
) -> ExtractedCitation:
    start = citing_document.index(matched_citation_text)
    return ExtractedCitation(
        citation_id="citation-1",
        citation_span=Span(start, start + len(matched_citation_text)),
        matched_locator_text=matched_locator_text,
        matched_citation_text=matched_citation_text,
        citation=FullCaseCitation(
            plaintiff="Example",
            defendant="Case",
            volume="2021",
            page="10",
            pin_cite=pin_cite,
            reporter=Reporter(edition_short_name="WL"),
        ),
    )


@pytest.mark.parametrize("match", [True, False, None])
def test_pin_cite_valid_accepts_nullable_boolean(match: bool | None) -> None:
    marker = "null" if match is None else '"*3"'
    excerpt = "null" if match is None else '"The pinpointed passage."'
    proposal = _proposal_from_output(
        f'{{"reason":"The pinpoint was compared with the attributed content.",'
        f'"pinpoint_marker":{marker},"pinpoint_excerpt":{excerpt},'
        f'"pin_cite_match":{_json_value(match)}}}',
        require_reason_first=True,
    )

    assert proposal.pin_cite_match is match


def test_output_requires_reason_before_judgment() -> None:
    validation = _validate_output_schema_and_order(
        _context(
            '{"pinpoint_excerpt":"The passage.","pin_cite_match":true,'
            '"pinpoint_marker":"*3",'
            '"reason":"The exact passage supports the proposition."}'
        )
    )

    assert not validation


def test_output_rejects_extra_fields() -> None:
    validation = _validate_output_schema_and_order(
        _context(
            '{"reason":"The passage supports the claim.",'
            '"pinpoint_marker":"*3",'
            '"pinpoint_excerpt":"The pinpointed passage.",'
            '"pin_cite_match":true,"confidence":0.9}'
        )
    )

    assert not validation


def test_prompt_derives_pinpoint_and_attribution_from_extracted_citation() -> None:
    citing_document = (
        "The court may grant a protective order. See "
        "Example v. Case, 2021 WL 10, at *3. The next proposition is unrelated."
    )
    prompt = render_pin_cite_prompt(
        citing_document=citing_document,
        citation=_citation(citing_document),
        cited_document="*3\nThe court grants the motion for a protective order.",
    )

    assert "cited_pin_cite:\n*3" in prompt
    assert "full_citation_text:\nExample v. Case, 2021 WL 10, at *3" in prompt
    assert "The court may grant a protective order" in prompt
    assert "exact pinpoint" in prompt
    assert "Content found only on another page" in prompt
    assert prompt.index('"reason"') < prompt.index('"pin_cite_match"')


def test_non_null_judgment_requires_grounded_marker_and_excerpt() -> None:
    cited_document = "*3\nThe court grants the motion."

    grounded = _validate_pinpoint_grounding(
        _context(
            '{"reason":"The pinpoint grants the motion.",'
            '"pinpoint_marker":"*3",'
            '"pinpoint_excerpt":"The court grants the motion.",'
            '"pin_cite_match":true}'
        ),
        cited_document,
    )
    ungrounded = _validate_pinpoint_grounding(
        _context(
            '{"reason":"The pinpoint grants the motion.",'
            '"pinpoint_marker":"*4",'
            '"pinpoint_excerpt":"The court denies the motion.",'
            '"pin_cite_match":true}'
        ),
        cited_document,
    )

    assert grounded
    assert not ungrounded


def test_missing_pin_cite_is_not_applicable_without_calling_llm() -> None:
    citing_document = "See Example v. Case, 2021 WL 10, at *3."
    result = asyncio.run(
        assess_found_locator_pin_cite(
            object(),  # type: ignore[arg-type]
            citing_document=citing_document,
            citation=_citation(citing_document, pin_cite=None),
            cited_document="*3\nText",
        )
    )

    assert result.status is PinCiteInferenceStatus.NOT_APPLICABLE
    assert result.validation is None
    assert result.error_message == "citation does not contain a pin cite"


def test_mismatched_upstream_citation_span_fails_without_calling_llm() -> None:
    citing_document = "See Example v. Case, 2021 WL 10, at *3."
    citation = _citation(citing_document)
    result = asyncio.run(
        assess_found_locator_pin_cite(
            object(),  # type: ignore[arg-type]
            citing_document=citing_document.replace("Example", "Sample"),
            citation=citation,
            cited_document="*3\nText",
        )
    )

    assert result.status is PinCiteInferenceStatus.FAILED
    assert result.validation is None
    assert result.error_message == ("matched_citation_text does not match citation_span in citing_document")


def _json_value(value: bool | None) -> str:
    if value is None:
        return "null"
    return "true" if value else "false"
