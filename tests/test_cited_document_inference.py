"""Tests for cited-document identity inference."""

from __future__ import annotations

import asyncio

from mellea.core.base import ModelOutputThunk
from mellea.stdlib.context import ChatContext

from mellea_lrc.cited_document_inference import (
    CitedDocumentIdentity,
    CitedDocumentInferenceStatus,
    CitedDocumentReference,
    _proposal_from_output,
    _validate_evidence_availability,
    _validate_evidence_grounding,
    _validate_output_schema,
    infer_cited_document,
    render_cited_document_prompt,
)


def _context(output: str) -> ChatContext:
    return ChatContext().add(ModelOutputThunk(value=output))


def test_proposal_parses_unwrapped_json_object() -> None:
    proposal = _proposal_from_output(
        '{"classification":"same_document","document_case_name":"Smith v. Jones",'
        '"evidence":["SMITH v. JONES"],"reason":"The caption aligns."}'
    )

    assert proposal.classification == CitedDocumentIdentity.SAME_DOCUMENT.value
    assert proposal.document_case_name == "Smith v. Jones"


def test_schema_requirement_rejects_unknown_fields() -> None:
    validation = _validate_output_schema(
        _context(
            '{"classification":"same_document","document_case_name":"Smith v. Jones",'
            '"evidence":["SMITH v. JONES"],"reason":"aligned","confidence":0.9}'
        )
    )

    assert not validation


def test_identity_judgment_requires_grounded_document_evidence() -> None:
    document = "IN THE SUPREME COURT\nSMITH v. JONES\nDecided March 1, 2020."
    context = _context(
        '{"classification":"same_document","document_case_name":"Smith v. Jones",'
        '"evidence":["SMITH v. JONES","Decided March 1, 2020."],"reason":"aligned"}'
    )

    assert _validate_evidence_availability(context)
    assert _validate_evidence_grounding(context, document)


def test_identity_judgment_rejects_paraphrased_evidence() -> None:
    document = "IN THE SUPREME COURT\nSMITH v. JONES\nDecided March 1, 2020."
    context = _context(
        '{"classification":"same_document","document_case_name":"Smith v. Jones",'
        '"evidence":["Smith versus Jones"],"reason":"aligned"}'
    )

    assert not _validate_evidence_grounding(context, document)


def test_insufficient_evidence_may_have_no_excerpt() -> None:
    context = _context(
        '{"classification":"insufficient_evidence","document_case_name":null,'
        '"evidence":[],"reason":"No caption or decision metadata is present."}'
    )

    assert _validate_evidence_availability(context)


def test_prompt_separates_document_from_citation_cues_and_scope() -> None:
    prompt = render_cited_document_prompt(
        candidate_document="SMITH v. JONES\nDecided March 1, 2020.",
        reference=CitedDocumentReference(
            citation_text="Smith v. Jones, 1 U.S. 2 (2020)",
            case_name="Smith v. Jones",
            locator="1 U.S. 2",
            court="scotus",
            decision_date="2020-03-01",
        ),
    )

    assert "candidate_document" in prompt
    assert "SMITH v. JONES" in prompt
    assert "1 U.S. 2" in prompt
    assert "document identity analysis only" in prompt
    assert "supports a proposition" in prompt


def test_empty_candidate_fails_without_calling_llm() -> None:
    result = asyncio.run(
        infer_cited_document(
            object(),  # type: ignore[arg-type]
            candidate_document="  ",
            reference=CitedDocumentReference(citation_text="Smith v. Jones, 1 U.S. 2"),
        )
    )

    assert result.status is CitedDocumentInferenceStatus.FAILED
    assert result.identity is None
    assert result.error_message == "candidate_document must not be empty"
