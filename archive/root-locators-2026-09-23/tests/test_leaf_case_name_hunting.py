"""Unread-name hunting is a separate, iterative leaf stage."""

from __future__ import annotations

import asyncio

import pytest

import mellea_lrc.extraction.adjudication.leaf_case_name_hunting as stage
from mellea_lrc.api import Document, grow_leaves, grow_roots
from mellea_lrc.extraction.adjudication.review.case_name import AdjudicatedCaseName, Reading
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind, SiteReview
from mellea_lrc.llm import IvrAttempt, IvrRun
from mellea_lrc.model.citations import CitationKind
from mellea_lrc.model.spans import Span
from mellea_lrc.serialization.ivr import deserialize_site_review_trace


def _run(*, success: bool = True) -> IvrRun:
    return IvrRun(
        success=success,
        selected_attempt=0,
        attempts=(IvrAttempt(output='{"reading":"short_form"}', requirements=()),),
        backend="test",
        model="test",
        model_options={},
        instruction="test",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )


def test_hunting_requires_leaf_growth() -> None:
    with pytest.raises(ValueError, match="requires leaf_growth"):
        asyncio.run(stage.hunt_leaf_case_names(Document.from_plain_text("No citations.")))


def test_bare_name_admission_updates_mask_and_is_checkpointable(monkeypatch) -> None:
    text = "Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007). Twombly explains the rule."
    document = asyncio.run(grow_leaves(asyncio.run(grow_roots(Document.from_plain_text(text)))))
    root = next(record for record in document.citations if record.is_root)
    begin = text.rindex("Twombly")
    span = Span(begin, begin + len("Twombly"))
    site = Candidate(
        generator="test",
        kind=CandidateKind.CASE_NAME,
        span=span,
        window=Span(max(0, begin - 40), len(text)),
        note="Unread bare case name.",
    )
    proposed: list[int] = []

    def sites(current):
        proposed.append(len(current.citations))
        if not any(record.kind is CitationKind.REFERENCE for record in current.citations):
            yield site

    async def review(current, candidate, *, session=None):
        del current, candidate, session
        return SiteReview(
            answer=AdjudicatedCaseName(
                span=span,
                name="Twombly",
                reading=Reading.SHORT_FORM,
                root_id=root.citation_id,
                defendant="Twombly",
                reason="This refers to the cited case.",
            ),
            reason="This refers to the cited case.",
            run=_run(),
        )

    monkeypatch.setattr(stage, "case_name_sites", sites)
    monkeypatch.setattr(stage, "adjudicate_case_name", review)
    hunted = asyncio.run(stage.hunt_leaf_case_names(document))

    assert proposed == [len(document.citations), len(document.citations) + 1]
    assert len(document.citations) + 1 == len(hunted.citations)
    leaf = next(record for record in hunted.citations if record.kind is CitationKind.REFERENCE)
    assert leaf.root_id == root.citation_id
    assert leaf.case_name is not None and leaf.case_name.text == "Twombly"
    assert deserialize_site_review_trace(leaf.trace[0].details).ivr.success
    assert hunted.passes[-1] == stage.STAGE
    assert Document.model_validate(hunted.model_dump(mode="json")) == hunted
    assert asyncio.run(stage.hunt_leaf_case_names(hunted)) is hunted


def test_exhausted_review_retains_evidence_without_admitting_a_leaf(monkeypatch) -> None:
    text = "Bell Atlantic Corp. v. Twombly, 550 U.S. 544 (2007). Twombly explains the rule."
    document = asyncio.run(grow_leaves(asyncio.run(grow_roots(Document.from_plain_text(text)))))
    root = next(record for record in document.citations if record.is_root)
    begin = text.rindex("Twombly")
    span = Span(begin, begin + len("Twombly"))
    site = Candidate(generator="test", kind=CandidateKind.CASE_NAME, span=span, window=span)

    monkeypatch.setattr(stage, "case_name_sites", lambda _document: iter((site,)))

    async def failed(_document, _site, *, session=None):
        del session
        return SiteReview(
            answer=AdjudicatedCaseName(
                span=span,
                name="Twombly",
                reading=Reading.SHORT_FORM,
                root_id=root.citation_id,
                defendant="Twombly",
            ),
            reason="The repair budget was exhausted.",
            run=_run(success=False),
        )

    monkeypatch.setattr(stage, "adjudicate_case_name", failed)
    hunted = asyncio.run(stage.hunt_leaf_case_names(document))

    assert len(hunted.citations) == len(document.citations)
    assert hunted.findings[-1].node_id == hunted.nodes[-1].node_id
    assert hunted.nodes[-1].outcome == "declined"
    assert not deserialize_site_review_trace(hunted.nodes[-1].details).ivr.success
