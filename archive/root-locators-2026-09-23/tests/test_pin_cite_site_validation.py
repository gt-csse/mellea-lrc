"""Pin-cite site validation is an independent, resumable document stage."""

from __future__ import annotations

import asyncio

import pytest

import mellea_lrc.extraction.adjudication.pin_cite_site_validation as stage
from mellea_lrc.api import Document, grow_leaves, grow_roots
from mellea_lrc.extraction.adjudication.review.pin_cite import AdjudicatedPinCite, Reading
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind, SiteReview
from mellea_lrc.llm import IvrAttempt, IvrRun
from mellea_lrc.model.citations import CitationField
from mellea_lrc.model.pin_cites import PinCite
from mellea_lrc.model.spans import Span
from mellea_lrc.serialization.ivr import deserialize_site_review_trace


def _run(*, success: bool = True) -> IvrRun:
    return IvrRun(
        success=success,
        selected_attempt=0,
        attempts=(IvrAttempt(output='{"reading":"no_page_claim"}', requirements=()),),
        backend="test",
        model="test",
        model_options={},
        instruction="test",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )


def _grown(text: str) -> Document:
    return asyncio.run(grow_leaves(asyncio.run(grow_roots(Document.from_plain_text(text)))))


def test_stage_requires_leaf_growth() -> None:
    with pytest.raises(ValueError, match="requires leaf_growth"):
        asyncio.run(stage.validate_pin_cite_sites(Document.from_plain_text("No citation.")))


def test_no_site_still_records_a_resumable_pass() -> None:
    document = _grown("A filing without citations.")

    checked = asyncio.run(stage.validate_pin_cite_sites(document))

    assert checked is not document
    assert checked.passes[-1] == stage.PIN_CITE_SITE_VALIDATION_STAGE
    assert asyncio.run(stage.validate_pin_cite_sites(checked)) is checked
    assert Document.model_validate(checked.model_dump(mode="json")) == checked


def test_review_updates_only_the_snapshot_and_keeps_grounded_run(monkeypatch) -> None:
    document = _grown("Milwaukee Police Ass'n v. Jones, 192 F.3d 742, 74950 (7th Cir. 1999).")
    (original,) = document.citations
    assert original.fields.pin_cite is None
    seen: list[str] = []

    async def review(text, site, record, *, session=None):
        del session
        seen.append(record.citation_id)
        return SiteReview(
            answer=AdjudicatedPinCite(
                citation_id=site.about,
                pin_cite=PinCite(
                    span=Span(text.index("74950"), text.index("74950") + 5),
                    text="74950",
                    pages=(),
                ),
                reading=Reading.WRITTEN_BUT_NO_PAGE,
                reason="The filing writes a damaged page claim.",
            ),
            reason="The filing writes a damaged page claim.",
            run=_run(),
        )

    monkeypatch.setattr(stage, "adjudicate_pin_cite", review)
    checked = asyncio.run(stage.validate_pin_cite_sites(document))
    (revised,) = checked.citations

    assert seen == [original.citation_id]
    assert original.fields.pin_cite is None
    assert revised.fields.pin_cite is not None
    assert revised.fields.pin_cite.text == "74950"
    assert revised.fields.pin_cite.pages == ()
    assert revised.field_updates[-1].field is CitationField.PIN_CITE
    node = revised.trace[-1]
    assert node.stage == stage.PIN_CITE_SITE_VALIDATION_STAGE
    assert node.outcome == Reading.WRITTEN_BUT_NO_PAGE.value
    trace = deserialize_site_review_trace(node.details)
    assert trace.candidate.about == original.citation_id
    assert trace.ivr.success
    assert Document.model_validate(checked.model_dump(mode="json")) == checked
    assert asyncio.run(stage.validate_pin_cite_sites(checked)) is checked


def test_failed_review_is_evidence_without_a_field_change(monkeypatch) -> None:
    document = _grown("Milwaukee Police Ass'n v. Jones, 192 F.3d 742, 74950 (7th Cir. 1999).")

    async def failed(text, site, record, *, session=None):
        del session
        return SiteReview(
            answer=AdjudicatedPinCite(
                citation_id=record.citation_id,
                pin_cite=PinCite(
                    span=Span(text.index("74950"), text.index("74950") + 5),
                    text="74950",
                    pages=(),
                ),
                reading=Reading.WRITTEN_BUT_NO_PAGE,
            ),
            reason="The quote validator failed.",
            run=_run(success=False),
        )

    monkeypatch.setattr(stage, "adjudicate_pin_cite", failed)
    checked = asyncio.run(stage.validate_pin_cite_sites(document))

    assert checked.citations[0].fields.pin_cite == document.citations[0].fields.pin_cite
    assert checked.citations[0].field_updates == document.citations[0].field_updates
    assert checked.citations[0].trace[-1].outcome == "failed"
    assert not deserialize_site_review_trace(checked.citations[0].trace[-1].details).ivr.success


def test_each_site_is_proposed_against_the_previous_review(monkeypatch) -> None:
    document = _grown("Milwaukee Police Ass'n v. Jones, 192 F.3d 742, 74950 (7th Cir. 1999).")
    record = document.citations[0]
    first = Candidate(
        generator="test",
        kind=CandidateKind.PIN_CITE,
        span=record.locator_span,
        window=record.full_span,
        about=record.citation_id,
    )
    second_span = Span(document.text.index("74950"), document.text.index("74950") + 5)
    second = Candidate(
        generator="test",
        kind=CandidateKind.PIN_CITE,
        span=second_span,
        window=record.full_span,
        about=record.citation_id,
    )
    proposed: list[bool] = []
    reviewed: list[Candidate] = []

    def sites(current):
        has_claim = current.citations[0].fields.pin_cite is not None
        proposed.append(has_claim)
        yield second if has_claim else first

    async def review(text, site, current_record, *, session=None):
        del text, session
        reviewed.append(site)
        claim = PinCite(span=second_span, text="74950", pages=()) if site is first else None
        return SiteReview(
            answer=AdjudicatedPinCite(
                citation_id=current_record.citation_id,
                pin_cite=claim,
                reading=Reading.WRITTEN_BUT_NO_PAGE if claim else Reading.NO_PAGE_CLAIM,
            ),
            reason="Reviewed the next state.",
            run=_run(),
        )

    monkeypatch.setattr(stage, "pin_cite_sites", sites)
    monkeypatch.setattr(stage, "adjudicate_pin_cite", review)
    checked = asyncio.run(stage.validate_pin_cite_sites(document))

    assert reviewed == [first, second]
    assert proposed == [False, True, False]
    assert checked.citations[0].fields.pin_cite is None
    assert (
        len(
            [
                node
                for node in checked.citations[0].trace
                if node.stage == stage.PIN_CITE_SITE_VALIDATION_STAGE
            ]
        )
        == 2
    )
