"""Docket site hunting admits locators before ordinary field parsing."""

from __future__ import annotations

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.extraction.adjudication import apply_docket_site_review, suspected_dockets
from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket
from mellea_lrc.extraction.adjudication.review.docket import (
    DOCKET_REVIEW_PREFIX,
    DOCKET_SITE_HUNTING_SESSION_ID,
    INSTRUCTION,
    RecoveredDocketLocator,
    _site_review,
)
from mellea_lrc.extraction.adjudication.types import SiteReview
from mellea_lrc.llm import IvrAttempt, IvrRequirementAttempt, IvrRun
from mellea_lrc.preprocessing import preprocess


def _accepted_review(locator: str, docket_number: str) -> SiteReview[RecoveredDocketLocator]:
    run = IvrRun(
        success=True,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output=(
                    '{"is_docket_citation":true,'
                    f'"locator":"{locator}",'
                    f'"docket_number":"{docket_number}",'
                    '"reason":"The filing cites a bankruptcy proceeding."}'
                ),
                requirements=(IvrRequirementAttempt("valid", True, None, None),),
            ),
        ),
        backend="test",
        model="test",
        model_options={},
        instruction="test",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )
    return SiteReview(
        answer=RecoveredDocketLocator(
            locator_text=locator,
            docket_number=docket_number,
            reason="The filing cites a bankruptcy proceeding.",
        ),
        reason="The filing cites a bankruptcy proceeding.",
        run=run,
    )


def test_site_admission_creates_a_minimal_docket_locator_then_rereads_fields() -> None:
    text = "Ascentra v. Example, No. 21-11854 (Bankr.  S.D.N.Y. Nov. 2, 2021)."
    document = grow_roots(preprocess(text), rules=stable())
    (site,) = suspected_dockets(document)

    updated = apply_docket_site_review(
        document,
        site,
        _accepted_review(site.locator_text, site.docket_number),
    )

    (record,) = updated.citations
    assert isinstance(record.source, DocketCitation)
    assert record.source.docket_number == "21-11854"
    # The site review carried no court. The normal court reader subsequently
    # resolves the literal court spelling despite the doubled whitespace.
    assert record.stated.court == "nysb"
    assert record.stated.court_text == "Bankr.  S.D.N.Y."
    assert record.stated.date is not None
    assert record.stated.date.year == "2021"
    assert record.stated.case_name is not None
    assert record.stated.case_name.text == "Ascentra v. Example"
    assert all("docket_audit" not in node.node_id for node in record.trace)
    assert record.trace[0].stage == "docket_site_hunting"


def test_failed_exact_grounding_is_a_declined_review_not_a_promotion() -> None:
    """A parseable final model answer is unsafe when its repair run failed."""
    run = IvrRun(
        success=False,
        selected_attempt=0,
        attempts=(
            IvrAttempt(
                output=(
                    '{"is_docket_citation":true,"locator":"No. 21-11854 typo",'
                    '"docket_number":"21-11854","reason":"quoted the wrong locator"}'
                ),
                requirements=(
                    IvrRequirementAttempt(
                        "Quote the complete docket locator exactly as written in the window.",
                        False,
                        "The locator does not exactly match the candidate.",
                        None,
                    ),
                ),
            ),
        ),
        backend="test",
        model="test",
        model_options={},
        instruction="test",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )

    review = _site_review(run)

    assert review.answer is None
    assert review.reason == "The locator does not exactly match the candidate."


def test_docket_review_uses_a_shared_prefix_and_dynamic_site_instruction(monkeypatch) -> None:
    """The repeated contract is cacheable; no docket convention is site-specific."""
    import asyncio
    from types import SimpleNamespace

    from mellea.stdlib.sampling import MultiTurnStrategy

    from mellea_lrc.extraction.adjudication.review.docket import adjudicate_docket

    captured: dict[str, object] = {}

    async def fake_run(_session: object, spec: object, **kwargs: object) -> IvrRun:
        captured["spec"] = spec
        captured["model_options"] = kwargs["model_options"]
        return IvrRun(
            success=True,
            selected_attempt=0,
            attempts=(
                IvrAttempt(
                    output=(
                        '{"is_docket_citation":false,"locator":null,'
                        '"docket_number":null,"reason":"not a case"}'
                    ),
                    requirements=(),
                ),
            ),
            backend="test",
            model="test",
            model_options={},
            instruction="test",
            prefix=None,
            grounding_context={},
            user_variables={},
            output_schema=None,
        )

    monkeypatch.setattr("mellea_lrc.extraction.adjudication.review.docket.run_instruct_ivr", fake_run)
    monkeypatch.setattr(
        "mellea_lrc.extraction.adjudication.review.docket.llm_api_config_from_env",
        lambda _environ: SimpleNamespace(mellea_call_options=lambda **_kwargs: {"max_tokens": 1200}),
    )
    site = SuspectedDocket(
        locator_span=Span(10, 22),
        locator_text="No. 21-11854",
        docket_number="21-11854",
        context_span=Span(0, 30),
        context="Case text No. 21-11854 more text",
    )

    asyncio.run(adjudicate_docket(site, session=SimpleNamespace()))

    spec = captured["spec"]
    assert getattr(spec, "prefix") == DOCKET_REVIEW_PREFIX
    assert getattr(spec, "description") == INSTRUCTION
    assert getattr(spec, "user_variables") == {
        "locator": "No. 21-11854",
        "window": "Case text No. 21-11854 more text",
    }
    assert captured["model_options"] == {
        "max_tokens": 1200,
        "extra_body": {"session_id": DOCKET_SITE_HUNTING_SESSION_ID},
    }
