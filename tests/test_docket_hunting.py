"""Docket site hunting admits locators before ordinary field parsing."""

from __future__ import annotations

from mellea_lrc.core.citations import DocketCitation
from mellea_lrc.extraction import grow_roots, stable
from mellea_lrc.extraction.adjudication import apply_docket_site_review, suspected_dockets
from mellea_lrc.extraction.adjudication.review.docket import RecoveredDocketLocator, _site_review
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
