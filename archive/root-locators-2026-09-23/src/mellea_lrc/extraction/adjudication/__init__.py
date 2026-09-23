"""Optional, checkpointable reviews of sites extraction did not settle.

Candidate generators propose sites; grounded reviewers read one site at a time.
The independent document stages in :mod:`mellea_lrc.api` apply decisions and
retain declined attempts. Docket hunting runs before root formation. Leaf
case-name hunting and pin-cite site validation run after leaf growth.
"""

from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket, suspected_dockets
from mellea_lrc.extraction.adjudication.candidates.pin_cite_sites import pin_cite_sites
from mellea_lrc.extraction.adjudication.candidates.reporter_sites import (
    SiteStage,
    SuspectedLocator,
    suspected_locators,
)
from mellea_lrc.extraction.adjudication.docket_hunting import apply_docket_site_review, hunt_docket_locators
from mellea_lrc.extraction.adjudication.masking import mask_full_spans, mask_locator_spans
from mellea_lrc.extraction.adjudication.promotion import (
    promote,
    promote_docket_locator,
    promote_locator,
    reread_site,
)
from mellea_lrc.extraction.adjudication.review.docket import RecoveredDocketLocator, adjudicate_docket
from mellea_lrc.extraction.adjudication.review.locator import adjudicate_locator
from mellea_lrc.extraction.adjudication.types import (
    Adjudication,
    Candidate,
    CandidateKind,
    SiteReview,
    Verdict,
)

__all__ = [
    "Adjudication",
    "Candidate",
    "CandidateKind",
    "RecoveredDocketLocator",
    "SiteReview",
    "SiteStage",
    "SuspectedDocket",
    "SuspectedLocator",
    "Verdict",
    "adjudicate_docket",
    "adjudicate_locator",
    "apply_docket_site_review",
    "hunt_docket_locators",
    "mask_full_spans",
    "mask_locator_spans",
    "pin_cite_sites",
    "promote",
    "promote_docket_locator",
    "promote_locator",
    "reread_site",
    "suspected_dockets",
    "suspected_locators",
]
