"""Everything after extraction's rules: candidates, and a judgement on them.

Extraction proper is deterministic. It reads what its patterns can read and
records nothing else, which is the property that lets its output be measured.
This is the stage that follows, inside the same layer because that is all it
serves -- nothing downstream reuses it -- and it is where every model call
before validation lives.

It depends on extraction and extraction does not depend on it. Nothing here is
imported by ``mellea_lrc.extraction`` itself, so a caller wanting only the
deterministic reading never loads a reviewer.

It has two halves and they are kept apart on purpose.

:mod:`~mellea_lrc.extraction.adjudication.candidates`
    Generators. Cheap, deterministic, each answering one question, none of them
    deciding anything. A generator that decides is a rule that belonged in
    extraction.

:mod:`~mellea_lrc.extraction.adjudication.review`
    Reviewers. One module per question, each given a candidate and the window
    around it. A reviewer never decides what is in the document, only whether
    characters already there mean what a generator proposed.

:mod:`~mellea_lrc.extraction.adjudication.promotion` closes the loop: an accepted candidate
is re-read through eyecite on its own window, with a rule widened for that span
alone, so it becomes an ordinary ``CitationRecord`` rather than a hand-built
object that would miss the court, date and party names the real pipeline
produces.

**Why the layer exists rather than more rules.** A fix that looks easy but that
the data cannot show generalises should not be hardened -- it should propose and
be reviewed, and the thinness of the evidence is exactly what makes the review
affordable. Making 67 reporter spellings case-insensitive to catch two citations
is the wrong trade; proposing those two is the right one.

:mod:`~mellea_lrc.extraction.adjudication.reviews`
    The one way in. :class:`Review` names each question this layer can be asked,
    every one of them opt-in, and :func:`adjudicate` runs the ones a caller
    named.

**A review writes to the record, so it runs one site at a time.** The model
underneath is asynchronous and nothing about the network stops several sites
being in flight together -- but a reviewer corrects a field on a record, and the
site after it is shown the corrected field. `adjudicate_case_name` is handed the
records precisely so it can say which citation a name belongs to; two sites
deciding about the same citation concurrently would each answer against a record
the other is rewriting, and which answer survives would depend on which call
returned first. That is a record that cannot be replayed, which is the one
property this whole design exists to keep.

So :func:`adjudicate` awaits each review in turn and each site within it in
turn. Concurrency here is not an optimisation to add later and forget to guard:
it needs a review that declares its sites independent of one another, and none
of them does yet.
"""

from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket, suspected_dockets
from mellea_lrc.extraction.adjudication.candidates.pin_cite_sites import pin_cite_sites
from mellea_lrc.extraction.adjudication.candidates.reporter_sites import (
    SiteStage,
    SuspectedLocator,
    suspected_locators,
)
from mellea_lrc.extraction.adjudication.masking import mask_full_spans, mask_locator_spans
from mellea_lrc.extraction.adjudication.promotion import promote, promote_locator, reread_site
from mellea_lrc.extraction.adjudication.review.docket import adjudicate_docket
from mellea_lrc.extraction.adjudication.review.locator import adjudicate_locator
from mellea_lrc.extraction.adjudication.reviews import DEFAULT_REVIEWS, Review, adjudicate
from mellea_lrc.extraction.adjudication.types import Adjudication, Candidate, CandidateKind, Verdict

__all__ = [
    "DEFAULT_REVIEWS",
    "Adjudication",
    "Candidate",
    "CandidateKind",
    "Review",
    "SiteStage",
    "SuspectedDocket",
    "SuspectedLocator",
    "Verdict",
    "adjudicate",
    "adjudicate_docket",
    "adjudicate_locator",
    "mask_full_spans",
    "mask_locator_spans",
    "pin_cite_sites",
    "promote",
    "promote_locator",
    "reread_site",
    "suspected_dockets",
    "suspected_locators",
]
