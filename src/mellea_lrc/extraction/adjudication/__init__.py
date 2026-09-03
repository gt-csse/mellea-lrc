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
alone, so it becomes an ordinary ``ExtractedCitation`` rather than a hand-built
object that would miss the court, date and party names the real pipeline
produces.

**Why the layer exists rather than more rules.** A fix that looks easy but that
the data cannot show generalises should not be hardened -- it should propose and
be reviewed, and the thinness of the evidence is exactly what makes the review
affordable. Making 67 reporter spellings case-insensitive to catch two citations
is the wrong trade; proposing those two is the right one. See
`exploration/notes/candidates-and-adjudication.md`.
"""

from mellea_lrc.extraction.adjudication.candidates.docket_sites import SuspectedDocket, suspected_dockets
from mellea_lrc.extraction.adjudication.candidates.reporter_sites import SuspectedLocator, suspected_locators
from mellea_lrc.extraction.adjudication.masking import mask_full_spans, mask_locator_spans
from mellea_lrc.extraction.adjudication.promotion import promote
from mellea_lrc.extraction.adjudication.review.docket import adjudicate_docket
from mellea_lrc.extraction.adjudication.review.locator import adjudicate_locator
from mellea_lrc.extraction.adjudication.types import Adjudication, Candidate, CandidateKind, Verdict

__all__ = [
    "Adjudication",
    "Candidate",
    "CandidateKind",
    "SuspectedDocket",
    "SuspectedLocator",
    "Verdict",
    "adjudicate_docket",
    "adjudicate_locator",
    "mask_full_spans",
    "mask_locator_spans",
    "promote",
    "suspected_dockets",
    "suspected_locators",
]
