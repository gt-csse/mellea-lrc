"""Adjudicate suspected citation sites against the text they were found in.

Runs after extraction, on what extraction left behind, and recovers citations
too damaged for a generated pattern to match:

1.  **Mask** every citation already found, so the next step can only look where
    the extractor came up empty.
2.  **Hunt** the masked text for candidate sites -- a reporter string sitting in
    a volume-and-page shape, or a docket-shaped string with courts written near
    it. This is a recall-oriented filter whose output is meant to be rejected
    freely.
3.  **Adjudicate** each site with a model, which reports an identifier only when
    the text states one completely, quoting it verbatim.

Every quote is then grounded back into the source text, so an answer that
cannot be found there is dropped rather than believed. That is what keeps a
model in the loop without letting it invent citations: it can only agree that
characters already in the document form an identifier.

The name says the two things that make this sound. **Adjudication**: the model
is handed a specific candidate and asked to rule on it, rather than asked what
a document contains. **Grounded**: its ruling is only accepted if the text it
quotes is found in the source.

Compare :mod:`mellea_lrc.experimental.llm_only_extraction`, where the model *is*
the extractor and neither property holds.
"""

from mellea_lrc.experimental.grounded_adjudication.docket_adjudication import (
    AdjudicatedDocket,
    adjudicate_docket,
)
from mellea_lrc.experimental.grounded_adjudication.docket_hunting import (
    CourtCandidate,
    SuspectedDocket,
    docket_context,
    suspected_dockets,
)
from mellea_lrc.experimental.grounded_adjudication.locator_adjudication import (
    AdjudicatedLocator,
    adjudicate_locator,
    reporter_context,
)
from mellea_lrc.experimental.grounded_adjudication.locator_hunting import SuspectedLocator, suspected_locators
from mellea_lrc.experimental.grounded_adjudication.masking import mask_full_spans, mask_locator_spans

__all__ = [
    "AdjudicatedDocket",
    "AdjudicatedLocator",
    "CourtCandidate",
    "SuspectedDocket",
    "SuspectedLocator",
    "adjudicate_docket",
    "adjudicate_locator",
    "docket_context",
    "mask_full_spans",
    "mask_locator_spans",
    "reporter_context",
    "suspected_dockets",
    "suspected_locators",
]
