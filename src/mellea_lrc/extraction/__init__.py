"""Citation extraction from Layer 2 preprocessed legal text.

**Preprocessing is a stage of its own and runs first.** It converts the source,
decides which page furniture is not the document's text, and in doing so settles
the coordinate space every span here will index. Extraction reads citations out
of what it produced and changes nothing about the text.

Two entrypoints:

``extract_citations``
    A :class:`~mellea_lrc.preprocessing.PreprocessedDocument`, whatever produced
    it. This is the stage's real signature: the preceding stage's output in, this
    stage's output out.

``extract_from_plain_text``
    Text already in hand, wrapped for you. A convenience over the above for a
    caller that has a string and no file.

Nothing here opens a PDF or chooses a layout rule. A caller starting from a file
runs :func:`~mellea_lrc.preprocessing.preprocess` first and passes the result,
so which furniture was removed is a decision made in the open rather than one
extraction makes on the caller's behalf.

Both take the same ``relaxation`` keyword, which is the one thing that
decides how much whitespace damage a citation may carry and still be found.
There is no separate relaxed extractor and no text repair anywhere behind them:
one code path, one parameter. See :class:`Relaxation` for the three levels and
what each one costs.

## How the layer is arranged

Four kinds of thing, and they were worth separating because they answer
different questions.

``eyecite_extractor``, ``types``
    The entrypoints and the vocabulary. `eyecite_extractor` is the one place
    that calls everything else, in the order the comments there give.

:mod:`~mellea_lrc.extraction.reading`
    What eyecite is made to read. Every correction to how the library parses
    text a PDF converter produced -- the reporter joins, pin cites, where the
    court-and-date scan stops, resolving a court, and reading a docket number
    as a locator.

:mod:`~mellea_lrc.extraction.structure`
    What the citations mean together once each is read: which occupy the same
    span, and which refer to the same authority. Neither looks at the text.

:mod:`~mellea_lrc.extraction.adjudication`
    What happens after the rules. Candidates the deterministic pass did not
    record, and the judgement that lets one into the record. Nothing above
    imports it, so a caller wanting only the deterministic reading never loads
    a reviewer.
"""

from mellea_lrc.extraction.eyecite_extractor import (
    extract_citations,
    extract_from_plain_text,
    grow_leaves,
    grow_roots,
)
from mellea_lrc.extraction.locator_stages import (
    COLOCATION_STAGE,
    DOCKET_RULE_STAGE,
    DOCKET_SITE_STAGE,
    REPORTER_RULE_STAGE,
    REPORTER_SITE_STAGE,
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
    start_locator_document,
)
from mellea_lrc.extraction.reading.relaxation import Relaxation
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE, form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.stages import (
    audit_dockets,
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.structure.attachment import Attachment
from mellea_lrc.extraction.structure.locator_layers import Locator, LocatorLayers, find_locators
from mellea_lrc.extraction.structure.withdrawal import withdraw_leaves_of_withdrawn_roots
from mellea_lrc.extraction.types import (
    CitationRecord,
    Document,
    ExtractionBackend,
    ExtractionMetadata,
)

__all__ = [
    "COLOCATION_STAGE",
    "DOCKET_RULE_STAGE",
    "DOCKET_SITE_STAGE",
    "REPORTER_RULE_STAGE",
    "REPORTER_SITE_STAGE",
    "ROOT_FORMATION_STAGE",
    "Attachment",
    "CitationRecord",
    "Document",
    "ExtractionBackend",
    "ExtractionMetadata",
    "ExtractionRules",
    "Locator",
    "LocatorLayers",
    "Relaxation",
    "audit_dockets",
    "extract_citations",
    "extract_from_plain_text",
    "find_docket_locators",
    "find_full_reporter_locators",
    "find_locators",
    "form_roots",
    "grow_leaves",
    "grow_roots",
    "mark_full_reporter_locator_hunting_skipped",
    "resolve_case_names",
    "resolve_colocations",
    "resolve_courts",
    "resolve_dates",
    "resolve_pin_cites",
    "stable",
    "start_locator_document",
    "withdraw_leaves_of_withdrawn_roots",
]
