r"""Ask a reader what a case name standing outside every citation actually is.

`case_name_sites` proposes a span; this answers it. Three readings, and the
document decides which:

``misread_citation``
    The name belongs to a citation read beside it. Extraction reached the
    locator and not the name -- `In re BYJU ' s Alpha, Inc. , 2024 WL 1455586`
    is recorded with the party `Alpha, Inc.` -- so the finding is a correction
    to a citation already in the record, not a new one.

``short_form``
    A proper Bluebook Rule 10.9 reference: the filing names a case it gave in
    full somewhere else and states no identifier here. The finding is a
    citation the record does not hold at all, and what makes it one is the
    root it reads back to, which is usually nowhere near the window.

``not_a_citation``
    The filing's own caption, a section heading, a roman-numeral list item.

The window is **not masked**. The locator reviewer blanks what was already read
so that a reviewer cannot quote a citation the record holds; here the
neighbouring citation is the question, so it has to be visible. What keeps the
answer honest instead is that the reader never returns offsets: it quotes the
name verbatim and picks a neighbour or a root **by index** from lists this
module built, so every part of the answer resolves back into the record
deterministically or fails to resolve at all.

A root is offered from the whole document, because Rule 10.9 puts no distance
limit on a short form -- document 022 writes `Doe v. Rose` eight thousand
characters after the table of authorities entry that gives it in full.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated

from eyecite.annotate import SpanUpdater
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.review.locator import _locate
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.extraction.adjudication.types import Candidate
    from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

MAX_TOKENS = 600
MAX_REPAIR_TURNS = 2

INSTRUCTION = """
Below is a window of text from a legal filing whose PDF extraction is often
damaged: spaces are lost or doubled, line breaks fall mid-citation, and table
columns interleave.

A case name was found at this position and no citation was read there:

    {{site}}

Decide what it is. There are exactly three answers.

"misread_citation" - the name belongs to a citation in the list below, which
    was read with the wrong name or with no name. The citation is beside the
    name in the window. Answer this when the name and one of those citations
    are one citation that the extraction split.

"short_form" - the filing is referring by name alone to a case it has already
    cited in full, which Bluebook Rule 10.9 permits. Choose which case from the
    roots list below. Answer this only when one of those roots is the same
    case: the party names must match, allowing for abbreviation and for a
    shortened name. Each root is marked with where it sits relative to this
    name. {{ordering}}

"uncited_case" - the filing OFFERS the case in support of something it is
    asserting - what a court held, what the law is, what standard applies, or
    that something happened - and nothing in the document cites it: no root in
    the list is this case, and there is no volume, reporter, page or docket
    number for it here either. This is a defect in the filing, and it is a
    different answer from "not_a_citation".

"not_a_citation" - the name appears for some reason other than relying on what
    the case decided. The filing's own caption naming its own parties; a
    proceeding someone names as part of their own history or involvement rather
    than for what was decided in it; a name in a heading; a roman numeral "v"
    that is not "versus"; or characters too damaged to be a case name at all.

Report:
- reading   one of the three answers above
- name      the case name quoted EXACTLY as written in the window, character
            for character, including any damage. For "misread_citation" quote
            the WHOLE name, including the part the citation already has.
- citation  for "misread_citation", the number of the citation from the list
- root      for "short_form", the number of the root from the list
- reason    one sentence

Rules:
- Quote "name" exactly. Do not tidy its spacing or punctuation. The quote is
  checked against the window and a repaired quote will not be found.
- Set "citation" only for "misread_citation" and "root" only for "short_form".
  Leave both null for "not_a_citation".
- A case the filing names and never cites anywhere is NOT a short form. If no
  root in the list is the same case, the answer is "uncited_case" when the
  filing is offering the case as authority, and "not_a_citation" otherwise.
- Between "uncited_case" and "not_a_citation", the question is what the
  sentence is doing, not whether the words look like a case name. Ask whether
  the sentence is leaning on the case for something it wants the reader to
  accept. If it is, and nothing cites it, that is "uncited_case", whether the
  point is legal or factual. If the case is named for another reason - whose
  matter it is, what a heading says, who the parties to this filing are - that
  is "not_a_citation".
- Do not guess. "not_a_citation" is a real answer, and it is the answer
  whenever none of the other three clearly fits.

{{neighbours}}

{{roots}}

window:
{{window}}
""".strip()


ORDERING = (
    "Each root is marked with where it sits relative to this name. Rule 10.9 "
    "permits a short form only AFTER the full citation has appeared, so a root "
    'marked "later in the document" does not make this a short form -- at that '
    "point nothing in the filing locates the case, and the answer is "
    '"uncited_case".'
)


class Reading(str, Enum):
    """What a reader says a case-name site is."""

    MISREAD_CITATION = "misread_citation"
    SHORT_FORM = "short_form"
    UNCITED_CASE = "uncited_case"
    """A case offered as authority that the document never cites."""
    NOT_A_CITATION = "not_a_citation"


class _Answer(BaseModel):
    """What a reader says, before any of it is grounded.

    Every field is required with no default: a structured-output schema has to
    list each of its properties as required, and a field the model may omit is
    a field it omits.
    """

    model_config = ConfigDict(extra="forbid")

    reading: Reading
    name: Annotated[str, StringConstraints(min_length=1)]
    citation: int | None
    root: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class AdjudicatedCaseName:
    """One case-name site and what a reader made of it.

    `span` is the name the reader quoted, grounded back into the document, and
    it is usually wider than the site: a site is what survived masking, and the
    name it belongs to may run into the citation beside it.
    """

    span: Span
    name: str
    reading: Reading
    citation_id: str | None = None
    """For `misread_citation`, the citation the name belongs to."""
    root_id: str | None = None
    """For `short_form`, the root the name reads back to."""
    reason: str = ""
    match_method: str = ""


_REPEATED_INLINE_WHITESPACE = re.compile(r"[ \t]{2,}")
_WORD = re.compile(r"[A-Za-z]{3,}")
# Words that carry no identity: every other case is against one of them.
_COMMON = frozenset(
    {
        "the",
        "and",
        "for",
        "inc",
        "llc",
        "ltd",
        "corp",
        "corporation",
        "company",
        "united",
        "states",
        "state",
        "city",
        "county",
        "department",
        "dept",
        "board",
        "commission",
        "district",
        "school",
        "bank",
        "national",
        "america",
        "american",
        "association",
        "assn",
        "group",
        "holdings",
        "services",
        "servs",
        "systems",
        "sys",
        "international",
        "intl",
        "university",
        "univ",
        "estate",
        "doe",
        "john",
        "jane",
        "director",
        "secretary",
        "commissioner",
        "sheriff",
        "police",
        "trump",
    }
)


def _identity_words(name: str) -> set[str]:
    """The words in a case name that could tell one case from another."""
    return {word.lower() for word in _WORD.findall(name) if word.lower() not in _COMMON}


def _citation_line(index: int, text: str, citation: ExtractedCitation, site: Span | None = None) -> str:
    name = " v. ".join(
        part
        for part in (
            getattr(citation.citation, "plaintiff", None),
            getattr(citation.citation, "defendant", None),
        )
        if part
    ) or getattr(citation.citation, "antecedent", None)
    locator = text[citation.locator_span.start : citation.locator_span.end]
    locator = _REPEATED_INLINE_WHITESPACE.sub(" ", locator.replace("\n", " ")).strip()
    read = f"   (read with the name {name!r})" if name else "   (read with no name)"
    if site is None:
        return f"  {index}. {locator}{read}"
    # Rule 10.9 is ordered: a short form stands only after the full citation.
    # A reader sees one window and cannot tell which came first, so it is told.
    where = "earlier in the document" if citation.locator_span.start < site.start else "later in the document"
    return f"  {index}. {locator}{read}   [{where}]"


def neighbours(document: ExtractedDocument, window: Span) -> tuple[ExtractedCitation, ...]:
    """Every citation read inside the window, in document order.

    These are what `misread_citation` chooses between. The list is the window's
    own, not the document's: a name belongs to a citation it is written beside,
    and offering a distant one invites an answer that cannot be true.
    """
    return tuple(
        citation
        for citation in document.citations
        if citation.full_span.start < window.end and window.start < citation.full_span.end
    )


def roots(document: ExtractedDocument) -> tuple[ExtractedCitation, ...]:
    """Every citation in the document that states an identifier of its own."""
    tree = build_citation_tree(document)
    return tuple(sorted((item.root for item in tree.roots), key=lambda c: c.locator_span.start))


def _ground(window: str, collapsed: str, offset: int, quote: str) -> tuple[Span, str, str] | None:
    """Resolve a quote against the collapsed window, then map back to the document."""
    located = _locate(collapsed, quote)
    if located is None:
        return None
    start, end, method = located
    if collapsed != window:
        updater = SpanUpdater(collapsed, window)
        start = updater.update(start, bisect_right)
        end = updater.update(end, bisect_left)
    return Span(start=offset + start, end=offset + end), window[start:end], method


def _parse(value: object) -> _Answer:
    return _Answer.model_validate_json(str(value))


_UNPARSEABLE = ValidationResult(
    result=False,
    reason=(
        "Return a JSON object with `reading`, `name`, `citation`, `root` and `reason`. "
        "An empty response is not an answer."
    ),
)


def _proposed(ctx: Context) -> tuple[_Answer | None, ValidationResult | None]:
    try:
        return _parse(ctx.last_output().value), None
    except ValidationError:
        return None, _UNPARSEABLE


def _validate_schema(ctx: Context) -> ValidationResult:
    _, failure = _proposed(ctx)
    if failure is not None:
        return failure
    return ValidationResult(result=True)


def _validate_choice(ctx: Context, citations: int, root_count: int) -> ValidationResult:
    """Require the reading and the index it needs to agree."""
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    if proposed.reading is Reading.MISREAD_CITATION:
        if proposed.citation is None or not 1 <= proposed.citation <= citations:
            return ValidationResult(
                result=False,
                reason=(
                    f"`misread_citation` needs `citation` set to one of the {citations} numbered "
                    f"citations in the window. If none of them is the name's citation, the "
                    f"reading is `short_form` or `not_a_citation`."
                ),
            )
        return ValidationResult(result=True)
    if proposed.reading is Reading.SHORT_FORM:
        if proposed.root is None or not 1 <= proposed.root <= root_count:
            return ValidationResult(
                result=False,
                reason=(
                    f"`short_form` needs `root` set to one of the {root_count} numbered roots. "
                    f"If no root is the same case, the filing never cites it in full and the "
                    f"reading is `not_a_citation`."
                ),
            )
        return ValidationResult(result=True)
    if proposed.citation is not None or proposed.root is not None:
        return ValidationResult(
            result=False,
            reason=(f"`{proposed.reading.value}` names neither a citation nor a root. Leave both null."),
        )
    return ValidationResult(result=True)


def _validate_grounding(ctx: Context, window: str) -> ValidationResult:
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    if _locate(window, proposed.name) is None:
        return ValidationResult(
            result=False,
            reason=(
                f"{proposed.name!r} does not resolve inside the window. Quote the name exactly "
                f"as written there, character for character, without repairing its spacing or "
                f"punctuation."
            ),
        )
    return ValidationResult(result=True)


def _same_case(name: str, other: str) -> bool:
    """Whether two case names could be the same case.

    A shortened name keeps at least one party that carries identity, so the two
    must share a word that is not `the`, `Inc.` or `United States`. This catches
    an index off by one, which reads as a confident answer about a case from
    another page; it does not try to settle whether the two names agree, which
    is the reader's judgement and validation's question afterwards.
    """
    words = _identity_words(name)
    return not words or not _identity_words(other) or bool(words & _identity_words(other))


async def adjudicate_case_name(
    document: ExtractedDocument,
    site: Candidate,
    *,
    session: MelleaSession | None = None,
    ordered: bool = True,
) -> AdjudicatedCaseName | None:
    """Return what a reader makes of one case-name site, or `None` on a decline.

    `ordered` applies Rule 10.9's ordering: a short form stands only after the
    full citation has appeared, so a name whose only matching root comes later
    in the document is an `uncited_case`. It is on by default because it is the
    rule, and because over this corpus it is what stops the reader inventing a
    short form -- but it also refuses three names the ground truth calls proper
    short forms, each of which the filing cites in full a sentence or two
    afterwards. Which of those readings is right is a question about the ground
    truth, so the switch is here rather than settled.
    """
    text = document.text
    window_text = text[site.window.start : site.window.end]
    collapsed = _REPEATED_INLINE_WHITESPACE.sub(" ", window_text)
    nearby = neighbours(document, site.window)
    document_roots = roots(document)

    neighbour_lines = "\n".join(
        _citation_line(index, text, citation) for index, citation in enumerate(nearby, start=1)
    )
    root_lines = "\n".join(
        _citation_line(index, text, citation, site.span if ordered else None)
        for index, citation in enumerate(document_roots, start=1)
    )

    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "site": _REPEATED_INLINE_WHITESPACE.sub(" ", text[site.span.start : site.span.end]),
                "ordering": ORDERING if ordered else "",
                "window": collapsed,
                "neighbours": (
                    f"citations already read in this window:\n{neighbour_lines}"
                    if nearby
                    else "no citation was read anywhere in this window."
                ),
                "roots": (
                    f"cases this document cites in full, anywhere:\n{root_lines}"
                    if document_roots
                    else "this document cites no case in full anywhere."
                ),
            },
            output_format=_Answer,
            requirements=[
                req("Return a valid answer.", validation_fn=_validate_schema),
                req(
                    "Name a citation for `misread_citation` and a root for `short_form`.",
                    validation_fn=lambda ctx: _validate_choice(ctx, len(nearby), len(document_roots)),
                ),
                req(
                    "Quote the name with the window's exact characters.",
                    validation_fn=lambda ctx: _validate_grounding(ctx, collapsed),
                ),
            ],
        ),
        strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
        model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
    )
    try:
        proposed = _parse(result.result.value)
    except ValidationError:
        return None

    grounded = _ground(window_text, collapsed, site.window.start, proposed.name)
    if grounded is None:
        return None
    span, name, method = grounded

    citation_id = root_id = None
    if proposed.reading is Reading.MISREAD_CITATION:
        if proposed.citation is None or not 1 <= proposed.citation <= len(nearby):
            return None
        citation_id = nearby[proposed.citation - 1].citation_id
    elif proposed.reading is Reading.SHORT_FORM:
        if proposed.root is None or not 1 <= proposed.root <= len(document_roots):
            return None
        chosen = document_roots[proposed.root - 1]
        beside = next((c for c in nearby if c.citation_id == chosen.citation_id), None)
        if beside is not None:
            # The root it named is the citation in the window. That is a name
            # and a locator the extraction split, whichever word was used for
            # it, so it is recorded as the one finding it is.
            return AdjudicatedCaseName(
                span=span,
                name=name,
                reading=Reading.MISREAD_CITATION,
                citation_id=beside.citation_id,
                reason=proposed.reason,
                match_method=method,
            )
        chosen_name = " ".join(
            part
            for part in (
                getattr(chosen.citation, "plaintiff", None),
                getattr(chosen.citation, "defendant", None),
                getattr(chosen.citation, "antecedent", None),
            )
            if part
        )
        if not _same_case(name, chosen_name):
            return None
        root_id = chosen.citation_id

    return AdjudicatedCaseName(
        span=span,
        name=name,
        reading=proposed.reading,
        citation_id=citation_id,
        root_id=root_id,
        reason=proposed.reason,
        match_method=method,
    )
