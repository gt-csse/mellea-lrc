r"""Ask a reader what a case name standing outside every citation actually is.

`case_name_sites` proposes a span; this answers it. Three readings, and the
document decides which:

``names_a_citation``
    The name is the case name of a citation already read. It is not a separate
    citation and nothing about it says the citation is wrong: a filing writes
    `In Boeser v. Sharp , the court recognized …` and then the citation, and
    both names are correct. What the finding carries is **where the name is
    written**, so a consumer can hold the fuller of the two -- `In re BYJU ' s
    Alpha, Inc.` against the `Alpha, Inc.` the parse reached, or
    `United States v. Hassan` against the `Hassan` the filing shortens to at
    the citation itself.

``short_form``
    The filing names a case it gives in full somewhere else and states no
    identifier here. The finding is a citation the record does not hold at all,
    and what makes it one is the root it reads back to, which is usually
    nowhere near the window.

``not_a_citation``
    The filing's own caption, a section heading, a roman-numeral list item.

The window is **not masked**. The locator reviewer blanks what was already read
so that a reviewer cannot quote a citation the record holds; here the
neighbouring citation is the question, so it has to be visible. What keeps the
answer honest instead is that the reader never returns offsets: it quotes the
name verbatim and picks a neighbour or a root **by index** from lists this
module built, so every part of the answer resolves back into the record
deterministically or fails to resolve at all.

A root is offered from the whole document, and from either direction. There is
no distance limit -- document 022 writes `Doe v. Rose` eight thousand characters
after the table of authorities entry that gives it in full -- and no ordering
either: what decides `uncited_case` is whether the document cites the case at
all, because that is what a reader needs to reach it.
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
from mellea_lrc.extraction.adjudication import ocr
from mellea_lrc.extraction.adjudication.review.locator import _locate
from mellea_lrc.extraction.reading.case_names import IDENTIFIER
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

"names_a_citation" - the name is the case name of one of the citations in the
    list below. They are one reference, not two: the filing names the case and
    then cites it, or cites it and then names it. Answer this whenever the name
    and one of those citations are the same case, whatever name that citation
    was read with - a citation read with a shorter name, a different form of
    the name, or no name at all is still the same reference.

"short_form" - the filing is referring by name alone to a case this document
    cites in full somewhere. Choose which case from the roots list below.
    Answer this only when one of those roots is the same case: the party names
    must match, allowing for abbreviation and for a shortened name. WHERE that
    root sits does not matter. A filing may name a case and cite it a sentence
    later, and a reader can still reach it, so it is cited.

"uncited_case" - the filing OFFERS the case in support of something it is
    asserting - what a court held, what the law is, what standard applies, or
    that something happened - and NOTHING ANYWHERE in the document cites it: no
    root in the list is this case, and there is no volume, reporter, page or
    docket number for it here either. Nothing a reader could look the case up
    with exists, which is what makes this a defect in the filing. It is a
    different answer from "not_a_citation".

"not_a_citation" - the name appears for some reason other than relying on what
    the case decided. The filing's own caption naming its own parties; a
    proceeding someone names as part of their own history or involvement rather
    than for what was decided in it; a name in a heading; a roman numeral "v"
    that is not "versus"; or characters too damaged to be a case name at all.

Report:
- reading   one of the four answers above
- name      the case name quoted EXACTLY as written in the window, character
            for character, including any damage. For "names_a_citation" quote
            the WHOLE name, including any part the citation already has.
- citation  for "names_a_citation", the number of the citation from the list
- root      for "short_form", the number of the root from the list
- plaintiff and defendant, for "names_a_citation" and "short_form": the name
            split into its two parties, with any extraction damage REPAIRED.
            `Bell Atl. Corp. v. Twombly` is plaintiff="Bell Atl. Corp.",
            defendant="Twombly". A case with no adverse party has no plaintiff:
            `In re Giftcraft Ltd.` is plaintiff=null, defendant="Giftcraft
            Ltd.", and `Ex parte Young` is plaintiff=null, defendant="Young".
            Leave both null for the other two readings.
- reason    one sentence

Rules:
- Quote "name" exactly. Do not tidy its spacing or punctuation. The quote is
  checked against the window and a repaired quote will not be found.
- Stop at the end of the name. A docket number, a reporter citation or a year
  after it is not part of it: `Boeser v. Sharp , No. CIVA03CV00031WDMMEH, 2007
  WL 1430100` is the name `Boeser v. Sharp` and then two identifiers.
- Set "citation" only for "names_a_citation" and "root" only for "short_form".
  Leave both null for the other two.
- "names_a_citation" comes first. If a citation in the window is the same case
  as this name, that is the answer, even when the case is also in the roots
  list: a name beside its own citation is not a short form of itself.
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
- Damage belongs in "name" and must be repaired in "plaintiff" and
  "defendant": quote "Ass ' n of Specialty Programs" as written and report the
  party as "Ass'n of Specialty Programs". Do not otherwise change the words -
  the parties are checked against the quote and a party that is not in it will
  not pass.
- Do not guess. "not_a_citation" is a real answer, and it is the answer
  whenever none of the other three clearly fits.

{{neighbours}}

{{roots}}

window:
{{window}}
""".strip()


class Reading(str, Enum):
    """What a reader says a case-name site is."""

    NAMES_A_CITATION = "names_a_citation"
    """The case name of a citation already read, not a citation of its own."""
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
    plaintiff: str | None
    defendant: str | None
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
    """For `names_a_citation`, the citation this is the case name of."""
    root_id: str | None = None
    """For `short_form`, the root the name reads back to."""
    plaintiff: str | None = None
    """The first party, repaired. `None` for a case with no adverse party."""
    defendant: str | None = None
    """The second party, or the whole name where there is only one.

    Which is eyecite's own convention: it parses `In re Flint Water Cases` to
    `defendant='Flint Water Cases'` and a single-party short form to
    `defendant='Hassan'`, leaving `plaintiff` unset. A patch that filled
    `plaintiff` instead would disagree with every citation the rules parsed.
    """
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


def _citation_line(index: int, text: str, citation: ExtractedCitation) -> str:
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
    return f"  {index}. {locator}{read}"


def neighbours(document: ExtractedDocument, window: Span) -> tuple[ExtractedCitation, ...]:
    """Every citation read inside the window, in document order.

    These are what `names_a_citation` chooses between. The list is the window's
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


# Abbreviations a case name ends in, where the final period belongs to the word.
# Everything else ending a quoted name is the sentence's period, not the name's.
_ABBREVIATION = frozenset(
    """admin assn auth bd bros cas cent cir cnty co comm commn constr corp cty dept
    dist div educ elec eng engrs enters equip fed fin found gen grp hldgs hosp inc
    indus ins intl invs liab ltd llc llp lp mfg mgmt mkts mortg mtge mut natl no pc
    pharm pharms plc pllc prods props res ry rr sch sec servs sols soc sys techs
    univ""".split()
)
_TRAILING_WORD = re.compile(r"([A-Za-z'’]+)\.$")


def trim_sentence_period(name: str) -> str:
    """Drop a final period that ends the sentence rather than the name.

    A filing writes `… including Dailey v. Integon and Murray v. Nationwide.`
    and the period is the sentence's, while `Andrade Gutierrez Engenharia S.A.`
    and `Weetabix Co.` end in one of their own. What tells them apart is the
    word in front of it: an abbreviation, or a whole word.
    """
    found = _TRAILING_WORD.search(name)
    if not found:
        return name
    word = found.group(1).replace("'", "").replace("\u2019", "").lower()
    if word in _ABBREVIATION or len(word) <= 2:
        return name
    return name[:-1]


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
    if proposed.reading is Reading.NAMES_A_CITATION:
        if proposed.citation is None or not 1 <= proposed.citation <= citations:
            return ValidationResult(
                result=False,
                reason=(
                    f"`names_a_citation` needs `citation` set to one of the {citations} numbered "
                    f"citations in the window. If none of them is the same case as this name, the "
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


def _validate_parties(ctx: Context) -> ValidationResult:
    """Require the parties to be the quoted name, and nothing else."""
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    if proposed.reading not in _NAMES_A_CASE:
        return ValidationResult(result=True)
    if parties_read_as(proposed.name, proposed.plaintiff, proposed.defendant):
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=(
            f"plaintiff={proposed.plaintiff!r} and defendant={proposed.defendant!r} do not read "
            f"as {proposed.name!r}. Split the name you quoted and nothing else, repairing the "
            f"damage but changing no words. A case with no adverse party has a defendant and no "
            f"plaintiff, and `In re` and `Ex parte` belong to neither."
        ),
    )


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


_NAMES_A_CASE = frozenset({Reading.NAMES_A_CITATION, Reading.SHORT_FORM})
# The opening of a case with no adverse party. Part of the name, and part of
# neither party -- which is eyecite's own reading, since it parses
# `In re Giftcraft Ltd.` to `defendant='Giftcraft Ltd.'`
_OPENS_WITH_NO_PARTY = re.compile(r"^(?:In\s+re|In\s+the\s+Matter\s+of|Matter\s+of|Ex\s+parte)\s+", re.I)
_VERSUS = re.compile(r"\bvs?\.?(?=\s|$)", re.I)
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _letters(value: str) -> str:
    """A name reduced to the characters that carry it."""
    return _NOT_ALPHANUMERIC.sub("", value.lower())


def parties_read_as(name: str, plaintiff: str | None, defendant: str | None) -> bool:
    """Whether the parties are what the quoted name says, damage forgiven.

    The quote is the document's characters and the parties are repaired, so the
    two are compared with punctuation, spacing and the `v.` removed, and with
    the characters a scanner confuses folded together -- `Ass ' n` against
    `Ass'n`, `l83` against `183`. Folding admits a substitution and never an
    insertion, because the two must already be the same length to compare.

    A name with no adverse party carries only a defendant, which is eyecite's
    convention and not a choice made here. Its opening words are not part of
    either party: `In re Giftcraft Ltd.` is `Giftcraft Ltd.`
    """
    if not defendant:
        return False
    quoted = _letters(_OPENS_WITH_NO_PARTY.sub("", _VERSUS.sub(" ", name), count=1))
    parts = _letters((plaintiff or "") + defendant)
    return quoted == parts or (len(quoted) == len(parts) and ocr.fold(quoted) == ocr.fold(parts))


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
) -> AdjudicatedCaseName | None:
    """Return what a reader makes of one case-name site, or `None` on a decline.

    A case is uncited when the document holds no citation of it, in either
    direction. Rule 10.9's ordering was tried here and is the wrong test: it
    governs short-form *citations*, which claim a page, and a name written in
    text claims nothing, so a filing may name a case and cite it a sentence
    later without writing anything improper. Reading those as uncited cost
    eight of this layer's thirteen errors on `extraction-eval-1`.
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
        _citation_line(index, text, citation) for index, citation in enumerate(document_roots, start=1)
    )

    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "site": _REPEATED_INLINE_WHITESPACE.sub(" ", text[site.span.start : site.span.end]),
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
                    "Name a citation for `names_a_citation` and a root for `short_form`.",
                    validation_fn=lambda ctx: _validate_choice(ctx, len(nearby), len(document_roots)),
                ),
                req(
                    "Split the name into its parties, repairing the damage.",
                    validation_fn=_validate_parties,
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
    # A quote asked for the WHOLE name comes back with the docket number after
    # it more often than not. The name stops where an identifier starts, the
    # same rule the rules-side reader applies.
    cut = IDENTIFIER.search(name)
    if cut:
        span, name = Span(start=span.start, end=span.start + cut.start()), name[: cut.start()]
    trimmed = trim_sentence_period(name.rstrip(" ,"))
    if trimmed != name:
        span, name = Span(start=span.start, end=span.start + len(trimmed)), trimmed

    citation_id = root_id = None
    if proposed.reading is Reading.NAMES_A_CITATION:
        if proposed.citation is None or not 1 <= proposed.citation <= len(nearby):
            return None
        citation_id = nearby[proposed.citation - 1].citation_id
    elif proposed.reading is Reading.SHORT_FORM:
        if proposed.root is None or not 1 <= proposed.root <= len(document_roots):
            return None
        chosen = document_roots[proposed.root - 1]
        beside = next((c for c in nearby if c.citation_id == chosen.citation_id), None)
        if beside is not None:
            # The root it named is a citation in the window, so the name and
            # that citation are one reference however the answer was worded.
            parties = parties_read_as(name, proposed.plaintiff, proposed.defendant)
            return AdjudicatedCaseName(
                span=span,
                name=name,
                reading=Reading.NAMES_A_CITATION,
                citation_id=beside.citation_id,
                plaintiff=proposed.plaintiff if parties else None,
                defendant=proposed.defendant if parties else None,
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

    parties = parties_read_as(name, proposed.plaintiff, proposed.defendant)
    return AdjudicatedCaseName(
        span=span,
        name=name,
        reading=proposed.reading,
        citation_id=citation_id,
        root_id=root_id,
        plaintiff=proposed.plaintiff if parties else None,
        defendant=proposed.defendant if parties else None,
        reason=proposed.reason,
        match_method=method,
    )
