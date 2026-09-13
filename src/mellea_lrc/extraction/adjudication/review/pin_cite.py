r"""Ask a reader what page a citation claims, where the rules are not sure.

`pin_cite_sites` proposes a position on a citation already in the record; this
answers it. Two readings, and the characters decide which:

``claims_a_page``
    The filing claims a page here, and the reader quotes the characters that
    say so. That is the whole answer -- which pages those characters mean is
    read from them afterwards by the same code that reads every other pin cite,
    so a reader cannot state a page the document does not carry.

``written_but_no_page``
    The filing wrote a page claim and no page numbering reaches it: `749-50`
    arriving as `74950` because the converter dropped the hyphen, digits spaced
    apart, a range that ends before it begins. The characters are kept and no
    page is claimed, because recording nothing would say the filing claims no
    page and recording 74,950 would state a page it does not claim.

``no_page_claim``
    There is no page claim at this position. The number after the citation is
    the next reporter's volume, a line number from the margin of pleading
    paper, a year, or the first word of the next sentence.

**The reader never returns an offset and never returns a number.** It quotes,
and the quote is searched for in the window; a quote that is not found is a
declined answer rather than an approximate one. What the reader is for is the
one judgement the characters do not settle on their own -- whether the digits
after a citation belong to it -- and every consequence of that judgement is
computed from the document.

**The answer corrects a record, and only ever one field.** Each site names the
citation it is about, so the finding is a `Correction` to that citation's
`pin_cite` and nothing else. A site that proposed a citation nobody read would
have to create a record instead, and there are none: finding a citation with no
locator means searching for a case name, which waits until validation has
resolved the roots. See `mellea_lrc.extraction.adjudication.reviews`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.core.pin_cites import PinCite
from mellea_lrc.core.spans import Span
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.adjudication.types import Candidate

MAX_TOKENS = 400
MAX_REPAIR_TURNS = 2

INSTRUCTION = """
Below is a window of text from a legal filing whose PDF extraction is often
damaged: spaces are lost or doubled, hyphens disappear, line breaks fall
mid-citation, and the numbered margin of pleading paper lands inside a line of
prose.

One citation in this window is:

    {{citation}}

The rules read this as the page it claims:

    {{read_as}}

That reading may be short, wrong, or missing, because: {{why}}

Say what page the citation claims. A **pin cite** is the page or pages of the
cited case that the filing is pointing the reader at. It is written after the
citation -- `, 570`, `, 570-71`, `at 546`, `at *3`, `¶ 26`, `570 n.10` -- and
it is a page of THIS case.

These are not pin cites, and each of them stands exactly where one would:

*   **The next reporter's volume.** One case reported twice is written as one
    citation: the page of the first reporter, then a comma, then the second
    reporter's volume. The number after the comma is a volume when a reporter
    name follows it.
*   **A line number from the margin.** Pleading paper numbers its lines, and
    the converter drops those numbers into the text. A number with no
    connector in front of it, sitting between the case name and the volume or
    between the page and the court, is margin furniture.
*   **A year.** A four-digit number in the seventeen to twenty hundreds, or one
    inside the parentheses that carry the court.
*   **The next sentence.** Prose that happens to begin with a number.

Answer with:

*   `reading`, one of:

    *   `claims_a_page` -- the characters state a page of this case.
    *   `written_but_no_page` -- the filing writes a page claim here and no
        page numbering reaches it. A run of digits the converter made by
        dropping a hyphen, so that `749-50` arrives as `74950`; digits it
        spaced apart; a range that ends before it begins. The claim is real and
        the characters are not a page.
    *   `no_page_claim` -- nothing at this position claims a page of this case.

*   `pin_cite`: for the first two, the characters of the page claim, copied
    from the window **exactly** as they appear -- every space, hyphen and
    period. Do not add a connector the filing does not write, do not repair a
    lost hyphen, and do not include the parenthesis, the court or the year.
    For `no_page_claim`, the empty string.
*   `reason`: one sentence saying what the characters are.

window:
{{window}}
""".strip()


class Reading(str, Enum):
    """What a reader says is at a pin-cite site.

    Three, because a record has three states and a reader that could only say
    two of them would have to round one of the others off. A filing that writes
    `192 F.3d 742, 74950` has made a page claim -- the characters are there, and
    they are `749-50` with the hyphen lost -- and no page numbering reaches
    them. Recording nothing would say the filing claims no page, which is false;
    recording page 74,950 would state a page the filing does not claim.
    """

    CLAIMS_A_PAGE = "claims_a_page"
    """The characters state a page, and quoting them is the answer."""
    WRITTEN_BUT_NO_PAGE = "written_but_no_page"
    """The characters are a page claim and no page numbering reaches them.

    A run of digits the converter made by dropping a hyphen, digits it spaced
    apart, a page range that ends before it begins.
    """
    NO_PAGE_CLAIM = "no_page_claim"
    """Nothing at this position claims a page of this case."""


class _Answer(BaseModel):
    """What a reader says, before any of it is grounded.

    Every field is required with no default: a structured-output schema has to
    list each of its properties as required, and a field the model may omit is
    a field it omits.
    """

    model_config = ConfigDict(extra="forbid")

    reading: Reading
    pin_cite: str
    reason: Annotated[str, StringConstraints(min_length=1)]


@dataclass(frozen=True, slots=True)
class AdjudicatedPinCite:
    """One pin-cite site and what a reader made of it."""

    citation_id: str
    """The citation this corrects. Every site names one."""

    pin_cite: PinCite | None
    """The page claim as read from the characters, or `None` for no claim."""

    reading: Reading
    reason: str = ""


_WHITESPACE = re.compile(r"\s+")
#: How far past the site a quote may be found. A pin cite sits against its
#: citation; a match further away is a different number that happens to read
#: the same, which is a decline rather than an answer.
REACH = 90


def _parse(raw: object) -> _Answer:
    return _Answer.model_validate_json(str(getattr(raw, "value", raw)))


def _proposed(ctx: Context) -> tuple[_Answer | None, ValidationResult | None]:
    try:
        return _parse(ctx.last_output().value), None
    except ValidationError as error:
        return None, ValidationResult(result=False, reason=str(error)[:400])


def _validate_schema(ctx: Context) -> ValidationResult:
    _, failure = _proposed(ctx)
    return failure or ValidationResult(result=True)


def _validate_quote(ctx: Context, window: str) -> ValidationResult:
    """A page claim has to be characters the window holds."""
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    if proposed.reading is Reading.NO_PAGE_CLAIM:
        if proposed.pin_cite:
            return ValidationResult(
                result=False, reason="`no_page_claim` quotes nothing; leave `pin_cite` empty."
            )
        return ValidationResult(result=True)
    if not proposed.pin_cite.strip():
        return ValidationResult(
            result=False,
            reason=f"`{proposed.reading.value}` has to quote the characters the filing wrote.",
        )
    if _find(window, proposed.pin_cite) is None:
        return ValidationResult(
            result=False,
            reason=f"{proposed.pin_cite!r} is not in the window. Copy it exactly as it appears.",
        )
    return ValidationResult(result=True)


def _find(window: str, quote: str) -> re.Match[str] | None:
    """Where a quote sits in the window, with any run of whitespace matching any.

    The converter's damage is in the spacing, and a reader asked to copy
    `1053 -54` will sometimes return `1053-54` -- and asked to copy `74950`
    will sometimes return `749 50`. The characters have to be the window's own,
    in the window's order; the spaces between them need not be.
    """
    characters = [character for character in quote if not character.isspace()]
    if not characters:
        return None
    pattern = re.compile(r"\s*".join(re.escape(character) for character in characters))
    return pattern.search(window)


async def adjudicate_pin_cite(
    text: str,
    site: Candidate,
    record: CitationRecord,
    *,
    session: MelleaSession | None = None,
) -> AdjudicatedPinCite | None:
    """Return what a reader makes of one pin-cite site, or `None` on a decline.

    `None` is a real answer and is recorded as one by the caller: a reader that
    cannot quote the characters has told us the rules' reading stands.
    """
    window = text[site.window.start : site.window.end]
    pin = record.pin_cite_span
    read_as = text[pin.start : pin.end] if pin is not None else ""

    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "citation": _WHITESPACE.sub(" ", record.matched_text),
                "read_as": f"`{read_as}`" if read_as else "no page at all",
                "why": site.note,
                "window": window,
            },
            output_format=_Answer,
            requirements=[
                req("Return a valid answer.", validation_fn=_validate_schema),
                req(
                    "Quote the page claim with the window's exact characters.",
                    validation_fn=lambda ctx: _validate_quote(ctx, window),
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

    if proposed.reading is Reading.NO_PAGE_CLAIM:
        return AdjudicatedPinCite(
            citation_id=site.about or record.citation_id,
            pin_cite=None,
            reading=proposed.reading,
            reason=proposed.reason,
        )

    found = _find(window, proposed.pin_cite)
    if found is None:
        return None
    span = Span(start=site.window.start + found.start(), end=site.window.start + found.end())
    # Against its own citation, not a number elsewhere in the window that reads
    # the same. The site is where the rules looked; the claim is beside it.
    if not (site.span.start - REACH <= span.start <= site.span.end + REACH):
        return None
    written = text[span.start : span.end]
    # `claims_a_page` reads the pages out of the characters; the reader never
    # states a number. `written_but_no_page` keeps the characters and claims no
    # page, which is the record saying what the filing wrote and that no page
    # numbering reaches it.
    claim = (
        PinCite.read(written, span)
        if proposed.reading is Reading.CLAIMS_A_PAGE
        else PinCite(span=span, text=written, pages=())
    )
    return AdjudicatedPinCite(
        citation_id=site.about or record.citation_id,
        pin_cite=claim,
        reading=proposed.reading,
        reason=proposed.reason,
    )
