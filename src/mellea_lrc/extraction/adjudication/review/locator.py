"""Ask a model to read a suspected site and report the locators actually there.

The model never returns offsets. It quotes each locator verbatim, and the quote
is then grounded deterministically back into the document, so a hallucinated or
paraphrased answer fails to resolve rather than producing a plausible-looking
span. This is the same contract the pinpoint evidence quote uses.

Two details matter for span fidelity:

*   The window shown to the model has its repeated inline whitespace collapsed.
    Layout-damaged text is full of column padding, and a model reading
    ``937   S.W.2d   796`` is markedly less likely to quote it back in a shape
    that can be found again. Collapsing first also makes the quote easier to
    ground.
*   Because the model sees collapsed text, a resolved span is in collapsed
    coordinates. It is mapped back onto the original window, and then to
    document coordinates, so callers receive offsets into the real document.

The model is asked about the whole window rather than the flagged position. A
locator whose volume is missing or displaced never carries the flag itself --
what surfaces is a court abbreviation in its own parenthetical -- so restricting
the question to the flagged span would make those unreachable.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from eyecite.annotate import SpanUpdater
from eyecite.tokenizers import EDITIONS_LOOKUP
from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from mellea_lrc.core.spans import Span
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.pinpoint_retrieval.evidence_quote import resolve_evidence_quote

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.extraction.adjudication.candidates.reporter_sites import SuspectedLocator

MAX_TOKENS = 320
MAX_REPAIR_TURNS = 2

INSTRUCTION = """
Below is a window of text taken from a legal filing whose PDF extraction is
often damaged: spaces are lost or doubled, line breaks fall mid-citation, and
table columns interleave.

{{reporter_context}}

List every citation LOCATOR that appears in this window. A locator is the
volume, reporter and page that together identify one case.

For each one, report four things:
- text    the locator quoted EXACTLY as written in the window, character for
          character, including any damage
- volume  the volume, as digits
- reporter the reporter
- page    the page, a single number

Damage belongs in "text" and must be repaired in the other three fields:

- "731P.2d902" -> text="731P.2d902", volume="731", reporter="P.2d", page="902"
- "88 Cal. Rptr.\n\n3d 411" -> text="88 Cal. Rptr.\n\n3d 411", volume="88",
  reporter="Cal. Rptr. 3d", page="411"
- "509 N.W.2d\n\n144" -> text="509 N.W.2d\n\n144", volume="509",
  reporter="N.W.2d", page="144"

A line break does not end a locator. If the volume, reporter or page continues
after a newline, quote straight through it, newlines and all.

Rules:
- Quote "text" exactly. Do not tidy its spacing or punctuation. The quote is
  checked against the window and a repaired quote will not be found.
- "text" must contain the volume, the reporter and the page and nothing else -
  no case name, no year, no pin cite. Start at the volume.
- Report a locator ONLY when all three parts are there to be quoted, in one
  run of text, however damaged. Never invent a missing part.
- If a part is genuinely not present - a volume that appears in a different
  table row or column, a page that was never written - report nothing for that
  locator. A part separated only by spaces or line breaks IS present; a part
  somewhere else in the document is not.
- A page is a SINGLE number. If a reporter is followed by a run of counting
  numbers ("1 2 3 4 5 ..."), those are PDF margin line numbers and not a page:
  the page is missing, so report nothing.
- Never report a bare number on its own. A pin cite such as "1187" is part of a
  citation, not a locator.
- Report nothing for statutes, rules, docket numbers, dates, court names, or
  street addresses.
- If the window contains no complete case locator, return an empty list.

window:
{{window}}
""".strip()


class _Locator(BaseModel):
    """One locator the model claims to see, parsed into its three parts.

    All three parts are required. A locator missing any of them is not a
    minimum sufficient identifier -- the span alone would not find the case --
    so the schema refuses it rather than accepting a fragment the caller would
    have to decide what to do with.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    # Non-empty, because a required field a model can satisfy with "" is not
    # required at all: asked for a part that is not in the window, it answers
    # with the empty string rather than omitting the locator.
    volume: Annotated[str, StringConstraints(min_length=1)]
    reporter: Annotated[str, StringConstraints(min_length=1)]
    page: Annotated[str, StringConstraints(min_length=1)]


class _Locators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locators: list[_Locator]


@dataclass(frozen=True, slots=True)
class AdjudicatedLocator:
    """One model-proposed locator, grounded back into the document.

    Every instance carries all three parts, because a locator missing one is
    not reported at all. A fragment cannot be looked up and its span would not
    identify the case, so there is nothing useful to hand back.
    """

    span: Span
    text: str
    volume: str
    reporter: str
    page: str
    match_method: str
    repaired: bool = False
    """Whether a character was read back, rather than only spacing forgiven.

    `833 F.2d l83` reported as page 183 is a **judgement**, not a parse: the
    document does not say 183 anywhere. Recorded so a consumer can tell a
    citation the rules read from one a reviewer repaired, and decline the second
    if it wants to.
    """


# The prompt window is collapsed before the model sees it, which is a
# presentation choice and not a repair: nothing here changes what extraction
# reads, and every span is mapped back before it leaves this module.
_REPEATED_INLINE_WHITESPACE = re.compile(r"[ \t]{2,}")

_NON_CASE_CITE_TYPES = frozenset({"leg_statute", "leg_session", "admin_compilation", "admin_docket"})
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_NUMERIC_TOKEN = re.compile(r"^\d+$")
_MAX_NUMERIC_TOKENS = 3


def implausible_locator(text: str) -> bool:
    """Reject shapes a locator cannot have, whatever the model claims.

    Three failures survive prompting and must be caught deterministically:

    *   A bare number is a pin cite -- it names no reporter.
    *   A reporter trailed by a run of counting numbers is PDF margin line
        numbering swept into the text. A page is one number, so four or more
        numeric tokens means the quote absorbed page furniture.
    *   A quote starting mid-token is a truncated locator. Reading a citation
        out of a table cell, a model may begin at "​. Supp. 3d 1140" and drop the
        "164 F" in front of it; the resulting span is wrong even though every
        character in it is real.
    """
    stripped = text.strip()
    if not stripped or not any(character.isalpha() for character in stripped):
        return True
    if not stripped[0].isalnum():
        return True
    numeric = sum(1 for token in stripped.split() if _NUMERIC_TOKEN.match(token.strip(",.*")))
    return numeric >= _MAX_NUMERIC_TOKENS + 1


def reporter_context(reporter: str) -> str:
    """Describe the flagged reporter using eyecite's own reporter database.

    A model told only "here is some text" must infer what counts as a locator.
    Told that ``S.W.2d`` is the South Western Reporter and that its citations
    read ``<volume> S.W.2d <page>``, it has the shape to look for -- which is
    what lets it recognise an occurrence too damaged for the regex that flagged
    the site in the first place.
    """
    editions = EDITIONS_LOOKUP.get(reporter)
    if not editions:
        return (
            f'"{reporter}" was flagged here but is not a reporter this database knows, '
            f"so this is probably a false flag."
        )
    edition = editions[0]
    cite_type = getattr(edition.reporter, "cite_type", "") or ""
    if cite_type in _NON_CASE_CITE_TYPES:
        # Naming a statute reporter without this caveat legitimises it, and the
        # model then reports "42 U.S.C. § 1983" as a locator despite the rule
        # against statutes -- the description contradicts the instruction.
        return (
            f'"{reporter}" is a statutory or regulatory source, not a case reporter. '
            f"Any citation to it is out of scope: report nothing for it."
        )
    canonical = getattr(edition, "short_name", reporter) or reporter
    reporter_name = getattr(edition.reporter, "name", None)
    described = f'"{reporter}" is {reporter_name}' if reporter_name else f'"{reporter}" is a case reporter'
    # A vendor-neutral reporter numbers its citations by year and sequence, so
    # the generic "<small volume> <reporter> <small page>" example misdescribes
    # it and invites the model to treat a four-digit year as something other
    # than the volume. Show the shape this reporter actually uses.
    example = f"1998 {canonical} 7654321" if cite_type == "specialty_west" else f"731 {canonical} 902"
    return (
        f"{described}. Its citations are written "
        f'"<volume> {canonical} <page>", for example "{example}". '
        f"Report a locator here only if the window actually shows that shape."
    )


def _parse(value: object) -> _Locators:
    return _Locators.model_validate_json(str(value))


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValidationError as error:
        return ValidationResult(result=False, reason=str(error))
    return ValidationResult(result=True)


_UNPARSEABLE = ValidationResult(
    result=False,
    reason=(
        "Return a JSON object with a `locators` list, and an empty list when the window "
        "holds no complete locator. An empty response is not an answer."
    ),
)


def _proposed(ctx: Context) -> tuple[_Locators | None, ValidationResult | None]:
    """The parsed sample, or the failure to report when there is not one.

    A validator that raises takes the whole call down with it, and the model
    does sometimes return an empty string -- which is how a refusal arrived as
    a crash and was counted as a provider failure. Unparseable output is a
    failed requirement: the repair loop gets a reason and another turn, and if
    the budget runs out the caller sees an empty result, which is a decline.
    """
    try:
        return _parse(ctx.last_output().value), None
    except ValidationError:
        return None, _UNPARSEABLE


def _validate_shape(ctx: Context) -> ValidationResult:
    """Reject shapes a locator cannot have, naming the one that failed."""
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    for locator in proposed.locators:
        if implausible_locator(locator.text):
            return ValidationResult(
                result=False,
                reason=(
                    f"{locator.text!r} is not a locator. A locator names a reporter and its "
                    f"page is a single number: a bare number is a pin cite, and a run of "
                    f"counting numbers after a reporter is PDF margin line numbering."
                ),
            )
    return ValidationResult(result=True)


def _identifier(value: str) -> str:
    """Reduce a locator or one of its parts to comparable characters."""
    return _NON_ALPHANUMERIC.sub("", value.lower())


# Letters a scanner reads for digits, and the digits they are read for. Only
# pairs that arise from shape -- `l` for one, `O` for zero -- and no pair that
# would let one reporter become another.
_CONFUSABLE = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5", "b": "8", "z": "2", "g": "9"})


def _repairable(value: str) -> str:
    """The identifier with every confusable character folded onto one side.

    So `833f2dl83` and `833f2d183` compare equal, and `833f2dl83` and
    `833f2d999` still do not. Folding is applied to both the quote and the
    parts, so it admits a *substitution* and never an insertion: the two must
    already be the same length to survive `_validate_parts`.
    """
    return _identifier(value).translate(_CONFUSABLE)


def _validate_parts(ctx: Context) -> ValidationResult:
    """Require the quote to be exactly its own volume, reporter and page.

    This is the check that makes a span minimum sufficient. Stripped of
    punctuation and whitespace, the quoted text must equal the three parts
    concatenated -- no more and no less. It catches both directions of error at
    once: a quote carrying a case name, year or pin cite has more than the
    identifier, and a locator whose volume is missing from the window has less,
    because the model can only satisfy it by inventing one, which the quote
    then contradicts.

    Damage is deliberately invisible here. "58  N.Y .2d  916" and "58 N.Y.2d
    916" reduce to the same characters, so the model may repair the parts while
    quoting the damage verbatim -- which is exactly what it is asked to do.
    """
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    for locator in proposed.locators:
        missing = [
            name
            for name, value in (
                ("volume", locator.volume),
                ("reporter", locator.reporter),
                ("page", locator.page),
            )
            if not _identifier(value)
        ]
        if missing:
            return ValidationResult(
                result=False,
                reason=(
                    f"{locator.text!r} was reported with an empty {' and '.join(missing)}. "
                    f"A locator missing part of itself is not a locator: either the part is "
                    f"written elsewhere in the window and you should quote the whole thing, or "
                    f"it is not there at all and you must not report this locator."
                ),
            )
        quoted = _identifier(locator.text)
        parts = _identifier(locator.volume) + _identifier(locator.reporter) + _identifier(locator.page)
        # Equal outright, or equal once the characters a scanner confuses are
        # folded together. That second reading is why this layer exists: the
        # rules read damage between the parts of a citation and cannot read
        # damage inside one, so `833 F.2d l83` is a citation only a reader
        # recovers. Admitting it here admits a substitution and nothing else --
        # an inserted word changes the length, which no folding hides.
        if quoted != parts and _repairable(locator.text) != _repairable(
            locator.volume + locator.reporter + locator.page
        ):
            return ValidationResult(
                result=False,
                reason=(
                    f"{locator.text!r} does not read as volume={locator.volume!r}, "
                    f"reporter={locator.reporter!r}, page={locator.page!r}. Quote exactly the "
                    f"volume, reporter and page and nothing else, and report those same three "
                    f"parts with the damage repaired. If any part is missing from the window, "
                    f"do not report this locator at all."
                ),
            )
    return ValidationResult(result=True)


# `at`, `p.` and `page` are how a short form writes its pin cite. None belongs
# inside a locator, and no reporter carries one as a whole word.
_PIN_CITE_MARKER = re.compile(r"\b(?:at|p|page)\b\.?", re.IGNORECASE)


def _grounded_more_than_a_locator(text: str, locator: _Locator) -> bool:
    """Whether the span that would enter the record holds more than the locator.

    The requirements check the model's **quote**; what is recorded is the
    **grounded span**, and the whitespace-relaxed match binds to real characters
    in the window, so the two can differ. `214 F.3d at 1071` arrived that way
    once -- a short form, whose `at 1071` is a pin cite and not a first page, and
    which would resolve to the wrong case or to none.

    Two things are refused, and neither refuses a repair. A pin-cite marker in
    the span is a short form. A span whose characters do not *count* the same as
    the three parts has gained or lost something, which an insertion does and a
    substitution does not -- so `833 F.2d l83` reported as page 183 survives,
    which is what `_validate_parts` now also allows over the confusable set.
    """
    if _PIN_CITE_MARKER.search(text):
        return True
    parts = _identifier(locator.volume) + _identifier(locator.reporter) + _identifier(locator.page)
    return len(_identifier(text)) != len(parts)


def _validate_grounding(ctx: Context, window: str) -> ValidationResult:
    """Require every quoted locator to be findable verbatim in the window.

    Naming the offending quote is what makes this repairable. The usual failure
    is a model that silently tidies the damage it was asked to preserve --
    answering "2016 WL 1448829" for a window reading "2016 WL1448829" -- and it
    can only correct that if told which quote failed to resolve.
    """
    proposed, failure = _proposed(ctx)
    if failure is not None:
        return failure
    for locator in proposed.locators:
        if _locate(window, locator.text) is None:
            return ValidationResult(
                result=False,
                reason=(
                    f"{locator.text!r} does not resolve uniquely inside the window. Quote the "
                    f"locator exactly as written there, character for character, without "
                    f"repairing its spacing or punctuation."
                ),
            )
    return ValidationResult(result=True)


def _locate(text: str, quote: str) -> tuple[int, int, str] | None:
    """Find a quote in text, tolerating whitespace the model added or dropped.

    ``resolve_evidence_quote`` handles a quote that differs from the source in
    spacing it can normalize, but not one that differs in whether a space is
    there at all. Asked for a locator's parts *repaired* and its text quoted
    *verbatim*, a model routinely repairs both -- answering "2010 WL 4722279"
    for a window reading "2010 WL4722279" -- and no amount of reminding stops
    it reliably.

    Since a locator's identity is compared with spacing removed anyway, the
    same tolerance belongs here: the quote is matched character by character
    with any amount of whitespace permitted between, so it resolves to the
    real span whichever way the model spaced it. The match is still anchored to
    real characters in real order, so a hallucinated quote does not resolve.
    """
    resolved = resolve_evidence_quote(text, quote)
    if resolved is not None:
        return resolved.span.start, resolved.span.end, resolved.method.value
    squeezed = "".join(quote.split())
    if not squeezed:
        return None
    pattern = re.compile(r"\s*".join(re.escape(character) for character in squeezed))
    match = pattern.search(text)
    if match is None:
        return None
    # A span far longer than the quote means the characters were matched across
    # unrelated text rather than found together.
    if match.end() - match.start() > 2 * len(squeezed):
        return None
    return match.start(), match.end(), "whitespace_relaxed"


def _ground(window: str, collapsed: str, offset: int, quote: str) -> tuple[Span, str, str] | None:
    """Resolve a quote against the collapsed window, then map back to the document."""
    located = _locate(collapsed, quote)
    if located is None:
        return None
    start, end, method = located
    if collapsed != window:
        # Collapsing whitespace is a net insertion going collapsed -> window,
        # so the start binds right and the end binds left; the reverse pairing
        # is off by one collapsed space at segment boundaries.
        updater = SpanUpdater(collapsed, window)
        start = updater.update(start, bisect_right)
        end = updater.update(end, bisect_left)
    return Span(start=offset + start, end=offset + end), window[start:end], method


async def adjudicate_locator(
    masked_text: str,
    site: SuspectedLocator,
    *,
    session: MelleaSession | None = None,
    context: int = 170,
) -> tuple[AdjudicatedLocator, ...]:
    """Return every locator the model finds in one suspected site's window.

    ``masked_text`` is the document with every already-extracted **locator**
    blanked by
    :func:`~mellea_lrc.extraction.adjudication.masking.mask_locator_spans`, which
    preserves every offset, so the spans returned here still index the original.

    Masked because the question is about *this* site and nothing else. A reviewer
    shown a neighbouring citation quotes it and returns a locator the record
    already holds, which is what happened when this was run against the bench.
    Blanking the locator is enough: a quote must carry a volume, a reporter and a
    page, and those characters are gone. Grounding is checked against the same
    window, so a masked citation cannot be quoted even if the model recalls it.

    Locators rather than full spans, because a full span covers text that is not
    a citation -- at `Relaxation.NONE` one span can reach across a whole sentence
    -- and blanking it would hide the very candidates this layer exists to
    find."""
    offset = max(0, site.span_start - context)
    window = masked_text[offset : site.span_end + context // 2]
    collapsed = _REPEATED_INLINE_WHITESPACE.sub(" ", window)

    resolved_session = session or start_mellea_session_from_env()
    result = await run_instruct_ivr(
        resolved_session,
        InstructIvrSpec(
            description=INSTRUCTION,
            user_variables={
                "window": collapsed,
                "reporter_context": reporter_context(site.reporter),
            },
            output_format=_Locators,
            requirements=[
                req("Return a valid locator list.", validation_fn=_validate_schema),
                req(
                    "Every reported locator must name a reporter and a single page.",
                    validation_fn=_validate_shape,
                ),
                req(
                    "Quote only the volume, reporter and page, and report all three.",
                    validation_fn=_validate_parts,
                ),
                # Mellea puts a requirement's description in the prompt, so the
                # wording steers the model as well as gating the result. Phrasing
                # this as exactness alone ("quote it exactly as written") reads as
                # a caution and suppresses damaged locators the layer exists to
                # find: at temperature 0 it consistently lost "WL9137645", whose
                # volume is displaced. Leading with completeness recovers it.
                req(
                    "Report every locator present, quoting each with the window's exact characters.",
                    validation_fn=lambda ctx: _validate_grounding(ctx, collapsed),
                ),
            ],
        ),
        strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
        model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
    )
    # A run that exhausts its repair budget still carries its last sample, and
    # that sample is often mostly right - one bad quote among several good ones
    # is enough to fail a requirement. Discarding it would lose the good ones,
    # so keep whatever survives the same checks applied per locator below. The
    # requirements steer the model; these filters are what make the result safe.
    try:
        proposed = _parse(result.result.value)
    except ValidationError:
        return ()

    found: list[AdjudicatedLocator] = []
    seen: set[tuple[int, int]] = set()
    for locator in proposed.locators:
        if implausible_locator(locator.text):
            continue
        grounded = _ground(window, collapsed, offset, locator.text)
        if grounded is None:
            continue
        span, text, method = grounded
        if _grounded_more_than_a_locator(text, locator):
            continue
        if (span.start, span.end) in seen:
            continue
        seen.add((span.start, span.end))
        found.append(
            AdjudicatedLocator(
                span=span,
                text=text,
                volume=locator.volume,
                reporter=locator.reporter,
                page=locator.page,
                match_method=method,
                repaired=_identifier(text)
                != _identifier(locator.volume) + _identifier(locator.reporter) + _identifier(locator.page),
            )
        )
    return tuple(found)
