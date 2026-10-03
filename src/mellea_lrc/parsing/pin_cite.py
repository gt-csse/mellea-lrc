"""One pinpoint grammar for exact source reading and normalization."""

import re

from mellea_lrc.matching.literal import fuzzy_literal

# PDF text extraction may insert horizontal space on either side of a range
# separator. Share this bounded relaxation with the reader so the exact quote
# it captures is also a quote this normalizer can interpret.
PIN_RANGE_JOIN = r"[^\S\r\n]*[-–][^\S\r\n]*"
_HSPACE = r"[^\S\r\n]"
_NUMBER = rf"(?:\*|¶{{1,2}})?{_HSPACE}*\d+(?:{PIN_RANGE_JOIN}\d+)?"
_NOTE = (
    rf"(?:{_HSPACE}+"
    rf"(?:(?:&|and){_HSPACE}*)?"
    rf"(?:n{{1,2}}\.|fn\.?)"
    rf"{_HSPACE}*\d+(?:{PIN_RANGE_JOIN}\d+)?)?"
)
_ITEM = rf"{_NUMBER}{_NOTE}"
# The extractor uses the same syntax to find a bounded candidate. The
# normalizer below still requires every character of that candidate to parse.
AT_JOIN = fuzzy_literal("at ", whitespace=True, newline=True)
PIN_PREFIX = re.compile(rf"^\s*,?\s*(?:{AT_JOIN})?(?P<pin>{_ITEM}(?:,\s*{_ITEM})*)", re.I)
AT_PREFIX = re.compile(AT_JOIN, re.I)
PIN_PART = re.compile(
    rf"(?P<label>\*|¶{{1,2}})?{_HSPACE}*(?P<first>\d+)"
    rf"(?:{PIN_RANGE_JOIN}(?P<last>\d+))?"
    rf"(?:{_HSPACE}+(?:(?:&|and){_HSPACE}*)?(?:n{{1,2}}\.|fn\.?)"
    rf"{_HSPACE}*(?P<footnote>\d+(?:{PIN_RANGE_JOIN}\d+)?))?\Z",
    re.I,
)


def pin_after(source: str, position: int, end: int) -> tuple[int, int] | None:
    """Return offsets of an immediately adjacent pinpoint."""
    match = PIN_PREFIX.match(source[position:end])
    if match is None:
        return None
    return position + match.start("pin"), position + match.end("pin")
