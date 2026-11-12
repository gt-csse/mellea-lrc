"""Pinpoint grammar shared by typed citation-field normalization."""

import re

# PDF text may insert horizontal space around range separators. Newlines
# within a range are not normalized into a continuous written target.
PIN_RANGE_JOIN = r"[^\S\r\n]*[-–][^\S\r\n]*"
_HSPACE = r"[^\S\r\n]"

# The settled reader's whitespace-relaxed literal "at " is exactly this
# pattern. Keep the field grammar independent of extraction or workflows.
_AT = r"at\s*"
_LABEL = r"\*|¶{1,2}"
_NOTE_START = rf"{_HSPACE}+(?:(?:&|and){_HSPACE}*)?(?:n{{1,2}}\.|fn\.?){_HSPACE}*"
_NUMBER = rf"(?:{_LABEL})?{_HSPACE}*\d+(?:{PIN_RANGE_JOIN}\d+)?"
_ITEM = rf"{_NUMBER}(?:{_NOTE_START}\d+(?:{PIN_RANGE_JOIN}\d+)?)?"
AT_PREFIX = re.compile(_AT, re.I)
PIN_PREFIX = re.compile(rf"^\s*,?\s*(?:{_AT})?(?P<pin>{_ITEM}(?:,\s*{_ITEM})*)", re.I)
PIN_PART = re.compile(
    rf"(?P<label>{_LABEL})?{_HSPACE}*(?P<first>\d+)"
    rf"(?:{PIN_RANGE_JOIN}(?P<last>\d+))?"
    rf"(?:{_NOTE_START}(?P<footnote>\d+(?:{PIN_RANGE_JOIN}\d+)?))?\Z",
    re.I,
)
