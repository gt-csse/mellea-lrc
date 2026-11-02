"""Pinpoint grammar shared by typed citation-field normalization."""

import re

# PDF text may insert horizontal space around range separators. Newlines
# within a range are not normalized into a continuous written target.
PIN_RANGE_JOIN = r"[^\S\r\n]*[-–][^\S\r\n]*"
_HSPACE = r"[^\S\r\n]"

# The settled reader's whitespace-relaxed literal "at " is exactly this
# pattern. Keep the field grammar independent of extraction or workflows.
AT_PREFIX = re.compile(r"at\s*", re.I)
PIN_PART = re.compile(
    rf"(?P<label>\*|¶{{1,2}})?{_HSPACE}*(?P<first>\d+)"
    rf"(?:{PIN_RANGE_JOIN}(?P<last>\d+))?"
    rf"(?:{_HSPACE}+(?:(?:&|and){_HSPACE}*)?(?:n{{1,2}}\.|fn\.?)"
    rf"{_HSPACE}*(?P<footnote>\d+(?:{PIN_RANGE_JOIN}\d+)?))?\Z",
    re.I,
)
