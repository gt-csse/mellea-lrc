"""Shared patterns for written citation dates."""

import re

YEAR_RE = re.compile(r"(?<!\d)(?:1[6789]\d{2}|20\d{2}|21\d{2})(?!\d)")
FULL_DATE_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>(?:1[6789]|20|21)\d{2})\b",
    re.I,
)
MONTH_YEAR_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<year>(?:1[6789]|20|21)\d{2})\b",
    re.I,
)
