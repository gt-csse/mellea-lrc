"""Is the adjacent-citation bug the same thing as the parallel-citation problem?"""

import contextlib
import io
import sys

sys.path.insert(0, ".")
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

CASES = [
    # The bug #339 names: two citations, no space after the comma.
    "See Brown, 347 U.S. 483,349 U.S. 294 (1955).",
    "See Brown, 347 U.S. 483, 349 U.S. 294 (1955).",
    # A parallel citation: the same case in three reporters.
    "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968).",
    "St. Amant v. Thompson, 390 U.S. 727, 88 S.Ct. 1323 (1968).",
    "Garrison v. Louisiana, 379 U.S. 64, 74, 85 S.Ct. 209, 13 L.Ed.2d 125 (1964).",
    # Doubled spacing, the corpus's own shape.
    "St. Amant v. Thompson,  390 U.S. 727,  731,  88 S.Ct. 1323,  20 L.Ed.2d 262 (1968).",
]

for text in CASES:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
    print(f"{text}")
    for citation in document.citations:
        kind = type(citation.citation).__name__
        pin = getattr(citation.citation, "pin_cite", None)
        extra = getattr(citation.citation, "extra", None)
        print(f"    {kind:<20}{' '.join(citation.matched_text.split())!r:<24} pin={pin!r} extra={extra!r}")
    print()
