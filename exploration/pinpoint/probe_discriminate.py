"""What separates a parallel citation from two cases written side by side.

**And the answer is that extraction should not try.** This file argued its way
to the wrong conclusion first, and the correction is the useful part.

The evidence below is real. Coinciding full spans do not tell a parallel cite
from two different cases:

    parallel, one case         11-78, 11-78, 12-78
    two cases, same reporter    4-44,  4-44      <- Brown I and Brown II
    two cases, diff reporters   4-61,  4-61
    two cases, semicolon        0-24, 26-51

`See Brown, 347 U.S. 483, 349 U.S. 294 (1955)` has one case name, one year
parenthetical, identical spans, and an `extra` holding nothing but a citation.
It is structurally identical to a parallel cite and is two decisions.

From that this file concluded span coincidence was unusable. That was judging
it against the wrong job. It is a poor rule for deciding **identity** and a good
signal for reporting **candidacy**, and those are different layers:

- **Extraction reports co-location.** Citations whose full spans coincide are a
  candidate parallel group. Deterministic, no lookup, no claim about whether
  they name the same case. It has no false negatives -- a parallel cite always
  co-locates -- which is the property a candidate signal needs.
- **Validation decides identity**, because it is the layer with the data.
  `CourtListenerOpinionCluster` carries a `cluster_id` and its own list of
  reporter citations, so two citations resolving to one cluster are one
  authority and two clusters are two. Brown I and Brown II separate there, on
  evidence, rather than here on a regex.

That also disposes of the two rules this file previously proposed. Reading
`extra` and checking reporter jurisdictions were both attempts to settle
identity during extraction, which is not extraction's question. Neither is
needed if the group is only a candidate.

Two findings from the failed attempts are still worth keeping.

**Party metadata is corrupted and must not be trusted.**
`is_parallel_citation` fires whenever two adjacent full citations share a
full-span start and copies plaintiff, defendant and year onto the later one --
without checking they are parallel. `652 F.3d 1202` is labelled
`defendant='Iqbal'` when it is Starr. Anything keyed on that field inherits the
error, here or elsewhere.

**Reporters_db jurisdiction lists are too broad to carry weight.** `U.S.` lists
`us;supreme.court` and also the 9th, 10th and other circuits, so it intersects
`F.` and would admit Iqbal and Starr as one case.

    uv run python -m exploration.pinpoint.probe_discriminate
"""

from __future__ import annotations

import contextlib
import io
import re

from mellea_lrc.extraction import Relaxation, extract_from_plain_text

CASES = [
    ("PARALLEL, one case", "St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)."),
    ("TWO CASES, same reporter", "See Brown, 347 U.S. 483, 349 U.S. 294 (1955)."),
    ("TWO CASES, different reporters", "See Iqbal, 556 U.S. 662, 678, and Starr, 652 F.3d 1202 (2011)."),
    ("TWO CASES, string cite", "Lacey, 693 F.3d 896, 912; Garmon, 828 F.3d 837, 843."),
]

LEFTOVER = re.compile(r"[^\s,;.()\[\]*\-–'’]")


def extra_is_only_citations(extra: str) -> bool:
    """Whether `extra` decomposes entirely into citations.

    Written as a question to eyecite rather than as a pattern. A hand-built
    regex for "looks like citations" was tried first and got both answers
    wrong: it rejected `20 L.Ed.2d 262`, because the series digit inside the
    reporter is not something a letters-only character class can hold.

    Asking the extractor what it finds, and then checking that nothing but
    punctuation is left over, needs no such guess.
    """
    if not extra.strip():
        return False
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        found = extract_from_plain_text(extra, relaxation=Relaxation.FULL)
    remaining = list(extra)
    for citation in found.citations:
        remaining[citation.full_span.start : citation.full_span.end] = " " * (
            citation.full_span.end - citation.full_span.start
        )
    return bool(found.citations) and not LEFTOVER.search("".join(remaining))


def main() -> None:
    """Print each shape with the three signals side by side."""
    for label, text in CASES:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        found = sorted(document.citations, key=lambda c: c.full_span.start)
        print(f"--- {label}\n    {text}")
        for citation in found:
            inner = citation.citation
            extra = str(getattr(inner, "extra", "") or "")
            print(
                f"      {' '.join(citation.matched_text.split())!r:<20}"
                f" span={citation.full_span.start}-{citation.full_span.end}"
                f" reporter={getattr(inner, 'reporter', None)!r:<10}"
                f" extra_is_pure={extra_is_only_citations(extra)}"
            )
        reporters = [str(getattr(c.citation, "reporter", "")) for c in found]
        distinct = len(set(reporters)) == len(reporters)
        pure = any(extra_is_only_citations(str(getattr(c.citation, "extra", "") or "")) for c in found)
        print(f"      distinct reporters: {distinct}   an extra is only citations: {pure}")
        print(f"      MERGE: {distinct and pure}\n")


if __name__ == "__main__":
    main()
