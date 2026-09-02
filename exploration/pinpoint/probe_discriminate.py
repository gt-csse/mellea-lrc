"""What separates a parallel citation from two cases written side by side.

The parallel-citation problem is a grouping problem: every reporter of a
parallel cite is extracted, but as its own authority, so one case becomes
three. Combining them is the right shape of fix -- a post-eyecite step reading
`extra` -- but the discriminator is harder than it looks, and three candidate
signals were tried before one held.

**Coinciding full spans do not discriminate.** They coincide for a parallel
cite and equally for two different cases separated by a comma:

    parallel, one case         11-78, 11-78, 12-78
    two cases, same reporter    4-44,  4-44
    two cases, diff reporters   4-61,  4-61
    two cases, semicolon        0-24, 26-51     <- only this one separates

Merging on span overlap would have merged `347 U.S. 483` with `349 U.S. 294`:
Brown I and Brown II, different cases, different years, different holdings.

**Shared courts from reporters_db do not discriminate.** `U.S.` lists
`us;supreme.court` and also `us:c9;court.appeals`, `us:c10;court.appeals` and
more, so it intersects `F.` and the rule admits Iqbal and Starr as one case.
The jurisdiction lists are far too broad to carry this weight.

**Party metadata does not discriminate, and is actively misleading.**
`is_parallel_citation` fires whenever two adjacent full citations share a
full-span start, and copies plaintiff, defendant and year onto the later one --
without checking that they are parallel at all. So `652 F.3d 1202` is labelled
`defendant='Iqbal'` when it is Starr. Every one of the four shapes reports a
single party set, because eyecite has overwritten the difference.

**What does hold: never merge citations sharing a reporter.** A case has one
first page in one reporter, so two `U.S.` cites are two cases. That refuses
Brown I/II and the string cite, needs no external data, and cannot be wrong.

**What is still open** is the case it does not decide: two different cases in
different reporters, side by side. `extra` carries the evidence -- Iqbal's is
`'and Starr, 652 F.3d 1202'`, with a case name in it, where a parallel cite's is
`'88 S.Ct. 1323, 20 L.Ed.2d 262'` and is nothing but citations. Asking eyecite
to decompose `extra` and checking what is left over gets three of the four
shapes right and still merges Iqbal with Starr, because the leftover test is
not yet reading the case name as disqualifying.

So the combine step should be built refusing by default: merge only on a
positive signal, and treat an `extra` containing anything that is not a
citation -- including a party name -- as a refusal to merge.

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
        remaining[citation.span.start : citation.span.end] = " " * (citation.span.end - citation.span.start)
    return bool(found.citations) and not LEFTOVER.search("".join(remaining))


def main() -> None:
    """Print each shape with the three signals side by side."""
    for label, text in CASES:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        found = sorted(document.citations, key=lambda c: c.span.start)
        print(f"--- {label}\n    {text}")
        for citation in found:
            inner = citation.citation
            extra = str(getattr(inner, "extra", "") or "")
            print(
                f"      {' '.join(citation.matched_text.split())!r:<20}"
                f" span={citation.span.start}-{citation.span.end}"
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
