"""Hunt the masked residue for return visits the extractor did not produce.

The locator hunt looks for a citation nobody found. This looks for a *return
visit* nobody found: an `id.`, a `supra`, a short form, or a case referred to by
party name. They matter for a different reason. A missed locator loses a claim;
a missed return visit loses a claim **and** lets the next `id.` attach itself to
whatever citation happens to precede it, which is how the one misattribution in
this corpus happened.

Same order as the locator hunt: extract with the widest deterministic setting,
mask every citation found, and read only what is left.

Three shapes, and they are not equally reliable:

    id./ibid./supra   a literal, and unambiguous when it survives masking
    short form        `<volume> <reporter> at <page>`, the shape eyecite parses
    party name        a plaintiff or defendant from a citation this document
                      already makes, appearing again in the residue

The third is the one worth having and the noisiest. A name is only looked for
if the document itself cites a case with that party, so it cannot invent a
reference -- but a common surname will match prose about anyone.

    uv run python -m exploration.locator_recall.hunt_secondary
    uv run python -m exploration.locator_recall.hunt_secondary --show 40
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.adjudication import mask_full_spans
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

BACK_REFERENCE = re.compile(r"\b(?:[Ii]d\.|[Ii]bid\.|supra)", re.MULTILINE)
# A back-reference carrying a paragraph pin cite points into a document's
# numbered allegations, not at a page of a reporter. Those are record
# references -- the filing's own complaint, the opposing brief, an ECF entry --
# and they are out of scope: they identify nothing to anyone outside this
# case's docket, so there is no authority to attach them to and nothing to
# verify them against. Labelled rather than dropped, so the count stays honest.
RECORD_REFERENCE = re.compile(r"^[\s.,)]*(?:\u00b6|\bat\s+\u00b6)", re.MULTILINE)
SHORT_FORM = re.compile(r"\b\d{1,4}\s+[A-Z][A-Za-z.'’ ]{0,18}[A-Za-z.]\s+at\s+\*?\d{1,5}")
# Corporate and procedural tails carry no identity of their own, and a party
# referred to later is referred to without them.
TAIL = re.compile(
    r"\b(?:Corp|Corporation|Inc|LLC|LLP|PLLC|Ltd|Co|Company|L\.P|N\.A|et al)\b\.?,?\s*$",
    re.IGNORECASE,
)
# A party name has to be distinctive enough that finding it again means
# something. One short word is not.
MIN_NAME = 6


def party_names(document) -> set[str]:
    """Distinctive plaintiff and defendant names from this document's citations."""
    names: set[str] = set()
    for citation in document.citations:
        for field in ("plaintiff", "defendant"):
            value = getattr(citation.citation, field, None)
            if not value:
                continue
            trimmed = TAIL.sub("", " ".join(str(value).split())).strip(" ,.")
            if len(trimmed) >= MIN_NAME and not trimmed.isdigit():
                names.add(trimmed)
    return names


def main() -> int:
    """Report every back-reference shape surviving the mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--show", type=int, default=25)
    args = parser.parse_args()

    documents = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    totals: Counter = Counter()
    findings: list[tuple[str, str, str, str]] = []

    for path in sorted(documents.glob("*.txt")):
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=Relaxation.FULL)
        masked = mask_full_spans(document)
        stem = path.stem[:20]

        for label, pattern in (("back-reference", BACK_REFERENCE), ("short form", SHORT_FORM)):
            for match in pattern.finditer(masked):
                shown = label
                if label == "back-reference" and RECORD_REFERENCE.match(
                    masked[match.end() : match.end() + 12]
                ):
                    shown = "record reference (out of scope)"
                totals[shown] += 1
                window = " ".join(text[max(0, match.start() - 90) : match.end() + 60].split())
                findings.append((stem, shown, match.group(), window))

        for name in sorted(party_names(document)):
            for match in re.finditer(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", masked):
                # A bare name in prose is not a citation -- a filing names its
                # own parties on every page. A name carrying a pin cite is:
                # `Caraway , at 1301` is a reference to a case this document
                # cites, written in the form eyecite is supposed to produce a
                # ReferenceCitation for.
                trailing = masked[match.end() : match.end() + 18]
                pinned = re.match(r"^[\s,]*at\s+\*?\d", trailing)
                label = "party name + pin cite" if pinned else "party name"
                totals[label] += 1
                window = " ".join(text[max(0, match.start() - 80) : match.end() + 80].split())
                findings.append((stem, label, name, window))

    print("| shape surviving the mask | count |")
    print("|---|---:|")
    for label in (
        "back-reference",
        "record reference (out of scope)",
        "short form",
        "party name + pin cite",
        "party name",
    ):
        print(f"| {label} | {totals[label]} |")
    print(f"\ntotal: {sum(totals.values())}\n")

    for stem, label, matched, window in findings[: args.show]:
        print(f"  [{stem:<20}] {label:<16}{matched!r}")
        print(f"      {window[:150]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
