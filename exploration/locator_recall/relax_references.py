"""Relax the whitespace in eyecite's reference pin-cite regex, and measure it.

`reference_pin_cite_re` composes a party name with `PIN_CITE_REGEX`, whose
separators are written as a single optional literal space, `\\ ?`. PDF extraction
of justified text leaves doubled spaces, so a reference vanishes:

    'Caraway ,  at  1301'   doubled   no match
    'Rafiyev at  861.'      doubled   no match
    'Bell at 546.'          single    matches

This is the same defect the reporter joins had, in a place the existing
relaxation does not reach: `mellea_lrc.extraction.relaxation` rebuilds the
reporter *extractors*, while reference extraction runs later in `find.py`
against a regex the tokenizer never sees.

The relaxation here is deliberately narrow. Each literal space becomes
horizontal whitespace only -- `[^\\S\\r\\n]` -- so a doubled or tabbed separator
matches and a paragraph break still does not. Doubled spaces are the defect
actually observed; crossing a newline is a wider claim that this corpus gives no
reason to make, and the branch's own history is that the bounded form was right
and the unbounded one bought errors.

It is applied by patching the module global for the duration of a run, because
`reference_pin_cite_re` reads it at call time. That is a blunt instrument and
the reason this lives in exploration rather than in the library: it is global,
so it cannot vary with `Relaxation` the way the reporter joins do.

    uv run python -m exploration.locator_recall.relax_references
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from collections import Counter
from pathlib import Path

import eyecite.regexes

from exploration.locator_recall.fuzzy_sites import body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text
from mellea_lrc.extraction.citation_tree import build_citation_tree

HORIZONTAL_OPTIONAL = r"[^\S\r\n]*"
HORIZONTAL_REQUIRED = r"[^\S\r\n]+"


def relaxed_pin_cite(pattern: str) -> str:
    """The same pattern with its literal spaces widened to horizontal whitespace."""
    # Order matters: the optional form contains the mandatory one.
    widened = pattern.replace("\\ ?", HORIZONTAL_OPTIONAL)
    return widened.replace("\\ ", HORIZONTAL_REQUIRED)


@contextlib.contextmanager
def references_relaxed():
    """Swap in the relaxed pin-cite regex for the duration of the block."""
    original = eyecite.regexes.PIN_CITE_REGEX
    eyecite.regexes.PIN_CITE_REGEX = relaxed_pin_cite(original)
    try:
        yield
    finally:
        eyecite.regexes.PIN_CITE_REGEX = original


def survey(documents: list[Path], level: Relaxation) -> tuple[Counter, list[tuple[str, str]], dict]:
    """Extract every document and record what the references and the tree do."""
    totals: Counter = Counter()
    references: list[tuple[str, str]] = []
    attribution: dict[tuple[str, int], str] = {}

    for path in documents:
        text = body(path)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = extract_from_plain_text(text, relaxation=level)
            tree = build_citation_tree(document)

        for citation in document.citations:
            totals[type(citation.citation).__name__] += 1
            if type(citation.citation).__name__ == "ReferenceCitation":
                references.append((path.stem[:20], " ".join(citation.matched_text.split())))

        for citation in document.citations:
            if type(citation.citation).__name__ in ("FullCaseCitation", "DocketCitation"):
                attribution[("SPAN", path.stem, citation.locator_span.start)] = (
                    f"{citation.locator_span.end}|{' '.join(citation.matched_text.split())}"
                )

        totals["authorities"] += len(tree.authorities)
        totals["unattributed"] += len(tree.unattributed)
        for authority in tree.authorities:
            for occurrence in authority.occurrences:
                key = (path.stem[:12], occurrence.citation.span.start)
                attribution[key] = " ".join(authority.root.matched_text.split())[:30]
    return totals, references, attribution


def main() -> int:
    """Compare extraction with and without the relaxed reference regex."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--level", default="full", choices=[level.value for level in Relaxation])
    args = parser.parse_args()

    directory = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    documents = sorted(directory.glob("*.txt"))
    level = Relaxation(args.level)
    print(f"{len(documents)} documents, relaxation={level.value}\n")

    before, before_refs, before_attr = survey(documents, level)
    with references_relaxed():
        after, after_refs, after_attr = survey(documents, level)

    kinds = sorted(set(before) | set(after))
    print(f"{'':<24}{'before':>10}{'after':>10}{'change':>9}")
    for kind in kinds:
        delta = after[kind] - before[kind]
        mark = f"{delta:+d}" if delta else ""
        print(f"{kind:<24}{before[kind]:>10}{after[kind]:>10}{mark:>9}")

    gained = [r for r in after_refs if r not in before_refs]
    lost = [r for r in before_refs if r not in after_refs]
    print(f"\nreferences gained: {len(gained)}")
    for stem, matched in gained:
        print(f"  [{stem:<20}] {matched!r}")
    print(f"references lost: {len(lost)}")
    for stem, matched in lost:
        print(f"  [{stem:<20}] {matched!r}")

    spans_before = {k: v for k, v in before_attr.items() if k[0] == "SPAN"}
    spans_after = {k: v for k, v in after_attr.items() if k[0] == "SPAN"}
    print(
        f"\nlocator spans: {len(spans_before)} before, {len(spans_after)} after, "
        f"identical: {spans_before == spans_after}"
    )

    moved = {
        key: (before_attr[key], after_attr[key])
        for key in set(before_attr) & set(after_attr)
        if key[0] != "SPAN" and before_attr[key] != after_attr[key]
    }
    print(f"\nattributions that changed authority: {len(moved)}")
    for (stem, start), (was, now) in sorted(moved.items()):
        print(f"  [{stem:<12}] at {start}: {was!r} -> {now!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
