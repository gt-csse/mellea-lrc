"""Read the pin cites the whitespace widening does not recover.

Widening the literal spaces in eyecite's pin-cite pattern takes the citations
carrying a bare page in `extra` from 68 to 42. This prints the 42 with the text
that follows them, so the shapes can be read and the decision about a model can
be made against what is actually there.

Nothing is judged automatically. Each row is the citation, what eyecite put in
`extra`, and the raw characters after the citation's locator -- the last being
what any rule would have to read.

    uv run python -m exploration.pinpoint.read_remaining
"""

from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

from exploration.pinpoint.relax_pin_cites import pin_cites_relaxed
from exploration.pinpoint.survey_extra import CARRIES_PIN, PARALLEL, PIN_SHAPED, body
from mellea_lrc.extraction import Relaxation, extract_from_plain_text

AFTER = 34


def main() -> int:
    """Print every citation still carrying a bare page in `extra`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--show", type=int, default=60)
    args = parser.parse_args()

    documents = args.documents or Path("data/false-citation-bench-locator-only-v2.0/documents_txt")
    rows: list[tuple[str, str, str, str]] = []
    reporters: Counter = Counter()

    with pin_cites_relaxed():
        for path in sorted(documents.glob("*.txt")):
            text = body(path)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                document = extract_from_plain_text(text, relaxation=Relaxation.FULL)

            for citation in document.citations:
                if type(citation.citation).__name__ not in CARRIES_PIN:
                    continue
                extra = getattr(citation.citation, "extra", None)
                if not extra:
                    continue
                written = str(extra).strip()
                if PARALLEL.search(written) or not PIN_SHAPED.match(written):
                    continue
                reporters[str(getattr(citation.citation, "reporter", ""))] += 1
                rows.append(
                    (
                        path.stem[:16],
                        " ".join(citation.matched_text.split())[:24],
                        written[:20],
                        text[citation.locator_span.end : citation.locator_span.end + AFTER],
                    )
                )

    print(f"{len(rows)} citations still carrying a bare page in `extra`\n")
    print(f"reporters: {dict(reporters.most_common(10))}\n")
    print(f"{'document':<18}{'citation':<26}{'extra':<22}raw text after the locator")
    for stem, matched, extra, after in rows[: args.show]:
        print(f"{stem:<18}{matched:<26}{extra:<22}{after!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
