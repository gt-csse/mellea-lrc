"""Read the two table-of-authorities citations the primary agent recorded as unreachable.

`exploration/notes/arm-disagreements-23aug.md` on the explorations branch says
the table reader emitted these rows with the columns out of order, so the page
number arrived before the volume and reporter and no tokenizer could recover
them. It also says the gold is right in both cases and stays.

One of them is in our bench and one is not. This prints what each version of
the text actually says, which decides whether that is a missing record or a
converter that has since been fixed.

    uv run python -m exploration.margin_rules.probe_table_cells
"""

from __future__ import annotations

import re
from pathlib import Path

VERSIONS = {
    "v1": Path("data/false-citation-bench/documents_txt"),
    "v1.1": Path("data/false-citation-bench-v1.1/documents_txt"),
    "v2.0": Path("data/false-citation-bench-v2.0/documents_txt"),
}
BODY_MARKER = "--- Plain text ---\n"

PROBES = [
    ("021", r"1013", "Loos v. Lowe's — 796 F. Supp. 2d 1013"),
    ("022", r"9137645", "Doe v. Rose — 2016 WL 9137645"),
]


def body(directory: Path, stem: str) -> str:
    path = next(directory.glob(f"{stem}*.txt"))
    _, marker, text = path.read_text(encoding="utf-8").partition(BODY_MARKER)
    return text if marker else path.read_text(encoding="utf-8")


def main() -> None:
    for stem, needle, label in PROBES:
        print(f"=== {label} (document {stem}) ===")
        for version, directory in VERSIONS.items():
            text = body(directory, stem)
            hits = list(re.finditer(needle, text))
            print(f"  --- {version}: {len(hits)} hit(s) for {needle!r}")
            for hit in hits[:3]:
                start = max(0, hit.start() - 130)
                window = " ".join(text[start : hit.end() + 90].split())
                print(f"      {window!r}")
        print()


if __name__ == "__main__":
    main()
