"""Cache Docling's page layout so margin rules can be tried without reconverting.

Converting the benchmark takes minutes and some of it needs OCR, which makes
iterating on a layout rule impossible if every variant pays for a conversion.
Everything a margin rule reads is in the text items -- their string, label,
content layer, page and bounding box -- so this records exactly that, once, and
every experiment afterwards runs against the cache in seconds.

The format matches `tests/fixtures/margin_pages.json`, so a page that turns out
to be interesting can be lifted straight into a regression fixture.

    uv run python scripts/dump_page_layout.py

Writes `data/page-layout-cache.json`. Run it again to refresh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _items(document: object) -> list[dict]:
    """Every text item, flattened to what a layout rule can see."""
    recorded = []
    for item in getattr(document, "texts", None) or []:
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            continue
        box = provenance[0].bbox
        recorded.append(
            {
                "text": getattr(item, "text", "") or "",
                "label": item.label.value,
                "layer": item.content_layer.value,
                "page": provenance[0].page_no,
                "l": box.l,
                "t": box.t,
                "r": box.r,
                "b": box.b,
            }
        )
    return recorded


def main() -> int:
    """Convert every PDF once and record its layout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", type=Path, default=Path("data/false-citation-bench/documents_pdf"))
    parser.add_argument("--out", type=Path, default=Path("data/page-layout-cache.json"))
    args = parser.parse_args()

    from docling.document_converter import DocumentConverter

    sources = sorted(args.pdfs.glob("*.pdf"))
    if not sources:
        print(f"{args.pdfs}: no PDFs found", file=sys.stderr)
        return 1

    converter = DocumentConverter()
    recorded = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.name}", file=sys.stderr, flush=True)
        result = converter.convert(str(source))
        items = _items(result.document)
        recorded.append(
            {
                "source": source.stem,
                "pages": sorted({item["page"] for item in items}),
                "items": items,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recorded), encoding="utf-8")
    print(f"wrote {args.out} ({len(recorded)} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
