# Pin-cite page resolution

Pin-cite assessment must not search an entire opinion for a page number and a
passage independently. Before any proposition-support inference runs, the
system must derive a bounded page region whose relationship to the cited pin is
auditable.

## Reporter-page method

For a reporter citation such as `162 F.R.D. 418, 422`:

1. Obtain the CourtListener-hosted opinion PDF from retrieved candidate
   metadata (`filepath_pdf_harvard`, then `filepath_pdf_scan`).
2. Preserve physical PDF pages during text extraction.
3. Locate a physical page whose first printed line equals the citation's base
   reporter page (`418`). The PDF may contain earlier reporter pages, so this
   anchor is not necessarily physical page one.
4. Map the pin by physical-page offset: `422 - 418` pages after the verified
   anchor.
5. Give only the resulting page or page range to later semantic inference.

This is a page-header anchor check. It is intentionally different from testing
whether `"422"` occurs somewhere in the opinion.

Abbreviated ranges such as `463-64` are expanded to `463-464`. Westlaw star
pages such as `at *3` do not use reporter-page arithmetic and currently return
`unsupported_pin_cite`.

## Real-data evaluation (2026-07-19)

The resolver was evaluated against all 36 unique, locator-found, numeric-pin
citations in the available snapshots that also exposed a CourtListener-hosted
Harvard/scan PDF:

- 33 resolved to bounded physical pages.
- 3 remained `unverified_pdf` because the PDF did not expose the cited
  reporter's pagination: `905 P.2d 559, 563`, `678 N.Y.S.2d 611, 612`, and
  `578 F.2d 411, 415`.

The first failure is a concrete parallel-pagination case: the retrieved PDF is
paginated in the Arizona reporter (starting at page 550), while the citation is
to Pacific Reporter page 559. The resolver deliberately does not assume a
constant offset between parallel reporters.

Remote sanity fixtures verify four real mappings, including a PDF that starts
before the cited case:

| Citation | Reporter pin | Physical PDF page |
| --- | ---: | ---: |
| `162 F.R.D. 418, 422` | 422 | 5 |
| `307 F.R.D. 1, 7` | 7 | 7 |
| `5 Cal. App. 5th 1069, 1091` | 1091 | 37 |
| `556 U.S. 662, 678` | 678 | 17 |

## Docling boundary

Born-digital Harvard reporter PDFs already expose page-faithful text through
PDFium, so full Docling conversion is unnecessary for the successful cases and
is substantially more expensive. Docling remains the OCR fallback for scanned
or textless pages.

When Docling is used, do not call one whole-document `export_to_text()` and
discard pagination. Use `extract_pdf_pages_with_docling`, which calls
`export_to_text(page_no=...)` for every physical page and returns an ordered page
tuple. A later node may retry `text_unavailable` results with this adapter.

Whole-document identity inference and the prior substring-based pin-cite
prototype were removed. This layer expresses only whether a reliable pin-cited
page region was recovered; it does not yet decide whether that region supports
the citing proposition.
