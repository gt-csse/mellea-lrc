# Pin-cite page-resolution handoff — 2026-07-20

## Current branch and state

- Branch: `woody/integration-citation-node-model`
- Working tree was clean when this handoff was started.
- Latest implementation commit: `f9268fd feat: resolve pin cites to reporter PDF pages`
- `ExtractedDocument` remains the upstream compatibility boundary. Do not
  redesign it as part of this work.
- The new page resolver is deliberately not wired into the citation graph or
  final assessment yet. It is a narrowly tested component awaiting an opinion
  PDF retrieval/download node.

## Decision reached

Pin-cite validation cannot be implemented by independently checking that a page
number and an excerpt occur somewhere in an opinion. For a citation such as
`162 F.R.D. 418, 422`, the system must first prove that a bounded physical PDF
page corresponds to reporter page 422. Only that bounded page region may later
be passed to proposition-support inference.

The current layer therefore answers only:

> Did we recover a page-faithful opinion region corresponding to the cited
> reporter pin?

It does not answer:

- whether the opinion is otherwise the correct authority;
- whether the page supports the citing proposition;
- whether the citation is legally appropriate.

The earlier whole-document identity inference and substring-based pin-cite LLM
prototype were deleted, together with their tests. Their commits remain in Git
history (`1e7d04a` and `bbcb9a2`), but neither implementation exists at branch
HEAD.

## Current algorithm

Implementation:
`src/mellea_lrc/assessment/fields/pin_cite/page_region.py`

For numeric reporter pins:

1. Read the upstream `ExtractedCitation`; do not reconstruct citation fields.
2. Obtain the candidate's CourtListener-hosted opinion PDF URL from
   `CourtListenerCitationRecord.extra_data`, preferring
   `filepath_pdf_harvard`, then `filepath_pdf_scan`.
3. Extract an ordered tuple of physical PDF pages with PDFium.
4. Locate a physical page whose first extracted line exactly equals the
   citation's reporter start page.
5. Treat that page as the verified base anchor.
6. Resolve the pin by reporter-page offset from the anchor.
7. Return only the computed physical page or page range in
   `PinCitePageRegion.pages`.

The mapping basis is recorded as
`located_reporter_start_page_plus_physical_page_offset`. Abbreviated page
ranges such as `463-64` expand to `463-464`.

Important: the resolver searches page headers to locate the base-page anchor;
it does not accept a page number merely because that number appears somewhere
in the opinion body.

## Result model

Successful output is shaped like:

```python
PinCitePageRegion(
    status=PinCitePageRegionStatus.RESOLVED,
    pin_cite="422",
    reporter_page_start=422,
    reporter_page_end=422,
    reporter_base_pdf_page=1,
    pdf_page_start=5,
    pdf_page_end=5,
    pages=(
        ReporterPage(
            reporter_page_number=422,
            pdf_page_number=5,
            text="422\n...text only from reporter page 422...",
            printed_page_label_observed=True,
        ),
    ),
    mapping_basis="located_reporter_start_page_plus_physical_page_offset",
    message="pin cite resolved to a bounded reporter-page region",
)
```

Non-success statuses are explicit:

- `not_applicable`
- `unsupported_pin_cite`
- `unverified_pdf`
- `out_of_range`
- `text_unavailable`

Westlaw star pages such as `at *3` currently return
`unsupported_pin_cite`; they must not be forced through reporter-page
arithmetic.

## Real-data findings

The available snapshots contained 36 unique citations satisfying all of these
conditions:

- exact locator lookup returned `found`;
- the citation had a numeric reporter pin;
- the retrieved cluster exposed a Harvard or scan PDF path.

Results:

- 33/36 resolved to bounded page regions (91.7%).
- 3/36 returned `unverified_pdf`:
  - `905 P.2d 559, 563`
  - `678 N.Y.S.2d 611, 612`
  - `578 F.2d 411, 415`

These failures are meaningful rather than implementation misses. At least the
first uses a PDF paginated in a parallel state reporter rather than the cited
Pacific Reporter. Do not infer a constant offset between parallel reporters;
their page breaks need not align.

The resolver originally assumed the PDF began at the cited case's base page.
Real data disproved that assumption: the PDF for
`5 Cal. App. 5th 1069, 1091` begins at reporter page 1055. The implementation
now locates page 1069 inside the PDF and correctly maps reporter page 1091 to
physical PDF page 37.

Four opt-in remote sanity cases download real CourtListener PDFs and verify the
page-local text:

| Citation | Physical PDF page |
| --- | ---: |
| `162 F.R.D. 418, 422` | 5 |
| `307 F.R.D. 1, 7` | 7 |
| `5 Cal. App. 5th 1069, 1091` | 37 |
| `556 U.S. 662, 678` | 17 |

Run them with:

```bash
uv run pytest tests/smoke/test_pin_cite_page_region_remote.py \
  --run-remote-smoke --no-cov
```

## Docling conclusion

Docling can preserve physical pages, but the existing whole-document call
`export_to_text()` flattens them. The new
`extract_pdf_pages_with_docling()` adapter calls
`export_to_text(page_no=...)` in page order.

Current policy:

- Use PDFium for born-digital CourtListener/Harvard PDFs. It is fast and
  already page-faithful.
- Use Docling only as an OCR fallback for `text_unavailable` results.
- Do not run full Docling conversion merely to preserve pages that PDFium
  already exposes. A five-page real conversion was materially slower and was
  stopped during investigation.

Docling fallback is implemented and unit-tested as an adapter, but it has not
yet been exercised end-to-end on a real textless cited opinion.

## Verification at handoff

- `uv run ruff check .` passed.
- Changed files passed `ruff format --check`.
- Ordinary suite: 224 passed, 15 opt-in skips.
- Real CourtListener PDF sanity suite: 4 passed.
- The 36-case snapshot research run produced 33 resolved and 3
  `unverified_pdf` results.

## Relevant files

- `src/mellea_lrc/assessment/fields/pin_cite/page_region.py`
  - page-region model, pin parsing, base-page anchoring, PDFium extraction.
- `src/mellea_lrc/assessment/fields/pin_cite/__init__.py`
  - narrow public exports.
- `src/mellea_lrc/preprocessing/docling.py`
  - page-preserving Docling OCR adapter.
- `tests/test_pin_cite_page_region.py`
  - deterministic mapping, ranges, rejection, and PDF URL tests.
- `tests/smoke/test_pin_cite_page_region_remote.py`
  - real public CourtListener PDF checks.
- `docs/architecture/pin-cite-page-resolution.md`
  - stable architectural explanation and evaluation summary.

## Recommended next steps

1. Add a narrow opinion-PDF retrieval boundary to the CourtListener client.
   It should consume the selected `RetrievedCandidate`, expose download
   provenance/status, and return bytes without placing raw PDF content in JSON
   snapshots.
2. Add a citation-node operation for page-region recovery after an exact
   locator-found result. Keep download, page extraction, and page mapping as
   separate traceable operations if their failure statuses need independent
   visibility.
3. Exercise the Docling adapter on several genuinely textless or scanned PDFs.
   Only then decide how OCR status and cost should appear in the graph.
4. After page-region recovery is stable, add the LLM proposition-support node.
   Its document input should be only `PinCitePageRegion.text` plus the upstream
   citing context—not the entire cited opinion.
5. Design WL star-page recovery separately. A commercial WL backend or a
   source retaining star-page markers may be required; do not conflate star
   pages with physical PDF pages.
6. Investigate parallel-reporter pagination as its own evidence problem. Keep
   `unverified_pdf` until an explicit cross-publication page map exists.

## Cautions for the next session

- Do not reintroduce whole-document identity inference at this layer.
- Do not validate a page using `page_number in document_text`.
- Do not treat matching proposition text elsewhere in the opinion as proof of
  a pin cite.
- Do not silently translate parallel-reporter pages using arithmetic.
- Do not serialize opinion PDF bytes or full page text into the existing
  retrieval candidate record without first designing the citation-node
  tracing/storage boundary.
