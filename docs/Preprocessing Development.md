---

## tags: [preprocessing, docling, tesseract, ocr, layer-2]
status: active
created: 2026-06-24

# Preprocessing Development

This document explains how we turn 3rd-layer court filings into 2nd-layer text,
why we moved away from CourtListener plain-text exports, and how to evaluate
preprocessing changes systematically.

See also [Benchmark](./Benchmark.md) for the three-layer dataset design and
[Extraction Model Development](./Extraction%20Model%20Development.md) for how
preprocessing quality affects citation extraction.

---

## Current Default: Docling + Tesseract CLI OCR

Our default preprocessing path is implemented in
`src/mellea_lrc/preprocessing/docling.py` via `preprocess_with_docling()`.

For PDFs we configure Docling with:

- `**PdfPipelineOptions.do_ocr = True**` — run OCR when embedded text is missing
or unreliable.
- `**TesseractCliOcrOptions(lang=["eng"])**` — use the system `tesseract`
binary rather than Docling's bundled OCR engines.

Non-PDF formats (DOCX, HTML, etc.) still go through Docling's default converter
without the PDF-specific OCR override.

Plain text is exported with `document.export_to_text()`. That keeps Layer 2 as
a single linear string, which matches what `eyecite` and our span annotations
expect.

### Why not CourtListener plain text?

Early local test data used RECAP-style `.txt` exports fetched from CourtListener
(`--- Plain text ---` header + body). That was convenient for bootstrapping,
but the body text is often **worse than re-parsing the original PDF**:

- Line breaks and column order from PACER/RECAP extraction do not match the
visual brief layout.
- Citation strings are split across lines or merged with neighboring tokens.
- OCR and copy/paste artifacts from upstream conversion propagate unchanged.

Because extraction failures of **Type 2** (ill-formatted text) are often
upstream problems, feeding the pipeline CourtListener plain text confuses
preprocessing quality with extraction quality. We now treat the RECAP PDF as the
3rd-layer source of truth and regenerate Layer 2 locally.

### Prerequisites

```bash
# Python deps (Docling + models)
uv sync --group preprocessing

# OCR engine used by Docling for PDFs
tesseract --version
```

Docling downloads layout models on first run (Hugging Face). Ensure network
access the first time you preprocess a PDF on a new machine.