# Preprocessing Development

How 3rd-layer filings become 2nd-layer text, and how to evaluate changes. See [Benchmark](./Benchmark.md) and [Extraction Model Development](./Extraction%20Model%20Development.md).

## Current default: Docling + Tesseract CLI OCR

Implemented in `src/mellea_lrc/preprocessing/docling.py` (`preprocess_with_docling()`). For PDFs:

- `PdfPipelineOptions.do_ocr = True` — OCR when embedded text is missing/unreliable.
- `TesseractCliOcrOptions(lang=["eng"])` — the system `tesseract` binary, not Docling's bundled engines.

Non-PDF formats use Docling's default converter. Text is exported with `document.export_to_text()`, keeping Layer 2 a single linear string as `eyecite` and our span offsets expect.

## Why not CourtListener plain text?

RECAP `.txt` exports were convenient for bootstrapping but are often worse than re-parsing the PDF: PACER line breaks/column order don't match the visual layout, citation strings split across lines, and upstream OCR/copy-paste artifacts propagate. Feeding them in conflates preprocessing quality with extraction quality (Type 2 failures), so we treat the RECAP PDF as source of truth and regenerate Layer 2 locally.

## Prerequisites

```bash
uv sync --group preprocessing   # Docling + models
tesseract --version             # OCR engine
```

Docling downloads layout models from Hugging Face on first run — ensure network access the first time on a new machine.
