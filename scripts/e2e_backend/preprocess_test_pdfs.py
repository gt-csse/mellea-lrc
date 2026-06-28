"""Preprocess local test PDFs with Docling + Tesseract CLI OCR.

Reads PDFs from ``local/test_data/pdfs/*.pdf`` and writes Layer 2 plain-text
exports to ``local/test_data/<pdf-stem>.txt``. Each file includes a short
provenance header followed by the shared ``--- Plain text ---`` marker used by
``preprocess_plain_text``.

Usage:
    uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs
    uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs --pdf 432895579
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mellea_lrc.preprocessing.docling import build_docling_converter, preprocess_with_docling
from mellea_lrc.preprocessing.types import PreprocessedDocument

DEFAULT_PDF_DIR = Path("local/test_data/pdfs")
DEFAULT_OUTPUT_DIR = Path("local/test_data")


def main() -> None:
    """Preprocess one or all PDFs in the local test corpus."""
    args = _parse_args()
    pdf_paths = _select_pdfs(args.pdf_dir, args.pdf)
    if not pdf_paths:
        _emit(f"No PDFs found under {args.pdf_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    converter = build_docling_converter(enable_pdf_ocr=True)
    failures: list[str] = []

    for pdf_path in pdf_paths:
        output_path = args.output_dir / f"{pdf_path.stem}.txt"
        try:
            document = preprocess_with_docling(pdf_path, converter=converter)
            output_path.write_text(
                _format_preprocessed_text_file(document, source_pdf=pdf_path),
                encoding="utf-8",
            )
            _emit(f"{pdf_path.name}: wrote {output_path} ({len(document.text)} chars)")
        except Exception as exc:  # noqa: BLE001 - batch driver should continue
            failures.append(f"{pdf_path.name}: {type(exc).__name__}: {exc}")
            _emit(f"{pdf_path.name}: FAILED ({type(exc).__name__}: {exc})")

    if failures:
        _emit(f"Completed with {len(failures)} failure(s).")
        sys.exit(1)


def _format_preprocessed_text_file(
    document: PreprocessedDocument,
    *,
    source_pdf: Path,
) -> str:
    header_lines = [
        f"Source PDF: {source_pdf}",
        f"Backend: {document.preprocessing_metadata.backend.value}",
    ]
    if document.preprocessing_metadata.backend_version:
        header_lines.append(
            f"Backend version: {document.preprocessing_metadata.backend_version}"
        )
    return "\n".join(header_lines) + f"\n\n--- Plain text ---\n{document.text}"


def _select_pdfs(pdf_dir: Path, stems: list[str] | None) -> list[Path]:
    if stems:
        return [pdf_dir / (stem if stem.endswith(".pdf") else f"{stem}.pdf") for stem in stems]
    return sorted(pdf_dir.glob("*.pdf"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=None,
        help="Specific PDF stems or filenames; default is all PDFs in --pdf-dir",
    )
    return parser.parse_args()


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
