"""Drive test-data documents through the pipeline and save per-module snapshots.

For every PDF in ``local/test_data/pdfs/*.pdf`` (or matching preprocessed
``.txt`` in ``local/test_data/``) this runs the full chain (preprocess ->
extract -> validate -> assess) and serializes each module output to:

    local/snapshots/<doc>/preprocessed.json
    local/snapshots/<doc>/extraction.json
    local/snapshots/<doc>/validation.json
    local/snapshots/<doc>/assessment.json

Layer 2 text should be regenerated from PDFs with:

    uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs

Validation hits the deployed CourtListener service and assessment hits the
configured LLM, so the run can be rate limited. On the first failure for a
document the run stops gracefully, preserving every snapshot already written.

Usage:
    uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus
    # optional: --docs 1 2   --snapshot-root local/snapshots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.assessment import run_assessment
from mellea_lrc.extraction import run_extraction
from mellea_lrc.preprocessing import preprocess_plain_text, run_preprocessing
from mellea_lrc.serialization import (
    serialize_assessed_document,
    serialize_extracted_document,
    serialize_validated_document,
    serialize_preprocessed_document,
)
from mellea_lrc.validation import run_validation
from scripts.e2e_backend.run_artifact_pipeline import _load_dotenv

if TYPE_CHECKING:
    from mellea_lrc.preprocessing.types import PreprocessedDocument

DEFAULT_TEST_DATA = Path("local/test_data")
DEFAULT_PDF_DIR = DEFAULT_TEST_DATA / "pdfs"
DEFAULT_SNAPSHOT_ROOT = Path("local/snapshots")

SNAPSHOT_STAGES = ("preprocessed", "extraction", "validation", "assessment")


def main() -> None:
    """Run the configured test corpus and write module snapshots."""
    args = _parse_args()
    _load_dotenv(Path(".env"))

    documents = _select_documents(args.test_data, args.pdf_dir, args.docs)
    if not documents:
        _emit("No matching test documents found.")
        return

    for input_path in documents:
        doc = input_path.stem
        try:
            _snapshot_document(
                input_path,
                test_data=args.test_data,
                snapshot_root=args.snapshot_root,
            )
            _emit(f"{doc}: done")
        except Exception as exc:
            _emit(f"{doc}: STOPPED ({type(exc).__name__}: {exc})")
            _emit("Stopping; snapshots written so far are preserved.")
            sys.exit(1)


def _snapshot_document(
    input_path: Path,
    *,
    test_data: Path,
    snapshot_root: Path,
) -> None:
    doc = input_path.stem
    doc_dir = snapshot_root / doc
    doc_dir.mkdir(parents=True, exist_ok=True)

    preprocessed = _load_preprocessed(input_path, test_data=test_data)
    _write_snapshot(doc_dir, "preprocessed", serialize_preprocessed_document(preprocessed))

    extraction = run_extraction(preprocessed)
    _write_snapshot(doc_dir, "extraction", serialize_extracted_document(extraction))

    validation = run_validation(extraction)
    _write_snapshot(doc_dir, "validation", serialize_validated_document(validation))

    assessment = run_assessment(validation)
    _write_snapshot(doc_dir, "assessment", serialize_assessed_document(assessment))


def _write_snapshot(doc_dir: Path, stage: str, payload: object) -> Path:
    if stage not in SNAPSHOT_STAGES:
        msg = f"Unknown snapshot stage: {stage}"
        raise ValueError(msg)
    path = doc_dir / f"{stage}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_preprocessed(source_path: Path, *, test_data: Path) -> PreprocessedDocument:
    """Load cached Layer 2 text or preprocess the source PDF on demand."""
    cached_txt = test_data / f"{source_path.stem}.txt"
    if cached_txt.exists():
        return preprocess_plain_text(cached_txt)
    if source_path.suffix.lower() == ".pdf":
        return run_preprocessing(source_path)
    if source_path.suffix.lower() == ".txt":
        return preprocess_plain_text(source_path)
    msg = f"No preprocessed text for {source_path.name}; run preprocess_test_pdfs first."
    raise FileNotFoundError(msg)


def _select_documents(test_data: Path, pdf_dir: Path, docs: list[str] | None) -> list[Path]:
    if docs:
        selected: list[Path] = []
        for name in docs:
            stem = Path(name).stem
            pdf_path = pdf_dir / f"{stem}.pdf"
            txt_path = test_data / f"{stem}.txt"
            if pdf_path.exists():
                selected.append(pdf_path)
            elif txt_path.exists():
                selected.append(txt_path)
            else:
                selected.append(pdf_path)
        return selected
    if pdf_dir.exists():
        return sorted(pdf_dir.glob("*.pdf"), key=_document_sort_key)
    return sorted(test_data.glob("*.txt"), key=_document_sort_key)


def _document_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (int(stem), stem)
    return (10**9, stem)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--docs",
        nargs="*",
        default=None,
        help="Specific document stems, e.g. 1",
    )
    return parser.parse_args()


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
