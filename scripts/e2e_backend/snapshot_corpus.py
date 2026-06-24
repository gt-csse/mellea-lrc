"""Drive test-data documents through the pipeline and save per-module snapshots.

For every PDF in ``local/test_data/pdfs/*.pdf`` (or matching preprocessed
``.txt`` in ``local/test_data/``) this runs the full chain (preprocess ->
extract -> validate -> assess) and serializes each module output to:

    local/snapshots/<doc>/<module>/<model>_<timestamp>.json

Layer 2 text should be regenerated from PDFs with:

    uv run --group preprocessing python -m scripts.e2e_backend.preprocess_test_pdfs

Validation hits the deployed CourtListener service and assessment hits the
configured LLM, so the run can be rate limited. On the first failure for a
document the run stops gracefully, preserving every snapshot already written.

Usage:
    uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus
    # optional: --docs 432895579 436876274   --snapshot-root local/snapshots   --max-mellea N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mellea_lrc.assessment import run_assessment
from mellea_lrc.extraction import run_extraction
from mellea_lrc.llm import llm_provider_config_from_env
from mellea_lrc.preprocessing import preprocess, preprocess_plain_text
from mellea_lrc.preprocessing.types import PreprocessedDocument
from mellea_lrc.serialization import (
    serialize_document_assessment,
    serialize_document_extraction,
    serialize_document_validation,
    serialize_preprocessed_document,
)
from mellea_lrc.validation import run_validation
from scripts.e2e_backend.run_artifact_pipeline import _load_dotenv

DEFAULT_TEST_DATA = Path("local/test_data")
DEFAULT_PDF_DIR = DEFAULT_TEST_DATA / "pdfs"
DEFAULT_SNAPSHOT_ROOT = Path("local/snapshots")

# Engine identifier recorded in the filename for the non-LLM modules.
STATIC_MODULE_MODELS = {
    "preprocessed": "docling-tesseract",
    "extraction": "eyecite",
    "validation": "cl-access",
}


def main() -> None:
    """Run the configured test corpus and write timestamped snapshots."""
    args = _parse_args()
    _load_dotenv(Path(".env"))

    documents = _select_documents(args.test_data, args.pdf_dir, args.docs)
    if not documents:
        _emit("No matching test documents found.")
        return

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assessment_model = llm_provider_config_from_env(os.environ).model

    for input_path in documents:
        doc = input_path.stem
        try:
            _snapshot_document(
                input_path,
                test_data=args.test_data,
                snapshot_root=args.snapshot_root,
                run_stamp=run_stamp,
                assessment_model=assessment_model,
                max_mellea=args.max_mellea,
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
    run_stamp: str,
    assessment_model: str,
    max_mellea: int | None,
) -> None:
    doc = input_path.stem

    preprocessed = _load_preprocessed(input_path, test_data=test_data)
    _write_snapshot(
        snapshot_root, doc, "preprocessed", STATIC_MODULE_MODELS["preprocessed"], run_stamp,
        serialize_preprocessed_document(preprocessed),
    )

    extraction = run_extraction(preprocessed)
    _write_snapshot(
        snapshot_root, doc, "extraction", STATIC_MODULE_MODELS["extraction"], run_stamp,
        serialize_document_extraction(extraction),
    )

    validation = run_validation(extraction)
    _write_snapshot(
        snapshot_root, doc, "validation", STATIC_MODULE_MODELS["validation"], run_stamp,
        serialize_document_validation(validation),
    )

    assessment = run_assessment(validation, max_mellea=max_mellea)
    _write_snapshot(
        snapshot_root, doc, "assessment", assessment_model, run_stamp,
        serialize_document_assessment(assessment),
    )


def _write_snapshot(
    snapshot_root: Path,
    doc: str,
    module: str,
    model: str,
    run_stamp: str,
    payload: object,
) -> Path:
    out_dir = snapshot_root / doc / module
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(model)}_{run_stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_preprocessed(source_path: Path, *, test_data: Path) -> PreprocessedDocument:
    """Load cached Layer 2 text or preprocess the source PDF on demand."""
    cached_txt = test_data / f"{source_path.stem}.txt"
    if cached_txt.exists():
        return preprocess_plain_text(cached_txt)
    if source_path.suffix.lower() == ".pdf":
        return preprocess(source_path)
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
        return sorted(pdf_dir.glob("*.pdf"))
    return sorted(test_data.glob("*.txt"))


def _slug(value: str) -> str:
    return value.replace("/", "-").replace(":", "-")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--docs",
        nargs="*",
        default=None,
        help="Specific CourtListener ids or stems, e.g. 432895579",
    )
    parser.add_argument("--max-mellea", type=int, default=None)
    return parser.parse_args()


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
