"""Run the E2E pipeline from typed snapshots without the frontend."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from mellea_lrc.assessment import run_assessment
from mellea_lrc.extraction import run_extraction
from mellea_lrc.preprocessing import preprocess, preprocess_plain_text
from mellea_lrc.serialization import (
    deserialize_document_validation,
    serialize_document_assessment,
    serialize_document_validation,
)
from mellea_lrc.validation import run_validation

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import DocumentExtraction
    from mellea_lrc.validation.types import DocumentValidation

DEFAULT_INPUT = Path("local/test_data/pdfs/432895579.pdf")
DEFAULT_FALLBACK_INPUT = Path("local/test_data/432895579.txt")
DEFAULT_SNAPSHOT_ROOT = Path("local/snapshots")


def main() -> None:
    """Run extraction, reusable validation, and optional Mellea assessment."""
    args = _parse_args()
    _load_dotenv(Path(".env"))

    input_path = args.input if args.input.exists() else DEFAULT_FALLBACK_INPUT
    if not input_path.exists():
        msg = (
            "No default test input found. Provide --input or run "
            "scripts.e2e_backend.preprocess_test_pdfs first."
        )
        raise FileNotFoundError(msg)
    snapshot_dir = args.snapshot_dir or (DEFAULT_SNAPSHOT_ROOT / input_path.stem)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    validation_path = snapshot_dir / "validation.json"
    assessment_path = snapshot_dir / "assessment.json"

    preprocessed = (
        preprocess_plain_text(input_path)
        if input_path.suffix.lower() == ".txt"
        else preprocess(input_path)
    )
    extraction = run_extraction(preprocessed)
    validation = _load_or_create_validation(
        validation_path,
        extraction,
        refresh=args.refresh_validation,
    )

    _emit_json(
        {
            "input": str(input_path),
            "validation_snapshot": str(validation_path),
            "assessment_snapshot": str(assessment_path),
            "citations": len(extraction.citations),
            "validations": _validation_counts(validation),
        }
    )

    if not args.assess_mellea:
        return

    assessment = run_assessment(validation, max_mellea=args.max_mellea, on_mellea_call=_emit_mellea_call)
    assessment_path.write_text(
        json.dumps(serialize_document_assessment(assessment), indent=2),
        encoding="utf-8",
    )
    _emit(f"wrote assessment snapshot: {assessment_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directory for validation.json and assessment.json (default: local/snapshots/<doc>)",
    )
    parser.add_argument("--refresh-validation", action="store_true")
    parser.add_argument("--assess-mellea", action="store_true")
    parser.add_argument("--max-mellea", type=int, default=None)
    return parser.parse_args()


def _load_or_create_validation(
    path: Path,
    extraction: DocumentExtraction,
    *,
    refresh: bool,
) -> DocumentValidation:
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return deserialize_document_validation(payload)

    validation = run_validation(extraction)
    path.write_text(json.dumps(serialize_document_validation(validation), indent=2), encoding="utf-8")
    return validation


def _validation_counts(validation: DocumentValidation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in validation.validations:
        counts[item.status.value] = counts.get(item.status.value, 0) + 1
    return counts


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _emit_mellea_call(payload: object) -> None:
    _emit_json(asdict(payload))


def _emit_json(payload: object) -> None:
    _emit(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
