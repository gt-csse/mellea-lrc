"""Drive test-data documents through the pipeline and save per-module snapshots.

For every ``local/test_data/test-*.txt`` document this runs the full chain
(preprocess -> extract -> validate -> assess) and serializes each module output
to:

    local/snapshots/<doc>/<module>/<model>_<timestamp>.json

Validation hits the deployed CourtListener service and assessment hits the
configured LLM, so the run can be rate limited. On the first failure for a
document the run stops gracefully, preserving every snapshot already written.

Usage:
    uv run --group modal --group assessment python -m scripts.e2e_backend.snapshot_corpus
    # optional: --docs test-1 test-2   --snapshot-root local/snapshots   --max-mellea N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mellea_lrc.extraction import run_extraction
from mellea_lrc.preprocessing import preprocess_plain_text
from mellea_lrc.serialization import (
    serialize_document_assessment,
    serialize_document_extraction,
    serialize_document_validation,
    serialize_preprocessed_document,
)
from mellea_lrc.validation import validate_extraction
from scripts.e2e_backend.run_artifact_pipeline import _load_dotenv, _run_assessment

DEFAULT_TEST_DATA = Path("local/test_data")
DEFAULT_SNAPSHOT_ROOT = Path("local/snapshots")

# Engine identifier recorded in the filename for the non-LLM modules.
STATIC_MODULE_MODELS = {
    "preprocessed": "plain-text",
    "extraction": "eyecite",
    "validation": "cl-access",
}


def main() -> None:
    args = _parse_args()
    _load_dotenv(Path(".env"))

    documents = _select_documents(args.test_data, args.docs)
    if not documents:
        _emit("No matching test documents found.")
        return

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assessment_model = os.environ.get("MELLEA_LRC_ASSESSMENT_MODEL", "unknown")

    for input_path in documents:
        doc = input_path.stem
        try:
            _snapshot_document(
                input_path,
                snapshot_root=args.snapshot_root,
                run_stamp=run_stamp,
                assessment_model=assessment_model,
                max_mellea=args.max_mellea,
            )
            _emit(f"{doc}: done")
        except Exception as exc:  # noqa: BLE001 - any failure (often rate limit) stops the run
            _emit(f"{doc}: STOPPED ({type(exc).__name__}: {exc})")
            _emit("Stopping; snapshots written so far are preserved.")
            sys.exit(1)


def _snapshot_document(
    input_path: Path,
    *,
    snapshot_root: Path,
    run_stamp: str,
    assessment_model: str,
    max_mellea: int | None,
) -> None:
    doc = input_path.stem

    preprocessed = preprocess_plain_text(input_path)
    _write_snapshot(
        snapshot_root, doc, "preprocessed", STATIC_MODULE_MODELS["preprocessed"], run_stamp,
        serialize_preprocessed_document(preprocessed),
    )

    extraction = run_extraction(preprocessed)
    _write_snapshot(
        snapshot_root, doc, "extraction", STATIC_MODULE_MODELS["extraction"], run_stamp,
        serialize_document_extraction(extraction),
    )

    validation = validate_extraction(extraction)
    _write_snapshot(
        snapshot_root, doc, "validation", STATIC_MODULE_MODELS["validation"], run_stamp,
        serialize_document_validation(validation),
    )

    assessment = _run_assessment(validation, max_mellea=max_mellea)
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


def _select_documents(test_data: Path, docs: list[str] | None) -> list[Path]:
    if docs:
        return [test_data / (name if name.endswith(".txt") else f"{name}.txt") for name in docs]
    return sorted(test_data.glob("test-*.txt"), key=_doc_sort_key)


def _doc_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.removeprefix("test-")
    return (int(suffix), path.stem) if suffix.isdigit() else (1 << 30, path.stem)


def _slug(value: str) -> str:
    return value.replace("/", "-").replace(":", "-")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--docs", nargs="*", default=None, help="Specific doc stems, e.g. test-1 test-2")
    parser.add_argument("--max-mellea", type=int, default=None)
    return parser.parse_args()


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
