"""Form document-internal roots from persisted complete-locator checkpoints.

The input is the cumulative ``colocation`` checkpoint. This runner applies the
deterministic field readers required to state a complete docket identifier, then
serializes the independently reusable ``root_formation`` document. It performs
no archive lookup and makes no model call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import Document, form_roots, resolve_case_names, resolve_courts, resolve_dates, stable


def _atomic_json(path: Path, value: object) -> None:
    """Write JSON without leaving a partial artifact behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return a checkpoint's byte hash for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _colocation_document(path: Path) -> Document:
    """Restore one cumulative locator-chain document at its colocation point."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_type") != "locator_checkpoint_chain":
        msg = f"{path}: expected a locator_checkpoint_chain artifact"
        raise ValueError(msg)
    checkpoints = artifact.get("checkpoints")
    if not isinstance(checkpoints, dict):
        msg = f"{path}: missing checkpoint map"
        raise ValueError(msg)
    payload = checkpoints.get("colocation")
    if not isinstance(payload, dict):
        msg = f"{path}: missing serialized colocation checkpoint"
        raise ValueError(msg)
    return Document.model_validate(payload)


def _prepare_root_fields(document: Document) -> Document:
    """Read the document fields that precede pure root graph formation."""
    rules = stable()
    document = resolve_case_names(document, rules=rules)
    document = resolve_courts(document, rules=rules)
    return resolve_dates(document, rules=rules)


def run(
    *, locator_checkpoints: Path, output: Path, start: int, limit: int, resume: bool
) -> dict[str, object]:
    """Materialize a deterministic sorted slice of root-formation artifacts."""
    paths = sorted(locator_checkpoints.glob("*.json"))[start : start + limit]
    if len(paths) != limit:
        msg = f"{locator_checkpoints} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    documents: list[dict[str, str]] = []
    root_count = 0
    locator_count = 0
    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            document = Document.model_validate(payload)
        else:
            document = form_roots(_prepare_root_fields(_colocation_document(path)))
            payload = document.model_dump(mode="json")
            Document.model_validate(payload)
            _atomic_json(result_path, payload)
        roots = [citation for citation in document.active_citations if citation.is_root]
        root_count += len(roots)
        locator_count += len(document.locators)
        documents.append(
            {
                "locator_checkpoint_artifact": path.name,
                "locator_checkpoint_artifact_sha256": _sha256(path),
                "root_formation_result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(roots)} roots from {len(document.locators)} locators")

    return {
        "artifact_type": "root_formation_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_locator_checkpoint_dir": str(locator_checkpoints),
        "input_stage": "colocation",
        "field_preparation": ["case_names", "courts", "dates"],
        "root_formation": "exact complete-reporter repetition and court-qualified docket repetition",
        "start_index": start,
        "documents": documents,
        "document_count": len(documents),
        "locator_occurrences": locator_count,
        "roots": root_count,
        "excluded_stages": ["identity_lookup", "docket_lookup", "leaf_growth", "pinpoint"],
    }


def main() -> None:
    """Run deterministic root formation over locator-chain artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locator-checkpoints",
        type=Path,
        required=True,
        help="Directory of locator_checkpoint_chain artifacts.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for root-formation artifacts.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based filing index in sorted artifacts.")
    parser.add_argument("--limit", type=int, required=True, help="Deterministic sorted filing count to run.")
    parser.add_argument("--resume", action="store_true", help="Reuse valid artifacts already in --output.")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    manifest = run(
        locator_checkpoints=args.locator_checkpoints,
        output=args.output,
        start=args.start,
        limit=args.limit,
        resume=args.resume,
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"locators": manifest["locator_occurrences"], "roots": manifest["roots"]}))


if __name__ == "__main__":
    main()
