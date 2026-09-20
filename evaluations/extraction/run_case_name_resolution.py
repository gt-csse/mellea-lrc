"""Materialize the standalone case-name field stage over serialized Documents.

This runner exists for evaluation and migration only. Production callers use
``resolve_case_names(document)`` through the public compositional API. Each
output is a normal serialized :class:`Document`, so a later stage resumes it
without bespoke checkpoint logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import Document, resolve_case_names, stable


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name_count(document: Document) -> int:
    return sum(
        citation.stated.case_name is not None
        for citation in document.active_citations
        if hasattr(citation.stated, "case_name")
    )


def run(*, documents: Path, output: Path, start: int, limit: int, resume: bool) -> dict[str, object]:
    """Apply the deterministic public case-name stage to one sorted slice."""
    inputs = sorted(documents.glob("*.json"))[start : start + limit]
    if len(inputs) != limit:
        msg = f"{documents} has {len(inputs)} documents in slice start={start}, limit={limit}"
        raise ValueError(msg)

    results: list[dict[str, str]] = []
    added_names = 0
    for index, source in enumerate(inputs, start=1):
        destination = output / "documents" / source.name
        if resume and destination.exists():
            document = Document.from_serialized(json.loads(destination.read_text(encoding="utf-8")))
        else:
            original = Document.from_serialized(json.loads(source.read_text(encoding="utf-8")))
            before = _name_count(original)
            document = resolve_case_names(original, rules=stable())
            added_names += _name_count(document) - before
            payload = document.serialize()
            Document.from_serialized(payload)
            _write_json(destination, payload)
        results.append(
            {
                "input_document": str(source),
                "input_sha256": _sha256(source),
                "result_document": str(destination.relative_to(output)),
            }
        )
        print(f"{index}/{len(inputs)} {source.stem}")

    return {
        "artifact_type": "case_name_resolution_evaluation",
        "created_at": datetime.now(UTC).isoformat(),
        "input_documents": str(documents),
        "stage": "case_name_resolution",
        "document_count": len(results),
        "case_names_added": added_names,
        "documents": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = run(
        documents=args.documents,
        output=args.output,
        start=args.start,
        limit=args.limit,
        resume=args.resume,
    )
    _write_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {"documents": manifest["document_count"], "case_names_added": manifest["case_names_added"]}
        )
    )


if __name__ == "__main__":
    main()
