"""Run the no-network reporter metadata-candidate resolution checkpoint.

This driver resumes documents after both provider metadata-search checkpoints
and after exact reporter unique/ambiguity identity. It does not replay a
provider request or a model call: it only assesses the candidates already
preserved in each document, then writes the next ordinary serialized
``Document`` checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from mellea_lrc.api import Document, resolve_full_reporter_search_candidates
from mellea_lrc.model.citations import FullCaseCitation
from mellea_lrc.model.record import Question


def _atomic_json(path: Path, value: object) -> None:
    """Write one complete JSON artifact without leaving a partial checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(
    *,
    documents: Path,
    output: Path,
    start: int,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Run the explicit candidate-resolution stage over one deterministic slice."""
    paths = sorted(documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{documents} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    outcomes: Counter[str] = Counter()
    results: list[dict[str, str]] = []
    for index, source in enumerate(paths, start=1):
        destination = output / "documents" / source.name
        if resume and destination.exists():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            document = Document.model_validate(payload)
        else:
            document = Document.model_validate(json.loads(source.read_text(encoding="utf-8")))
            document = await resolve_full_reporter_search_candidates(document)
            payload = document.model_dump(mode="json")
            Document.model_validate(payload)
            _atomic_json(destination, payload)
        outcomes.update(_identity_outcomes(document))
        results.append(
            {
                "input_document": source.name,
                "input_document_sha256": _sha256(source),
                "result": str(destination.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {source.stem}")

    return {
        "artifact_type": "full_reporter_search_candidate_resolution_stage",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_documents": str(documents),
        "start_index": start,
        "document_count": len(results),
        "documents": results,
        "identity_outcomes": dict(sorted(outcomes.items())),
    }


def _identity_outcomes(document: Document) -> Counter[str]:
    outcomes: Counter[str] = Counter()
    for record in document.active_citations:
        if record.is_root and isinstance(record.fields, FullCaseCitation):
            outcomes[record.judgement(Question.IDENTITY).outcome] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    manifest = asyncio.run(
        run(
            documents=args.documents,
            output=args.output,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["identity_outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
