"""Run one checkpointable full-reporter-locator validation stage.

Every invocation accepts serialized ``Document`` artifacts and writes the next
serialized ``Document`` artifacts.  It never silently composes later stages:
call this module once for exact lookup, once for unique-candidate identity, and
once for bounded ambiguity resolution.  That makes each result independently
inspectable and resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from mellea_lrc.api import (
    Document,
    lookup_full_reporter_locators_exact,
    resolve_full_reporter_locator_ambiguities,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

Stage = Literal["exact-lookup", "unique-identity", "ambiguity-resolution"]


def _atomic_json(path: Path, value: object) -> None:
    """Write one complete JSON checkpoint or leave the preceding artifact intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(
    *,
    stage: Stage,
    documents: Path,
    output: Path,
    start: int,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Run exactly one validation stage over a deterministic slice of documents."""
    paths = sorted(documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{documents} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    service = CourtListenerClient()
    session = start_mellea_session_from_env() if stage != "exact-lookup" else None
    outcomes: Counter[str] = Counter()
    result_paths: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            document = Document.from_serialized(payload)
        else:
            document = Document.from_serialized(json.loads(path.read_text(encoding="utf-8")))
            document = await _run_stage(stage, document, service=service, session=session)
            payload = document.serialize()
            Document.from_serialized(payload)
            _atomic_json(result_path, payload)
        outcomes.update(_outcomes(payload, stage=stage))
        result_paths.append(
            {
                "input_document": path.name,
                "input_document_sha256": _sha256(path),
                "result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(document.citations)} citations")

    return {
        "artifact_type": "full_reporter_locator_validation_stage",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "input_documents": str(documents),
        "start_index": start,
        "document_count": len(result_paths),
        "documents": result_paths,
        "outcomes": dict(sorted(outcomes.items())),
        **(
            {"model": llm_api_config_from_env(os.environ).model}
            if stage != "exact-lookup"
            else {"model": None}
        ),
    }


async def _run_stage(
    stage: Stage,
    document: Document,
    *,
    service: CourtListenerClient,
    session: object | None,
) -> Document:
    if stage == "exact-lookup":
        return await lookup_full_reporter_locators_exact(document, client=service)
    if stage == "unique-identity":
        return await validate_unique_full_reporter_locator_identities(
            document, client=service, session=session
        )
    if stage == "ambiguity-resolution":
        return await resolve_full_reporter_locator_ambiguities(document, client=service, session=session)
    msg = f"Unsupported full-reporter-locator validation stage: {stage!r}"
    raise ValueError(msg)


def _outcomes(payload: dict[str, object], *, stage: Stage) -> Counter[str]:
    """Read the first-class judgement written by this stage, not its trace graph."""
    question = "locator_lookup" if stage == "exact-lookup" else "identity"
    outcomes: Counter[str] = Counter()
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return outcomes
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("root_id") != citation.get("citation_id"):
            continue
        source = citation.get("source")
        if not isinstance(source, dict) or source.get("citation_type") != "FullCaseCitation":
            continue
        judgements = citation.get("judgements")
        judgement = judgements.get(question) if isinstance(judgements, dict) else None
        if isinstance(judgement, dict) and isinstance(judgement.get("outcome"), str):
            outcomes[str(judgement["outcome"])] += 1
    return outcomes


def main() -> None:
    """Run the requested stage and persist its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("exact-lookup", "unique-identity", "ambiguity-resolution"), required=True
    )
    parser.add_argument("--documents", type=Path, required=True, help="Input serialized Document directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output stage-artifact directory.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    load_dotenv(".env")
    manifest = asyncio.run(
        run(
            stage=args.stage,
            documents=args.documents,
            output=args.output,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
