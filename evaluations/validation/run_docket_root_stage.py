"""Run one checkpointable docket-root identity stage over serialized Documents.

Each invocation reads one directory of serialized ``Document`` checkpoints and
writes the next one.  Search, zero/one-candidate resolution, and bounded
ambiguity resolution are separate runs, so a saved search result can be
examined or resumed without repeating a CourtListener request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from mellea_lrc.api import (
    Document,
    resolve_docket_root_ambiguities,
    search_docket_roots,
    validate_unique_docket_root_identities,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

Stage = Literal["search", "unique-identity", "ambiguity-resolution"]


class _PacedDocketSearchClient:
    """Evaluation-only CourtListener wrapper that respects a configured request interval."""

    def __init__(self, client: CourtListenerClient, *, minimum_interval_seconds: float) -> None:
        self.client = client
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_at: float | None = None

    def search(
        self,
        query: str,
        search_type: Literal["r", "rd", "d", "o"],
        cursor: str | None = None,
        *,
        semantic: bool = False,
    ):
        if self._last_request_at is not None:
            remaining = self.minimum_interval_seconds - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        result = self.client.search(query, search_type, cursor=cursor, semantic=semantic)
        self._last_request_at = time.monotonic()
        return result


def _atomic_json(path: Path, value: object) -> None:
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
    retry_failed: bool,
    minimum_search_interval_seconds: float,
) -> dict[str, object]:
    """Run exactly one docket-root stage over a deterministic source slice."""
    paths = sorted(documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{documents} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    service = _PacedDocketSearchClient(
        CourtListenerClient(), minimum_interval_seconds=minimum_search_interval_seconds
    )
    session = start_mellea_session_from_env() if stage != "search" else None
    outcomes: Counter[str] = Counter()
    result_paths: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        retry = False
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            retry = retry_failed and stage == "search" and _has_failed_search(payload)
            if not retry:
                document = Document.from_serialized(payload)
        if not resume or not result_path.exists() or retry:
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
        "artifact_type": "docket_root_validation_stage",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "input_documents": str(documents),
        "start_index": start,
        "document_count": len(result_paths),
        "documents": result_paths,
        "outcomes": dict(sorted(outcomes.items())),
        **({"model": llm_api_config_from_env(os.environ).model} if stage != "search" else {"model": None}),
    }


def _has_failed_search(payload: dict[str, object]) -> bool:
    """Whether a serialized search checkpoint has a retryable stage failure."""
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return False
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        source = citation.get("source")
        if not isinstance(source, dict) or source.get("citation_type") != "DocketCitation":
            continue
        trace = citation.get("trace")
        if not isinstance(trace, list):
            continue
        for node in trace:
            if not isinstance(node, dict) or node.get("stage") != "docket_root_search":
                continue
            details = node.get("details")
            validation = details.get("validation") if isinstance(details, dict) else None
            if isinstance(validation, dict) and validation.get("outcome") == "failed":
                return True
    return False


async def _run_stage(
    stage: Stage,
    document: Document,
    *,
    service: _PacedDocketSearchClient,
    session: object | None,
) -> Document:
    if stage == "search":
        return await search_docket_roots(document, client=service)
    if stage == "unique-identity":
        return await validate_unique_docket_root_identities(document, session=session)
    if stage == "ambiguity-resolution":
        return await resolve_docket_root_ambiguities(document, session=session)
    msg = f"Unsupported docket-root validation stage: {stage!r}"
    raise ValueError(msg)


def _outcomes(payload: dict[str, object], *, stage: Stage) -> Counter[str]:
    """Read first-class docket-root judgements rather than inferring trace state."""
    question = "docket_lookup" if stage == "search" else "identity"
    outcomes: Counter[str] = Counter()
    citations = payload.get("citations")
    if not isinstance(citations, list):
        return outcomes
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("root_id") != citation.get("citation_id"):
            continue
        source = citation.get("source")
        if not isinstance(source, dict) or source.get("citation_type") != "DocketCitation":
            continue
        judgements = citation.get("judgements")
        judgement = judgements.get(question) if isinstance(judgements, dict) else None
        if isinstance(judgement, dict) and isinstance(judgement.get("outcome"), str):
            outcomes[str(judgement["outcome"])] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("search", "unique-identity", "ambiguity-resolution"), required=True
    )
    parser.add_argument("--documents", type=Path, required=True, help="Input serialized Document directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output stage-artifact directory.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume and --stage search, recompute only Documents containing a failed search.",
    )
    parser.add_argument(
        "--minimum-search-interval-seconds",
        type=float,
        default=0.0,
        help="Evaluation-only minimum spacing between CourtListener search requests.",
    )
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")
    if args.minimum_search_interval_seconds < 0:
        parser.error("--minimum-search-interval-seconds must be non-negative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    load_dotenv(".env")
    manifest = asyncio.run(
        run(
            stage=args.stage,
            documents=args.documents,
            output=args.output,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
            retry_failed=args.retry_failed,
            minimum_search_interval_seconds=args.minimum_search_interval_seconds,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
