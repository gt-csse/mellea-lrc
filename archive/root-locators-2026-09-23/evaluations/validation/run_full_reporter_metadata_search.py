"""Run one bounded full-reporter metadata-discovery stage over Documents.

This is an evaluation driver, not a pipeline.  It reads an exact-lookup
checkpoint and writes the next ordinary serialized ``Document`` checkpoint,
leaving every source-grounded term-plan, provider query, result, and failure in
the document trace.  CourtListener and GovInfo are separate invocations so an
artifact can be inspected or resumed without replaying the other provider.
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
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

Stage = Literal["courtlistener", "govinfo"]

_STAGE_NAMES: dict[Stage, str] = {
    "courtlistener": "courtlistener_full_reporter_metadata_search",
    "govinfo": "govinfo_full_reporter_metadata_search",
}


class _PacedCourtListenerClient:
    """Evaluation-only wrapper that spaces provider requests deterministically."""

    def __init__(self, client: CourtListenerClient, *, minimum_interval_seconds: float) -> None:
        self.client = client
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_at: float | None = None

    def lookup_citation(self, volume: str, reporter: str, page: str):
        return self.client.lookup_citation(volume, reporter, page)

    def search(self, query: str, search_type: Literal["r", "rd", "d", "o"], cursor=None, *, semantic=False):
        self._wait()
        result = self.client.search(query, search_type, cursor=cursor, semantic=semantic)
        self._last_request_at = time.monotonic()
        return result

    def get_docket(self, docket_id: str):
        return self.client.get_docket(docket_id)

    def get_opinion(self, opinion_id: str):
        return self.client.get_opinion(opinion_id)

    def _wait(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.minimum_interval_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)


class _PacedGovInfoClient:
    """Evaluation-only wrapper that spaces USCOURTS searches deterministically."""

    def __init__(self, client: GovInfoClient, *, minimum_interval_seconds: float) -> None:
        self.client = client
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_request_at: float | None = None

    def search_uscourts(self, query: str, *, page_size: int):
        if self._last_request_at is not None:
            remaining = self.minimum_interval_seconds - (time.monotonic() - self._last_request_at)
            if remaining > 0:
                time.sleep(remaining)
        result = self.client.search_uscourts(query, page_size=page_size)
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
    """Run exactly one provider stage over a deterministic document slice."""
    paths = sorted(documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{documents} has {len(paths)} artifacts in slice start={start}, limit={limit}"
        raise ValueError(msg)

    courtlistener = _PacedCourtListenerClient(
        CourtListenerClient(), minimum_interval_seconds=minimum_search_interval_seconds
    )
    govinfo = _PacedGovInfoClient(GovInfoClient(), minimum_interval_seconds=minimum_search_interval_seconds)
    session = start_mellea_session_from_env() if stage == "courtlistener" else None
    outcomes: Counter[str] = Counter()
    results: list[dict[str, str]] = []
    for index, source in enumerate(paths, start=1):
        destination = output / "documents" / source.name
        payload: dict[str, object]
        rerun = False
        if resume and destination.exists():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            rerun = retry_failed and _has_failed_provider_search(payload, stage=stage)
            if not rerun:
                Document.model_validate(payload)
        if not resume or not destination.exists() or rerun:
            document = Document.model_validate(json.loads(source.read_text(encoding="utf-8")))
            document = await _run_stage(
                stage,
                document,
                courtlistener=courtlistener,
                govinfo=govinfo,
                session=session,
            )
            payload = document.model_dump(mode="json")
            Document.model_validate(payload)
            _atomic_json(destination, payload)
        outcomes.update(_outcomes(payload, stage=stage))
        results.append(
            {
                "input_document": source.name,
                "input_document_sha256": _sha256(source),
                "result": str(destination.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {source.stem}")

    return {
        "artifact_type": "full_reporter_metadata_search_stage",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "input_documents": str(documents),
        "start_index": start,
        "document_count": len(results),
        "documents": results,
        "outcomes": dict(sorted(outcomes.items())),
        "model": llm_api_config_from_env(os.environ).model if stage == "courtlistener" else None,
        "minimum_search_interval_seconds": minimum_search_interval_seconds,
    }


async def _run_stage(
    stage: Stage,
    document: Document,
    *,
    courtlistener: _PacedCourtListenerClient,
    govinfo: _PacedGovInfoClient,
    session: object | None,
) -> Document:
    if stage == "courtlistener":
        return await search_courtlistener_full_reporter_roots(document, client=courtlistener, session=session)
    if stage == "govinfo":
        return await search_govinfo_full_reporter_roots(document, client=govinfo, session=session)
    msg = f"Unsupported full-reporter metadata-search stage: {stage!r}"
    raise ValueError(msg)


def _has_failed_provider_search(payload: dict[str, object], *, stage: Stage) -> bool:
    expected = _STAGE_NAMES[stage]
    for citation in payload.get("citations", []):
        if not isinstance(citation, dict):
            continue
        source = citation.get("fields")
        if not isinstance(source, dict) or source.get("kind") != "FullCaseCitation":
            continue
        for node in citation.get("trace", []):
            if not isinstance(node, dict) or node.get("stage") != expected:
                continue
            details = node.get("details")
            validation = details.get("validation") if isinstance(details, dict) else None
            if isinstance(validation, dict) and validation.get("outcome") == "failed":
                return True
    return False


def _outcomes(payload: dict[str, object], *, stage: Stage) -> Counter[str]:
    outcomes: Counter[str] = Counter()
    expected = _STAGE_NAMES[stage]
    for citation in payload.get("citations", []):
        if not isinstance(citation, dict) or citation.get("root_id") != citation.get("citation_id"):
            continue
        source = citation.get("fields")
        if not isinstance(source, dict) or source.get("kind") != "FullCaseCitation":
            continue
        for node in citation.get("trace", []):
            if not isinstance(node, dict) or node.get("stage") != expected:
                continue
            details = node.get("details")
            validation = details.get("validation") if isinstance(details, dict) else None
            outcome = validation.get("outcome") if isinstance(validation, dict) else None
            if isinstance(outcome, str):
                outcomes[outcome] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("courtlistener", "govinfo"), required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--minimum-search-interval-seconds", type=float, default=0.0)
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
