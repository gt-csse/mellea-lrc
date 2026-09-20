"""Run one checkpointable docket-root identity stage over serialized Documents.

Each invocation reads one directory of serialized ``Document`` checkpoints and
writes the next one.  Initial search and programmatic resolution are separate runs. The bounded
extraction-review-and-requeue stage records its own review and requeued-search
nodes; the second programmatic resolution runs and semantic candidate review
remain separate. A saved checkpoint can therefore be examined or resumed
without repeating an earlier CourtListener request or model call.
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
    lookup_govinfo_docket_roots,
    resolve_docket_root_ambiguities,
    resolve_docket_root_semantics,
    resolve_govinfo_docket_root_ambiguities,
    resolve_requeued_docket_root_ambiguities,
    review_and_requeue_unresolved_docket_roots,
    search_courtlistener_docket_roots,
    search_docket_roots,
    search_govinfo_docket_roots,
    shortlist_docket_root_metadata_candidates,
    validate_unique_docket_root_identities,
    validate_unique_govinfo_docket_root_identities,
    validate_unique_requeued_docket_root_identities,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env

Stage = Literal[
    "search",
    "unique-identity",
    "ambiguity-resolution",
    "govinfo-search",
    "govinfo-unique-identity",
    "govinfo-ambiguity-resolution",
    "docket-extraction-review-and-requeue",
    "requeued-unique-identity",
    "requeued-ambiguity-resolution",
    "metadata-shortlist",
    "semantic-resolution",
    "courtlistener-metadata-search",
    "govinfo-metadata-search",
]

_SEARCH_STAGES = frozenset(
    {"search", "govinfo-search", "courtlistener-metadata-search", "govinfo-metadata-search"}
)
_MODEL_STAGES = frozenset(
    {
        "docket-extraction-review-and-requeue",
        "semantic-resolution",
        "courtlistener-metadata-search",
    }
)


class _PacedGovInfoClient:
    """Evaluation-only GovInfo wrapper that avoids burst-rate failures."""

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
    govinfo_service = _PacedGovInfoClient(
        GovInfoClient(), minimum_interval_seconds=minimum_search_interval_seconds
    )
    session = start_mellea_session_from_env() if stage in _MODEL_STAGES else None
    outcomes: Counter[str] = Counter()
    result_paths: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        retry = False
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            retry = retry_failed and stage in _SEARCH_STAGES and _has_failed_search(payload, stage=stage)
            if not retry:
                document = Document.from_serialized(payload)
        if not resume or not result_path.exists() or retry:
            document = Document.from_serialized(json.loads(path.read_text(encoding="utf-8")))
            document = await _run_stage(
                stage,
                document,
                service=service,
                govinfo_service=govinfo_service,
                session=session,
            )
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
        **(
            {"model": llm_api_config_from_env(os.environ).model}
            if stage in _MODEL_STAGES
            else {"model": None}
        ),
    }


def _has_failed_search(payload: dict[str, object], *, stage: Stage) -> bool:
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
        expected_stage = {
            "govinfo-search": "govinfo_docket_root_search",
            "courtlistener-metadata-search": "courtlistener_docket_search",
            "govinfo-metadata-search": "govinfo_docket_search",
        }.get(stage, "docket_root_search")
        for node in trace:
            if not isinstance(node, dict) or node.get("stage") != expected_stage:
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
    govinfo_service: _PacedGovInfoClient,
    session: object | None,
) -> Document:
    if stage == "search":
        return await search_docket_roots(document, client=service)
    if stage == "unique-identity":
        return await validate_unique_docket_root_identities(document)
    if stage == "ambiguity-resolution":
        return await resolve_docket_root_ambiguities(document)
    if stage == "govinfo-search":
        return await lookup_govinfo_docket_roots(document, client=govinfo_service)
    if stage == "govinfo-unique-identity":
        return await validate_unique_govinfo_docket_root_identities(document)
    if stage == "govinfo-ambiguity-resolution":
        return await resolve_govinfo_docket_root_ambiguities(document)
    if stage == "docket-extraction-review-and-requeue":
        return await review_and_requeue_unresolved_docket_roots(document, client=service, session=session)
    if stage == "requeued-unique-identity":
        return await validate_unique_requeued_docket_root_identities(document)
    if stage == "requeued-ambiguity-resolution":
        return await resolve_requeued_docket_root_ambiguities(document)
    if stage == "metadata-shortlist":
        return await shortlist_docket_root_metadata_candidates(document)
    if stage == "semantic-resolution":
        return await resolve_docket_root_semantics(document, session=session)
    if stage == "courtlistener-metadata-search":
        return await search_courtlistener_docket_roots(document, client=service, session=session)
    if stage == "govinfo-metadata-search":
        return await search_govinfo_docket_roots(document, client=govinfo_service, session=session)
    msg = f"Unsupported docket-root validation stage: {stage!r}"
    raise ValueError(msg)


def _outcomes(payload: dict[str, object], *, stage: Stage) -> Counter[str]:
    """Read first-class docket-root judgements rather than inferring trace state."""
    if stage == "metadata-shortlist":
        return _metadata_shortlist_outcomes(payload)
    if stage in _SEARCH_STAGES:
        question = "docket_lookup"
    elif stage == "docket-extraction-review-and-requeue":
        question = "extraction_review"
    else:
        question = "identity"
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


def _metadata_shortlist_outcomes(payload: dict[str, object]) -> Counter[str]:
    """Count persisted shortlist decisions without inventing an identity judgement."""
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
        trace = citation.get("trace")
        if not isinstance(trace, list):
            continue
        for node in trace:
            if not isinstance(node, dict) or node.get("stage") != "docket_root_metadata_shortlist":
                continue
            details = node.get("details")
            validation = details.get("validation") if isinstance(details, dict) else None
            outcome = validation.get("outcome") if isinstance(validation, dict) else None
            if isinstance(outcome, str):
                outcomes[outcome] += 1
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "search",
            "unique-identity",
            "ambiguity-resolution",
            "govinfo-search",
            "govinfo-unique-identity",
            "govinfo-ambiguity-resolution",
            "docket-extraction-review-and-requeue",
            "requeued-unique-identity",
            "requeued-ambiguity-resolution",
            "metadata-shortlist",
            "semantic-resolution",
            "courtlistener-metadata-search",
            "govinfo-metadata-search",
        ),
        required=True,
    )
    parser.add_argument("--documents", type=Path, required=True, help="Input serialized Document directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output stage-artifact directory.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "With --resume and a search stage, recompute only Documents containing a failed "
            "retrieval request."
        ),
    )
    parser.add_argument(
        "--minimum-search-interval-seconds",
        type=float,
        default=0.0,
        help="Evaluation-only minimum spacing between provider search requests.",
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
