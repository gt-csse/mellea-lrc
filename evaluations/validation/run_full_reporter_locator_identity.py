"""Evaluate reporter identity from persisted locator-chain artifacts.

Each source artifact supplies its cumulative ``colocation`` ``Document``.
This runner resumes that document, applies the deterministic case-name, court,
and date readers, then invokes only reporter identity. Every completed filing
is a standalone serialized ``Document`` that a later stage can deserialize.
Pinpoint page retrieval is intentionally outside this run.
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
from typing import Any

from dotenv import load_dotenv

from mellea_lrc.api import (
    Document,
    full_reporter_locator_identity,
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    stable,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env


def _atomic_json(path: Path, value: object) -> None:
    """Write JSON without leaving a partial run artifact behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """Return a source checkpoint's byte hash for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _colocation_document(path: Path) -> Document:
    """Restore the final locator document from one locator-chain artifact."""
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
    return Document.from_serialized(payload)


def _prepare_identity_fields(document: Document) -> Document:
    """Fill only the deterministic fields reporter identity may compare."""
    rules = stable()
    document = resolve_case_names(document, rules=rules)
    document = resolve_courts(document, rules=rules)
    return resolve_dates(document, rules=rules)


def _summary(payload: dict[str, Any]) -> Counter[str]:
    """Count the terminal identity decision, keeping evidence summaries separate."""
    outcomes: Counter[str] = Counter()
    for citation in payload["citations"]:
        nodes = _validation_nodes(citation)
        lookup = next((node for node in nodes if node["node_type"] == "ExactLocatorLookupNode"), None)
        resolution = next(
            (node for node in nodes if node["node_type"] == "LocatorIdentityResolutionNode"),
            None,
        )
        summary = next(
            (
                node
                for node in nodes
                if node["node_type"] in {"LocatorCitationSummaryNode", "SearchCitationSummaryNode"}
            ),
            None,
        )
        if resolution is not None:
            outcomes[f"identity_resolution:{resolution['outcome']}"] += 1
        elif summary is not None:
            outcomes[f"identity_summary:{summary['overall_outcome']}"] += 1
        elif lookup is not None:
            outcomes[f"lookup:{lookup['outcome']}"] += 1
        else:
            outcomes["outside_full_reporter_locator_scope"] += 1
    return outcomes


def _model_statistics(payload: dict[str, Any]) -> Counter[str]:
    """Count persisted IVR attempts rather than inferring model work from nodes."""
    statistics: Counter[str] = Counter()
    for citation in payload["citations"]:
        for node in _validation_nodes(citation):
            run = node.get("run")
            if not isinstance(run, dict):
                continue
            attempts = run.get("attempts")
            if not isinstance(attempts, list):
                continue
            statistics["ivr_runs"] += 1
            statistics["provider_attempts"] += len(attempts)
            if len(attempts) > 1:
                statistics["resampled_runs"] += 1
            if not run.get("success"):
                statistics["failed_runs"] += 1
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                for requirement in attempt.get("requirements", []):
                    if not isinstance(requirement, dict) or requirement.get("passed"):
                        continue
                    description = requirement.get("description")
                    if description == "Return exactly one JSON object matching the required output schema.":
                        statistics["schema_rejected_attempts"] += 1
                    elif description == "parties must be copied before locator in local_context":
                        statistics["grounding_rejected_attempts"] += 1
    return statistics


def _validation_nodes(citation: dict[str, Any]) -> list[dict[str, Any]]:
    """Read validation payloads held by this citation's stage-neutral trace."""
    nodes: list[dict[str, Any]] = []
    for trace_node in citation.get("trace", []):
        details = trace_node.get("details")
        if not isinstance(details, dict):
            continue
        node_type = details.get("validation_node_type")
        payload = details.get("validation")
        if not isinstance(node_type, str) or not isinstance(payload, dict):
            continue
        nodes.append({"node_type": node_type, **payload})
    return nodes


async def run(
    *,
    locator_checkpoints: Path,
    output: Path,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Resume a deterministic first slice of locator-chain artifacts."""
    paths = sorted(locator_checkpoints.glob("*.json"))[:limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{locator_checkpoints} has {len(paths)} artifacts, fewer than requested limit={limit}"
        raise ValueError(msg)

    service = CourtListenerClient()
    session = start_mellea_session_from_env()
    result_paths: list[dict[str, str]] = []
    outcomes: Counter[str] = Counter()
    model_statistics: Counter[str] = Counter()

    for index, path in enumerate(paths, start=1):
        result_path = output / "documents" / path.name
        if resume and result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            Document.from_serialized(payload)
        else:
            document = _prepare_identity_fields(_colocation_document(path))
            document = await full_reporter_locator_identity(document, client=service, session=session)
            payload = document.serialize()
            Document.from_serialized(payload)
            _atomic_json(result_path, payload)
        outcomes.update(_summary(payload))
        model_statistics.update(_model_statistics(payload))
        result_paths.append(
            {
                "locator_checkpoint_artifact": path.name,
                "locator_checkpoint_artifact_sha256": _sha256(path),
                "full_reporter_locator_identity_result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(payload['citations'])} locators")

    config = llm_api_config_from_env(os.environ)
    return {
        "artifact_type": "full_reporter_locator_identity_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_locator_checkpoint_dir": str(locator_checkpoints),
        "input_stage": "colocation",
        "field_preparation": ["case_names", "courts", "dates"],
        "documents": result_paths,
        "document_count": len(result_paths),
        "full_reporter_locator_identity_scope": (
            "full reporter exact lookup, bounded candidate field checks, and "
            "deterministic or grounded model candidate choice"
        ),
        "excluded_stages": ["lookup-miss search", "docket lookup", "leaf_growth", "pinpoint"],
        "model": config.model,
        "model_statistics": dict(sorted(model_statistics.items())),
        "outcomes": dict(sorted(outcomes.items())),
    }


def main() -> None:
    """Run the bounded checkpoint evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locator-checkpoints",
        type=Path,
        required=True,
        help="Directory of locator_checkpoint_chain document artifacts.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for full-reporter-locator identity artifacts."
    )
    parser.add_argument("--limit", type=int, default=5, help="Deterministic sorted filing count to run.")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse validated documents already in --output."
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    load_dotenv(".env")
    manifest = asyncio.run(
        run(
            locator_checkpoints=args.locator_checkpoints,
            output=args.output,
            limit=args.limit,
            resume=args.resume,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
