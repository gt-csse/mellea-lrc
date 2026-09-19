"""Evaluate reporter-root identity from persisted root-formation documents.

Each source document already carries the filing-internal root graph and its
deterministic case-name, court, and date readings. This runner performs only
reporter-root identity validation. Every completed filing remains a standalone,
serializable ``Document`` for the next stage.
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
    root_formation_documents: Path,
    output: Path,
    start: int,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Resume a deterministic first slice of root-formation documents."""
    paths = sorted(root_formation_documents.glob("*.json"))[start : start + limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = (
            f"{root_formation_documents} has {len(paths)} artifacts in "
            f"slice start={start}, limit={limit}"
        )
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
            document = Document.from_serialized(json.loads(path.read_text(encoding="utf-8")))
            document = await full_reporter_locator_identity(document, client=service, session=session)
            payload = document.serialize()
            Document.from_serialized(payload)
            _atomic_json(result_path, payload)
        outcomes.update(_summary(payload))
        model_statistics.update(_model_statistics(payload))
        result_paths.append(
            {
                "root_formation_document": path.name,
                "root_formation_document_sha256": _sha256(path),
                "full_reporter_locator_identity_result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(payload['citations'])} locators")

    config = llm_api_config_from_env(os.environ)
    return {
        "artifact_type": "full_reporter_locator_identity_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_root_formation_document_dir": str(root_formation_documents),
        "input_stage": "root_formation",
        "start_index": start,
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
        "--root-formation-documents",
        type=Path,
        required=True,
        help="Directory of serialized root-formation Documents.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for full-reporter-locator identity artifacts."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Zero-based filing index in sorted checkpoint artifacts."
    )
    parser.add_argument("--limit", type=int, default=5, help="Deterministic sorted filing count to run.")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse validated documents already in --output."
    )
    args = parser.parse_args()
    if args.start < 0 or args.limit < 1:
        parser.error("--start must be non-negative and --limit must be positive")

    load_dotenv(".env")
    manifest = asyncio.run(
        run(
            root_formation_documents=args.root_formation_documents,
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
