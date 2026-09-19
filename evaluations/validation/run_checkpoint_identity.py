"""Run a bounded identity-only evaluation from persisted locator checkpoints.

The source checkpoint has already completed locator admission, colocation, and
the deterministic court, date, and case-name readers.  This runner resumes
those exact ``Document`` objects, invokes only the identity progression, and
writes one recoverable ``ValidatedDocument`` trace per filing.  Pinpoint page
retrieval is intentionally outside this run.
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

from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.llm import llm_api_config_from_env, start_mellea_session_from_env
from mellea_lrc.serialization import (
    deserialize_document,
    deserialize_validated_document,
    serialize_validated_document,
)
from mellea_lrc.validation import validate_document_identity


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
        nodes = citation["nodes"]
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
            outcomes["missing_identity_result"] += 1
    return outcomes


def _model_statistics(payload: dict[str, Any]) -> Counter[str]:
    """Count persisted IVR attempts rather than inferring model work from nodes."""
    statistics: Counter[str] = Counter()
    for citation in payload["citations"]:
        for node in citation["nodes"]:
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


async def run(
    *,
    checkpoints: Path,
    output: Path,
    limit: int,
    resume: bool,
) -> dict[str, object]:
    """Resume a deterministic first slice of checkpoint documents."""
    paths = sorted(checkpoints.glob("*.json"))[:limit]  # noqa: ASYNC240
    if len(paths) != limit:
        msg = f"{checkpoints} has {len(paths)} checkpoints, fewer than requested limit={limit}"
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
            deserialize_validated_document(payload)
        else:
            document = deserialize_document(json.loads(path.read_text(encoding="utf-8")))
            validated = await validate_document_identity(document, client=service, session=session)
            payload = serialize_validated_document(validated)
            deserialize_validated_document(payload)
            _atomic_json(result_path, payload)
        outcomes.update(_summary(payload))
        model_statistics.update(_model_statistics(payload))
        result_paths.append(
            {
                "checkpoint": path.name,
                "checkpoint_sha256": _sha256(path),
                "identity_result": str(result_path.relative_to(output)),
            }
        )
        print(f"{index}/{len(paths)} {path.stem}: {len(payload['citations'])} locators")

    config = llm_api_config_from_env(os.environ)
    return {
        "artifact_type": "checkpoint_identity_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "input_checkpoint_dir": str(checkpoints),
        "input_stage": "case_names",
        "documents": result_paths,
        "document_count": len(result_paths),
        "identity_scope": (
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
        "--checkpoints", type=Path, required=True, help="Directory of case-name checkpoint documents."
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for identity-only artifacts.")
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
            checkpoints=args.checkpoints,
            output=args.output,
            limit=args.limit,
            resume=args.resume,
        )
    )
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest["outcomes"], sort_keys=True))


if __name__ == "__main__":
    main()
