"""Resume root body corroboration from a saved Document checkpoint.

Example:
    uv run python scripts/run_root_body_checkpoint.py \
      --checkpoint /path/to/root_body_corroboration_search \
      --artifacts /path/to/new-run \
      --document 011

The serialized checkpoint contains all earlier extraction and identity
decisions. This runner does not call any upstream stage. ``--include-search``
starts with a metadata-resolution checkpoint. ``--retry-from`` reuses clean
body results and reruns documents whose earlier body pass had transient errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mellea_lrc.api import resolve_root_body_corroboration, search_root_body_corroboration
from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerConfig
from mellea_lrc.courtlistener.client import body_text_client
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.model.document import Document
from mellea_lrc.serialization import artifact_directory, serialize
from mellea_lrc.serialization.source_provenance import (
    attach_source_provenance,
    read_source_provenance,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient


@serialize()
async def search_saved_body_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    govinfo_client: GovInfoClient,
    retrospective_date: date | None = None,
) -> Document:
    """Refresh body candidates from the saved pre-search Document."""
    return await search_root_body_corroboration(
        document,
        client=client,
        govinfo_client=govinfo_client,
        retrospective_date=retrospective_date,
    )


@serialize()
async def resolve_saved_body_document(
    document: Document,
    *,
    session: MelleaSession | None,
    client: CourtListenerServiceClient,
    retrospective_date: date | None = None,
) -> Document:
    """Persist a body-stage Document using the same standalone stage API."""
    return await resolve_root_body_corroboration(
        document, session=session, client=client, retrospective_date=retrospective_date
    )


def _checkpoint_paths(directory: Path, selectors: tuple[str, ...]) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.glob("*.json")))
    if not paths:
        raise ValueError(f"No serialized Documents found in {directory}")
    if not selectors:
        return paths
    selected = tuple(
        path
        for path in paths
        if any(path.stem == selector or path.stem.startswith(f"{selector}__") for selector in selectors)
    )
    missing = [
        selector
        for selector in selectors
        if not any(path.stem == selector or path.stem.startswith(f"{selector}__") for path in selected)
    ]
    if missing:
        raise ValueError(f"No checkpoint Document matched: {', '.join(missing)}")
    return selected


def _retryable_body_failure(payload: dict[str, object]) -> bool:
    """Whether a saved body pass lost evidence to a transient provider error."""
    for citation in payload.get("citations", []):
        if not isinstance(citation, dict):
            continue
        for node in citation.get("trace", []):
            if not isinstance(node, dict) or not str(node.get("stage", "")).startswith("root_body_"):
                continue
            details = node.get("details", {})
            if not isinstance(details, dict):
                continue
            validation = details.get("validation", {})
            if isinstance(validation, dict):
                if _transient_error(validation.get("error")):
                    return True
                attempts = validation.get("search_attempts", [])
                if isinstance(attempts, list) and any(
                    isinstance(attempt, dict) and _transient_error(attempt.get("error"))
                    for attempt in attempts
                ):
                    return True
            fetches = details.get("fetches", [])
            if isinstance(fetches, list) and any(
                isinstance(fetch, dict)
                and fetch.get("outcome") == "failed"
                and _transient_error(fetch.get("error"))
                for fetch in fetches
            ):
                return True
    return False


def _transient_error(value: object) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.casefold()
    return any(fragment in lowered for fragment in ("429", "500", "502", "503", "504", "timed out"))


def _require_retry_directory(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"Prior body-resolution directory does not exist: {path}")


def _prior_checkpoint(retry_from: Path) -> Path:
    """Read the earlier run's input so changed upstream decisions are replayed."""
    manifest_path = retry_from.parent / "run-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Prior run has no input-checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_checkpoint")
    if not isinstance(source, str):
        raise ValueError(f"Prior run manifest has no source_checkpoint: {manifest_path}")
    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"Prior input checkpoint no longer exists: {source_path}")
    return source_path


def _presearch_prior_checkpoint(retry_from: Path) -> Path:
    """Follow one saved stage boundary back to the matching pre-search input.

    A body-resolution replay can start from saved body-search Documents. A
    later search refresh must compare against the input *before* that search,
    or every clean filing appears changed and wastes provider quota.
    """
    prior = _prior_checkpoint(retry_from)
    sample = next(iter(sorted(prior.glob("*.json"))), None)
    if sample is None:
        raise ValueError(f"Prior source checkpoint is empty: {prior}")
    payload = json.loads(sample.read_text(encoding="utf-8"))
    if "root_body_corroboration_search" in payload.get("passes", []):
        return _prior_checkpoint(prior)
    return prior


def _write_run_manifest(artifacts: Path, details: dict[str, object]) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "run-manifest.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _resume_run_manifest(artifacts: Path, expected: dict[str, object]) -> tuple[list[str], list[str]]:
    """Carry document provenance across a resumed, partially completed run."""
    path = artifacts / "run-manifest.json"
    if not path.is_file():
        return [], []
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Existing run manifest has a different {key}: {path}")
    return list(saved.get("rerun", [])), list(saved.get("reused", []))


async def run_checkpoint(
    checkpoint: Path,
    artifacts: Path,
    *,
    documents: tuple[str, ...] = (),
    retrospective_date: date | None = None,
    include_search: bool = False,
    courtlistener_pool: str | None = None,
    retry_from: Path | None = None,
    source_provenance: Path | None = None,
) -> tuple[Path, ...]:
    """Run body search when requested, then resolve selected Documents."""
    load_dotenv()
    paths = _checkpoint_paths(checkpoint, documents)
    if retry_from is not None and not include_search:
        raise ValueError("--retry-from requires --include-search so failed searches can be refreshed")
    if retry_from is not None and retrospective_date is not None:
        raise ValueError("A retry from older artifacts cannot introduce a different retrospective date")
    if retry_from is not None:
        _require_retry_directory(retry_from)
    prior_checkpoint = _presearch_prior_checkpoint(retry_from) if retry_from is not None else None
    if retrospective_date is not None and len(paths) != 1:
        raise ValueError("One retrospective date can only be applied to one selected checkpoint")
    config = CourtListenerConfig.from_env()
    if courtlistener_pool is not None:
        config = replace(config, token=None, pool=courtlistener_pool, retry_short_rate_limits=True)
    search_client = CourtListenerClient(config=config)
    body_client = body_text_client(search_client)
    govinfo = GovInfoClient()
    session = start_mellea_session_from_env()
    completed: list[Path] = []
    provenance = None
    provenance_details: dict[str, str] = {}
    if source_provenance is not None:
        provenance, provenance_details = read_source_provenance(source_provenance)
    prior_has_same_provenance = False
    if retry_from is not None and provenance is not None:
        prior_manifest = json.loads((retry_from.parent / "run-manifest.json").read_text(encoding="utf-8"))
        prior_has_same_provenance = (
            prior_manifest.get("source_provenance_sha256") == provenance_details["source_provenance_sha256"]
        )
    manifest_fields = {
        "source_checkpoint": str(checkpoint),
        "prior_resolution": str(retry_from) if retry_from is not None else None,
        "include_search": include_search,
        "courtlistener_search_pool": courtlistener_pool,
        "body_fetch_pool": config.pool or "proxy_default",
        "retrospective_date": retrospective_date.isoformat() if retrospective_date else None,
        **provenance_details,
    }
    rerun, reused = _resume_run_manifest(artifacts, manifest_fields)
    with artifact_directory(artifacts):
        for index, path in enumerate(paths, start=1):
            target = artifacts / "root_body_corroboration_resolution" / path.name
            if target.exists() and provenance is not None:
                saved = Document.model_validate(json.loads(target.read_text(encoding="utf-8")))
                expected = attach_source_provenance(saved, provenance)
                if expected.source_metadata != saved.source_metadata:
                    raise ValueError(f"{target}: existing resolution lacks the requested source provenance")
            if target.exists() and retrospective_date is not None:
                raise ValueError(
                    f"{target}: existing resolution cannot be reused with an explicit retrospective date"
                )
            if not target.exists():
                source_payload = json.loads(path.read_text(encoding="utf-8"))
                document = Document.model_validate(source_payload)
                if provenance is not None:
                    document = attach_source_provenance(document, provenance)
                    source_payload = document.model_dump(mode="json")
                previous = retry_from / path.name if retry_from is not None else None
                if previous is not None:
                    if not previous.is_file():
                        raise ValueError(f"Prior body-resolution Document is missing: {previous}")
                    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
                    if source_payload.get("text") != previous_payload.get("text"):
                        raise ValueError(f"Retry source text differs from saved resolution for {path.name}")
                    assert prior_checkpoint is not None
                    prior_source = prior_checkpoint / path.name
                    if not prior_source.is_file():
                        raise ValueError(f"Prior input Document is missing: {prior_source}")
                    prior_source_payload = json.loads(prior_source.read_text(encoding="utf-8"))
                    # Compare reconstructed Documents rather than raw JSON.
                    # Adding an optional serialized field to a node must not
                    # force clean filings through provider search again.
                    prior_source_payload = Document.model_validate(prior_source_payload).model_dump(
                        mode="json"
                    )
                    if prior_has_same_provenance:
                        # The prior runner attached this identical sidecar in
                        # memory. Compare effective inputs so a quota retry
                        # reuses clean results instead of replaying every
                        # document solely because its raw checkpoint predates
                        # that attachment.
                        prior_source_payload = attach_source_provenance(
                            Document.model_validate(prior_source_payload), provenance
                        ).model_dump(mode="json")
                    source_changed = source_payload != prior_source_payload
                    if previous_payload.get("source_metadata") != source_payload.get("source_metadata"):
                        source_changed = True
                    if not source_changed and not _retryable_body_failure(previous_payload):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(previous, target)
                        previous_search = retry_from.parent / "root_body_corroboration_search" / path.name
                        if previous_search.is_file():
                            search_target = artifacts / "root_body_corroboration_search" / path.name
                            search_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(previous_search, search_target)
                        reused.append(path.name)
                        completed.append(target)
                        print(f"{index}/{len(paths)} {path.stem}: reused {target}", flush=True)
                        continue
                if include_search:
                    document = await search_saved_body_document(
                        document,
                        client=search_client,
                        govinfo_client=govinfo,
                        retrospective_date=retrospective_date,
                    )
                await resolve_saved_body_document(
                    document,
                    session=session,
                    client=body_client,
                    retrospective_date=retrospective_date,
                )
                rerun.append(path.name)
            completed.append(target)
            print(f"{index}/{len(paths)} {path.stem}: {target}", flush=True)
    _write_run_manifest(
        artifacts,
        {
            **manifest_fields,
            "rerun": rerun,
            "reused": reused,
        },
    )
    return tuple(completed)


def main() -> None:
    """Select saved documents and run only the body resolution stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="Optional evaluation sidecar with filings[.txt filename].docket_id; recorded in the run manifest.",
    )
    parser.add_argument("--document", action="append", default=[], dest="documents")
    parser.add_argument(
        "--include-search",
        action="store_true",
        help="Input is a full-reporter metadata-resolution checkpoint; rerun body search before resolution.",
    )
    parser.add_argument(
        "--courtlistener-pool",
        choices=("main", "reserved"),
        help="Select a Modal proxy pool for searches. Body-text fetches use the proxy's main pool.",
    )
    parser.add_argument(
        "--retry-from",
        type=Path,
        help="Prior body-resolution directory; rerun only documents with transient provider errors.",
    )
    parser.add_argument(
        "--retrospective-date",
        help="For one selected document, reject evidence dated after YYYY-MM-DD; "
        "saved checkpoints already carry their original cutoff.",
    )
    args = parser.parse_args()
    cutoff = None
    if args.retrospective_date is not None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.retrospective_date) is None:
            parser.error("--retrospective-date must be YYYY-MM-DD")
        try:
            cutoff = date.fromisoformat(args.retrospective_date)
        except ValueError:
            parser.error("--retrospective-date must be a valid YYYY-MM-DD date")
    asyncio.run(
        run_checkpoint(
            args.checkpoint,
            args.artifacts,
            documents=tuple(args.documents),
            retrospective_date=cutoff,
            include_search=args.include_search,
            courtlistener_pool=args.courtlistener_pool,
            retry_from=args.retry_from,
            source_provenance=args.source_provenance,
        )
    )


if __name__ == "__main__":
    main()
