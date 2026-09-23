"""Run the explicit root pipeline and persist one Document artifact per stage.

Example:
    uv run python scripts/run_root_identity_pipeline.py \\
      --source /path/to/filing.txt \\
      --artifacts /path/to/run-artifacts/my-filing

Every ``@serialize()`` stage returns a Document and writes that same result to
``<artifacts>/<stage>/<source-stem>.json``. Any artifact can be restored with
``Document.model_validate(...)`` and passed to the next public stage.
Research runs are unrestricted unless a retrospective date is supplied.
For retrospective corpus evaluation, supply a JSON map of source filenames to
their actual drafting dates. A filing date is not assumed to be the draft date.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mellea_lrc.api import (
    Document,
    grow_roots,
    lookup_full_reporter_locators_exact,
    resolve_docket_root_identities,
    resolve_full_reporter_locator_ambiguities,
    resolve_full_reporter_search_candidates,
    resolve_open_web_root_identities,
    resolve_root_body_corroboration,
    review_full_reporter_exact_dates,
    review_shared_body_evidence,
    search_courtlistener_full_reporter_roots,
    search_govinfo_full_reporter_roots,
    search_open_web_roots,
    search_root_body_corroboration,
    stable,
    validate_unique_full_reporter_locator_identities,
)
from mellea_lrc.courtlistener import CourtListenerClient
from mellea_lrc.courtlistener.client import body_text_client
from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.serialization import artifact_directory, serialize
from mellea_lrc.serialization.source_provenance import (
    attach_source_provenance,
    read_source_provenance,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient


def _write_source_provenance_manifest(artifacts: Path, details: Mapping[str, str]) -> None:
    """Record exactly which opt-in sidecar shaped this evaluation run."""
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / "run-manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key, value in details.items():
        if key in existing and existing[key] != value:
            raise ValueError(f"{path}: existing run has a different {key}")
    existing.update(details)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@serialize()
async def grow_roots_document(
    document: Document,
    *,
    hunt_dockets: bool,
    session: MelleaSession | None,
) -> Document:
    """Extract complete locators, read fields, and form roots."""
    return await grow_roots(document, rules=stable(), hunt_dockets=hunt_dockets, session=session)


@serialize()
def persist_root_checkpoint(document: Document) -> Document:
    """Carry an already-grown native root checkpoint into this run's artifacts."""
    return document


@serialize()
async def resolve_docket_root_identities_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    govinfo_client: GovInfoClient,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Run the complete docket query-and-resolution drill."""
    return await resolve_docket_root_identities(
        document,
        client=client,
        govinfo_client=govinfo_client,
        session=session,
        retrospective_date=retrospective_date,
    )


@serialize()
async def lookup_full_reporter_locators_exact_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    retrospective_date: date | None = None,
) -> Document:
    """Retrieve exact reporter-citation candidates without judging them."""
    return await lookup_full_reporter_locators_exact(
        document, client=client, retrospective_date=retrospective_date
    )


@serialize()
async def validate_unique_full_reporter_locator_identities_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Validate exact reporter lookups that returned one candidate."""
    return await validate_unique_full_reporter_locator_identities(
        document, client=client, session=session, retrospective_date=retrospective_date
    )


@serialize()
async def review_full_reporter_exact_dates_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    retrospective_date: date | None = None,
) -> Document:
    """Persist the reserved date boundary after simple exact-date comparison."""
    return await review_full_reporter_exact_dates(
        document, client=client, retrospective_date=retrospective_date
    )


@serialize()
async def resolve_full_reporter_locator_ambiguities_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Resolve bounded ambiguous exact reporter-citation candidates."""
    return await resolve_full_reporter_locator_ambiguities(
        document, client=client, session=session, retrospective_date=retrospective_date
    )


@serialize()
async def search_courtlistener_full_reporter_roots_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Discover CourtListener metadata for unresolved reporter roots."""
    return await search_courtlistener_full_reporter_roots(
        document, client=client, session=session, retrospective_date=retrospective_date
    )


@serialize()
async def search_govinfo_full_reporter_roots_document(
    document: Document,
    *,
    client: GovInfoClient,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Discover GovInfo metadata for unresolved reporter roots."""
    return await search_govinfo_full_reporter_roots(
        document, client=client, session=session, retrospective_date=retrospective_date
    )


@serialize()
async def resolve_full_reporter_search_candidates_document(
    document: Document,
    *,
    session: MelleaSession | None,
    retrospective_date: date | None = None,
) -> Document:
    """Judge the metadata candidates collected for reporter roots."""
    return await resolve_full_reporter_search_candidates(
        document, session=session, retrospective_date=retrospective_date
    )


@serialize()
async def search_root_body_corroboration_document(
    document: Document,
    *,
    client: CourtListenerServiceClient,
    govinfo_client: GovInfoClient,
    retrospective_date: date | None = None,
) -> Document:
    """Search CourtListener and GovInfo bodies for unresolved roots."""
    return await search_root_body_corroboration(
        document,
        client=client,
        govinfo_client=govinfo_client,
        retrospective_date=retrospective_date,
    )


@serialize()
async def resolve_root_body_corroboration_document(
    document: Document,
    *,
    session: MelleaSession | None,
    client: CourtListenerServiceClient | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Finalize provider-backed body identity decisions in this runner."""
    return await resolve_root_body_corroboration(
        document,
        session=session,
        client=client,
        retrospective_date=retrospective_date,
    )


@serialize()
async def review_shared_body_evidence_document(
    document: Document, *, session: MelleaSession | None
) -> Document:
    """Rereview deferred roots using grounded citations already saved for a neighboring root."""
    return await review_shared_body_evidence(document, session=session)


@serialize()
async def search_open_web_roots_document(document: Document) -> Document:
    """Collect public search results for roots deferred by provider corpora."""
    return await search_open_web_roots(document)


@serialize()
async def resolve_open_web_root_identities_document(
    document: Document,
    *,
    session: MelleaSession | None,
) -> Document:
    """Fetch and ground public result pages, then write identity decisions."""
    return await resolve_open_web_root_identities(document, session=session)


async def run(
    source: Path,
    *,
    artifacts: Path,
    root_checkpoint: Path | None = None,
    hunt_dockets: bool = True,
    stop_after_root_formation: bool = False,
    stop_after_docket_identity: bool = False,
    client: CourtListenerServiceClient | None = None,
    govinfo_client: GovInfoClient | None = None,
    session: MelleaSession | None = None,
    retrospective_date: date | None = None,
    source_provenance: Path | None = None,
) -> Document:
    """Execute the explicit root pipeline shown in the project API design."""
    courtlistener = client if client is not None else CourtListenerClient()
    govinfo = govinfo_client if govinfo_client is not None else GovInfoClient()
    mellea = session if session is not None else start_mellea_session_from_env()

    provenance = None
    if source_provenance is not None:
        provenance, manifest_details = read_source_provenance(source_provenance)
        _write_source_provenance_manifest(artifacts, manifest_details)

    with artifact_directory(artifacts):
        document = Document.from_source(source)
        if provenance is not None:
            document = attach_source_provenance(document, provenance)

        # Extraction and root formation.
        if root_checkpoint is None:
            document = await grow_roots_document(document, hunt_dockets=hunt_dockets, session=mellea)
        else:
            saved_text = await asyncio.to_thread(root_checkpoint.read_text, encoding="utf-8")
            saved = Document.model_validate(json.loads(saved_text))
            if saved.passes[-1:] != ("root_formation",):
                raise ValueError(f"Not a root-formation checkpoint: {root_checkpoint}")
            if provenance is not None:
                # Root formation may have been saved before the evaluation
                # sidecar was attached. Apply the same trusted source metadata
                # before comparing and persisting the resumed checkpoint.
                saved = attach_source_provenance(saved, provenance)
            if saved.text != document.text or saved.source_metadata != document.source_metadata:
                raise ValueError(f"Root checkpoint differs from source or provenance: {root_checkpoint}")
            document = persist_root_checkpoint(saved)
        if stop_after_root_formation:
            return document

        # Docket roots: direct retrieval, source re-reading, metadata discovery,
        # and bounded semantic selection are one ordered identity stage.
        document = await resolve_docket_root_identities_document(
            document,
            client=courtlistener,
            govinfo_client=govinfo,
            session=mellea,
            retrospective_date=retrospective_date,
        )
        if stop_after_docket_identity:
            return document

        # Reporter roots: exact retrieval, unique identity, and ambiguity.
        document = await lookup_full_reporter_locators_exact_document(
            document, client=courtlistener, retrospective_date=retrospective_date
        )
        document = await validate_unique_full_reporter_locator_identities_document(
            document,
            client=courtlistener,
            session=mellea,
            retrospective_date=retrospective_date,
        )
        document = await review_full_reporter_exact_dates_document(
            document, client=courtlistener, retrospective_date=retrospective_date
        )
        document = await resolve_full_reporter_locator_ambiguities_document(
            document,
            client=courtlistener,
            session=mellea,
            retrospective_date=retrospective_date,
        )

        # Reporter metadata discovery and assessment.
        document = await search_courtlistener_full_reporter_roots_document(
            document,
            client=courtlistener,
            session=mellea,
            retrospective_date=retrospective_date,
        )
        document = await search_govinfo_full_reporter_roots_document(
            document,
            client=govinfo,
            session=mellea,
            retrospective_date=retrospective_date,
        )
        document = await resolve_full_reporter_search_candidates_document(
            document, session=mellea, retrospective_date=retrospective_date
        )

        # Corpus body corroboration is this runner's final identity stage.
        # Open-web functions remain independently callable for a deliberate
        # follow-up, but broad web results are not automatically consulted.
        document = await search_root_body_corroboration_document(
            document,
            client=courtlistener,
            govinfo_client=govinfo,
            retrospective_date=retrospective_date,
        )
        document = await resolve_root_body_corroboration_document(
            document,
            session=mellea,
            client=body_text_client(courtlistener),
            retrospective_date=retrospective_date,
        )
        document = await review_shared_body_evidence_document(document, session=mellea)
        return document


async def run_corpus(
    sources: tuple[Path, ...],
    *,
    artifacts: Path,
    hunt_dockets: bool = True,
    stop_after_root_formation: bool = False,
    stop_after_docket_identity: bool = False,
    retrospective_dates: Mapping[str, date] | None = None,
    require_retrospective_dates: bool = False,
    source_provenance: Path | None = None,
) -> tuple[Document, ...]:
    """Run the same explicit pipeline over clean corpus text files."""
    if require_retrospective_dates and retrospective_dates is None:
        raise ValueError(
            "Corpus identity evaluation requires a drafting-date map; "
            "pass retrospective_dates for every source document"
        )
    if retrospective_dates is not None:
        source_names = {source.name for source in sources}
        missing = sorted(source_names - retrospective_dates.keys())
        extra = sorted(retrospective_dates.keys() - source_names)
        if missing or extra:
            msg = f"Retrospective dates must match corpus filenames; missing={missing}, extra={extra}"
            raise ValueError(msg)
        if any(not isinstance(value, date) for value in retrospective_dates.values()):
            raise ValueError("Retrospective dates must be datetime.date values")
    courtlistener = CourtListenerClient()
    govinfo = GovInfoClient()
    mellea = start_mellea_session_from_env()
    documents: list[Document] = []
    for index, source in enumerate(sources, start=1):
        document = await run(
            source,
            artifacts=artifacts,
            hunt_dockets=hunt_dockets,
            stop_after_root_formation=stop_after_root_formation,
            stop_after_docket_identity=stop_after_docket_identity,
            client=courtlistener,
            govinfo_client=govinfo,
            session=mellea,
            retrospective_date=retrospective_dates[source.name] if retrospective_dates is not None else None,
            source_provenance=source_provenance,
        )
        documents.append(document)
        print(f"{index}/{len(sources)} {source.stem}: {len(document.citations)} citations")
    return tuple(documents)


def _clean_text_sources(directory: Path) -> tuple[Path, ...]:
    """Return benchmark text files, excluding disclosed pre-clean originals."""
    sources = tuple(path for path in sorted(directory.glob("*.txt")) if "(before clean)" not in path.name)
    if not sources:
        msg = f"{directory}: no clean .txt sources found"
        raise ValueError(msg)
    return sources


def _parse_retrospective_date(value: str) -> date:
    """Accept only a calendar date in the CLI's documented ISO format."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"Expected a retrospective date in YYYY-MM-DD format, got {value!r}")
    return date.fromisoformat(value)


def _read_retrospective_dates(path: Path) -> dict[str, date]:
    """Read a corpus map keyed by each source file's exact name."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in payload.items()
    ):
        raise ValueError(f"{path}: expected a JSON object mapping .txt filenames to YYYY-MM-DD dates")
    return {name: _parse_retrospective_date(value) for name, value in payload.items()}


def main() -> None:
    """Run one filing or one clean corpus through the persisted root pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="One source filing or plain-text document.")
    inputs.add_argument(
        "--documents",
        type=Path,
        help="Directory of corpus .txt files; disclosed '(before clean)' originals are skipped.",
    )
    parser.add_argument("--artifacts", type=Path, required=True, help="Directory for stage checkpoints.")
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="Optional evaluation sidecar with filings[.txt filename].docket_id; recorded in the run manifest.",
    )
    parser.add_argument(
        "--retrospective-date",
        type=_parse_retrospective_date,
        help="For --source, reject evidence dated after this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--retrospective-dates-json",
        type=Path,
        help="For --documents, JSON object mapping every .txt filename to its YYYY-MM-DD cutoff.",
    )
    parser.add_argument(
        "--retrospective-evaluation",
        action="store_true",
        help="Require an explicit cutoff for each document in a retrospective evaluation.",
    )
    parser.add_argument(
        "--no-hunt-dockets",
        action="store_true",
        help="Skip optional docket site hunting during root growth.",
    )
    parser.add_argument(
        "--stop-after-root-formation",
        action="store_true",
        help="Persist and stop after extraction has formed roots.",
    )
    parser.add_argument(
        "--stop-after-docket-identity",
        action="store_true",
        help="Persist and stop after the unified docket-root identity checkpoint.",
    )
    args = parser.parse_args()
    if args.source is not None and args.retrospective_dates_json is not None:
        parser.error("--retrospective-dates-json requires --documents")
    if args.documents is not None and args.retrospective_date is not None:
        parser.error("--retrospective-date requires --source")
    if args.retrospective_evaluation and args.source is not None and args.retrospective_date is None:
        parser.error("--retrospective-evaluation with --source requires --retrospective-date")
    if args.retrospective_evaluation and args.documents is not None and args.retrospective_dates_json is None:
        parser.error("--retrospective-evaluation with --documents requires --retrospective-dates-json")
    retrospective_dates = (
        _read_retrospective_dates(args.retrospective_dates_json)
        if args.retrospective_dates_json is not None
        else None
    )
    load_dotenv(".env")
    if args.source is not None:
        document = asyncio.run(
            run(
                args.source,
                artifacts=args.artifacts,
                hunt_dockets=not args.no_hunt_dockets,
                stop_after_root_formation=args.stop_after_root_formation,
                stop_after_docket_identity=args.stop_after_docket_identity,
                retrospective_date=args.retrospective_date,
                source_provenance=args.source_provenance,
            )
        )
        print(f"Completed {len(document.citations)} citation records; final pass: {document.passes[-1]}.")
        return
    if args.documents is None:
        msg = "Specify exactly one of --source or --documents"
        raise ValueError(msg)
    documents = asyncio.run(
        run_corpus(
            _clean_text_sources(args.documents),
            artifacts=args.artifacts,
            hunt_dockets=not args.no_hunt_dockets,
            stop_after_root_formation=args.stop_after_root_formation,
            stop_after_docket_identity=args.stop_after_docket_identity,
            retrospective_dates=retrospective_dates,
            require_retrospective_dates=args.retrospective_evaluation,
            source_provenance=args.source_provenance,
        )
    )
    print(f"Completed {len(documents)} documents; final pass: {documents[-1].passes[-1]}.")


if __name__ == "__main__":
    main()
