"""Opt-in source provenance sidecar support for evaluation runners.

Core stages never read sidecars. Callers explicitly load one and attach its
known source docket ID before a stage begins.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from mellea_lrc.model.document import Document


def read_source_provenance(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read the evaluation sidecar keyed by exact source filename."""
    content = path.read_bytes()
    payload = json.loads(content)
    filings = payload.get("filings") if isinstance(payload, dict) else None
    if not isinstance(filings, dict):
        raise ValueError(f"{path}: expected a JSON object with a 'filings' mapping")
    docket_ids: dict[str, str] = {}
    for name, entry in filings.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError(f"{path}: invalid filing entry")
        docket_id = entry.get("docket_id")
        if docket_id is None:
            continue
        if not isinstance(docket_id, str) or not docket_id.strip():
            raise ValueError(f"{path}: {name!r} has an invalid docket_id")
        docket_ids[name] = docket_id
    return docket_ids, {
        "source_provenance_path": str(path.resolve()),
        "source_provenance_sha256": hashlib.sha256(content).hexdigest(),
    }


def attach_source_provenance(document: Document, docket_ids: Mapping[str, str]) -> Document:
    """Attach known source provenance without changing unmatched documents."""
    source_path = document.source_metadata.path
    docket_id = docket_ids.get(Path(source_path).name) if source_path is not None else None
    if docket_id is None or document.source_metadata.courtlistener_docket_id == docket_id:
        return document
    return document.evolve(
        source_metadata=replace(document.source_metadata, courtlistener_docket_id=docket_id),
    )
