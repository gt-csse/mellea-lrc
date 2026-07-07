"""Regenerate strict pipeline snapshots without manually running the notebook.

Only two CLI options are supported:

    uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
        --file bookmarked --phase assessment
    uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
        --file 3 --phase retrieval
    uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus

``--file`` accepts ``bookmarked`` or a positive numeric text-fixture stem. When
it is omitted, the inclusive numeric range in ``CONFIG`` is processed. All
filesystem paths, the batch range, and assessment concurrency live in config;
they are deliberately not CLI options.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, TypeVar, cast

from mellea_lrc.assessment import (
    CitationAssessmentResult,
    MelleaCallContext,
    run_assessment_async,
)
from mellea_lrc.core.env import load_env_file
from mellea_lrc.extraction import run_extraction
from mellea_lrc.jurisdiction_inference import infer_jurisdiction
from mellea_lrc.llm import llm_api_config_from_env
from mellea_lrc.preprocessing import run_preprocessing
from mellea_lrc.retrieval import run_retrieval
from mellea_lrc.serialization import (
    deserialize_assessed_document,
    deserialize_extracted_document,
    deserialize_inferred_document,
    deserialize_preprocessed_document,
    deserialize_retrieved_document,
    serialize_assessed_document,
    serialize_extracted_document,
    serialize_inferred_document,
    serialize_preprocessed_document,
    serialize_retrieved_document,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from enum import Enum

    from mellea_lrc.assessment.types import AssessedDocument
    from mellea_lrc.retrieval.types import RetrievedDocument

Phase: TypeAlias = Literal["preprocessed", "extraction", "inferred", "retrieval", "assessment"]
T = TypeVar("T")
PHASES: tuple[Phase, ...] = (
    "preprocessed",
    "extraction",
    "inferred",
    "retrieval",
    "assessment",
)


class _HasStatus(Protocol):
    status: Enum


@dataclass(frozen=True, slots=True)
class SnapshotConfig:
    """All non-operational inputs for snapshot regeneration."""

    env_path: Path
    test_data_dir: Path
    bookmarked_path: Path
    snapshot_root: Path
    batch_start: int
    batch_end: int
    mellea_concurrency: int = 5

    def __post_init__(self) -> None:
        if self.batch_start < 1 or self.batch_end < self.batch_start:
            msg = "Snapshot batch range must be positive and inclusive"
            raise ValueError(msg)
        if self.mellea_concurrency < 1:
            msg = "mellea_concurrency must be positive"
            raise ValueError(msg)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = SnapshotConfig(
    env_path=PROJECT_ROOT / ".env",
    test_data_dir=PROJECT_ROOT / "local/test_data",
    bookmarked_path=PROJECT_ROOT / "fixtures/bookmarked/bookmarked.txt",
    snapshot_root=PROJECT_ROOT / "local/snapshots",
    batch_start=1,
    batch_end=10,
    mellea_concurrency=5,
)


def main() -> None:
    """Regenerate configured snapshots through the requested phase."""
    args = _parse_args()
    load_env_file(CONFIG.env_path, override=True)
    phase = cast("Phase", args.phase)
    paths = select_documents(CONFIG, args.file)
    _emit_json(
        {
            "phase": phase,
            "documents": [path.name for path in paths],
            "snapshot_root": str(CONFIG.snapshot_root),
        }
    )
    for path in paths:
        run_document(path, phase=phase, config=CONFIG)


def select_documents(config: SnapshotConfig, file_name: str | None) -> tuple[Path, ...]:
    """Resolve one CLI selector or the configured inclusive text-fixture range."""
    if file_name is not None:
        paths = (_path_for_selector(config, file_name),)
    else:
        paths = tuple(
            config.test_data_dir / f"{index}.txt"
            for index in range(config.batch_start, config.batch_end + 1)
        )
    missing = [path for path in paths if not path.is_file()]
    empty = [
        path
        for path in paths
        if path.is_file() and not path.read_text(encoding="utf-8").strip()
    ]
    if missing or empty:
        details = [f"missing: {path}" for path in missing]
        details.extend(f"empty: {path}" for path in empty)
        raise FileNotFoundError("Configured snapshot inputs are unavailable: " + "; ".join(details))
    return paths


def run_document(path: Path, *, phase: Phase, config: SnapshotConfig) -> None:
    """Run one text document from preprocessing through the selected phase."""
    snapshot_dir = config.snapshot_root / path.stem
    shutil.rmtree(snapshot_dir, ignore_errors=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"[{path.name}] start through {phase}")

    preprocessed_raw = run_preprocessing(path)
    preprocessed = _round_trip(
        snapshot_dir,
        "preprocessed",
        serialize_preprocessed_document(preprocessed_raw),
        deserialize_preprocessed_document,
    )
    _emit(f"[{path.name}] preprocessed chars={len(preprocessed.text)}")
    if phase == "preprocessed":
        return

    extraction_raw = run_extraction(preprocessed)
    extraction = _round_trip(
        snapshot_dir,
        "extraction",
        serialize_extracted_document(extraction_raw),
        deserialize_extracted_document,
    )
    _emit(f"[{path.name}] extraction citations={len(extraction.citations)}")
    if phase == "extraction":
        return

    inferred_raw = infer_jurisdiction(extraction)
    inferred = _round_trip(
        snapshot_dir,
        "inferred",
        serialize_inferred_document(inferred_raw),
        deserialize_inferred_document,
    )
    _emit(f"[{path.name}] inferred jurisdictions={len(inferred.jurisdictions)}")
    if phase == "inferred":
        return

    retrieval_raw = run_retrieval(inferred)
    retrieval = _round_trip(
        snapshot_dir,
        "retrieval",
        serialize_retrieved_document(retrieval_raw),
        deserialize_retrieved_document,
    )
    _emit_json({"document": path.name, "retrieval": _status_counts(retrieval.retrievals)})
    if phase == "retrieval":
        return

    assessment = asyncio.run(_assess(path, retrieval, config=config))
    llm_config = llm_api_config_from_env(os.environ)
    _emit_json(
        {
            "document": path.name,
            "assessment": _status_counts(assessment.assessments),
            "assessment_complete": assessment.assessment_complete,
            "api_base": llm_config.api_base,
            "model": llm_config.model,
        }
    )


async def _assess(
    path: Path,
    retrieval: RetrievedDocument,
    *,
    config: SnapshotConfig,
) -> AssessedDocument:
    started: dict[str, float] = {}

    def on_call(ctx: MelleaCallContext) -> None:
        started[ctx.citation_id] = time.perf_counter()
        _emit(
            f"[mellea] start doc={path.name} id={ctx.citation_id} "
            f"extracted={ctx.extracted_case_name!r}"
        )

    def on_done(ctx: MelleaCallContext, item: CitationAssessmentResult) -> None:
        elapsed = time.perf_counter() - started[ctx.citation_id]
        _emit(
            f"[mellea] done doc={path.name} id={ctx.citation_id} "
            f"case={item.case_name.initial.status.value} "
            f"followup={item.case_name.followup.status.value} ({elapsed:.1f}s)"
        )

    raw = await run_assessment_async(
        retrieval,
        mellea_concurrency=config.mellea_concurrency,
        on_mellea_call=on_call,
        on_mellea_done=on_done,
    )
    return _round_trip(
        config.snapshot_root / path.stem,
        "assessment",
        serialize_assessed_document(raw),
        deserialize_assessed_document,
    )


def _round_trip(
    snapshot_dir: Path,
    stage: Phase,
    payload: dict[str, object],
    deserialize: Callable[[dict[str, object]], T],
) -> T:
    """Write and immediately validate a strict current-schema snapshot."""
    snapshot_path = snapshot_dir / f"{stage}.json"
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    restored_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return deserialize(restored_payload)


def _path_for_selector(config: SnapshotConfig, selector: str) -> Path:
    if selector == "bookmarked":
        return config.bookmarked_path
    if not selector.isascii() or not selector.isdecimal() or int(selector) < 1:
        msg = "--file must be 'bookmarked' or a positive integer such as 1, 2, or 3"
        raise ValueError(msg)
    return config.test_data_dir / f"{int(selector)}.txt"


def _status_counts(items: tuple[_HasStatus, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = item.status.value
        counts[status] = counts.get(status, 0) + 1
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=None, help="bookmarked or a numeric text fixture stem")
    parser.add_argument("--phase", choices=PHASES, default="assessment")
    return parser.parse_args()


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _emit_json(payload: object) -> None:
    _emit(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
