"""Persist and evaluate the complete-locator chain one document stage at a time.

Each document artifact holds four *cumulative*, citation-centric ``Document``
payloads.  A checkpoint can be deserialized and handed directly to the next
stage; the runner deliberately does so after every write.

The configured chain is::

    full_reporter_locator_rule
      -> docket_locator_rule
      -> full_reporter_locator_site_hunting  # retained, intentionally not run
      -> docket_locator_site_hunting

Reporter site hunting is represented as a documented no-op because it has low
recovery yield for its model cost. The final checkpoint holds every locator
occurrence before the independent co-location projection. The separate
court/date report resumes from it, forms co-location once, and never re-runs a
locator reader or calls a model.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mellea_lrc.core.citations import DocketCitation, FullCaseCitation
from mellea_lrc.extraction import (
    DOCKET_RULE_STAGE,
    DOCKET_SITE_STAGE,
    REPORTER_RULE_STAGE,
    REPORTER_SITE_STAGE,
    find_docket_locators,
    find_full_reporter_locators,
    mark_full_reporter_locator_hunting_skipped,
    resolve_colocations,
    resolve_courts,
    resolve_dates,
    stable,
    start_locator_document,
)
from mellea_lrc.extraction.adjudication import hunt_docket_locators
from mellea_lrc.extraction.types import Document
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.serialization import deserialize_document, serialize_document

DATASETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
)
CHECKPOINTS = (
    REPORTER_RULE_STAGE,
    DOCKET_RULE_STAGE,
    REPORTER_SITE_STAGE,
    DOCKET_SITE_STAGE,
)
_SCHEMA_VERSION = 2
_REPORTER_SITE_REASON = (
    "Disabled for this run: reporter site hunting has low recovery yield relative to model cost."
)

SpanKey = tuple[str, int, int]
FieldKey = tuple[str, int, int, str]


@dataclass(frozen=True, slots=True)
class Corpus:
    """The active filing text and optional annotation files for one dataset."""

    name: str
    texts: Path
    annotations: Path


def _corpus(data: Path, name: str) -> Corpus:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}")
    root = data / name
    return Corpus(
        name=name,
        texts=root / ("documents_txt" if name == "primary" else "filings_txt"),
        annotations=root / "documents",
    )


def _atomic_json(path: Path, value: object) -> None:
    """Write one recoverable artifact without leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _source_metadata(path: Path, data: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(data)),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "length": len(text),
    }


def _artifact_path(run_root: Path, document: str) -> Path:
    return run_root / "documents" / f"{Path(document).stem}.json"


def _new_artifact(path: Path, *, data: Path) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": "locator_checkpoint_chain",
        "document": path.name,
        "text": _source_metadata(path, data),
        "checkpoints": {},
    }


def _read_artifact(path: Path) -> dict[str, Any] | None:
    """Read a possibly partial chain and validate every persisted handoff."""
    if not path.exists():
        return None
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema_version") != _SCHEMA_VERSION
        or artifact.get("artifact_type") != "locator_checkpoint_chain"
    ):
        raise ValueError(f"{path}: not a locator-checkpoint-chain artifact")
    checkpoints = artifact.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError(f"{path}: checkpoints must be an object")
    unknown = set(checkpoints) - set(CHECKPOINTS)
    if unknown:
        raise ValueError(f"{path}: unknown checkpoint names: {sorted(unknown)!r}")
    for stage, payload in checkpoints.items():
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: {stage} is not a serialized Document")
        deserialize_document(payload)
    _assert_checkpoint_prefix(path, checkpoints)
    return artifact


def _assert_checkpoint_prefix(path: Path, checkpoints: dict[str, object]) -> None:
    """A resumed artifact must contain a contiguous prefix of the four stages."""
    present = set(checkpoints)
    expected = set(CHECKPOINTS[: len(present)])
    if present != expected:
        raise ValueError(f"{path}: checkpoints must be a contiguous chain, got {sorted(present)!r}")


def _stored_document(artifact: dict[str, Any], stage: str) -> Document | None:
    payload = artifact["checkpoints"].get(stage)
    return deserialize_document(payload) if isinstance(payload, dict) else None


def _store_checkpoint(
    artifact_path: Path,
    artifact: dict[str, Any],
    stage: str,
    document: Document,
) -> Document:
    """Persist a stage and immediately rehydrate the exact next-stage input."""
    checkpoints = artifact["checkpoints"]
    if stage not in CHECKPOINTS:
        raise ValueError(f"Unexpected locator checkpoint {stage!r}")
    expected = CHECKPOINTS[len(checkpoints)] if stage not in checkpoints else stage
    if expected != stage:
        raise ValueError(f"Cannot write {stage!r}; next checkpoint is {expected!r}")
    checkpoints[stage] = serialize_document(document)
    _atomic_json(artifact_path, artifact)
    return deserialize_document(checkpoints[stage])


async def run_document(
    path: Path,
    *,
    data: Path,
    artifact_path: Path,
    resume: bool,
) -> dict[str, Any]:
    """Run or resume the four checkpoint stages for one filing."""
    artifact = _read_artifact(artifact_path) if resume else None
    if artifact is None:
        artifact = _new_artifact(path, data=data)
    elif artifact.get("text") != _source_metadata(path, data):
        raise ValueError(f"{path}: text changed since its locator checkpoint was written")

    document = _stored_document(artifact, DOCKET_SITE_STAGE)
    if document is not None:
        return artifact

    rules = stable()
    document = _stored_document(artifact, REPORTER_SITE_STAGE)
    if document is None:
        document = _stored_document(artifact, DOCKET_RULE_STAGE)
    if document is None:
        document = _stored_document(artifact, REPORTER_RULE_STAGE)
    if document is None:
        document = start_locator_document(preprocess(path), rules=rules)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = find_full_reporter_locators(document, rules=rules)
        document = _store_checkpoint(artifact_path, artifact, REPORTER_RULE_STAGE, document)

    if DOCKET_RULE_STAGE not in artifact["checkpoints"]:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            document = find_docket_locators(document, rules=rules)
        document = _store_checkpoint(artifact_path, artifact, DOCKET_RULE_STAGE, document)

    if REPORTER_SITE_STAGE not in artifact["checkpoints"]:
        document = mark_full_reporter_locator_hunting_skipped(document, reason=_REPORTER_SITE_REASON)
        document = _store_checkpoint(artifact_path, artifact, REPORTER_SITE_STAGE, document)

    if DOCKET_SITE_STAGE not in artifact["checkpoints"]:
        document = await hunt_docket_locators(document, rules=rules)
        _store_checkpoint(artifact_path, artifact, DOCKET_SITE_STAGE, document)

    return artifact


def _span(row: dict[str, Any], key: str = "locator") -> tuple[int, int] | None:
    value = row.get(key)
    if not isinstance(value, dict) or "start" not in value or "end" not in value:
        return None
    return int(value["start"]), int(value["end"])


def _annotation_gold(corpus: Corpus) -> tuple[dict[str, set[tuple[int, int]]], dict[FieldKey, str]]:
    """Read locator spans plus explicitly annotated courts and dates."""
    locators: dict[str, set[tuple[int, int]]] = {}
    fields: dict[FieldKey, str] = {}
    if not corpus.annotations.exists():
        return locators, fields
    for path in sorted(corpus.annotations.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or rows[0].get("unit") != "header":
            continue
        document = str(rows[0]["document"])
        for row in rows[1:]:
            if row.get("unit") != "citation" or row.get("kind") not in {"FullCaseCitation", "DocketCitation"}:
                continue
            span = _span(row)
            if span is None:
                continue
            locators.setdefault(document, set()).add(span)
            court = row.get("court")
            if isinstance(court, dict) and isinstance(court.get("id"), str):
                fields[(document, *span, "court")] = court["id"]
            date = row.get("date")
            if isinstance(date, dict) and isinstance(date.get("normalized"), str):
                fields[(document, *span, "date")] = date["normalized"]
    return locators, fields


def _locator_spans(
    document: Document, kind: type[FullCaseCitation] | type[DocketCitation]
) -> set[tuple[int, int]]:
    return {
        (record.locator_span.start, record.locator_span.end)
        for record in document.active_citations
        if isinstance(record.stated, kind)
    }


def _score(predicted: set[SpanKey], gold: set[SpanKey]) -> dict[str, int | float]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "tp": true_positive,
        "fp": len(predicted - gold),
        "fn": len(gold - predicted),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _date_value(record: Any) -> str | None:
    date = getattr(record.stated, "date", None)
    if date is None or not date.year:
        return None
    if not date.month:
        return str(date.year)
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    month_text = str(date.month).strip().casefold().rstrip(".")
    month = int(month_text) if month_text.isdigit() else months.get(month_text[:3])
    if month is None:
        return str(date.year)
    if not date.day:
        return f"{date.year}-{month:02d}"
    return f"{date.year}-{month:02d}-{int(date.day):02d}"


def _field_rows(
    artifacts: list[dict[str, Any]],
    gold: dict[FieldKey, str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int | float]]]:
    """Resume from final locator documents and score only court/date readings."""
    rows: list[dict[str, Any]] = []
    predicted: dict[str, set[FieldKey]] = {"court": set(), "date": set()}
    correct: dict[str, set[FieldKey]] = {"court": set(), "date": set()}
    for artifact in artifacts:
        document_name = str(artifact["document"])
        locator_document = _stored_document(artifact, DOCKET_SITE_STAGE)
        if locator_document is None:
            continue
        # This is deliberately a continuation from checkpoint four, never a
        # second extraction. Both readers are deterministic and independent.
        grouped = resolve_colocations(locator_document, rules=stable())
        resolved = resolve_dates(resolve_courts(grouped, rules=stable()), rules=stable())
        for record in resolved.active_citations:
            if not isinstance(record.stated, (FullCaseCitation, DocketCitation)):
                continue
            span = record.locator_span
            values = {"court": record.stated.court, "date": _date_value(record)}
            for field, value in values.items():
                key = (document_name, span.start, span.end, field)
                expected = gold.get(key)
                if value is not None:
                    predicted[field].add(key)
                if value is not None and value == expected:
                    correct[field].add(key)
                rows.append(
                    {
                        "document": document_name,
                        "locator": {"start": span.start, "end": span.end},
                        "kind": type(record.stated).__name__,
                        "field": field,
                        "predicted": value,
                        "annotated": expected,
                        "outcome": (
                            "correct"
                            if value is not None and value == expected
                            else "missing"
                            if value is None and expected is not None
                            else "unannotated"
                            if expected is None
                            else "wrong"
                        ),
                    }
                )
    metric: dict[str, dict[str, int | float]] = {}
    for field in ("court", "date"):
        gold_keys = {key for key in gold if key[-1] == field}
        true_positive = len(correct[field])
        read = len(predicted[field])
        precision = true_positive / read if read else 0.0
        recall = true_positive / len(gold_keys) if gold_keys else 0.0
        metric[field] = {
            "gold": len(gold_keys),
            "read": read,
            "tp": true_positive,
            "fp": read - true_positive,
            "fn": len(gold_keys - correct[field]),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    return sorted(rows, key=lambda row: (row["document"], row["locator"]["start"], row["field"])), metric


def _commit() -> str | None:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True)
    return completed.stdout.strip() or None


def _write_reports(
    *,
    run_root: Path,
    corpus: Corpus,
    artifacts: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    locator_gold, field_gold = _annotation_gold(corpus)
    reporter_rule: set[SpanKey] = set()
    docket_rule: set[SpanKey] = set()
    docket_final: set[SpanKey] = set()
    for artifact in artifacts:
        name = str(artifact["document"])
        reporter = _stored_document(artifact, REPORTER_RULE_STAGE)
        docket = _stored_document(artifact, DOCKET_RULE_STAGE)
        final = _stored_document(artifact, DOCKET_SITE_STAGE)
        if reporter is not None:
            reporter_rule |= {(name, *span) for span in _locator_spans(reporter, FullCaseCitation)}
        if docket is not None:
            docket_rule |= {(name, *span) for span in _locator_spans(docket, DocketCitation)}
        if final is not None:
            docket_final |= {(name, *span) for span in _locator_spans(final, DocketCitation)}

    reporter_gold = {
        (name, start, end)
        for name, spans in locator_gold.items()
        for start, end in spans
        if _gold_kind(corpus, name, start, end) == "FullCaseCitation"
    }
    docket_gold = {
        (name, start, end)
        for name, spans in locator_gold.items()
        for start, end in spans
        if _gold_kind(corpus, name, start, end) == "DocketCitation"
    }
    field_rows, field_metrics = _field_rows(artifacts, field_gold)
    _atomic_jsonl(run_root / "court-date-readings.jsonl", field_rows)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_type": "locator_checkpoint_chain",
        "dataset": corpus.name,
        "created_at": started_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "source_commit": _commit(),
        "chain": list(CHECKPOINTS),
        "reporter_site_hunting": {"enabled": False, "reason": _REPORTER_SITE_REASON},
        "documents_completed": len(artifacts),
        "metrics": {
            "full_reporter_locator_rule": _score(reporter_rule, reporter_gold),
            "docket_locator_rule": _score(docket_rule, docket_gold),
            "docket_locator_after_site_hunting": _score(docket_final, docket_gold),
            "court_date_from_final_locator_checkpoint": field_metrics,
        },
        "court_date_readings": "court-date-readings.jsonl",
        "document_artifacts": "documents/",
    }
    _atomic_json(run_root / "manifest.json", manifest)
    return manifest


def _gold_kind(corpus: Corpus, document: str, start: int, end: int) -> str | None:
    """Read a locator annotation's kind without relying on predicted data."""
    source = corpus.annotations / f"{Path(document).stem}.jsonl"
    if not source.exists():
        return None
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        span = _span(row)
        if span == (start, end):
            kind = row.get("kind")
            return kind if isinstance(kind, str) else None
    return None


async def run(*, data: Path, dataset: str, output: Path, resume: bool) -> dict[str, Any]:
    """Run every active filing, retaining an artifact after each completed stage."""
    load_dotenv(".env")
    corpus = _corpus(data, dataset)
    paths = [path for path in sorted(corpus.texts.glob("*.txt")) if " (before clean)" not in path.name]
    if not paths:
        raise ValueError(f"{corpus.texts}: no active filing texts")
    run_root = output / dataset
    started_at = datetime.now(UTC).isoformat()
    artifacts: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        artifact = await run_document(
            path,
            data=data,
            artifact_path=_artifact_path(run_root, path.name),
            resume=resume,
        )
        artifacts.append(artifact)
        final = _stored_document(artifact, DOCKET_SITE_STAGE)
        print(
            f"[{index}/{len(paths)}] {dataset} {path.name[:36]:<36} "
            f"locators={len(final.locators) if final else 0:3}",
            flush=True,
        )
    return _write_reports(run_root=run_root, corpus=corpus, artifacts=artifacts, started_at=started_at)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/run-artifacts/1-roots/locator-checkpoint-chain"),
    )
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    manifest = asyncio.run(
        run(
            data=arguments.data,
            dataset=arguments.dataset,
            output=arguments.output,
            resume=not arguments.no_resume,
        )
    )
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
