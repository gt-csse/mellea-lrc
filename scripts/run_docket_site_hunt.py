"""Run and resume live docket site hunting on selected annotated documents.

Only manifest-listed text and index spans reach the rule readers and reviewer.
Each completed Document is saved before its annotation file is opened. A failed
provider request stops the run; previously completed documents remain reusable.

Examples, from the repository root::

    uv run python scripts/run_docket_site_hunt.py --set primary --plan
    uv run python scripts/run_docket_site_hunt.py --set primary --document 001 --service-tier default
    uv run python scripts/run_docket_site_hunt.py --set primary --service-tier default
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from mellea_lrc.extraction import find_docket_locators, find_full_reporter_locators, hunt_docket_locators
from mellea_lrc.extraction.docket_hunting import STAGE, DocketSiteReviewer
from mellea_lrc.llm.docket_review import OpenRouterDocketReviewer
from mellea_lrc.model import Document, FullDocketCitation, Span
from mellea_lrc.model.preprocessed_document import PreprocessingBackend, PreprocessingMetadata
from mellea_lrc.model.source import SourceFormat, SourceMetadata
from mellea_lrc.preprocessing.document_index import is_within

SETS = (
    "primary",
    "hallucination-set-1",
    "hallucination-set-2",
    "reliable-high-profile",
    "reliable-low-profile",
    "final-held-out",
)
COUNT_FIELDS = (
    "documents",
    "gold_docket_locators",
    "rule_locators",
    "rule_exact",
    "reviewed_sites",
    "hunt_admissions",
    "hunt_exact",
    "hunt_nonmatching",
    "hunt_declined",
    "hunt_failed",
    "combined_locators",
    "combined_exact",
    "combined_nonmatching",
    "remaining_gold_misses",
)


def _span(raw: dict[str, Any]) -> Span:
    return Span(start=int(raw["start"]), end=int(raw["end"]))


def _manifest(data_root: Path, name: str) -> dict[str, dict[str, Any]]:
    return json.loads((data_root / name / "documents.json").read_text(encoding="utf-8"))["documents"]


def _matches(filename: str, selector: str) -> bool:
    return filename == selector or Path(filename).stem == selector or filename.startswith(f"{selector}__")


def select_documents(
    data_root: Path,
    names: tuple[str, ...],
    selectors: tuple[str, ...] = (),
    max_documents: int | None = None,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    """Select only filenames listed in the chosen sets' documents.json files."""
    if not names or any(name not in SETS for name in names):
        raise ValueError("Select one or more known annotated sets")
    if max_documents is not None and max_documents < 1:
        raise ValueError("--max-documents must be positive")
    manifests = {name: _manifest(data_root, name) for name in dict.fromkeys(names)}
    all_items = [
        (name, filename, metadata)
        for name, manifest in manifests.items()
        for filename, metadata in sorted(manifest.items())
    ]
    if selectors:
        chosen: set[tuple[str, str]] = set()
        for raw in selectors:
            scoped_name, separator, selector = raw.partition("/")
            if separator and (scoped_name not in manifests or not selector):
                raise ValueError(f"Unknown document selector: {raw}")
            matches = [
                (name, filename)
                for name, filename, _ in all_items
                if (not separator or name == scoped_name)
                and _matches(filename, selector if separator else raw)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Document selector {raw!r} matched {len(matches)} manifest files; use SET/NAME"
                )
            chosen.add(matches[0])
        all_items = [item for item in all_items if (item[0], item[1]) in chosen]
    return tuple(all_items[:max_documents])


def _text_path(data_root: Path, name: str, filename: str) -> Path:
    directory = "documents_txt" if name == "primary" else "filings_txt"
    return data_root / name / directory / filename


def _source_document(data_root: Path, name: str, filename: str, metadata: dict[str, Any]) -> Document:
    path = _text_path(data_root, name, filename)
    source = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if digest != metadata["sha256"] or len(source) != metadata["length"]:
        raise ValueError(f"{path}: text differs from documents.json")
    return Document(
        source_metadata=SourceMetadata(path=str(path), format=SourceFormat.TEXT),
        text=source,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend(metadata.get("backend", "plain_text")),
            backend_version=metadata.get("backend_version"),
        ),
        index_spans=tuple(_span(raw) for raw in metadata.get("index_spans", ())),
    )


def _gold_spans(path: Path, *, filename: str, digest: str, index_spans: tuple[Span, ...]) -> set[Span]:
    """Read annotation data only after predictions are complete and persisted."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty annotation file")
    header = json.loads(lines[0])
    if header.get("unit") != "header" or header.get("document") != filename:
        raise ValueError(f"{path}: annotation header does not match {filename}")
    if header.get("text", {}).get("sha256") != digest:
        raise ValueError(f"{path}: annotation text hash differs from documents.json")
    gold: set[Span] = set()
    for line in lines[1:]:
        row = json.loads(line)
        if row.get("unit") != "citation" or row.get("kind") != "DocketCitation":
            continue
        locator = row.get("locator")
        if not isinstance(locator, dict):
            raise ValueError(f"{path}: docket citation lacks a locator span")
        span = _span(locator)
        if not is_within(span, index_spans):
            gold.add(span)
    return gold


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _run_config(output_dir: Path, config: dict[str, Any] | None) -> None:
    if config is None:
        return
    path = output_dir / "run.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"{path}: model or review settings differ; choose a new output directory")
    else:
        _write_json(path, config)


def _checkpoint(output_dir: Path, name: str, filename: str) -> Path:
    return output_dir / "documents" / name / f"{filename}.json"


def _summary_path(output_dir: Path, name: str, filename: str) -> Path:
    return output_dir / "summaries" / name / f"{filename}.json"


def _validate_checkpoint(saved: Document, source: Document, path: Path) -> None:
    if (
        STAGE not in saved.stage_runs
        or saved.text != source.text
        or saved.index_spans != source.index_spans
        or saved.preprocessing_metadata != source.preprocessing_metadata
    ):
        raise ValueError(f"{path}: saved Document does not match this completed input")


def _score_document(document: Document, gold: set[Span]) -> dict[str, int]:
    rule = {
        citation.locator_span
        for citation in document.citations
        if isinstance(citation, FullDocketCitation) and citation.nodes[0].stage == "docket_locators"
    }
    admitted = {
        citation.locator_span
        for citation in document.citations
        if isinstance(citation, FullDocketCitation) and citation.nodes[0].stage == STAGE
    }
    reviews = tuple(review for review in document.site_reviews if review.stage == STAGE)
    combined = rule | admitted
    return {
        "documents": 1,
        "gold_docket_locators": len(gold),
        "rule_locators": len(rule),
        "rule_exact": len(rule & gold),
        "reviewed_sites": len(reviews),
        "hunt_admissions": len(admitted),
        "hunt_exact": len(admitted & gold),
        "hunt_nonmatching": len(admitted - gold),
        "hunt_declined": sum(review.outcome == "declined" for review in reviews),
        "hunt_failed": sum(review.outcome == "failed" for review in reviews),
        "combined_locators": len(combined),
        "combined_exact": len(combined & gold),
        "combined_nonmatching": len(combined - gold),
        "remaining_gold_misses": len(gold - combined),
    }


def _overall_summary(
    items: tuple[tuple[str, str, dict[str, Any]], ...],
    sets: dict[str, dict[str, int]],
    resumed: int,
    *,
    complete: bool,
) -> dict[str, Any]:
    totals = {field: sum(counts[field] for counts in sets.values()) for field in COUNT_FIELDS}
    return {
        "status": "complete" if complete else "in_progress",
        "metric": "Exact annotated DocketCitation locator spans outside manifest index_spans",
        "selected_documents": len(items),
        "completed_documents": totals["documents"],
        "resumed_documents": resumed,
        "sets": sets,
        "totals": totals,
    }


async def evaluate(
    data_root: Path,
    output_dir: Path,
    items: tuple[tuple[str, str, dict[str, Any]], ...],
    *,
    reviewer: DocketSiteReviewer | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume completed documents, then write per-document and aggregate scores."""
    _run_config(output_dir, config)
    sets: dict[str, dict[str, int]] = {}
    resumed = 0
    _write_json(output_dir / "summary.json", _overall_summary(items, sets, resumed, complete=False))
    for name, filename, metadata in items:
        source = _source_document(data_root, name, filename, metadata)
        checkpoint = _checkpoint(output_dir, name, filename)
        cached = checkpoint.exists()
        if cached:
            document = Document.model_validate_json(checkpoint.read_text(encoding="utf-8"))
            _validate_checkpoint(document, source, checkpoint)
            resumed += 1
        else:
            # Eyecite can emit benign overlap diagnostics for valid text.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                document = find_docket_locators(find_full_reporter_locators(source))
            document = await hunt_docket_locators(document, reviewer=reviewer)
            _atomic_write(checkpoint, document.model_dump_json(indent=2) + "\n")

        # Keep this read below the completed prediction checkpoint.
        gold = _gold_spans(
            data_root / name / "documents" / f"{Path(filename).stem}.jsonl",
            filename=filename,
            digest=metadata["sha256"],
            index_spans=source.index_spans,
        )
        counts = _score_document(document, gold)
        _write_json(
            _summary_path(output_dir, name, filename), {"set": name, "document": filename, "counts": counts}
        )
        set_counts = sets.setdefault(name, dict.fromkeys(COUNT_FIELDS, 0))
        for field in COUNT_FIELDS:
            set_counts[field] += counts[field]
        _write_json(output_dir / "summary.json", _overall_summary(items, sets, resumed, complete=False))
        print(f"{name}/{filename}: {'resumed' if cached else 'saved'}", file=sys.stderr)

    summary = _overall_summary(items, sets, resumed, complete=True)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/docket-site-hunt"))
    parser.add_argument("--set", dest="sets", action="append", choices=SETS, required=True)
    parser.add_argument(
        "--document",
        dest="documents",
        action="append",
        default=[],
        metavar="[SET/]NAME",
        help="Select an exact manifest filename, its stem, or a numbered prefix such as 001",
    )
    parser.add_argument("--max-documents", type=int, help="Run at most this many selected documents")
    parser.add_argument("--service-tier", help="Override this run's service tier; 'default' omits the field")
    parser.add_argument(
        "--plan", action="store_true", help="List the selection and saved checkpoints without calls"
    )
    args = parser.parse_args()
    try:
        items = select_documents(
            args.data_root,
            tuple(args.sets),
            tuple(args.documents),
            args.max_documents,
        )
        if args.plan:
            print(
                json.dumps(
                    {
                        "selected_documents": len(items),
                        "documents": [
                            {
                                "set": name,
                                "document": filename,
                                "saved": _checkpoint(args.output_dir, name, filename).exists(),
                            }
                            for name, filename, _ in items
                        ],
                    },
                    indent=2,
                )
            )
            return
        configured = OpenRouterDocketReviewer.from_env()
        reviewer = (
            replace(configured, service_tier=None if args.service_tier == "default" else args.service_tier)
            if args.service_tier is not None
            else configured
        )
        config = {
            "schema_version": 1,
            "reviewer": {
                "base_url": reviewer.base_url,
                "model": reviewer.model,
                "service_tier": reviewer.service_tier,
                "temperature": reviewer.temperature,
                "max_tokens": reviewer.max_tokens,
                "max_attempts": reviewer.max_attempts,
            },
        }
        print(
            json.dumps(
                asyncio.run(
                    evaluate(args.data_root, args.output_dir, items, reviewer=reviewer, config=config)
                ),
                indent=2,
            )
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
