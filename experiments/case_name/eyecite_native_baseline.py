"""Compare eyecite's native name reading with the project's case-name reader.

Locator sites and pre-locator windows come from an existing evaluation run.
For every reporter or docket site, the known locator is passed as an anchor to
eyecite's unmodified ``find_case_name`` helper. The prefix is tokenized by
eyecite's default tokenizer; project name rules are never invoked. Eyecite
does not expose a standalone case-name span, so a fixed adapter trims only
boundary whitespace and comma from the pre-locator title interval.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

from eyecite.helpers import find_case_name
from eyecite.models import CitationToken, FullCaseCitation
from eyecite.models import Document as EyeciteDocument
from eyecite.tokenizers import default_tokenizer

from evaluations.annotations import citation_annotations
from mellea_lrc.config.extraction import stable
from mellea_lrc.model import Document, FullReporterCitation
from mellea_lrc.model.citation_windows import before

CASE_NAME_SUBSTAGE = "grow_roots.field_reading.case_names"


def field_source(row: dict[str, Any]) -> dict[str, Any]:
    field = row["case_name"]
    if not isinstance(field, dict):
        raise ValueError(f"{row['id']}: missing case-name gold")
    return field.get("source", field)


def gold_span(row: dict[str, Any]) -> tuple[int, int] | None:
    source = field_source(row)
    if source.get("kind") == "quoted" or "start" in source:
        return source["start"], source["end"]
    if source["kind"] in {"not_stated", "inferred"}:
        return None
    raise ValueError(f"Unsupported gold source kind: {source['kind']}")


def outcome_agrees(prediction: tuple[int, int] | None, row: dict[str, Any]) -> bool:
    source = field_source(row)
    if source.get("kind") == "quoted" or "start" in source:
        return prediction == gold_span(row)
    if source["kind"] == "not_stated":
        return prediction is None
    if source["kind"] == "inferred":
        return False  # This span-only eyecite adapter cannot infer a name.
    raise ValueError(f"Unsupported gold source kind: {source['kind']}")


def overlaps(left: tuple[int, int] | None, right: tuple[int, int] | None) -> bool:
    return left is not None and right is not None and left[0] < right[1] and right[0] < left[1]


def eyecite_name_span(
    document: Document, citation: Any, limit: int
) -> tuple[tuple[int, int] | None, bool, bool]:
    """Return a native name-span proxy, given the project's locator anchor."""
    prefix, prefix_start = before(document, citation, limit)
    locator_quote = citation.locator[-1].quote
    source = prefix + locator_quote
    # Supply only the already-detected locator token, then call eyecite's
    # native backward name heuristic for either locator kind.
    eyecite_document = EyeciteDocument(plain_text=source)
    eyecite_document.words, _ = default_tokenizer.tokenize(prefix)
    token = CitationToken(data=locator_quote, start=len(prefix), end=len(source))
    index = len(eyecite_document.words)
    eyecite_document.words.append(token)
    parsed = FullCaseCitation(token=token, index=index)
    find_case_name(parsed, eyecite_document)
    metadata = parsed.metadata
    has_party = bool(metadata.plaintiff or metadata.defendant)
    start = parsed.full_span_start
    if not has_party or start is None or not (0 <= start < len(prefix)):
        return None, True, has_party
    end = len(prefix)
    while start < end and source[start] in " \t\r\n,":
        start += 1
    while end > start and source[end - 1] in " \t\r\n,":
        end -= 1
    if end <= start:
        return None, True, has_party
    return (prefix_start + start, prefix_start + end), True, has_party


def evaluate(documents_dir: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    document_paths = sorted(documents_dir.glob("*.json"))
    if not document_paths:
        raise ValueError(f"No saved documents in {documents_dir}")
    limit = stable().case_name_window
    for path in document_paths:
        final = Document.model_validate_json(path.read_text(encoding="utf-8"))
        document = final.get_substage(CASE_NAME_SUBSTAGE)
        rows = {
            (
                row["kind"],
                row["locator"].get("source", row["locator"])["start"],
                row["locator"].get("source", row["locator"])["end"],
            ): row
            for row in citation_annotations(document)
            if row.get("kind") in {"FullCaseCitation", "DocketCitation"}
        }
        counts["documents"] += 1
        for citation in document.full_locators:
            kind = "FullCaseCitation" if isinstance(citation, FullReporterCitation) else "DocketCitation"
            key = (kind, citation.locator_span.start, citation.locator_span.end)
            row = rows.get(key)
            counts["all_detected_locators"] += 1
            if row is None:
                counts["unmatched_gold_locator"] += 1
            current = (
                (citation.case_name[-1].span.start, citation.case_name[-1].span.end)
                if citation.case_name and citation.case_name[-1].span is not None
                else None
            )
            if row is not None and outcome_agrees(current, row):
                counts["current_all_correct"] += 1
            citation_type = "reporter" if isinstance(citation, FullReporterCitation) else "docket"
            counts[f"{citation_type}_locators"] += 1
            raw, locator_found, has_party = eyecite_name_span(document, citation, limit)
            counts[f"eyecite_{citation_type}_anchor_available"] += int(locator_found)
            counts["eyecite_party_available"] += int(has_party)
            counts["eyecite_name_span_available"] += int(raw is not None)
            if row is None:
                errors.append(
                    {"document": path.name, "locator": list(key), "reason": "unmatched_gold_locator"}
                )
                continue
            gold = gold_span(row)
            source = field_source(row)
            counts[f"gold_{source.get('kind', 'quoted')}"] += 1
            counts[f"current_{citation_type}_correct"] += int(outcome_agrees(current, row))
            counts[f"eyecite_{citation_type}_correct"] += int(outcome_agrees(raw, row))
            counts["eyecite_all_correct"] += int(outcome_agrees(raw, row))
            counts[f"current_{citation_type}_overlap"] += int(overlaps(current, gold))
            counts[f"eyecite_{citation_type}_overlap"] += int(overlaps(raw, gold))
            if not outcome_agrees(raw, row):
                errors.append(
                    {
                        "document": path.name,
                        "id": row["id"],
                        "kind": citation_type,
                        "gold": list(gold) if gold is not None else None,
                        "eyecite": list(raw) if raw is not None else None,
                        "current": list(current) if current is not None else None,
                        "gold_quote": document.text[gold[0] : gold[1]] if gold is not None else None,
                        "eyecite_quote": document.text[raw[0] : raw[1]] if raw is not None else None,
                    }
                )
    return {
        "protocol": {
            "documents_dir": str(documents_dir.resolve()),
            "eyecite_version": version("eyecite"),
            "case_name_window_chars": limit,
            "locator": "saved project reporter and docket locator anchors",
            "name_reader": "eyecite default prefix tokenizer and native find_case_name for both kinds",
            "anchor": "real pipeline-detected locator quote supplied as citation token",
            "name_span": "eyecite full_span_start to locator start; boundary whitespace/comma trim only",
        },
        "counts": dict(sorted(counts.items())),
        "eyecite_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents_dir", type=Path, help="Saved run's documents directory")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = evaluate(args.documents_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
