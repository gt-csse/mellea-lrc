"""Score only the new product of a completed extraction stage.

Run from the repository root, for example::

    uv run python -m evaluations.score_stages --run-dir local/run --stage docket_entries

The input is a saved Document, not an extraction command. A final Document may
contain later stages: ``get_stage`` and each reading's node recover the exact
stage product. Missing artifacts or unrun stages are errors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeAlias

from evaluations.annotations import SETS, annotated_documents, span
from evaluations.stage_products import StageProduct, stage_product
from mellea_lrc.extraction.context import (
    CASE_NAMES_STAGE,
    COURTS_STAGE,
    DATES_STAGE,
    PIN_CITES_STAGE,
)
from mellea_lrc.extraction.docket_entries import STAGE as DOCKET_ENTRIES_STAGE
from mellea_lrc.extraction.docket_hunting import STAGE as DOCKET_HUNT_STAGE
from mellea_lrc.extraction.locators import (
    DOCKET_LOCATORS_STAGE,
    FULL_REPORTER_LOCATORS_STAGE,
    SHORT_REPORTER_CITATIONS_STAGE,
)
from mellea_lrc.extraction.roots import STAGE as ROOTS_STAGE
from mellea_lrc.extraction.structure import STAGE as COLOCATIONS_STAGE
from mellea_lrc.model import Document, FullDocketCitation, FullReporterCitation, ShortReporterCitation, Span
from mellea_lrc.model.citations import CitationVariant
from mellea_lrc.model.citations.fields.base import CitationField

GoldKey: TypeAlias = tuple[str, Span]
NO_GOLD = object()
FIELD_STAGES = {
    FULL_REPORTER_LOCATORS_STAGE: "locator",
    DOCKET_LOCATORS_STAGE: "locator",
    DOCKET_HUNT_STAGE: "locator",
    DOCKET_ENTRIES_STAGE: "docket_entry",
    CASE_NAMES_STAGE: "case_name",
    COURTS_STAGE: "court",
    DATES_STAGE: "date",
    PIN_CITES_STAGE: "pin_cite",
    SHORT_REPORTER_CITATIONS_STAGE: "short_locator",
}
RELATIONSHIP_STAGES = {
    COLOCATIONS_STAGE: "colocation_id",
    ROOTS_STAGE: "root_id",
}
STAGES = (*FIELD_STAGES, *RELATIONSHIP_STAGES)
ELIGIBILITY = {
    FULL_REPORTER_LOCATORS_STAGE: "All unmasked annotated full reporter locators",
    DOCKET_LOCATORS_STAGE: "All unmasked annotated docket locators",
    DOCKET_HUNT_STAGE: "Annotated docket locators not already found by the rule stage",
    DOCKET_ENTRIES_STAGE: "Annotated entries whose docket locator existed before this stage",
    CASE_NAMES_STAGE: "Annotated case names whose full locator existed before this stage",
    COURTS_STAGE: "Annotated courts whose full locator existed before this stage",
    DATES_STAGE: "Annotated dates whose full locator existed before this stage",
    PIN_CITES_STAGE: "Annotated pin cites whose full locator existed before this stage",
    SHORT_REPORTER_CITATIONS_STAGE: "All unmasked annotated short reporter citations",
    COLOCATIONS_STAGE: "Annotated full locators present before colocation",
    ROOTS_STAGE: "Annotated full locators present before root formation",
}


def _citation_key(citation: CitationVariant) -> GoldKey:
    if isinstance(citation, FullReporterCitation):
        return "FullCaseCitation", citation.locator_span
    if isinstance(citation, FullDocketCitation):
        return "DocketCitation", citation.locator_span
    if isinstance(citation, ShortReporterCitation):
        return "ShortCaseCitation", citation.site_span
    raise TypeError(f"Unscored citation type: {type(citation).__name__}")


def _gold_by_key(rows: tuple[dict[str, Any], ...]) -> dict[GoldKey, dict[str, Any]]:
    result = {}
    for row in rows:
        if row["kind"] not in {"FullCaseCitation", "DocketCitation", "ShortCaseCitation"}:
            continue
        key = row["kind"], span(row["locator"])
        if key in result:
            raise ValueError(f"Duplicate annotated locator: {key}")
        result[key] = row
    return result


def _gold_field(row: dict[str, Any], field: str) -> dict[str, Any] | None:
    raw = row.get("locator" if field == "short_locator" else field)
    return raw if isinstance(raw, dict) else None


def _gold_span(raw: dict[str, Any]) -> Span | None:
    return span(raw) if "start" in raw and "end" in raw else None


def _span_json(value: Span | None) -> dict[str, int] | None:
    return {"start": value.start, "end": value.end} if value is not None else None


def _reporter_label(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _predicted_normalized(field: str, reading: CitationField) -> Any:
    value = reading.get_normalized()
    if field == "locator" and hasattr(value, "docket_number"):
        return value.docket_number
    if field == "locator":
        return (value.volume, _reporter_label(value.edition), value.page)
    if field == "short_locator":
        return (value.volume, _reporter_label(value.edition), value.pin_page)
    if field == "court":
        return value.id
    if field == "date":
        parts = [f"{value.year:04d}"]
        if value.month is not None:
            parts.append(f"{value.month:02d}")
        if value.day is not None:
            parts.append(f"{value.day:02d}")
        return "-".join(parts)
    if field == "pin_cite":
        return [item.model_dump(mode="json", exclude_none=True) for item in value]
    if field == "case_name":
        return value.model_dump(mode="json")
    return value


def _gold_normalized(row: dict[str, Any], field: str) -> Any:
    if field == "locator":
        identifier = row.get("identifier")
        if not isinstance(identifier, dict):
            return NO_GOLD
        if identifier.get("kind") == "docket":
            return identifier.get("docket_number", NO_GOLD)
        if identifier.get("kind") == "reporter" and all(
            identifier.get(part) is not None for part in ("volume", "reporter", "page")
        ):
            return (
                int(identifier["volume"]),
                _reporter_label(str(identifier["reporter"])),
                str(identifier["page"]),
            )
        return NO_GOLD
    gold = _gold_field(row, field)
    if gold is None:
        return NO_GOLD
    if field == "court":
        return gold.get("id", NO_GOLD)
    if field == "date":
        return gold.get("normalized", NO_GOLD)
    if field == "pin_cite":
        return gold.get("normalized", NO_GOLD)
    if field == "docket_entry":
        return str(gold["number"]) if "number" in gold else NO_GOLD
    # Gold has no independent normalized parties or short-reporter identity.
    return NO_GOLD


def _eligible_field_gold(
    product: StageProduct, rows: dict[GoldKey, dict[str, Any]], field: str
) -> tuple[dict[GoldKey, dict[str, Any]], int]:
    stage = product.stage
    if stage == FULL_REPORTER_LOCATORS_STAGE:
        relevant = {key: row for key, row in rows.items() if key[0] == "FullCaseCitation"}
    elif stage in {DOCKET_LOCATORS_STAGE, DOCKET_HUNT_STAGE, DOCKET_ENTRIES_STAGE}:
        relevant = {key: row for key, row in rows.items() if key[0] == "DocketCitation"}
    elif stage == SHORT_REPORTER_CITATIONS_STAGE:
        relevant = {key: row for key, row in rows.items() if key[0] == "ShortCaseCitation"}
    else:
        relevant = {key: row for key, row in rows.items() if key[0] in {"FullCaseCitation", "DocketCitation"}}
    relevant = {key: row for key, row in relevant.items() if _gold_field(row, field) is not None}
    if stage in {FULL_REPORTER_LOCATORS_STAGE, DOCKET_LOCATORS_STAGE, SHORT_REPORTER_CITATIONS_STAGE}:
        return relevant, len(relevant)
    if product.before is None:
        raise ValueError(f"{stage} needs a preceding locator checkpoint")
    before_keys = {_citation_key(citation) for citation in product.before.citations}
    if stage == DOCKET_HUNT_STAGE:
        eligible = {key: row for key, row in relevant.items() if key not in before_keys}
        return eligible, len(relevant)
    eligible = {key: row for key, row in relevant.items() if key in before_keys}
    return eligible, len(relevant)


def _score_fields(
    product: StageProduct, rows: dict[GoldKey, dict[str, Any]]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    field = FIELD_STAGES[product.stage]
    unexpected = {item.name for item in product.fields if item.name != field}
    if unexpected or product.relationships:
        raise ValueError(f"{product.stage} wrote unscored fields or relationships: {unexpected}")
    eligible, total_gold = _eligible_field_gold(product, rows, field)
    counts: Counter[str] = Counter(
        documents=1,
        gold_fields=total_gold,
        eligible_gold=len(eligible),
        gold_outside_stage_scope=total_gold - len(eligible),
        predicted=len(product.fields),
    )
    counts["gold_with_span"] = sum(
        _gold_span(_gold_field(row, field)) is not None for row in eligible.values()
    )
    counts["gold_without_span"] = len(eligible) - counts["gold_with_span"]
    counts["gold_with_normalization"] = sum(
        _gold_normalized(row, field) is not NO_GOLD for row in eligible.values()
    )
    for review in product.reviews:
        counts[f"site_review_{review.outcome}"] += 1
    matched: set[GoldKey] = set()
    details = []
    for item in product.fields:
        key = _citation_key(item.citation)
        reading = item.reading
        predicted_span = getattr(reading, "span", None)
        counts["normalizable"] += int(reading.normalizable)
        counts["normalization_errors"] += int(not reading.normalizable)
        row = rows.get(key)
        gold = eligible.get(key)
        normalization_outcome = "not_evaluated"
        normalization_target = None
        if gold is None:
            if row is None:
                outcome = "unmatched_locator"
            elif _gold_field(row, field) is None:
                outcome = "unlabeled_field"
                if field == "docket_entry" and predicted_span is not None:
                    cited_as = row.get("cited_as")
                    if isinstance(cited_as, dict) and (
                        span(cited_as).start <= predicted_span.start
                        and predicted_span.end <= span(cited_as).end
                    ):
                        outcome = "unlabeled_in_citation"
            else:
                outcome = "outside_stage_scope"
        elif key in matched:
            outcome = "duplicate_prediction"
        elif predicted_span != _gold_span(_gold_field(gold, field)):
            outcome = "wrong_span" if predicted_span is not None else "missing_span"
        else:
            matched.add(key)
            outcome = "matched"
            counts["exact_spans" if predicted_span is not None else "matched_without_span"] += 1
            target = _gold_normalized(gold, field)
            if target is not NO_GOLD:
                counts["normalization_checked"] += 1
                normalization_target = target
                if reading.normalizable and _predicted_normalized(field, reading) == target:
                    counts["normalization_correct"] += 1
                    normalization_outcome = "correct"
                else:
                    counts["normalization_wrong"] += 1
                    normalization_outcome = "wrong"
            else:
                normalization_outcome = "no_independent_gold"
        if outcome != "matched":
            counts[outcome] += 1
        details.append(
            {
                "citation_id": item.citation.id,
                "gold_id": row.get("id") if row else None,
                "parent_locator_span": _span_json(key[1]),
                "field": field,
                "span": _span_json(predicted_span),
                "quote": getattr(reading, "quote", None),
                "normalizable": reading.normalizable,
                "normalized": _predicted_normalized(field, reading) if reading.normalizable else None,
                "gold_normalized": normalization_target,
                "normalization_outcome": normalization_outcome,
                "normalization_error": reading.normalization_error,
                "outcome": outcome,
            }
        )
    details.extend(
        {
            "product": "site_review",
            "citation_id": review.citation_id,
            "candidate_span": _span_json(review.candidate_span),
            "candidate_text": review.candidate_text,
            "outcome": review.outcome,
            "reason": review.reason,
            "attempts": len(review.attempts),
        }
        for review in product.reviews
    )
    counts["missed_gold"] = len(eligible) - len(matched)
    details.extend(
        {
            "product": "gold_miss",
            "gold_id": row.get("id"),
            "parent_locator_span": _span_json(key[1]),
            "field": field,
            "gold_span": _span_json(_gold_span(_gold_field(row, field))),
        }
        for key, row in eligible.items()
        if key not in matched
    )
    return counts, details


def _groups(pairs: list[tuple[str, GoldKey]], *, singletons: bool) -> set[frozenset[GoldKey]]:
    grouped: dict[str, set[GoldKey]] = defaultdict(set)
    for group_id, key in pairs:
        grouped[group_id].add(key)
    return {frozenset(keys) for keys in grouped.values() if singletons or len(keys) > 1}


def _pairs(groups: set[frozenset[GoldKey]]) -> set[frozenset[GoldKey]]:
    result = set()
    for group in groups:
        ordered = sorted(group, key=lambda key: (key[0], key[1].start, key[1].end))
        result.update(
            frozenset((left, right)) for index, left in enumerate(ordered) for right in ordered[index + 1 :]
        )
    return result


def _score_relationships(
    product: StageProduct, rows: dict[GoldKey, dict[str, Any]]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    field = RELATIONSHIP_STAGES[product.stage]
    unexpected = {item.name for item in product.relationships if item.name != field}
    if unexpected or product.fields:
        raise ValueError(f"{product.stage} wrote unscored fields or relationships: {unexpected}")
    if product.before is None:
        raise ValueError(f"{product.stage} needs a preceding locator checkpoint")
    before_keys = {_citation_key(citation) for citation in product.before.full_locators}
    eligible = {
        key: row
        for key, row in rows.items()
        if key in before_keys and key[0] in {"FullCaseCitation", "DocketCitation"}
    }
    singletons = field == "root_id"
    gold_pairs = [(str(row[field]), key) for key, row in eligible.items() if row.get(field) is not None]
    predicted_pairs = [
        (str(item.update.value), _citation_key(item.citation))
        for item in product.relationships
        if item.name == field and item.update.value is not None
    ]
    gold_groups = _groups(gold_pairs, singletons=singletons)
    predicted_groups = _groups(predicted_pairs, singletons=singletons)
    counts: Counter[str] = Counter(
        documents=1,
        gold_locators=len([key for key in rows if key[0] in {"FullCaseCitation", "DocketCitation"}]),
        eligible_locators=len(eligible),
        predicted_assignments=len(predicted_pairs),
        gold_groups=len(gold_groups),
        predicted_groups=len(predicted_groups),
        exact_groups=len(gold_groups & predicted_groups),
        unmatched_assignments=sum(key not in rows for _, key in predicted_pairs),
    )
    if product.stage == COLOCATIONS_STAGE:
        gold_links = _pairs(gold_groups)
        predicted_links = _pairs(predicted_groups)
        counts.update(
            gold_links=len(gold_links),
            predicted_links=len(predicted_links),
            correct_links=len(gold_links & predicted_links),
        )
    # Root pairwise linkage belongs to the later leaf-attachment evaluation.
    details = [
        {
            "citation_id": item.citation.id,
            "parent_locator_span": _span_json(_citation_key(item.citation)[1]),
            "relationship": field,
            "value": item.update.value,
            "gold_id": rows.get(_citation_key(item.citation), {}).get("id"),
        }
        for item in product.relationships
    ]
    details.extend(
        {
            "product": "gold_group_miss",
            "members": [_span_json(key[1]) for key in sorted(group, key=lambda key: key[1].start)],
        }
        for group in gold_groups - predicted_groups
    )
    details.extend(
        {
            "product": "predicted_group_extra",
            "members": [_span_json(key[1]) for key in sorted(group, key=lambda key: key[1].start)],
        }
        for group in predicted_groups - gold_groups
    )
    return counts, details


def score_document(
    document: Document, stage: str, gold_rows: tuple[dict[str, Any], ...]
) -> tuple[Counter[str], list[dict[str, Any]]]:
    """Evaluate one stage's new field readings or relationship assignments."""
    if stage not in STAGES:
        raise ValueError(f"Unsupported extraction stage: {stage}")
    product = stage_product(document, stage)
    rows = _gold_by_key(gold_rows)
    if stage in FIELD_STAGES:
        return _score_fields(product, rows)
    return _score_relationships(product, rows)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _summary(counts: Counter[str], stage: str) -> dict[str, Any]:
    result: dict[str, Any] = dict(counts)
    if stage in FIELD_STAGES:
        scored_predictions = sum(
            counts[name]
            for name in (
                "exact_spans",
                "wrong_span",
                "missing_span",
                "duplicate_prediction",
                "unmatched_locator",
                "outside_stage_scope",
            )
        )
        precision = _ratio(counts["exact_spans"], scored_predictions)
        recall = _ratio(counts["exact_spans"], counts["gold_with_span"])
        result["span_precision_on_scored_predictions"] = precision
        result["span_recall"] = recall
        result["span_f1_on_scored_predictions"] = (
            round(2 * precision * recall / (precision + recall), 4)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        result["normalization_accuracy_on_matched_evidence"] = _ratio(
            counts["normalization_correct"], counts["normalization_checked"]
        )
        result["normalization_recall"] = _ratio(
            counts["normalization_correct"], counts["gold_with_normalization"]
        )
    else:
        result["exact_group_recall"] = _ratio(counts["exact_groups"], counts["gold_groups"])
        if stage == COLOCATIONS_STAGE:
            result["link_precision"] = _ratio(counts["correct_links"], counts["predicted_links"])
            result["link_recall"] = _ratio(counts["correct_links"], counts["gold_links"])
    return result


def evaluate(data_root: Path, run_dir: Path, stage: str, sets: tuple[str, ...]) -> dict[str, Any]:
    """Score saved documents; no extraction or model call occurs here."""
    if stage not in STAGES:
        raise ValueError(f"Unsupported extraction stage: {stage}")
    by_set: dict[str, Counter[str]] = {}
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for name, filename, document, rows in annotated_documents(data_root, run_dir, sets):
        counts, details = score_document(document, stage, rows)
        by_set.setdefault(name, Counter()).update(counts)
        occurrences[f"{name}/{filename}"] = details
    totals: Counter[str] = Counter()
    for counts in by_set.values():
        totals.update(counts)
    return {
        "stage": stage,
        "basis": "Only readings or relationships written by this stage; normalized agreement requires matched source evidence",
        "eligibility": ELIGIBILITY[stage],
        "sets": {name: _summary(counts, stage) for name, counts in by_set.items()},
        "totals": _summary(totals, stage),
        "occurrences": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--set", dest="sets", action="append", choices=SETS)
    parser.add_argument("--output-dir", type=Path, help="Write summary.json and occurrences.json here")
    args = parser.parse_args()
    result = evaluate(args.data_root, args.run_dir, args.stage, tuple(args.sets or SETS))
    occurrences = result.pop("occurrences")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "occurrences.json").write_text(
            json.dumps(occurrences, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
