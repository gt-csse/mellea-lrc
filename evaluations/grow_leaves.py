"""Independent incremental leaf scorers and their workflow summary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluations.grow_roots import FieldScore, Precision, StageScore
from mellea_lrc.model import Document
from mellea_lrc.model.citations import (
    AttributionResult,
    FullDocketCitation,
    FullReporterCitation,
    IdCitation,
    LeafCitation,
    ReferenceCitation,
    ShortReporterCitation,
    SupraCitation,
    latest,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID

_SET = "primary"
SHORT_STAGE = "28_short_reporter_citations"
SUPRA_STAGE = "29_supra_citations"
ID_STAGE = "30_id_citations"
REFERENCE_STAGE = "31_reference_citations"
NAME_STAGE = "32_leaf_case_names"
PIN_STAGE = "33_leaf_pin_cites"
RULE_STAGE = "34_leaf_attribution_rule"
REVIEW_STAGE = "35_leaf_attribution_llm"
ID_ATTRIBUTION_STAGE = "36_id_attribution"
ID_REVIEW_STAGE = "37_id_attribution_llm"


def _rows(document: Document) -> tuple[dict[str, Any], ...]:
    if document.source_path is None:
        raise ValueError("Leaf evaluation needs an official source path")
    source = Path(document.source_path).resolve()
    dataset = source.parent.parent
    lines = (dataset / "documents" / f"{source.stem}.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    text = header.get("text", {})
    if (
        header.get("unit") != "header"
        or header.get("dataset") != dataset.name
        or header.get("document") != source.name
        or text.get("sha256") != hashlib.sha256(document.text.encode()).hexdigest()
        or text.get("length") != len(document.text)
        or (dataset.parent / text.get("path", "")).resolve() != source
    ):
        raise ValueError("Leaf annotation does not match the Document source")
    return tuple(row for line in lines[1:] if (row := json.loads(line)).get("unit") == "citation")


def _span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        return None
    value = value.get("source", value)
    return (value["start"], value["end"]) if "start" in value else None


def _kind(citation: Any) -> str:
    return {
        FullReporterCitation: "FullCaseCitation",
        FullDocketCitation: "DocketCitation",
        ShortReporterCitation: "ShortCaseCitation",
        IdCitation: "IdCitation",
        ReferenceCitation: "ReferenceCitation",
        SupraCitation: "SupraCitation",
    }[type(citation)]


def _gold_key(row: dict[str, Any]) -> tuple[str, int, int]:
    field = (
        "case_name"
        if row["kind"] == "ReferenceCitation"
        else (
            "locator"
            if row["kind"] in {"FullCaseCitation", "DocketCitation", "ShortCaseCitation"}
            else "cited_as"
        )
    )
    span = _span(row.get(field))
    if span is None:
        raise ValueError(f"{row['id']}: missing leaf source span")
    return (row["kind"], *span)


def _key(citation: Any) -> tuple[str, int, int]:
    return (_kind(citation), citation.site_span.start, citation.site_span.end)


def _gold(document: Document) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = _rows(document)
    keyed = {_gold_key(row): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate gold citation site")
    return keyed


def _attribution_rows(citations: list[Any], gold: dict) -> list[tuple[Any, dict | None]]:
    """Align occurrence identity independently of exact name-field boundaries.

    Name-only sites use a unique overlapping gold mention; their exact name
    span is scored separately. Each gold occurrence receives credit once.
    Locator and Id sites retain exact identifier-site matching.
    """
    result = []
    used: set[str] = set()
    for citation in citations:
        row = gold.get(_key(citation))
        if row is None and isinstance(citation, ReferenceCitation):
            matches = [
                candidate
                for (kind, start, end), candidate in gold.items()
                if kind == "ReferenceCitation"
                and start < citation.site_span.end
                and citation.site_span.start < end
            ]
            if len(matches) > 1:
                raise ValueError("Reference site overlaps multiple gold occurrences")
            row = matches[0] if matches else None
        if row is not None:
            if row["id"] in used:
                row = None
            else:
                used.add(row["id"])
        result.append((citation, row))
    return result


def _root_agrees(document: Document, citation: Any, row: dict[str, Any] | None) -> bool:
    if row is None or row["is_root"]:
        return False
    roots = {c.id: c for c in document.roots}
    target = roots.get(latest(citation.root_id))
    if target is None:
        return False
    rows = {r["id"]: r for r in _rows(document)}
    return _key(target) == _gold_key(rows[row["root_id"]])


def _field_normalization(reading: Any, target: Any) -> bool:
    if reading is None or not reading.normalizable or not isinstance(target, dict):
        return False
    value = reading.get_normalized()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json") == target.get("normalized")
    return [item.model_dump(mode="json", exclude_none=True) for item in value] == target.get("normalized")


def score_short_reporter_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(SHORT_STAGE)
    gold = _gold(checkpoint)
    citations = checkpoint.short_reporters
    # The annotation has no independent short reporter normalization target.
    return StageScore(
        SHORT_STAGE,
        {
            "span": Precision(sum(_key(c) in gold for c in citations), len(citations)),
            "normalization": Precision(),
        },
    )


def score_supra_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(SUPRA_STAGE)
    gold = _gold(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, SupraCitation)]
    return StageScore(
        SUPRA_STAGE,
        {
            "span": Precision(sum(_key(c) in gold for c in citations), len(citations)),
            "normalization": Precision(),
        },
    )


def score_id_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(ID_STAGE)
    gold = _gold(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, IdCitation)]
    return StageScore(
        ID_STAGE,
        {
            "span": Precision(sum(_key(c) in gold for c in citations), len(citations)),
            "normalization": Precision(),
        },
    )


def score_reference_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(REFERENCE_STAGE)
    gold = _gold(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, ReferenceCitation)]
    correct = sum(_key(c) in gold for c in citations)
    normalized = [
        c for c in citations if gold.get(_key(c), {}).get("case_name", {}).get("normalized") is not None
    ]
    return StageScore(
        REFERENCE_STAGE,
        {
            "span": Precision(correct, len(citations)),
            "normalization": Precision(
                sum(
                    _field_normalization(c.reference_name[-1], gold[_key(c)]["case_name"]) for c in normalized
                ),
                len(normalized),
            ),
        },
    )


def score_leaf_case_names(document: Document) -> StageScore:
    checkpoint = document.get_stage(NAME_STAGE)
    gold = _gold(checkpoint)
    readings = [
        (c, reading)
        for c in checkpoint.short_citations
        for reading in c.case_name
        if next(n.stage for n in c.nodes if n.id == reading.node_id) == NAME_STAGE
    ]
    correct = sum(
        _span(gold.get(_key(c), {}).get("case_name")) == (f.span.start, f.span.end) for c, f in readings
    )
    normalized = [
        (c, f)
        for c, f in readings
        if gold.get(_key(c), {}).get("case_name", {}).get("normalized") is not None
    ]
    return StageScore(
        NAME_STAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(
                sum(_field_normalization(f, gold[_key(c)]["case_name"]) for c, f in normalized),
                len(normalized),
            ),
        },
    )


def score_leaf_pin_cites(document: Document) -> StageScore:
    checkpoint = document.get_stage(PIN_STAGE)
    gold = _gold(checkpoint)
    readings = [
        (c, f)
        for c in checkpoint.short_citations
        for f in c.pin_cite
        if next(n.stage for n in c.nodes if n.id == f.node_id) == PIN_STAGE
    ]
    correct = sum(
        _span(gold.get(_key(c), {}).get("pin_cite")) == (f.span.start, f.span.end) for c, f in readings
    )
    normalized = [
        (c, f) for c, f in readings if gold.get(_key(c), {}).get("pin_cite", {}).get("normalized") is not None
    ]
    return StageScore(
        PIN_STAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(
                sum(_field_normalization(f, gold[_key(c)]["pin_cite"]) for c, f in normalized),
                len(normalized),
            ),
        },
    )


def score_leaf_attribution_rule(document: Document) -> StageScore:
    checkpoint = document.get_stage(RULE_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions and c.attributions[-1].result == AttributionResult.ATTACHED
    ]
    return StageScore(
        RULE_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, gold)),
                len(attached),
            )
        },
    )


def score_leaf_attribution_llm(document: Document) -> StageScore:
    checkpoint = document.get_stage(REVIEW_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.stage for n in c.nodes if n.id == c.attributions[-1].node_id) == REVIEW_STAGE
    ]
    return StageScore(
        REVIEW_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, gold)),
                len(attached),
            )
        },
    )


def score_id_attribution(document: Document) -> StageScore:
    checkpoint = document.get_stage(ID_ATTRIBUTION_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if isinstance(c, IdCitation) and latest(c.root_id) not in {None, WITHDRAWN_ROOT_ID}
    ]
    return StageScore(
        ID_ATTRIBUTION_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, gold.get(_key(c))) for c in attached), len(attached)
            )
        },
    )


def score_id_attribution_llm(document: Document) -> StageScore:
    checkpoint = document.get_stage(ID_REVIEW_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if isinstance(c, IdCitation)
        and c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.stage for n in c.nodes if n.id == c.attributions[-1].node_id) == ID_REVIEW_STAGE
    ]
    return StageScore(
        ID_REVIEW_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, gold.get(_key(c))) for c in attached), len(attached)
            )
        },
    )


GROW_LEAVES_SCORERS = {
    SHORT_STAGE: score_short_reporter_citations,
    SUPRA_STAGE: score_supra_citations,
    ID_STAGE: score_id_citations,
    REFERENCE_STAGE: score_reference_citations,
    NAME_STAGE: score_leaf_case_names,
    PIN_STAGE: score_leaf_pin_cites,
    RULE_STAGE: score_leaf_attribution_rule,
    REVIEW_STAGE: score_leaf_attribution_llm,
    ID_ATTRIBUTION_STAGE: score_id_attribution,
    ID_REVIEW_STAGE: score_id_attribution_llm,
}


@dataclass(frozen=True)
class WorkflowScore:
    stages: tuple[StageScore, ...]
    leaf_spans: dict[str, FieldScore]
    leaf_attribution: dict[str, FieldScore]

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(s.stage for s in self.stages) != tuple(s.stage for s in other.stages):
            raise ValueError("Cannot combine different leaf workflow stages")
        return WorkflowScore(
            tuple(a + b for a, b in zip(self.stages, other.stages, strict=True)),
            {k: v + other.leaf_spans[k] for k, v in self.leaf_spans.items()},
            {k: v + other.leaf_attribution[k] for k, v in self.leaf_attribution.items()},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.as_dict() for s in self.stages],
            "leaf_spans": {k: v.as_dict() for k, v in self.leaf_spans.items()},
            "leaf_attribution": {k: v.as_dict() for k, v in self.leaf_attribution.items()},
        }


def score_grow_leaves(document: Document) -> WorkflowScore:
    checkpoint = document.get_stage(
        ID_REVIEW_STAGE if ID_REVIEW_STAGE in document.stage_runs else ID_ATTRIBUTION_STAGE
    )
    required = set(GROW_LEAVES_SCORERS) - {REVIEW_STAGE, ID_REVIEW_STAGE}
    if not required.issubset(checkpoint.stage_runs):
        raise ValueError("Incomplete grow_leaves workflow")
    stages = tuple(
        scorer(checkpoint) for stage, scorer in GROW_LEAVES_SCORERS.items() if stage in checkpoint.stage_runs
    )
    gold = _gold(checkpoint)
    leaves = {k: r for k, r in gold.items() if not r["is_root"]}
    types = (
        "FullCaseCitation",
        "DocketCitation",
        "ShortCaseCitation",
        "SupraCitation",
        "IdCitation",
        "ReferenceCitation",
    )
    spans: dict[str, FieldScore] = {}
    assignments: dict[str, FieldScore] = {}
    for kind in types:
        predicted = [
            c
            for c in checkpoint.citations
            if _kind(c) == kind
            and latest(c.root_id) != WITHDRAWN_ROOT_ID
            and (isinstance(c, LeafCitation) or latest(c.root_id) != c.id)
        ]
        target = {k: r for k, r in leaves.items() if k[0] == kind}
        spans[kind] = FieldScore(sum(_key(c) in target for c in predicted), len(predicted), len(target))
        attached = [c for c in predicted if latest(c.root_id) not in {None, WITHDRAWN_ROOT_ID}]
        assignments[kind] = FieldScore(
            sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, target)),
            len(attached),
            len(target),
        )
    spans["all_leaves"] = FieldScore(
        sum(v.correct for v in spans.values()), sum(v.predicted for v in spans.values()), len(leaves)
    )
    assignments["all_leaves"] = FieldScore(
        sum(v.correct for v in assignments.values()),
        sum(v.predicted for v in assignments.values()),
        len(leaves),
    )
    return WorkflowScore(stages, spans, assignments)


def render_leaf_stage(score: StageScore) -> str:
    # Rendering is shared text formatting, not a shared stage scoring policy.
    rows = [f"## {score.stage}", "", "| Metric | Precision |", "| --- | ---: |"]
    for name, value in score.metrics.items():
        cell = (
            f"{value.correct}/{value.total} ({value.correct / value.total:.1%})"
            if value.total
            else ("— (no independent gold targets)" if name == "normalization" else "— (no decisions)")
        )
        rows.append(f"| {name} | {cell} |")
    return "\n".join(rows)


def render_short_reporter_citations(score: StageScore) -> str:
    if score.stage != SHORT_STAGE:
        raise ValueError("Expected short reporter discovery score")
    return render_leaf_stage(score)


def render_supra_citations(score: StageScore) -> str:
    if score.stage != SUPRA_STAGE:
        raise ValueError("Expected supra discovery score")
    return render_leaf_stage(score)


def render_id_citations(score: StageScore) -> str:
    if score.stage != ID_STAGE:
        raise ValueError("Expected Id. discovery score")
    return render_leaf_stage(score)


def render_reference_citations(score: StageScore) -> str:
    if score.stage != REFERENCE_STAGE:
        raise ValueError("Expected reference discovery score")
    return render_leaf_stage(score)


def render_leaf_case_names(score: StageScore) -> str:
    if score.stage != NAME_STAGE:
        raise ValueError("Expected leaf case-name reading score")
    return render_leaf_stage(score)


def render_leaf_pin_cites(score: StageScore) -> str:
    if score.stage != PIN_STAGE:
        raise ValueError("Expected leaf pin reading score")
    return render_leaf_stage(score)


def render_leaf_attribution_rule(score: StageScore) -> str:
    if score.stage != RULE_STAGE:
        raise ValueError("Expected rule attribution score")
    return render_leaf_stage(score)


def render_leaf_attribution_llm(score: StageScore) -> str:
    if score.stage != REVIEW_STAGE:
        raise ValueError("Expected semantic attribution score")
    return render_leaf_stage(score)


def render_id_attribution(score: StageScore) -> str:
    if score.stage != ID_ATTRIBUTION_STAGE:
        raise ValueError("Expected Id. rule attribution score")
    return render_leaf_stage(score)


def render_id_attribution_llm(score: StageScore) -> str:
    if score.stage != ID_REVIEW_STAGE:
        raise ValueError("Expected Id. semantic attribution score")
    return render_leaf_stage(score)


GROW_LEAVES_RENDERERS = {
    SHORT_STAGE: render_short_reporter_citations,
    SUPRA_STAGE: render_supra_citations,
    ID_STAGE: render_id_citations,
    REFERENCE_STAGE: render_reference_citations,
    NAME_STAGE: render_leaf_case_names,
    PIN_STAGE: render_leaf_pin_cites,
    RULE_STAGE: render_leaf_attribution_rule,
    REVIEW_STAGE: render_leaf_attribution_llm,
    ID_ATTRIBUTION_STAGE: render_id_attribution,
    ID_REVIEW_STAGE: render_id_attribution_llm,
}


def render_grow_leaves(score: WorkflowScore, *, set_name: str = _SET) -> str:
    sections = [
        f"# Grow-leaves evaluation: {set_name}",
        "Stage tables score only that stage's decisions. Workflow recall uses annotated leaves, including repeated full citations. Normalization is scored only where independent normalized gold exists. Withdrawn reference proposals remain in history but are excluded from the final leaf tables.",
    ]
    sections.extend(GROW_LEAVES_RENDERERS[s.stage](s) for s in score.stages)
    for title, values in [
        ("Leaf source spans", score.leaf_spans),
        ("Leaf attribution", score.leaf_attribution),
    ]:
        rows = [f"## {title}", "", "| Kind | Precision | Recall |", "| --- | ---: | ---: |"]
        for kind, value in values.items():
            precision = (
                f"{value.correct}/{value.predicted} ({value.correct / value.predicted:.1%})"
                if value.predicted
                else "—"
            )
            recall = f"{value.correct}/{value.gold} ({value.correct / value.gold:.1%})" if value.gold else "—"
            rows.append(f"| {kind} | {precision} | {recall} |")
        sections.append("\n".join(rows))
    return "\n\n".join(sections) + "\n"
