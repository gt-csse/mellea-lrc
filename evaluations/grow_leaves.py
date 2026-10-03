"""Independent incremental leaf scorers and their workflow summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluations.annotations import citation_annotations
from evaluations.score_types import FieldScore, Precision, StageScore
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
SHORT_COLOCATION_STAGE = "28.1_short_reporter_colocations"
SHORT_NAME_STAGE = "28.2_short_reporter_case_names"
SHORT_ATTRIBUTION_STAGE = "29_short_reporter_attribution"
REFERENCE_STAGE = "30_reference_citations"
REFERENCE_ATTRIBUTION_STAGE = "31_reference_attribution"
ID_STAGE = "32_id_citations"
ID_ATTRIBUTION_STAGE = "33_id_attribution"
SUPRA_STAGE = "34_supra_citations"
SUPRA_NAME_STAGE = "35_supra_case_names"
SUPRA_PIN_STAGE = "36_supra_pin_cites"
SUPRA_RULE_STAGE = "37_supra_attribution_rule"
SUPRA_REVIEW_STAGE = "38_supra_attribution_llm"


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
    # The native annotation unit supplies evaluation scope. Bare references
    # remain in the dataset as out_of_scope_citation rows, outside _rows.
    rows = citation_annotations(document)
    for row in rows:
        if row["kind"] == "ReferenceCitation" and _span(row.get("pin_cite")) is None:
            raise ValueError(f"{row['id']}: reference without a pinpoint must be out_of_scope_citation")
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
    rows = {r["id"]: r for r in citation_annotations(document)}
    return _key(target) == _gold_key(rows[row["root_id"]])


def _field_normalization(reading: Any, target: Any) -> bool:
    if not isinstance(target, dict):
        raise ValueError("Missing explicit field normalization gold")
    source = target.get("source", target)
    source_kind = source.get("kind", "quoted" if "start" in source else None)
    if source_kind not in {"quoted", "not_stated"}:
        raise ValueError("Field normalization gold needs quoted or not_stated source")
    if "normalization" in target:
        normalization = target["normalization"]
        if not isinstance(normalization, dict):
            raise ValueError("Missing explicit field normalization gold")
        kind = normalization.get("kind")
        if kind == "unavailable":
            if source_kind == "not_stated":
                return reading is None
            if source_kind == "quoted":
                return (
                    reading is not None
                    and not reading.normalizable
                    and _span(target) == (reading.span.start, reading.span.end)
                )
            raise ValueError("Unavailable normalization needs quoted or not_stated source")
        if kind != "value" or normalization.get("value") is None:
            raise ValueError("Missing explicit field normalization gold value")
        if source_kind == "not_stated":
            raise ValueError("A not_stated source requires unavailable normalization")
        expected = normalization["value"]
    else:
        if source_kind == "not_stated":
            raise ValueError("A not_stated source requires explicit unavailable normalization")
        if target.get("normalized") is None:
            raise ValueError("Missing explicit field normalization gold")
        expected = target["normalized"]
    if reading is None or not reading.normalizable:
        return False
    value = reading.get_normalized()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json") == expected
    return [item.model_dump(mode="json", exclude_none=True) for item in value] == expected


def score_short_reporter_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(SHORT_STAGE)
    gold = _gold(checkpoint)
    citations = checkpoint.short_reporters
    # Creation returns a locator and pin outcome for every citation.
    # Absence is an outcome too: it is correct only against explicit
    # not_stated gold. Never filter predictions by reading or gold presence.
    correct_locators = sum(
        _field_normalization(c.short_locator[-1], gold[_key(c)].get("locator"))
        for c in citations
        if _key(c) in gold
    )
    correct_pins = correct_pin_spans = 0
    for citation in citations:
        row = gold.get(_key(citation))
        if row is None:
            continue
        pin = citation.pin_cite[-1] if citation.pin_cite is not None else None
        pin_target = row.get("pin_cite")
        correct_pins += _field_normalization(pin, pin_target)
        correct_pin_spans += (
            pin_target.get("source", {}).get("kind") == "not_stated"
            if pin is None
            else _span(pin_target) == (pin.span.start, pin.span.end)
        )
    return StageScore(
        SHORT_STAGE,
        {
            "locator_span": Precision(sum(_key(c) in gold for c in citations), len(citations)),
            "locator_normalization": Precision(correct_locators, len(citations)),
            "pin_cite_span": Precision(correct_pin_spans, len(citations)),
            "pin_cite_normalization": Precision(correct_pins, len(citations)),
        },
    )


def score_short_reporter_colocations(document: Document) -> StageScore:
    document.get_stage(SHORT_COLOCATION_STAGE)
    # Current annotations do not independently label short reporter groups.
    # Do not derive a gold denominator from our own grouping or root choices.
    return StageScore(SHORT_COLOCATION_STAGE, {})


def score_short_reporter_case_names(document: Document) -> StageScore:
    checkpoint = document.get_stage(SHORT_NAME_STAGE)
    gold = _gold(checkpoint)
    citations = checkpoint.short_reporters
    correct_names = correct_spans = 0
    for citation in citations:
        row = gold.get(_key(citation))
        if row is None:
            continue
        name = citation.case_name[-1] if citation.case_name else None
        target = row.get("case_name")
        correct_names += _field_normalization(name, target)
        correct_spans += (
            target.get("source", {}).get("kind") == "not_stated"
            if name is None
            else _span(target) == (name.span.start, name.span.end)
        )
    return StageScore(
        SHORT_NAME_STAGE,
        {
            "case_name_span": Precision(correct_spans, len(citations)),
            "case_name_normalization": Precision(correct_names, len(citations)),
        },
    )


def score_short_reporter_attribution(document: Document) -> StageScore:
    checkpoint = document.get_stage(SHORT_ATTRIBUTION_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_reporters
        if c.attributions and c.attributions[-1].result == AttributionResult.ATTACHED
    ]
    return StageScore(
        SHORT_ATTRIBUTION_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, gold.get(_key(c))) for c in attached), len(attached)
            )
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
    correct_pin_spans = correct_pin_normalizations = 0
    for citation in citations:
        row = gold.get(_key(citation))
        if row is None:
            continue
        pin = citation.pin_cite[-1] if citation.pin_cite is not None else None
        target = row.get("pin_cite")
        correct_pin_normalizations += _field_normalization(pin, target)
        correct_pin_spans += (
            target.get("source", {}).get("kind") == "not_stated"
            if pin is None
            else _span(target) == (pin.span.start, pin.span.end)
        )
    return StageScore(
        ID_STAGE,
        {
            "citation_span": Precision(sum(_key(c) in gold for c in citations), len(citations)),
            "pin_cite_span": Precision(correct_pin_spans, len(citations)),
            "pin_cite_normalization": Precision(correct_pin_normalizations, len(citations)),
        },
    )


def score_reference_citations(document: Document) -> StageScore:
    checkpoint = document.get_stage(REFERENCE_STAGE)
    gold = _gold(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, ReferenceCitation)]
    correct = sum(_key(c) in gold for c in citations)
    pins = [(c, c.pin_cite[-1] if c.pin_cite is not None else None) for c in citations]
    return StageScore(
        REFERENCE_STAGE,
        {
            "case_name_span": Precision(correct, len(citations)),
            "case_name_normalization": Precision(
                sum(
                    _field_normalization(c.case_name[-1], gold[_key(c)].get("case_name"))
                    for c in citations
                    if _key(c) in gold
                ),
                len(citations),
            ),
            "pin_cite_span": Precision(
                sum(
                    f is not None
                    and _span(gold.get(_key(c), {}).get("pin_cite")) == (f.span.start, f.span.end)
                    for c, f in pins
                ),
                len(pins),
            ),
            "pin_cite_normalization": Precision(
                sum(
                    _field_normalization(f, gold[_key(c)].get("pin_cite")) for c, f in pins if _key(c) in gold
                ),
                len(pins),
            ),
        },
    )


def score_reference_attribution(document: Document) -> StageScore:
    checkpoint = document.get_stage(REFERENCE_ATTRIBUTION_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if isinstance(c, ReferenceCitation)
        and c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
    ]
    return StageScore(
        REFERENCE_ATTRIBUTION_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, gold)),
                len(attached),
            )
        },
    )


def score_supra_case_names(document: Document) -> StageScore:
    checkpoint = document.get_stage(SUPRA_NAME_STAGE)
    gold = _gold(checkpoint)
    readings = []
    for citation in checkpoint.short_citations:
        node_ids = {node.id for node in citation.nodes if node.stage == SUPRA_NAME_STAGE}
        reading = next((f for f in reversed(citation.case_name) if f.node_id in node_ids), None)
        # The reader decides a name outcome for every supra, even when it
        # writes no quote. Named short reporters and references were read at
        # creation; they belong here only if this stage actually reread them.
        if isinstance(citation, SupraCitation) or reading is not None:
            readings.append((citation, reading))
    correct = normalized = 0
    for citation, reading in readings:
        row = gold.get(_key(citation))
        if row is None:
            continue
        target = row.get("case_name")
        normalized += _field_normalization(reading, target)
        correct += (
            target.get("source", {}).get("kind") == "not_stated"
            if reading is None
            else _span(target) == (reading.span.start, reading.span.end)
        )
    return StageScore(
        SUPRA_NAME_STAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(normalized, len(readings)),
        },
    )


def score_supra_pin_cites(document: Document) -> StageScore:
    checkpoint = document.get_stage(SUPRA_PIN_STAGE)
    gold = _gold(checkpoint)
    readings = []
    for citation in checkpoint.short_citations:
        node_ids = {node.id for node in citation.nodes if node.stage == SUPRA_PIN_STAGE}
        reading = next((f for f in reversed(citation.pin_cite or ()) if f.node_id in node_ids), None)
        # Supra is eligible even when no pinpoint was found. Id. pinpoints
        # belong to creation and must not be counted again in this stage.
        if isinstance(citation, SupraCitation) or reading is not None:
            readings.append((citation, reading))
    correct = normalized = 0
    for citation, reading in readings:
        row = gold.get(_key(citation))
        if row is None:
            continue
        target = row.get("pin_cite")
        normalized += _field_normalization(reading, target)
        correct += (
            target.get("source", {}).get("kind") == "not_stated"
            if reading is None
            else _span(target) == (reading.span.start, reading.span.end)
        )
    return StageScore(
        SUPRA_PIN_STAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(normalized, len(readings)),
        },
    )


def score_supra_attribution_rule(document: Document) -> StageScore:
    checkpoint = document.get_stage(SUPRA_RULE_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.stage for n in c.nodes if n.id == c.attributions[-1].node_id) == SUPRA_RULE_STAGE
    ]
    return StageScore(
        SUPRA_RULE_STAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, gold)),
                len(attached),
            )
        },
    )


def score_supra_attribution_llm(document: Document) -> StageScore:
    checkpoint = document.get_stage(SUPRA_REVIEW_STAGE)
    gold = _gold(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.stage for n in c.nodes if n.id == c.attributions[-1].node_id) == SUPRA_REVIEW_STAGE
    ]
    return StageScore(
        SUPRA_REVIEW_STAGE,
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


GROW_LEAVES_SCORERS = {
    SHORT_STAGE: score_short_reporter_citations,
    SHORT_COLOCATION_STAGE: score_short_reporter_colocations,
    SHORT_NAME_STAGE: score_short_reporter_case_names,
    SHORT_ATTRIBUTION_STAGE: score_short_reporter_attribution,
    REFERENCE_STAGE: score_reference_citations,
    REFERENCE_ATTRIBUTION_STAGE: score_reference_attribution,
    ID_STAGE: score_id_citations,
    ID_ATTRIBUTION_STAGE: score_id_attribution,
    SUPRA_STAGE: score_supra_citations,
    SUPRA_NAME_STAGE: score_supra_case_names,
    SUPRA_PIN_STAGE: score_supra_pin_cites,
    SUPRA_RULE_STAGE: score_supra_attribution_rule,
    SUPRA_REVIEW_STAGE: score_supra_attribution_llm,
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
    completed = [stage for stage in GROW_LEAVES_SCORERS if stage in document.stage_runs]
    if not completed:
        raise ValueError("No grow_leaves stage has run")
    last = completed[-1]
    checkpoint = document.get_stage(last)
    ordered = tuple(GROW_LEAVES_SCORERS)
    required = set(ordered[: ordered.index(last) + 1]) - {SUPRA_REVIEW_STAGE}
    if not required.issubset(checkpoint.stage_runs):
        raise ValueError("Incomplete grow_leaves workflow")
    stages = tuple(
        scorer(checkpoint) for stage, scorer in GROW_LEAVES_SCORERS.items() if stage in checkpoint.stage_runs
    )
    gold = _gold(checkpoint)
    # A bounded run does not score unrun discovery types as missed citations.
    # Repeated full citations are inherited from the formed source roots.
    types = tuple(
        kind
        for kind, creation_stage in (
            ("FullCaseCitation", None),
            ("DocketCitation", None),
            ("ShortCaseCitation", SHORT_STAGE),
            ("ReferenceCitation", REFERENCE_STAGE),
            ("IdCitation", ID_STAGE),
            ("SupraCitation", SUPRA_STAGE),
        )
        if creation_stage is None or creation_stage in checkpoint.stage_runs
    )
    leaves = {k: r for k, r in gold.items() if not r["is_root"] and k[0] in types}
    spans: dict[str, FieldScore] = {}
    assignments: dict[str, FieldScore] = {}
    for kind in types:
        # The source-span summary describes citations outside the dummy-head
        # collection, regardless of why attribution placed them there. Keep
        # the independent annotated recall denominator unchanged.
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
        if ID_ATTRIBUTION_STAGE in checkpoint.stage_runs:
            assignments[kind] = FieldScore(
                sum(_root_agrees(checkpoint, c, row) for c, row in _attribution_rows(attached, target)),
                len(attached),
                len(target),
            )
    spans["all_leaves"] = FieldScore(
        sum(v.correct for v in spans.values()), sum(v.predicted for v in spans.values()), len(leaves)
    )
    if assignments:
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
            else "— (no decisions)"
        )
        rows.append(f"| {name} | {cell} |")
    return "\n".join(rows)


def render_short_reporter_citations(score: StageScore) -> str:
    if score.stage != SHORT_STAGE:
        raise ValueError("Expected short reporter discovery score")
    return render_leaf_stage(score)


def render_short_reporter_colocations(score: StageScore) -> str:
    if score.stage != SHORT_COLOCATION_STAGE:
        raise ValueError("Expected short reporter colocation checkpoint")
    return (
        f"## {score.stage}\n\n"
        "No precision score: independent short-reporter group annotations are not defined."
    )


def render_short_reporter_case_names(score: StageScore) -> str:
    if score.stage != SHORT_NAME_STAGE:
        raise ValueError("Expected short reporter case-name score")
    return render_leaf_stage(score)


def render_short_reporter_attribution(score: StageScore) -> str:
    if score.stage != SHORT_ATTRIBUTION_STAGE:
        raise ValueError("Expected short reporter attribution score")
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


def render_reference_attribution(score: StageScore) -> str:
    if score.stage != REFERENCE_ATTRIBUTION_STAGE:
        raise ValueError("Expected reference attribution score")
    return render_leaf_stage(score)


def render_supra_case_names(score: StageScore) -> str:
    if score.stage != SUPRA_NAME_STAGE:
        raise ValueError("Expected leaf case-name reading score")
    return render_leaf_stage(score)


def render_supra_pin_cites(score: StageScore) -> str:
    if score.stage != SUPRA_PIN_STAGE:
        raise ValueError("Expected leaf pin reading score")
    return render_leaf_stage(score)


def render_supra_attribution_rule(score: StageScore) -> str:
    if score.stage != SUPRA_RULE_STAGE:
        raise ValueError("Expected rule attribution score")
    return render_leaf_stage(score)


def render_supra_attribution_llm(score: StageScore) -> str:
    if score.stage != SUPRA_REVIEW_STAGE:
        raise ValueError("Expected semantic attribution score")
    return render_leaf_stage(score)


def render_id_attribution(score: StageScore) -> str:
    if score.stage != ID_ATTRIBUTION_STAGE:
        raise ValueError("Expected Id. attribution score")
    return render_leaf_stage(score)


GROW_LEAVES_RENDERERS = {
    SHORT_STAGE: render_short_reporter_citations,
    SHORT_COLOCATION_STAGE: render_short_reporter_colocations,
    SHORT_NAME_STAGE: render_short_reporter_case_names,
    SHORT_ATTRIBUTION_STAGE: render_short_reporter_attribution,
    REFERENCE_STAGE: render_reference_citations,
    REFERENCE_ATTRIBUTION_STAGE: render_reference_attribution,
    ID_STAGE: render_id_citations,
    ID_ATTRIBUTION_STAGE: render_id_attribution,
    SUPRA_STAGE: render_supra_citations,
    SUPRA_NAME_STAGE: render_supra_case_names,
    SUPRA_PIN_STAGE: render_supra_pin_cites,
    SUPRA_RULE_STAGE: render_supra_attribution_rule,
    SUPRA_REVIEW_STAGE: render_supra_attribution_llm,
}


def render_grow_leaves(score: WorkflowScore, *, set_name: str = _SET) -> str:
    sections = [
        f"# Grow-leaves evaluation: {set_name}",
        f"Through `{score.stages[-1].stage}`. Only completed stages are scored. The workflow attribution summary appears after Id. attribution has run.",
        "Stage tables score only that stage's decisions. Creation stages score every field outcome for each created citation, including not_stated or null outcomes. Absence is correct only when independently annotated as not_stated. Later reading stages likewise count each eligible citation's field outcome, including absence. Unmatched predictions and failed normalizations remain in the denominator. A matched annotation missing a required normalization target raises instead of narrowing the denominator. Workflow recall uses annotated leaves of completed discovery types, including inherited repeated full citations; unrun discovery types are omitted. The citation unit defines annotation scope; bare references remain as out_of_scope_citation rows and are not scored. All citations attached to the dummy head are excluded from the workflow source-span and attribution summary predictions. Their histories and creation-stage scores remain intact; annotated recall denominators remain unchanged.",
    ]
    sections.extend(GROW_LEAVES_RENDERERS[s.stage](s) for s in score.stages)
    for title, values in [
        ("Leaf source spans", score.leaf_spans),
        ("Leaf attribution", score.leaf_attribution),
    ]:
        if not values:
            continue
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
