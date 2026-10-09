"""Independent incremental leaf scorers and their workflow summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluations.annotations import (
    align_citation_annotations,
    annotation_kind,
    annotation_site,
    annotation_span,
    annotations_by_site,
    citation_annotations,
    citation_site,
)
from evaluations.score_types import (
    FieldScore,
    Precision,
    SubstageScore,
    group_substage_records,
    render_stage_sections,
    substage_heading,
)
from mellea_lrc.model import Document
from mellea_lrc.model.citations import (
    AttributionResult,
    IdCitation,
    LeafCitation,
    ReferenceCitation,
    SupraCitation,
    latest,
)
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID

_SET = "primary"
SHORT_SUBSTAGE = "grow_leaves.short_reporter_citations.discovery"
SHORT_COLOCATION_SUBSTAGE = "grow_leaves.short_reporter_citations.colocations"
SHORT_NAME_SUBSTAGE = "grow_leaves.short_reporter_citations.case_names"
SHORT_ATTRIBUTION_SUBSTAGE = "grow_leaves.short_reporter_citations.attribution"
REFERENCE_SUBSTAGE = "grow_leaves.reference_citations.discovery"
REFERENCE_ATTRIBUTION_SUBSTAGE = "grow_leaves.reference_citations.attribution"
ID_SUBSTAGE = "grow_leaves.id_citations.discovery"
ID_ATTRIBUTION_SUBSTAGE = "grow_leaves.id_citations.attribution"
SUPRA_SUBSTAGE = "grow_leaves.supra_citations.discovery"
SUPRA_NAME_SUBSTAGE = "grow_leaves.supra_citations.case_names"
SUPRA_PIN_SUBSTAGE = "grow_leaves.supra_citations.pin_cites"
SUPRA_RULE_SUBSTAGE = "grow_leaves.supra_citations.rule_attribution"
SUPRA_REVIEW_SUBSTAGE = "grow_leaves.supra_citations.llm_attribution"
LEAF_CORRECTION_SUBSTAGE = "grow_leaves.leaf_field_correction.review"


def _root_agrees(document: Document, citation: Any, row: dict[str, Any] | None) -> bool:
    if row is None or row["is_root"]:
        return False
    roots = {c.id: c for c in document.roots}
    target = roots.get(latest(citation.root_id))
    if target is None:
        return False
    rows = {r["id"]: r for r in citation_annotations(document)}
    return citation_site(target) == annotation_site(rows[row["root_id"]])


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
                    and annotation_span(target) == (reading.span.start, reading.span.end)
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


def score_short_reporter_citations(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SHORT_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    citations = checkpoint.short_reporters
    # Creation returns a locator and pin outcome for every citation.
    # Absence is an outcome too: it is correct only against explicit
    # not_stated gold. Never filter predictions by reading or gold presence.
    correct_locators = sum(
        _field_normalization(c.short_locator[-1], gold[citation_site(c)].get("locator"))
        for c in citations
        if citation_site(c) in gold
    )
    correct_pins = correct_pin_spans = 0
    for citation in citations:
        row = gold.get(citation_site(citation))
        if row is None:
            continue
        pin = citation.pin_cite[-1] if citation.pin_cite is not None else None
        pin_target = row.get("pin_cite")
        correct_pins += _field_normalization(pin, pin_target)
        correct_pin_spans += (
            pin_target.get("source", {}).get("kind") == "not_stated"
            if pin is None
            else annotation_span(pin_target) == (pin.span.start, pin.span.end)
        )
    return SubstageScore(
        SHORT_SUBSTAGE,
        {
            "locator_span": Precision(sum(citation_site(c) in gold for c in citations), len(citations)),
            "locator_normalization": Precision(correct_locators, len(citations)),
            "pin_cite_span": Precision(correct_pin_spans, len(citations)),
            "pin_cite_normalization": Precision(correct_pins, len(citations)),
        },
    )


def score_short_reporter_colocations(document: Document) -> SubstageScore:
    document.get_substage(SHORT_COLOCATION_SUBSTAGE)
    # Current annotations do not independently label short reporter groups.
    # Do not derive a gold denominator from our own grouping or root choices.
    return SubstageScore(SHORT_COLOCATION_SUBSTAGE, {})


def score_short_reporter_case_names(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SHORT_NAME_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    citations = checkpoint.short_reporters
    correct_names = correct_spans = 0
    for citation in citations:
        row = gold.get(citation_site(citation))
        if row is None:
            continue
        name = citation.case_name[-1] if citation.case_name else None
        target = row.get("case_name")
        correct_names += _field_normalization(name, target)
        correct_spans += (
            target.get("source", {}).get("kind") == "not_stated"
            if name is None
            else annotation_span(target) == (name.span.start, name.span.end)
        )
    return SubstageScore(
        SHORT_NAME_SUBSTAGE,
        {
            "case_name_span": Precision(correct_spans, len(citations)),
            "case_name_normalization": Precision(correct_names, len(citations)),
        },
    )


def score_short_reporter_attribution(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SHORT_ATTRIBUTION_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    attached = [
        c
        for c in checkpoint.short_reporters
        if c.attributions and c.attributions[-1].result == AttributionResult.ATTACHED
    ]
    return SubstageScore(
        SHORT_ATTRIBUTION_SUBSTAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, gold.get(citation_site(c))) for c in attached), len(attached)
            )
        },
    )


def score_supra_citations(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SUPRA_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, SupraCitation)]
    return SubstageScore(
        SUPRA_SUBSTAGE,
        {
            "span": Precision(sum(citation_site(c) in gold for c in citations), len(citations)),
            "normalization": Precision(),
        },
    )


def score_id_citations(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(ID_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, IdCitation)]
    correct_pin_spans = correct_pin_normalizations = 0
    for citation in citations:
        row = gold.get(citation_site(citation))
        if row is None:
            continue
        pin = citation.pin_cite[-1] if citation.pin_cite is not None else None
        target = row.get("pin_cite")
        correct_pin_normalizations += _field_normalization(pin, target)
        correct_pin_spans += (
            target.get("source", {}).get("kind") == "not_stated"
            if pin is None
            else annotation_span(target) == (pin.span.start, pin.span.end)
        )
    return SubstageScore(
        ID_SUBSTAGE,
        {
            "citation_span": Precision(sum(citation_site(c) in gold for c in citations), len(citations)),
            "pin_cite_span": Precision(correct_pin_spans, len(citations)),
            "pin_cite_normalization": Precision(correct_pin_normalizations, len(citations)),
        },
    )


def score_reference_citations(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(REFERENCE_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    citations = [c for c in checkpoint.short_citations if isinstance(c, ReferenceCitation)]
    correct = sum(citation_site(c) in gold for c in citations)
    pins = [(c, c.pin_cite[-1] if c.pin_cite is not None else None) for c in citations]
    return SubstageScore(
        REFERENCE_SUBSTAGE,
        {
            "case_name_span": Precision(correct, len(citations)),
            "case_name_normalization": Precision(
                sum(
                    _field_normalization(c.case_name[-1], gold[citation_site(c)].get("case_name"))
                    for c in citations
                    if citation_site(c) in gold
                ),
                len(citations),
            ),
            "pin_cite_span": Precision(
                sum(
                    f is not None
                    and annotation_span(gold.get(citation_site(c), {}).get("pin_cite"))
                    == (f.span.start, f.span.end)
                    for c, f in pins
                ),
                len(pins),
            ),
            "pin_cite_normalization": Precision(
                sum(
                    _field_normalization(f, gold[citation_site(c)].get("pin_cite"))
                    for c, f in pins
                    if citation_site(c) in gold
                ),
                len(pins),
            ),
        },
    )


def score_reference_attribution(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(REFERENCE_ATTRIBUTION_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if isinstance(c, ReferenceCitation)
        and c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
    ]
    return SubstageScore(
        REFERENCE_ATTRIBUTION_SUBSTAGE,
        {
            "attribution": Precision(
                sum(
                    _root_agrees(checkpoint, c, row) for c, row in align_citation_annotations(attached, gold)
                ),
                len(attached),
            )
        },
    )


def score_supra_case_names(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SUPRA_NAME_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    readings = []
    for citation in checkpoint.short_citations:
        node_ids = {node.id for node in citation.nodes if node.substage == SUPRA_NAME_SUBSTAGE}
        reading = next((f for f in reversed(citation.case_name) if f.node_id in node_ids), None)
        # The reader decides a name outcome for every supra, even when it
        # writes no quote. Named short reporters and references were read at
        # creation; they belong here only if this substage actually reread them.
        if isinstance(citation, SupraCitation) or reading is not None:
            readings.append((citation, reading))
    correct = normalized = 0
    for citation, reading in readings:
        row = gold.get(citation_site(citation))
        if row is None:
            continue
        target = row.get("case_name")
        normalized += _field_normalization(reading, target)
        correct += (
            target.get("source", {}).get("kind") == "not_stated"
            if reading is None
            else annotation_span(target) == (reading.span.start, reading.span.end)
        )
    return SubstageScore(
        SUPRA_NAME_SUBSTAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(normalized, len(readings)),
        },
    )


def score_supra_pin_cites(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SUPRA_PIN_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    readings = []
    for citation in checkpoint.short_citations:
        node_ids = {node.id for node in citation.nodes if node.substage == SUPRA_PIN_SUBSTAGE}
        reading = next((f for f in reversed(citation.pin_cite or ()) if f.node_id in node_ids), None)
        # Supra is eligible even when no pinpoint was found. Id. pinpoints
        # belong to creation and must not be counted again in this substage.
        if isinstance(citation, SupraCitation) or reading is not None:
            readings.append((citation, reading))
    correct = normalized = 0
    for citation, reading in readings:
        row = gold.get(citation_site(citation))
        if row is None:
            continue
        target = row.get("pin_cite")
        normalized += _field_normalization(reading, target)
        correct += (
            target.get("source", {}).get("kind") == "not_stated"
            if reading is None
            else annotation_span(target) == (reading.span.start, reading.span.end)
        )
    return SubstageScore(
        SUPRA_PIN_SUBSTAGE,
        {
            "span": Precision(correct, len(readings)),
            "normalization": Precision(normalized, len(readings)),
        },
    )


def score_supra_attribution_rule(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SUPRA_RULE_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.substage for n in c.nodes if n.id == c.attributions[-1].node_id) == SUPRA_RULE_SUBSTAGE
    ]
    return SubstageScore(
        SUPRA_RULE_SUBSTAGE,
        {
            "attribution": Precision(
                sum(
                    _root_agrees(checkpoint, c, row) for c, row in align_citation_annotations(attached, gold)
                ),
                len(attached),
            )
        },
    )


def score_supra_attribution_llm(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(SUPRA_REVIEW_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if c.attributions
        and c.attributions[-1].result == AttributionResult.ATTACHED
        and next(n.substage for n in c.nodes if n.id == c.attributions[-1].node_id) == SUPRA_REVIEW_SUBSTAGE
    ]
    return SubstageScore(
        SUPRA_REVIEW_SUBSTAGE,
        {
            "attribution": Precision(
                sum(
                    _root_agrees(checkpoint, c, row) for c, row in align_citation_annotations(attached, gold)
                ),
                len(attached),
            )
        },
    )


def score_id_attribution(document: Document) -> SubstageScore:
    checkpoint = document.get_substage(ID_ATTRIBUTION_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    attached = [
        c
        for c in checkpoint.short_citations
        if isinstance(c, IdCitation) and latest(c.root_id) not in {None, WITHDRAWN_ROOT_ID}
    ]
    return SubstageScore(
        ID_ATTRIBUTION_SUBSTAGE,
        {
            "attribution": Precision(
                sum(_root_agrees(checkpoint, c, gold.get(citation_site(c))) for c in attached), len(attached)
            )
        },
    )


def score_leaf_field_corrections(document: Document) -> SubstageScore:
    """Score each reviewed field outcome against the existing field targets."""
    checkpoint = document.get_substage(LEAF_CORRECTION_SUBSTAGE)
    gold = annotations_by_site(checkpoint)
    reviewed = [
        citation
        for citation in checkpoint.leaves
        if any(
            review.decision is not None
            and next(node.substage for node in citation.nodes if node.id == review.node_id)
            == LEAF_CORRECTION_SUBSTAGE
            for review in citation.leaf_field_correction_reviews
        )
    ]
    metrics = {}
    for field in ("case_name", "pin_cite", "court", "date"):
        outcomes = [
            citation
            for citation in reviewed
            if field in {window.field for window in citation.leaf_field_correction_reviews[-1].windows}
        ]
        if not outcomes:
            continue
        spans = normalized = 0
        for citation in outcomes:
            row = gold.get(citation_site(citation))
            if row is None:
                continue
            readings = getattr(citation, field, None)
            reading = readings[-1] if readings else None
            target = row.get(field)
            if not isinstance(target, dict):
                continue
            normalized += _field_normalization(reading, target)
            spans += (
                target.get("source", {}).get("kind") == "not_stated"
                if reading is None
                else reading.span is not None
                and annotation_span(target) == (reading.span.start, reading.span.end)
            )
        metrics[f"{field}_span"] = Precision(spans, len(outcomes))
        metrics[f"{field}_normalization"] = Precision(normalized, len(outcomes))
    return SubstageScore(LEAF_CORRECTION_SUBSTAGE, metrics)


GROW_LEAVES_SCORERS = {
    SHORT_SUBSTAGE: score_short_reporter_citations,
    SHORT_COLOCATION_SUBSTAGE: score_short_reporter_colocations,
    SHORT_NAME_SUBSTAGE: score_short_reporter_case_names,
    SHORT_ATTRIBUTION_SUBSTAGE: score_short_reporter_attribution,
    REFERENCE_SUBSTAGE: score_reference_citations,
    REFERENCE_ATTRIBUTION_SUBSTAGE: score_reference_attribution,
    ID_SUBSTAGE: score_id_citations,
    ID_ATTRIBUTION_SUBSTAGE: score_id_attribution,
    SUPRA_SUBSTAGE: score_supra_citations,
    SUPRA_NAME_SUBSTAGE: score_supra_case_names,
    SUPRA_PIN_SUBSTAGE: score_supra_pin_cites,
    SUPRA_RULE_SUBSTAGE: score_supra_attribution_rule,
    SUPRA_REVIEW_SUBSTAGE: score_supra_attribution_llm,
    LEAF_CORRECTION_SUBSTAGE: score_leaf_field_corrections,
}


@dataclass(frozen=True)
class WorkflowScore:
    substages: tuple[SubstageScore, ...]
    leaf_spans: dict[str, FieldScore]
    leaf_attribution: dict[str, FieldScore]
    completed_stages: tuple[str, ...] = ()

    def __add__(self, other: WorkflowScore) -> WorkflowScore:
        if tuple(s.substage for s in self.substages) != tuple(s.substage for s in other.substages):
            raise ValueError("Cannot combine different leaf workflow substages")
        return WorkflowScore(
            tuple(a + b for a, b in zip(self.substages, other.substages, strict=True)),
            {k: v + other.leaf_spans[k] for k, v in self.leaf_spans.items()},
            {k: v + other.leaf_attribution[k] for k, v in self.leaf_attribution.items()},
            tuple(stage for stage in self.completed_stages if stage in other.completed_stages),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "grow_leaves",
            "stages": group_substage_records(
                "grow_leaves", (item.as_dict() for item in self.substages), self.completed_stages
            ),
            "leaf_spans": {k: v.as_dict() for k, v in self.leaf_spans.items()},
            "leaf_attribution": {k: v.as_dict() for k, v in self.leaf_attribution.items()},
        }


def score_grow_leaves(document: Document) -> WorkflowScore:
    completed = [substage for substage in document.substage_runs if substage in GROW_LEAVES_SCORERS]
    if not completed:
        raise ValueError("No grow_leaves substage has run")
    last = completed[-1]
    checkpoint = document.get_substage(last)
    ordered = tuple(GROW_LEAVES_SCORERS)
    required = set(ordered[: ordered.index(last) + 1]) - {SUPRA_REVIEW_SUBSTAGE}
    if not required.issubset(checkpoint.substage_runs):
        raise ValueError("Incomplete grow_leaves workflow")
    stages = tuple(GROW_LEAVES_SCORERS[substage](checkpoint) for substage in completed)
    gold = annotations_by_site(checkpoint)
    # A bounded run does not score unrun discovery types as missed citations.
    # Repeated full citations are inherited from the formed source roots.
    types = tuple(
        kind
        for kind, creation_stage in (
            ("FullCaseCitation", None),
            ("DocketCitation", None),
            ("ShortCaseCitation", SHORT_SUBSTAGE),
            ("ReferenceCitation", REFERENCE_SUBSTAGE),
            ("IdCitation", ID_SUBSTAGE),
            ("SupraCitation", SUPRA_SUBSTAGE),
        )
        if creation_stage is None or creation_stage in checkpoint.substage_runs
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
            if annotation_kind(c) == kind
            and latest(c.root_id) != WITHDRAWN_ROOT_ID
            and (isinstance(c, LeafCitation) or latest(c.root_id) != c.id)
        ]
        target = {k: r for k, r in leaves.items() if k[0] == kind}
        spans[kind] = FieldScore(
            sum(citation_site(c) in target for c in predicted), len(predicted), len(target)
        )
        attached = [c for c in predicted if latest(c.root_id) not in {None, WITHDRAWN_ROOT_ID}]
        if ID_ATTRIBUTION_SUBSTAGE in checkpoint.substage_runs:
            assignments[kind] = FieldScore(
                sum(
                    _root_agrees(checkpoint, c, row)
                    for c, row in align_citation_annotations(attached, target)
                ),
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
    return WorkflowScore(
        stages,
        spans,
        assignments,
        tuple(stage for stage in document.stage_runs if stage.startswith("grow_leaves.")),
    )


def render_leaf_substage(score: SubstageScore) -> str:
    # Rendering is shared text formatting, not a shared substage scoring policy.
    rows = [f"{substage_heading(score.substage)}", "", "| Metric | Precision |", "| --- | ---: |"]
    for name, value in score.metrics.items():
        cell = (
            f"{value.correct}/{value.total} ({value.correct / value.total:.1%})"
            if value.total
            else "— (no decisions)"
        )
        rows.append(f"| {name} | {cell} |")
    return "\n".join(rows)


def render_short_reporter_citations(score: SubstageScore) -> str:
    if score.substage != SHORT_SUBSTAGE:
        raise ValueError("Expected short reporter discovery score")
    return render_leaf_substage(score)


def render_short_reporter_colocations(score: SubstageScore) -> str:
    if score.substage != SHORT_COLOCATION_SUBSTAGE:
        raise ValueError("Expected short reporter colocation checkpoint")
    return (
        f"{substage_heading(score.substage)}\n\n"
        "No precision score: independent short-reporter group annotations are not defined."
    )


def render_short_reporter_case_names(score: SubstageScore) -> str:
    if score.substage != SHORT_NAME_SUBSTAGE:
        raise ValueError("Expected short reporter case-name score")
    return render_leaf_substage(score)


def render_short_reporter_attribution(score: SubstageScore) -> str:
    if score.substage != SHORT_ATTRIBUTION_SUBSTAGE:
        raise ValueError("Expected short reporter attribution score")
    return render_leaf_substage(score)


def render_supra_citations(score: SubstageScore) -> str:
    if score.substage != SUPRA_SUBSTAGE:
        raise ValueError("Expected supra discovery score")
    return render_leaf_substage(score)


def render_id_citations(score: SubstageScore) -> str:
    if score.substage != ID_SUBSTAGE:
        raise ValueError("Expected Id. discovery score")
    return render_leaf_substage(score)


def render_reference_citations(score: SubstageScore) -> str:
    if score.substage != REFERENCE_SUBSTAGE:
        raise ValueError("Expected reference discovery score")
    return render_leaf_substage(score)


def render_reference_attribution(score: SubstageScore) -> str:
    if score.substage != REFERENCE_ATTRIBUTION_SUBSTAGE:
        raise ValueError("Expected reference attribution score")
    return render_leaf_substage(score)


def render_supra_case_names(score: SubstageScore) -> str:
    if score.substage != SUPRA_NAME_SUBSTAGE:
        raise ValueError("Expected leaf case-name reading score")
    return render_leaf_substage(score)


def render_supra_pin_cites(score: SubstageScore) -> str:
    if score.substage != SUPRA_PIN_SUBSTAGE:
        raise ValueError("Expected leaf pin reading score")
    return render_leaf_substage(score)


def render_supra_attribution_rule(score: SubstageScore) -> str:
    if score.substage != SUPRA_RULE_SUBSTAGE:
        raise ValueError("Expected rule attribution score")
    return render_leaf_substage(score)


def render_supra_attribution_llm(score: SubstageScore) -> str:
    if score.substage != SUPRA_REVIEW_SUBSTAGE:
        raise ValueError("Expected semantic attribution score")
    return render_leaf_substage(score)


def render_id_attribution(score: SubstageScore) -> str:
    if score.substage != ID_ATTRIBUTION_SUBSTAGE:
        raise ValueError("Expected Id. attribution score")
    return render_leaf_substage(score)


def render_leaf_field_corrections(score: SubstageScore) -> str:
    if score.substage != LEAF_CORRECTION_SUBSTAGE:
        raise ValueError("Expected leaf field-correction checkpoint")
    return render_leaf_substage(score)


GROW_LEAVES_RENDERERS = {
    SHORT_SUBSTAGE: render_short_reporter_citations,
    SHORT_COLOCATION_SUBSTAGE: render_short_reporter_colocations,
    SHORT_NAME_SUBSTAGE: render_short_reporter_case_names,
    SHORT_ATTRIBUTION_SUBSTAGE: render_short_reporter_attribution,
    REFERENCE_SUBSTAGE: render_reference_citations,
    REFERENCE_ATTRIBUTION_SUBSTAGE: render_reference_attribution,
    ID_SUBSTAGE: render_id_citations,
    ID_ATTRIBUTION_SUBSTAGE: render_id_attribution,
    SUPRA_SUBSTAGE: render_supra_citations,
    SUPRA_NAME_SUBSTAGE: render_supra_case_names,
    SUPRA_PIN_SUBSTAGE: render_supra_pin_cites,
    SUPRA_RULE_SUBSTAGE: render_supra_attribution_rule,
    SUPRA_REVIEW_SUBSTAGE: render_supra_attribution_llm,
    LEAF_CORRECTION_SUBSTAGE: render_leaf_field_corrections,
}


def render_grow_leaves(score: WorkflowScore, *, set_name: str = _SET) -> str:
    sections = [
        f"# Grow-leaves evaluation: {set_name}",
        f"Through `{score.substages[-1].substage}`. Only completed substages are scored. The workflow attribution summary appears after Id. attribution has run.",
        "Substage tables score only that substage's decisions. Creation substages score every field outcome for each created citation, including not_stated or null outcomes. Absence is correct only when independently annotated as not_stated. Later reading substages likewise count each eligible citation's field outcome, including absence. Unmatched predictions and failed normalizations remain in the denominator. A matched annotation missing a required normalization target raises instead of narrowing the denominator. Workflow recall uses annotated leaves of completed discovery types, including inherited repeated full citations; unrun discovery types are omitted. The citation unit defines annotation scope; bare references remain as out_of_scope_citation rows and are not scored. All citations attached to the dummy head are excluded from the workflow source-span and attribution summary predictions. Their histories and creation-substage scores remain intact; annotated recall denominators remain unchanged.",
    ]
    sections.extend(
        render_stage_sections(
            "grow_leaves",
            ((item.substage, GROW_LEAVES_RENDERERS[item.substage](item)) for item in score.substages),
            score.completed_stages,
        )
    )
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
