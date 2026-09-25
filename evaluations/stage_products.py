"""Read the durable citation work committed by one stage."""

from __future__ import annotations

from dataclasses import dataclass

from mellea_lrc.model.citations import CitationVariant
from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.history import RelationshipUpdate
from mellea_lrc.model.citations.judgments import (
    IdentityJudgment,
    ReporterExactCaseNameJudgment,
    ReporterExactCourtJudgment,
    ReporterExactDateJudgment,
)
from mellea_lrc.model.citations.reporter_lookup import ReporterExactDocket, ReporterExactLookup
from mellea_lrc.model.document import Document
from mellea_lrc.model.site_review import SiteReview


@dataclass(frozen=True)
class FieldProduct:
    citation: CitationVariant
    name: str
    reading: CitationField


@dataclass(frozen=True)
class RelationshipProduct:
    citation: CitationVariant
    name: str
    update: RelationshipUpdate


ValidationRecord = (
    ReporterExactLookup
    | ReporterExactDocket
    | ReporterExactCaseNameJudgment
    | ReporterExactCourtJudgment
    | ReporterExactDateJudgment
    | IdentityJudgment
)


@dataclass(frozen=True)
class RecordProduct:
    citation: CitationVariant
    name: str
    record: ValidationRecord


@dataclass(frozen=True)
class StageProduct:
    stage: str
    before: Document | None
    after: Document
    created: tuple[CitationVariant, ...]
    fields: tuple[FieldProduct, ...]
    relationships: tuple[RelationshipProduct, ...]
    records: tuple[RecordProduct, ...]
    reviews: tuple[SiteReview, ...]


def stage_product(document: Document, stage: str) -> StageProduct:
    """Recover a checkpoint and select entries whose nodes belong to it.

    Selecting entries by node preserves repeated and same-value updates. A
    difference between latest field values would silently lose both.
    """
    after = document.get_stage(stage)
    position = after.stage_runs.index(stage)
    before = after.get_stage(after.stage_runs[position - 1]) if position else None
    created: list[CitationVariant] = []
    fields: list[FieldProduct] = []
    relationships: list[RelationshipProduct] = []
    records: list[RecordProduct] = []
    for citation in after.citations:
        node_ids = {node.id for node in citation.nodes if node.stage == stage}
        if not node_ids:
            continue
        if citation.nodes[0].stage == stage:
            created.append(citation)
        for name in type(citation).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            value = getattr(citation, name)
            entries = value if isinstance(value, tuple) else (value,) if value is not None else ()
            for entry in entries:
                if entry.node_id not in node_ids:
                    continue
                if isinstance(entry, CitationField):
                    fields.append(FieldProduct(citation, name, entry))
                elif isinstance(entry, RelationshipUpdate):
                    relationships.append(RelationshipProduct(citation, name, entry))
                elif isinstance(
                    entry,
                    (
                        ReporterExactLookup,
                        ReporterExactDocket,
                        ReporterExactCaseNameJudgment,
                        ReporterExactCourtJudgment,
                        ReporterExactDateJudgment,
                        IdentityJudgment,
                    ),
                ):
                    records.append(RecordProduct(citation, name, entry))
                else:
                    raise TypeError(f"Unknown citation history entry: {type(entry).__name__}")
    return StageProduct(
        stage=stage,
        before=before,
        after=after,
        created=tuple(created),
        fields=tuple(fields),
        relationships=tuple(relationships),
        records=tuple(records),
        reviews=tuple(review for review in after.site_reviews if review.stage == stage),
    )
