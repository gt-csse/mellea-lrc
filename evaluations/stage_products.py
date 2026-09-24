"""Read only the durable work committed by one extraction stage."""

from __future__ import annotations

from dataclasses import dataclass

from mellea_lrc.model.citations import CitationVariant
from mellea_lrc.model.citations.fields.base import CitationField
from mellea_lrc.model.citations.history import RelationshipUpdate
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


@dataclass(frozen=True)
class StageProduct:
    stage: str
    before: Document | None
    after: Document
    created: tuple[CitationVariant, ...]
    fields: tuple[FieldProduct, ...]
    relationships: tuple[RelationshipProduct, ...]
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
    for citation in after.citations:
        node_ids = {node.id for node in citation.nodes if node.stage == stage}
        if not node_ids:
            continue
        if citation.nodes[0].stage == stage:
            created.append(citation)
        for name in type(citation).model_fields:
            if name in {"id", "kind", "nodes"}:
                continue
            for entry in getattr(citation, name):
                if entry.node_id not in node_ids:
                    continue
                if isinstance(entry, CitationField):
                    fields.append(FieldProduct(citation, name, entry))
                elif isinstance(entry, RelationshipUpdate):
                    relationships.append(RelationshipProduct(citation, name, entry))
                else:
                    raise TypeError(f"Unknown citation history entry: {type(entry).__name__}")
    return StageProduct(
        stage=stage,
        before=before,
        after=after,
        created=tuple(created),
        fields=tuple(fields),
        relationships=tuple(relationships),
        reviews=tuple(review for review in after.site_reviews if review.stage == stage),
    )
