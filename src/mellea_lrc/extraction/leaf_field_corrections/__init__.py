"""Append validation-assisted corrections to attached occurrences' own readings."""

from __future__ import annotations

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.leaf_field_corrections.evidence import root_validation_evidence
from mellea_lrc.extraction.leaf_field_corrections.reviewer import (
    IvrLeafFieldCorrectionReviewer,
    LeafFieldCorrectionContext,
    LeafFieldCorrectionOutcome,
    LeafFieldCorrectionReviewer,
)
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations import Citation, latest
from mellea_lrc.model.citations.leaf_field_correction import LeafFieldCorrectionReview
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.leaf_field_correction.review"


def _apply(
    citation: Citation,
    context: LeafFieldCorrectionContext,
    outcome: LeafFieldCorrectionOutcome,
) -> Citation:
    decision, failure = outcome.decision, outcome.failure_reason
    if outcome.run is not None and not outcome.run.success:
        failure = failure or outcome.run.failure_reason or "Leaf field correction IVR failed"
    grounded = None
    if decision is not None and failure is None:
        grounded, failure = context.grounded_corrections(decision)
    if decision is None and failure is None:
        failure = "Leaf field correction review returned no decision"
    corrected = citation
    results = []
    if failure is None and grounded is not None:
        for field, result in grounded.items():
            history = getattr(corrected, field, None)
            prior = history[-1] if history else None
            proposal = next(item for item in decision.fields if item.field == field)
            normalized = proposal.normalized
            applied = (
                prior is None
                or prior.span != result.span
                or (
                    normalized is not None
                    and (not prior.normalizable or prior.get_normalized() != normalized)
                )
            )
            if applied:
                # Each named method quotes the original filing and invokes the
                # existing field normalizer. The model supplies no pin values.
                if field == "case_name":
                    corrected = corrected.with_case_name(context.source, result.span, normalized=normalized)
                else:
                    corrected = getattr(corrected, f"with_{field}")(context.source, result.span)
            history = getattr(corrected, field)
            results.append(result.model_copy(update={"reading_index": len(history) - 1, "applied": applied}))
    return corrected.with_leaf_field_correction_review(
        LeafFieldCorrectionReview(
            node_id=corrected.nodes[-1].id,
            root_id=context.root_id,
            evidence_refs=context.evidence_refs,
            windows=context.windows,
            decision=None if failure else decision,
            grounded=tuple(results),
            ivr=outcome.run,
            failure_reason=failure,
        )
    )


async def correct_leaf_fields(
    document: Document,
    *,
    review: bool = True,
    reviewer: LeafFieldCorrectionReviewer | None = None,
) -> Document:
    """Reread attached leaves from saved root validation, without identity work.

    Every reviewed occurrence, including a repeated full citation, gets an
    append-only decision or failure. No reviewed validation evidence means no
    model call. Rule-only compositions complete an empty checkpoint.
    """
    require_leaves(document, SUBSTAGE)
    if "validate_pincite.citation_preparation.page_resolution" in document.substage_runs:
        raise ValueError("Correct leaf extraction before resolving citation pages")
    if not review:
        return document.complete_substage(SUBSTAGE)
    roots = {root.id: root for root in document.roots}
    service = reviewer
    for leaf in document.leaves:
        root = roots[latest(leaf.root_id)]
        evidence = root_validation_evidence(root)
        if evidence is None:
            continue
        context = LeafFieldCorrectionContext.from_document(document, leaf, root, evidence)
        if service is None:
            service = IvrLeafFieldCorrectionReviewer.from_profile(load_profile(SUBSTAGE))
        try:
            answer = await service(context)
            outcome = (
                answer
                if isinstance(answer, LeafFieldCorrectionOutcome)
                else LeafFieldCorrectionOutcome(answer)
            )
        except Exception as exc:
            # Keep the node-bound failure just as an unsuccessful IVR run is
            # retained. A transport failure supplies no extraction correction.
            outcome = LeafFieldCorrectionOutcome(None, failure_reason=f"{type(exc).__name__}: {exc}")
        document = document.replace_citation(_apply(leaf.record(SUBSTAGE), context, outcome))
    return document.complete_substage(SUBSTAGE)
