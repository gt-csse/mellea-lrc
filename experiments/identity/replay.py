"""Materialize one probe outcome through its production review substage offline.

Other citations reuse their saved baseline outcomes. This mixed-baseline result
is for scoped integration checks only, not a replacement full-model corpus run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import TypeAdapter

from mellea_lrc.model import Document
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review import (
    docket_root_lookup_courtlistener_llm_review,
)
from mellea_lrc.validation.docket_root_lookup_courtlistener_llm_review.reviewer import (
    DocketLookupReviewContext,
    DocketLookupReviewOutcome,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review import (
    docket_root_lookup_govinfo_llm_review,
)
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review.reviewer import (
    GovInfoDocketReviewContext,
    GovInfoDocketReviewOutcome,
)
from mellea_lrc.validation.locator_body_llm_judgment import locator_body_llm_judgment
from mellea_lrc.validation.locator_body_llm_judgment.reviewer import (
    BodyCorroborationContext,
    BodyCorroborationOutcome,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment import (
    reporter_root_lookup_ambiguous_llm_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_ambiguous_llm_judgment.reviewer import (
    ReporterAmbiguousReviewContext,
    ReporterAmbiguousReviewOutcome,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment import (
    reporter_root_lookup_unique_llm_judgment,
)
from mellea_lrc.validation.reporter_root_lookup_unique_llm_judgment.reviewer import (
    ReporterUniqueReviewContext,
    ReporterUniqueReviewOutcome,
)

# Explicit routes mirror compare_models.ROUTES without importing its runner.
_REVIEWS = {
    "unique": (
        "validate_roots.reporter_lookup.ambiguous_rule_judgment",
        "validate_roots.reporter_lookup.unique_llm_judgment",
        reporter_root_lookup_unique_llm_judgment,
        ReporterUniqueReviewContext,
        ReporterUniqueReviewOutcome,
        "reporter_unique_review",
    ),
    "ambiguous": (
        "validate_roots.reporter_lookup.unique_llm_judgment",
        "validate_roots.reporter_lookup.ambiguous_llm_judgment",
        reporter_root_lookup_ambiguous_llm_judgment,
        ReporterAmbiguousReviewContext,
        ReporterAmbiguousReviewOutcome,
        "reporter_ambiguous_review",
    ),
    "docket_cl": (
        "validate_roots.docket_lookup.courtlistener_retrieval",
        "validate_roots.docket_lookup.courtlistener_review",
        docket_root_lookup_courtlistener_llm_review,
        DocketLookupReviewContext,
        DocketLookupReviewOutcome,
        "docket_lookup_review",
    ),
    "docket_govinfo": (
        "validate_roots.docket_lookup.govinfo_retrieval",
        "validate_roots.docket_lookup.govinfo_review",
        docket_root_lookup_govinfo_llm_review,
        GovInfoDocketReviewContext,
        GovInfoDocketReviewOutcome,
        "govinfo_docket_review",
    ),
    "body": (
        "validate_roots.locator_body_corroboration.govinfo_opinion_retrieval",
        "validate_roots.locator_body_corroboration.llm_judgment",
        locator_body_llm_judgment,
        BodyCorroborationContext,
        BodyCorroborationOutcome,
        "body_reviews",
    ),
}


async def replay_review(
    original: Document,
    route: str,
    citation_id: str,
    outcome: (
        ReporterUniqueReviewOutcome
        | ReporterAmbiguousReviewOutcome
        | DocketLookupReviewOutcome
        | GovInfoDocketReviewOutcome
        | BodyCorroborationOutcome
    ),
) -> Document:
    """Recover the saved input and apply one typed outcome through production.

    The injected reviewer returns saved typed outcomes for every other citation;
    it never creates a model reviewer or performs retrieval. Contexts must match
    their native before-review snapshots uniquely, and the target must reach the
    reviewer exactly once. The returned native Document ends at the review's
    atomic checkpoint, with real corrections, judgments, and routing applied.
    """
    if route not in _REVIEWS:
        raise ValueError(f"Unknown identity review route: {route}")
    before, review, stage, context_type, outcome_type, field = _REVIEWS[route]
    if not isinstance(outcome, outcome_type):
        raise TypeError(f"Route {route} requires {outcome_type.__name__}")
    checkpoint = original.get_substage(before)
    after = original.get_substage(review)
    roots = {root.id: root for root in checkpoint.roots}
    if citation_id not in roots:
        raise ValueError(f"Target is not a root at the saved input: {citation_id}")
    contexts = {}
    baselines = {}
    for root in after.roots:
        record = getattr(root, field, None)
        if field == "body_reviews":
            stage_nodes = {node.id for node in root.nodes if node.substage == review}
            record = next((item for item in reversed(record or ()) if item.node_id in stage_nodes), None)
        if record is None:
            continue
        if root.id not in roots:
            raise ValueError(f"Saved baseline root is absent before review: {root.id}")
        contexts[root.id] = context_type.from_document(checkpoint, roots[root.id])
        baselines[root.id] = outcome_type(
            decision=record.decision,
            run=record.ivr,
            failure_reason=record.failure_reason,
        )
    if citation_id not in baselines:
        raise ValueError(f"Target has no saved {route} baseline review: {citation_id}")
    seen = set()

    async def injected_reviewer(context):
        if isinstance(context, BodyCorroborationContext):
            identifier = context.root.id
            if identifier not in contexts or context != contexts[identifier]:
                raise ValueError("Body review context differs from its saved input")
        else:
            matches = [identifier for identifier, known in contexts.items() if context == known]
            if len(matches) != 1:
                raise ValueError(f"Review context matches {len(matches)} saved roots; expected one")
            identifier = matches[0]
        if identifier in seen:
            raise ValueError(f"Production stage reviewed a root more than once: {identifier}")
        seen.add(identifier)
        return outcome if identifier == citation_id else baselines[identifier]

    result = await stage(checkpoint, reviewer=injected_reviewer)
    if citation_id not in seen:
        raise ValueError(f"Target did not reach the production reviewer: {citation_id}")
    if result.substage_runs[-1] != review:
        raise ValueError(f"Production stage did not complete {review}")
    return result


async def replay_saved_outcomes(destination: Path) -> None:
    """Materialize saved probe outcomes without generation or retrieval.

    Native fields describe only the cohort target after production applies its
    outcome. Other citations retain mixed baseline outcomes for this scoped
    check. Replay errors and native review failures remain explicit in the
    outcome JSON; no aggregate identity checkpoint is invented.
    """
    from experiments.identity.compare_models import fields_for, save

    originals: dict[str, Document] = {}
    for path in sorted((destination / "outcomes").glob("*.json")):
        saved = json.loads(await asyncio.to_thread(path.read_text))
        native_path = destination / "documents" / path.name
        native_reference = str(native_path.relative_to(destination))
        if (
            saved.get("native_document") == native_reference
            and "native_fields" in saved
            and "native_failure" in saved
            and native_path.is_file()
        ):
            continue
        saved.update(native_document=None, native_fields=None)
        try:
            case = saved["case"]
            route = case["route"]
            outcome_type = _REVIEWS[route][4]
            outcome = TypeAdapter(outcome_type).validate_python(
                {
                    "decision": saved.get("decision"),
                    "run": saved.get("run"),
                    "failure_reason": saved.get("failure"),
                }
            )
            source = case["document"]
            if source not in originals:
                source_json = await asyncio.to_thread(Path(source).read_text)
                originals[source] = Document.model_validate_json(source_json)
            replayed = await replay_review(originals[source], route, case["citation_id"], outcome)
            native = Document.model_validate_json(replayed.model_dump_json())
            if native.model_dump_json() != replayed.model_dump_json():
                raise ValueError("Native review checkpoint changed on JSON roundtrip")
            root = next(root for root in native.roots if root.id == case["citation_id"])
            review = getattr(root, _REVIEWS[route][5])
            if route == "body":
                review = review[-1]
            native_fields = fields_for(review.decision, route, root)
            save(native_path, native.model_dump(mode="json"))
            saved.update(
                native_document=native_reference,
                native_fields=native_fields,
                native_failure=review.failure_reason,
            )
        except Exception as error:
            saved["native_failure"] = f"{type(error).__name__}: {error}"
        save(path, saved)
