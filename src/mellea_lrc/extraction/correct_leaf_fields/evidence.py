"""Reuse saved root decisions and records without retrieving an authority."""

from __future__ import annotations

from mellea_lrc.model.citations import FullCitation, FullReporterCitation
from mellea_lrc.model.citations.leaf_field_correction import RootValidationEvidenceReference
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactCandidateDocket,
    ReporterExactDocket,
    ReporterExactLookupOutcome,
)

_REVIEWS = (
    "case_name_judgments",
    "court_judgments",
    "date_judgments",
    "identity_judgments",
    "reporter_exact_ambiguity_resolution",
    "reporter_unique_review",
    "reporter_ambiguous_review",
    "docket_lookup_review",
    "govinfo_docket_review",
    "body_reviews",
    "intended_case_reviews",
)
_RECORDS = (
    "reporter_exact_lookup",
    "reporter_exact_docket",
    "reporter_exact_candidate_dockets",
    "docket_lookup",
    "govinfo_docket_lookup",
)


def root_validation_evidence(
    root: FullCitation,
) -> tuple[tuple[RootValidationEvidenceReference, ...], tuple[dict[str, object], ...]] | None:
    """Return exact history pointers and native evidence after a saved review.

    Extraction-only roots and retrieval-only roots have no reviewed validation
    evidence. An unresolved or wrong-identity review remains useful evidence
    about extraction, without establishing or altering the root's identity.
    """
    if not any(getattr(root, name, None) for name in _REVIEWS):
        return None
    references: list[RootValidationEvidenceReference] = []
    records: list[dict[str, object]] = []
    for name in (*_RECORDS, *_REVIEWS, "locator", "case_name", "court", "date"):
        history = getattr(root, name, None)
        if history is None:
            continue
        entries = tuple(enumerate(history)) if isinstance(history, tuple) else ((None, history),)
        # Current corrected readings are the source facts used for comparison.
        if name in {"locator", "case_name", "court", "date"}:
            entries = entries[-1:]
        for index, record in entries:
            selected = _selected_index(root, name)
            reference = RootValidationEvidenceReference(
                history=name, record_index=index, node_id=record.node_id, selected_candidate_index=selected
            )
            references.append(reference)
            records.append(
                {
                    "reference": reference.model_dump(mode="json"),
                    "record": _prompt_record(name, record, selected),
                }
            )
    return tuple(references), tuple(records)


def _prompt_record(name: str, record: object, selected: int | None) -> dict[str, object]:
    """Keep cacheable evidence concise; full traces stay behind native pointers."""
    if isinstance(record, ReporterExactDocket):
        docket = record.response
        return {
            "candidate_index": record.candidate_index
            if isinstance(record, ReporterExactCandidateDocket)
            else None,
            "docket": {"id": docket.id, "court": docket.court, "court_id": docket.court_id}
            if docket
            else None,
            "failure_reason": None,
        }
    data = record.model_dump(mode="json")
    if name == "reporter_exact_lookup":
        clusters = data.get("response", {}).get("clusters", []) if data.get("response") else []
        fields = ("id", "case_name", "case_name_full", "court", "court_id", "date_filed", "docket_id")
        return {
            "outcome": data["outcome"],
            "selected_candidate_index": selected,
            "candidates": [{key: item.get(key) for key in fields} for item in clusters],
        }
    if name in {"docket_lookup", "govinfo_docket_lookup"}:
        if selected is None:
            return {
                "selected_candidate_index": None,
                "shortlisted_candidate_indices": data["shortlisted_candidate_indices"],
            }
        candidate = record.candidates[selected]
        raw = record.attempts[candidate.attempt_index].pages[candidate.page_index]["results"][
            candidate.result_index
        ]
        keys = (
            "id",
            "docket_id",
            "cluster_id",
            "caseName",
            "caseNameFull",
            "case_name",
            "court",
            "court_id",
            "dateFiled",
            "date_filed",
            "docketNumber",
            "packageId",
            "granuleId",
            "title",
        )
        return {
            "selected_candidate_index": selected,
            "selected_record": {key: raw[key] for key in keys if key in raw},
        }
    data.pop("ivr", None)
    for key in ("grounded_quote", "grounded_context", "quote_span", "context_span"):
        data.pop(key, None)
    if name in {"body_reviews", "intended_case_reviews"} and data.get("decision"):
        decision = data["decision"]
        # Body passages and proposed pin/locator readings do not help reread
        # the leaf; its already saved identity result and reason do.
        keys = ("identity_verdict", "confidence", "reason", "source", "evidence_index")
        data["decision"] = {key: decision[key] for key in keys if key in decision}
    return data


def _selected_index(root: FullCitation, history: str) -> int | None:
    if history == "reporter_exact_lookup" and isinstance(root, FullReporterCitation):
        for review in (root.reporter_ambiguous_review, root.reporter_exact_ambiguity_resolution):
            decision = getattr(review, "decision", review)
            selected = getattr(decision, "selected_candidate_index", None)
            if selected is not None:
                return selected
        lookup = root.reporter_exact_lookup
        if lookup is not None and lookup.outcome is ReporterExactLookupOutcome.UNIQUE:
            return 0
    if history in {"docket_lookup", "govinfo_docket_lookup"}:
        review_name = "docket_lookup_review" if history == "docket_lookup" else "govinfo_docket_review"
        review = getattr(root, review_name, None)
        return getattr(getattr(review, "decision", None), "selected_candidate_index", None)
    return None
