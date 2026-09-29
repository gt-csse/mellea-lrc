"""Search GovInfo opinion bodies for the case name printed near a failed locator."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date

from mellea_lrc.govinfo import GovInfoClient
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
from mellea_lrc.model.document import Document
from mellea_lrc.validation.body_search.common import field_query_name
from mellea_lrc.validation.body_search.govinfo import GovInfoBodyClient, _problem, _search_root

STAGE = "26_govinfo_opinion_field_body_search"
ROUTE = "case_name_body_discovery"


def govinfo_opinion_field_body_search(
    document: Document,
    *,
    client: GovInfoBodyClient | None = None,
    retrospective_date: date | None = None,
) -> Document:
    """Save dated granule excerpts anchored to the printed case name.

    A name hit identifies a possible intended case. It does not establish that
    the filing's reporter or docket locator identifies that case.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "23_locator_body_review" not in document.stage_runs:
        raise ValueError("Complete locator body review before field body search")
    with ExitStack() as stack:
        service = client
        for root in document.roots:
            if root.next_stage != ROUTE:
                continue
            recorded = root.record(STAGE)
            query_name = field_query_name(root)
            if query_name is None:
                result = FieldBodySearch(
                    node_id=recorded.nodes[-1].id,
                    source=BodySource.GOVINFO_OPINION,
                    retrospective_date=retrospective_date,
                    query_name=None,
                    failures=(_problem("unsearchable_case_name", "Citation has no searchable case name"),),
                )
            else:
                if service is None:
                    service = stack.enter_context(GovInfoClient())
                search = _search_root(
                    document,
                    recorded,
                    service,
                    retrospective_date,
                    query_text=query_name,
                    anchor_kind="case_name",
                )
                if not isinstance(search, FieldBodySearch):
                    raise ValueError("A case-name search must return field evidence")
                result = search
            document = document.replace_citation(recorded.with_field_body_search(result))
    return document.complete(STAGE)
