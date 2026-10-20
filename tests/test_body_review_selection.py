"""Independent body evidence survives filtering before the review budget is applied."""

import pytest

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.body_evidence import BodySearch, BodySource
from mellea_lrc.model.citations.field_body_evidence import FieldBodySearch
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span
from mellea_lrc.validation.body_search.common import make_body_evidences, source_copy_bodies
from mellea_lrc.validation.review_intended_cases_llm.reviewer import IntendedCaseContext
from mellea_lrc.validation.review_locator_bodies_llm.reviewer import BodyCorroborationContext


@pytest.mark.parametrize("anchor_kind", ("locator", "case_name"))
def test_source_copies_cannot_exhaust_review_budget_or_renumber_saved_evidence(anchor_kind):
    citation = "Smith v. Jones, No. 05-4206 (2d Cir. 2007)."
    source = citation + " This is the source filing argument with a detailed factual account." * 30
    copied_body = source + " Extra appended material that is absent from the filing." * 100
    independent_body = "A different court discusses " + citation + " in addressing another question."
    anchor = "05-4206" if anchor_kind == "locator" else "Smith"

    def excerpts(body_id, body_text):
        return make_body_evidences(
            body_id=body_id,
            parent_id=None,
            url=None,
            issued_on=None,
            date_basis=None,
            metadata={},
            body_text=body_text,
            locator=anchor,
            source_text=source,
            anchor_kind=anchor_kind,
        )

    copies = tuple(item for i in range(6) for item in excerpts(f"copy-{i}", copied_body))
    independent = excerpts("independent", independent_body)
    evidence = copies + independent
    provider = BodySource.COURTLISTENER_OPINION
    assert len(copies) == 6 and len(independent) == 1
    assert len(source_copy_bodies(source, tuple((provider, item) for item in evidence))) == 6

    start = source.index("05-4206")
    root = FullDocketCitation.from_locator(
        citation_id="root",
        substage="sites",
        source=source,
        span=Span(source.index("No."), start + len("05-4206")),
        number_span=Span(start, start + len("05-4206")),
    )
    document = Document.from_source(source).add_citation(root).complete_substage("sites")
    root = root.record("body-search")
    if anchor_kind == "locator":
        root = root.with_body_search(
            BodySearch(node_id=root.nodes[-1].id, source=provider, retrospective_date=None, evidence=evidence)
        )
        context_type = BodyCorroborationContext
    else:
        root = root.with_field_body_search(
            FieldBodySearch(
                node_id=root.nodes[-1].id,
                source=provider,
                retrospective_date=None,
                query_name=anchor,
                evidence=evidence,
            )
        )
        context_type = IntendedCaseContext
    document = document.replace_citation(root).complete_substage("body-search")
    context = context_type.from_document(document, root)

    assert context.evidence == ((provider, 6, independent[0]),)
    assert '"evidence_index": 6' in context.prompt_evidence()
    assert Document.model_validate_json(document.model_dump_json()) == document
