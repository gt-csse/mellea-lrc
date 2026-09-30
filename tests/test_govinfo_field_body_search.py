"""GovInfo name discovery keeps possible case identities separate from locator proof."""

from __future__ import annotations

from datetime import date

import pytest

from mellea_lrc.providers.govinfo import GovInfoSearchPage
from mellea_lrc.model.citations.body_evidence import BodySource
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.intended_case_govinfo_opinion_retrieval import (
    STAGE,
    intended_case_govinfo_opinion_retrieval,
)
from mellea_lrc.validation.body_search import locator_body_govinfo_opinion_retrieval as govinfo_module

SOURCE = "Acme v. Smith, 30 F.3d 100 (2d Cir. 1994)."
PACKAGE = "USCOURTS-nyd-1_20-cv-1"


def _document(*, case_name: bool = True, routed: bool = True) -> Document:
    locator = "30 F.3d 100"
    start = SOURCE.index(locator)
    root = FullReporterCitation.from_locator(
        citation_id="root:1", stage="01_extract", source=SOURCE, span=Span(start, start + len(locator))
    )
    document = Document.from_source(SOURCE).add_citation(root).complete("01_extract")
    root = root.record("02_roots").with_root(root.id)
    document = document.replace_citation(root).complete("02_roots")
    if case_name:
        root = root.record("03_case_name").with_case_name(SOURCE, Span(0, len("Acme v. Smith")))
        document = document.replace_citation(root).complete("03_case_name")
    if routed:
        root = root.record("23_locator_body_llm_judgment").with_route("case_name_body_discovery")
        document = document.replace_citation(root).complete("23_locator_body_llm_judgment")
    else:
        document = document.complete("23_locator_body_llm_judgment")
    return document


class FakeGovInfoClient:
    def __init__(self, bodies: dict[str, str], dates: dict[str, str]) -> None:
        self.bodies = bodies
        self.dates = dates
        self.search_calls: list[str] = []
        self.summary_calls: list[str] = []
        self.download_calls: list[str] = []

    def search(
        self, query: str, *, offset_mark: str = "*", page_size: int = 100, result_level: str = "package"
    ) -> GovInfoSearchPage:
        assert offset_mark == "*"
        assert result_level == "default"
        self.search_calls.append(query)
        results = tuple(
            {"packageId": PACKAGE, "granuleId": granule, "dateIssued": "1990-01-01"}
            for granule in self.bodies
        )
        return GovInfoSearchPage(
            raw_json={"count": len(results), "results": list(results)},
            results=results,
            count=len(results),
            next_offset_mark=None,
        )

    def get_granule_summary(self, package_id: str, granule_id: str) -> dict:
        assert package_id == PACKAGE
        self.summary_calls.append(granule_id)
        return {
            "packageId": PACKAGE,
            "granuleId": granule_id,
            "dateIssued": self.dates[granule_id],
            "download": {"pdfLink": f"https://api.govinfo.gov/{granule_id}/pdf"},
        }

    def download_pdf(self, pdf_url: str) -> bytes:
        self.download_calls.append(pdf_url)
        granule_id = pdf_url.split("/")[-2]
        return self.bodies[granule_id].encode("utf-8")

    def list_granules(self, package_id: str, *, offset_mark: str = "*", page_size: int = 100):
        raise AssertionError(f"Direct granule results should not list package {package_id}")


def test_name_hit_with_another_locator_preserves_evidence_without_confirming_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(govinfo_module, "_pdf_text", lambda data: data.decode("utf-8"))
    client = FakeGovInfoClient(
        {"other": "A later opinion discusses Acme v. Smith, 77 F.3d 999 (1995)."},
        {"other": "2000-01-01"},
    )
    before = _document()
    after = intended_case_govinfo_opinion_retrieval(before, client=client)

    assert STAGE == "26_intended_case_govinfo_opinion_retrieval"
    assert after.stage_runs == (*before.stage_runs, STAGE)
    assert after.get_stage("23_locator_body_llm_judgment") == before
    assert client.search_calls == ["collection:uscourts and Acme and Smith"]
    assert client.summary_calls == ["other"]
    search = after.roots[0].field_body_searches[0]
    assert search.source is BodySource.GOVINFO_OPINION
    assert search.query_name == "Acme"
    assert search.attempts[0].pages[0]["count"] == 1
    assert len(search.evidence) == 1
    evidence = search.evidence[0]
    assert evidence.anchor_kind == "case_name"
    assert evidence.excerpt[evidence.anchor_span.start : evidence.anchor_span.end] == "Acme"
    assert "77 F.3d 999" in evidence.excerpt
    assert "30 F.3d 100" not in evidence.excerpt
    assert after.roots[0].locator == before.roots[0].locator
    assert after.roots[0].identity_judgments == before.roots[0].identity_judgments == ()
    assert after.roots[0].next_stage == "case_name_body_discovery"
    assert Document.model_validate_json(after.model_dump_json()) == after


def test_precise_granule_date_controls_cutoff_and_unrelated_body_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(govinfo_module, "_pdf_text", lambda data: data.decode("utf-8"))
    client = FakeGovInfoClient(
        {
            "late": "A later court cites Acme v. Smith, 77 F.3d 999.",
            "unrelated": "This granule has no citation to that case.",
        },
        {"late": "2026-01-01", "unrelated": "2024-01-01"},
    )
    after = intended_case_govinfo_opinion_retrieval(
        _document(), client=client, retrospective_date=date(2025, 1, 1)
    )
    search = after.roots[0].field_body_searches[0]

    assert search.retrospective_date == date(2025, 1, 1)
    assert search.evidence == ()
    assert [(failure.failure_type, failure.item_id) for failure in search.failures] == [
        ("ineligible_date", "late"),
        ("no_case_name", "unrelated"),
    ]
    assert client.download_calls == ["https://api.govinfo.gov/unrelated/pdf"]
    assert after.roots[0].identity_judgments == ()


def test_two_party_search_stops_after_four_unhelpful_pdfs_without_broad_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(govinfo_module, "_pdf_text", lambda data: data.decode("utf-8"))
    client = FakeGovInfoClient(
        {str(index): "An unrelated opinion." for index in range(1, 6)},
        {str(index): "1990-01-01" for index in range(1, 6)},
    )

    after = intended_case_govinfo_opinion_retrieval(_document(), client=client)

    assert client.search_calls == ["collection:uscourts and Acme and Smith"]
    assert len(client.download_calls) == 4
    search = after.roots[0].field_body_searches[0]
    assert search.query_name == "Acme"
    assert search.evidence == ()
    assert len(search.attempts) == 1
    assert search.failures[-1].failure_type == "fetch_limit"


def test_missing_name_and_unrouted_root_make_no_provider_calls() -> None:
    client = FakeGovInfoClient({}, {})
    unnamed = intended_case_govinfo_opinion_retrieval(_document(case_name=False), client=client)
    search = unnamed.roots[0].field_body_searches[0]
    assert search.query_name is None
    assert search.attempts == ()
    assert search.evidence == ()
    assert search.failures[0].failure_type == "unsearchable_case_name"
    assert client.search_calls == []

    unqueued = intended_case_govinfo_opinion_retrieval(_document(routed=False), client=client)
    assert unqueued.roots[0].field_body_searches == ()
    assert client.search_calls == []


def test_field_search_needs_locator_review_and_cannot_run_twice() -> None:
    with pytest.raises(ValueError, match="locator body review"):
        intended_case_govinfo_opinion_retrieval(Document.from_source("No citations"))
    completed = intended_case_govinfo_opinion_retrieval(
        _document(routed=False), client=FakeGovInfoClient({}, {})
    )
    with pytest.raises(ValueError, match="already completed"):
        intended_case_govinfo_opinion_retrieval(completed, client=FakeGovInfoClient({}, {}))
