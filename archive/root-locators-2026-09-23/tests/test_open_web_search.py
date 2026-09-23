"""Contract tests for the retrieval-only open-web root stage."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from mellea_lrc.api import form_roots
from mellea_lrc.extraction.root_stages import ROOT_FORMATION_STAGE
from mellea_lrc.llm import IvrAttempt, IvrRun
from mellea_lrc.model.citations import FullCaseCitation, Reporter, placed
from mellea_lrc.model.document import Document
from mellea_lrc.model.extraction_metadata import ExtractionMetadata
from mellea_lrc.model.operations import judge_citation
from mellea_lrc.model.record import CitationRecord, Node, Question, Reads
from mellea_lrc.model.spans import Span
from mellea_lrc.preprocessing import preprocess
from mellea_lrc.validation.root_identity.body import ROOT_BODY_CORROBORATION_RESOLUTION_STAGE
from mellea_lrc.validation.root_identity.open_web import (
    OPEN_WEB_ROOT_IDENTITY_STAGE,
    OPEN_WEB_ROOT_SEARCH_STAGE,
    OpenWebPage,
    OpenWebResult,
    OpenWebSearchResult,
    _parse_bing_results,
    resolve_open_web_root_identities,
    search_open_web_roots,
)
from tests.record_fixtures import read_citation


class _OpenWebClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> OpenWebSearchResult:
        self.queries.append(query)
        return OpenWebSearchResult(
            query=query,
            results=(
                OpenWebResult(
                    title="Smith v. Jones — Example Reporter",
                    url="https://example.test/smith-jones",
                    snippet="A public result mentioning the citation.",
                ),
            ),
        )

    def fetch(self, result_index: int, result: OpenWebResult) -> OpenWebPage:
        return OpenWebPage(
            result_index=result_index,
            title=result.title,
            url=result.url,
            text="Smith v. Jones, 123 F.3d 456 (Example Court 2024).",
        )


def _ready_document() -> Document:
    text = "Smith v. Jones, 123 F.3d 456."
    locator = "123 F.3d 456"
    preprocessed = preprocess(text)
    record = read_citation(
        citation_id="cite-1",
        fields=placed(
            FullCaseCitation(
                plaintiff="Smith",
                defendant="Jones",
                volume="123",
                reporter=Reporter(as_written="F.3d", short_name="F.3d"),
                page="456",
            ),
            span=Span(0, len(text)),
            locator_span=Span(text.index(locator), text.index(locator) + len(locator)),
            matched_text=locator,
        ),
    )
    formed = form_roots(
        Document(
            source_metadata=preprocessed.source_metadata,
            text=text,
            preprocessing_metadata=preprocessed.preprocessing_metadata,
            citations=(record,),
            extraction_metadata=ExtractionMetadata(),
        )
    )
    root = formed.citations[0]
    judge_citation(
        root,
        Node("previous", Reads.RECORD, "identity", "test", "deferred"),
        Question.IDENTITY,
        "deferred_to_open_web_search",
        message="Provider routes were exhausted.",
    )
    return formed.evolve(passes=(ROOT_FORMATION_STAGE, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE))


def _successful_run(output: str) -> IvrRun:
    return IvrRun(
        success=True,
        selected_attempt=0,
        attempts=(IvrAttempt(output=output, requirements=()),),
        backend="test",
        model="test",
        model_options={},
        instruction="",
        prefix=None,
        grounding_context={},
        user_variables={},
        output_schema=None,
    )


def test_open_web_search_persists_a_public_result_before_page_review() -> None:
    client = _OpenWebClient()
    completed = asyncio.run(search_open_web_roots(_ready_document(), client=client))
    restored = Document.model_validate(completed.model_dump(mode="json"))
    root = restored.citations[0]
    node = next(node for node in root.trace if node.stage == OPEN_WEB_ROOT_SEARCH_STAGE)

    assert client.queries == ['"Smith" "Jones"', '"123 F.3d 456"']
    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_open_web_search"
    assert root.judgement(Question.LOCATOR_LOOKUP).outcome == "open_web_found"
    assert node.details["results"] == [
        {
            "title": "Smith v. Jones — Example Reporter",
            "url": "https://example.test/smith-jones",
            "snippet": "A public result mentioning the citation.",
        }
    ]


def test_open_web_page_review_resolves_a_grounded_authority(monkeypatch) -> None:
    client = _OpenWebClient()
    searched = asyncio.run(search_open_web_roots(_ready_document(), client=client))

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            '{"decision":"admit","result_index":1,"locator_quote":"123 F.3d 456",'
            '"case_name_quote":"Smith v. Jones","rationale":"The case heading and locator agree."}'
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.open_web.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_open_web_root_identities(searched, client=client, session=object()))
    root = Document.model_validate(completed.model_dump(mode="json")).citations[0]

    assert OPEN_WEB_ROOT_IDENTITY_STAGE in completed.passes
    assert root.judgement(Question.IDENTITY).outcome == "resolved"
    assert root.authority_id == "open_web:https://example.test/smith-jones"
    assert root.found is not None
    assert root.found.case_name == "Smith v. Jones"


def test_open_web_page_review_does_not_admit_an_ungrounded_quote(monkeypatch) -> None:
    client = _OpenWebClient()
    searched = asyncio.run(search_open_web_roots(_ready_document(), client=client))

    async def _review(*args, **kwargs):
        del args, kwargs
        return _successful_run(
            '{"decision":"admit","result_index":1,"locator_quote":"not on this page",'
            '"case_name_quote":"Smith v. Jones","rationale":"unsupported"}'
        )

    monkeypatch.setattr("mellea_lrc.validation.root_identity.open_web.run_instruct_ivr", _review)
    monkeypatch.setenv("MELLEA_LRC_LLM_MODEL", "test")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("MELLEA_LRC_LLM_API_KEY", "test")
    completed = asyncio.run(resolve_open_web_root_identities(searched, client=client, session=object()))
    root = completed.citations[0]

    assert root.judgement(Question.IDENTITY).outcome == "deferred_to_future_implementation"
    assert root.authority_id is None


def test_bing_parser_extracts_a_canonical_url_and_nested_text() -> None:
    html = """
    <li class="b_algo"><h2><a href="https://example.test/case">Smith <strong>v.</strong> Jones</a></h2>
    <div class="b_caption"><p class="b_lineclamp2">A <strong>useful</strong> result.</p></div></li>
    """

    assert _parse_bing_results(html) == (
        OpenWebResult(
            title="Smith v. Jones",
            url="https://example.test/case",
            snippet="A useful result.",
        ),
    )
