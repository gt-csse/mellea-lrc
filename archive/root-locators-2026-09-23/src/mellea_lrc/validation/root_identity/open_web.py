"""Open-web retrieval and grounded page review for unresolved roots.

Search cards are only leads.  The second public stage fetches a bounded set of
result pages, grounds a model's copied locator to a selected page, and writes
the resulting admission or deferment to the citation record.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, replace
from html import unescape
from html.parser import HTMLParser
from itertools import zip_longest
from typing import TYPE_CHECKING, Literal, Protocol
from urllib.parse import parse_qs, urlparse

import requests
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.llm.grounding import fuzzy_find
from mellea_lrc.model.citations import DocketCitation, FullCaseCitation
from mellea_lrc.model.fuzziness import FuzzinessOption
from mellea_lrc.model.operations import (
    attribute_authority,
    judge_citation,
    observe_citation,
    resolve_citation,
)
from mellea_lrc.model.record import Node, Question, Reads, Resolution
from mellea_lrc.serialization._json import serialize_dataclass
from mellea_lrc.validation.root_identity.body import ROOT_BODY_CORROBORATION_RESOLUTION_STAGE
from mellea_lrc.validation.types import LocatorIdentityResolutionOutcome

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.model.document import Document
    from mellea_lrc.model.record import CitationRecord


OPEN_WEB_ROOT_SEARCH_STAGE = "open_web_root_search"
OPEN_WEB_ROOT_IDENTITY_STAGE = "open_web_root_identity"
_MADE_BY = "mellea_lrc.validation.root_identity.open_web"
_BING_SEARCH_URL = "https://www.bing.com/search"
_USER_AGENT = "mellea-lrc research (+https://github.com/gt-csse/mellea-lrc)"
_RESULT_LIMIT = 10
_PAGE_RESULT_LIMIT = 5
_PAGE_TEXT_LIMIT = 12_000
_PAGE_REVIEW_TEXT_LIMIT = 2_400
_PAGE_FETCH_TIMEOUT_SECONDS = 10
_SEARCH_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class OpenWebResult:
    """One parsed open-web result with only public display metadata."""

    title: str
    url: str
    snippet: str | None


@dataclass(frozen=True, slots=True)
class OpenWebSearchResult:
    """One complete bounded web-search response."""

    query: str
    results: tuple[OpenWebResult, ...]


@dataclass(frozen=True, slots=True)
class OpenWebPage:
    """Bounded visible text fetched from one public search result."""

    result_index: int
    title: str
    url: str
    text: str


class OpenWebSearchClient(Protocol):
    """Minimal seam for a public web-search backend."""

    def search(self, query: str) -> OpenWebSearchResult: ...

    def fetch(self, result_index: int, result: OpenWebResult) -> OpenWebPage: ...


class BingWebSearchClient:
    """Small HTML-search client requiring no separate commercial API key."""

    def __init__(self, *, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def search(self, query: str) -> OpenWebSearchResult:
        try:
            response = self.session.get(
                _BING_SEARCH_URL,
                params={"q": query, "count": str(_RESULT_LIMIT)},
                headers={"User-Agent": _USER_AGENT},
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OpenWebSearchError(f"Open-web search failed: {exc}") from exc
        return OpenWebSearchResult(query=query, results=tuple(_parse_bing_results(response.text)))

    def fetch(self, result_index: int, result: OpenWebResult) -> OpenWebPage:
        try:
            response = self.session.get(
                result.url,
                headers={"User-Agent": _USER_AGENT},
                timeout=_PAGE_FETCH_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OpenWebSearchError(f"Open-web page fetch failed: {exc}") from exc
        return OpenWebPage(
            result_index=result_index,
            title=result.title,
            url=result.url,
            text=_visible_page_text(response.text)[:_PAGE_TEXT_LIMIT],
        )


class OpenWebSearchError(RuntimeError):
    """A failed public web-search request or response parse."""


async def search_open_web_roots(
    document: Document,
    *,
    client: OpenWebSearchClient | None = None,
) -> Document:
    """Retrieve public search results for roots deferred by body corroboration.

    Search engines often tokenize punctuation-heavy legal locators into plain
    numbers.  Each root therefore first uses two source-copied case-name terms
    as a retrieval cue, then falls back to the exact locator query.  Retrieval
    is still not admission: page review must reproduce the source locator.
    """
    _require_stage(document, ROOT_BODY_CORROBORATION_RESOLUTION_STAGE)
    if OPEN_WEB_ROOT_SEARCH_STAGE in document.passes:
        return document
    _reject_partial_stage(document)

    service = client if client is not None else BingWebSearchClient()
    for record in _open_web_roots(document):
        queries = _queries(record)
        try:
            result, errors = _search(queries, service)
            node = _node(
                record,
                outcome="found" if result.results else "not_found",
                query=result.query,
                queries=queries,
                results=result.results,
                message=(
                    f"Open-web search returned {len(result.results)} public result(s)."
                    if result.results
                    else "Open-web search returned no public results."
                ),
                error="; ".join(errors) or None,
            )
        except Exception as exc:
            node = _node(
                record,
                outcome="failed",
                query=queries[0],
                queries=queries,
                results=(),
                message="Open-web search did not produce usable result evidence.",
                error=f"{type(exc).__name__}: {exc}",
            )
        observe_citation(record, node)
        judge_citation(
            record,
            node,
            _lookup_question(record),
            f"open_web_{node.outcome}",
            message=node.message,
        )
    return document.evolve(passes=(*document.passes, OPEN_WEB_ROOT_SEARCH_STAGE))


class _OpenWebIdentityProposal(BaseModel):
    """A grounded review decision over fetched public pages."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["admit", "defer"]
    result_index: int | None
    locator_quote: str | None
    case_name_quote: str | None
    rationale: str


async def resolve_open_web_root_identities(
    document: Document,
    *,
    session: MelleaSession | None = None,
    client: OpenWebSearchClient | None = None,
) -> Document:
    """Fetch, parse, ground, and decide identities from open-web results.

    A public search card is never evidence by itself.  This stage fetches a
    bounded result set, gives the model only those page texts and the filing's
    stated fields, grounds its quoted locator back to the chosen page, then
    writes either a ``resolved`` identity or an explicit remaining deferment.
    """
    _require_stage(document, OPEN_WEB_ROOT_SEARCH_STAGE)
    if OPEN_WEB_ROOT_IDENTITY_STAGE in document.passes:
        return document
    service = client if client is not None else BingWebSearchClient()
    for record in _open_web_roots(document):
        pages = _fetch_pages(record, service)
        node = await _review_pages(record, pages, session=session)
        observe_citation(record, node)
        if node.outcome == "resolved":
            selected = node.details["selected_page"]
            assert isinstance(selected, dict)
            case_name = node.details.get("case_name")
            resolution = Resolution(
                cluster_id=None,
                case_name=case_name if isinstance(case_name, str) else None,
                date_filed=None,
                court_id=None,
                node_id=node.node_id,
            )
            resolve_citation(record, node, resolution)
            attribute_authority(record, node, f"open_web:{selected['url']}")
            judge_citation(record, node, Question.IDENTITY, "resolved", message=node.message)
        else:
            judge_citation(
                record, node, Question.IDENTITY, "deferred_to_future_implementation", message=node.message
            )
    return document.evolve(passes=(*document.passes, OPEN_WEB_ROOT_IDENTITY_STAGE))


def _fetch_pages(record: CitationRecord, service: OpenWebSearchClient) -> tuple[OpenWebPage, ...]:
    search = _saved_search_node(record)
    raw = search.details.get("results", [])
    pages: list[OpenWebPage] = []
    if not isinstance(raw, list):
        return ()
    for index, item in enumerate(raw[:_PAGE_RESULT_LIMIT], start=1):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("title"), str)
            or not isinstance(item.get("url"), str)
        ):
            continue
        result = OpenWebResult(
            item["title"], item["url"], item.get("snippet") if isinstance(item.get("snippet"), str) else None
        )
        try:
            page = service.fetch(index, result)
        except Exception:
            continue
        if page.text:
            pages.append(page)
    return tuple(pages)


async def _review_pages(
    record: CitationRecord, pages: tuple[OpenWebPage, ...], *, session: MelleaSession | None
) -> Node:
    if not pages:
        return _review_node(record, "deferred", (), None, "No public result page yielded readable text.")
    source_locator = record.matched_text.strip()
    try:
        proposal_run = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            InstructIvrSpec(
                prefix=(
                    "Decide whether one fetched public page directly identifies the authority stated in the filing. "
                    "Admit only when its text identifies the same authority and contains the filing locator. "
                    "A later document merely citing the locator is not the authority. For an admission, copy both "
                    "the locator_quote and a case_name_quote from the selected page. Otherwise defer."
                ),
                description="source_locator:\n{{source_locator}}\n\nstated_case_name:\n{{stated_case_name}}\n\npages_json:\n{{pages_json}}",
                user_variables={
                    "source_locator": source_locator,
                    "stated_case_name": _stated_case_name(record),
                    "pages_json": __import__("json").dumps(
                        [_page_review_payload(page, source_locator) for page in pages], ensure_ascii=False
                    ),
                },
                output_format=_OpenWebIdentityProposal,
            ),
            strategy=MultiTurnStrategy(loop_budget=2),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=384),
        )
        if not proposal_run.success:
            return _review_node(
                record,
                "deferred",
                pages,
                None,
                "Open-web identity review exhausted its repair budget.",
                error=proposal_run.failure_reason,
                run=proposal_run,
            )
        proposal = _OpenWebIdentityProposal.model_validate_json(proposal_run.output)
    except (Exception, ValidationError) as exc:
        return _review_node(
            record,
            "deferred",
            pages,
            None,
            "Open-web identity review did not complete.",
            error=f"{type(exc).__name__}: {exc}",
        )
    selected = next((page for page in pages if page.result_index == proposal.result_index), None)
    grounded = (
        selected is not None
        and proposal.locator_quote is not None
        and fuzzy_find(
            proposal.locator_quote,
            selected.text,
            FuzzinessOption.edit_distance(similarity_percent=90),
        )
        is not None
    )
    case_name = _grounded_case_name(selected, proposal.case_name_quote)
    if proposal.decision == "admit" and grounded and selected is not None and case_name is not None:
        return _review_node(
            record, "resolved", pages, selected, proposal.rationale, run=proposal_run, case_name=case_name
        )
    return _review_node(record, "deferred", pages, None, proposal.rationale, run=proposal_run)


def _review_node(
    record: CitationRecord,
    outcome: str,
    pages: tuple[OpenWebPage, ...],
    selected: OpenWebPage | None,
    message: str,
    *,
    error: str | None = None,
    run: object | None = None,
    case_name: str | None = None,
) -> Node:
    return Node(
        node_id=f"{record.citation_id}:open_web_identity",
        reads=Reads.RECORD,
        stage=OPEN_WEB_ROOT_IDENTITY_STAGE,
        made_by=_MADE_BY,
        outcome=outcome,
        message=message,
        details={
            "pages": [serialize_dataclass(page) for page in pages],
            "selected_page": serialize_dataclass(selected) if selected else None,
            "case_name": case_name,
            "error": error,
            "ivr": serialize_dataclass(run) if run else None,
        },
    )


def _saved_search_node(record: CitationRecord) -> Node:
    return next(node for node in record.trace if node.stage == OPEN_WEB_ROOT_SEARCH_STAGE)


def _stated_case_name(record: CitationRecord) -> str:
    citation = record.fields
    if isinstance(citation, FullCaseCitation):
        return " v. ".join(part for part in (citation.plaintiff, citation.defendant) if part) or ""
    return ""


def _grounded_case_name(page: OpenWebPage | None, quote: str | None) -> str | None:
    """Require a copied case-name phrase on the same page as the locator."""
    if page is None or quote is None or not quote.strip():
        return None
    match = fuzzy_find(quote, page.text, FuzzinessOption.edit_distance(similarity_percent=90))
    return match if match is not None else None


def _page_review_payload(page: OpenWebPage, locator: str) -> dict[str, object]:
    """Send bounded source-locator excerpts while retaining full page trace.

    Search-result pages can be lengthy opinions. The model needs the heading
    and the text around the cited locator to distinguish an authority from a
    later discussion; it does not need the rest of the page. The full bounded
    fetch stays on the node for audit and for grounding the returned quotes.
    """
    lowered = page.text.casefold()
    needle = locator.casefold()
    start = lowered.find(needle) if needle else -1
    pieces = [page.text[:800]]
    if start >= 0:
        pieces.append(page.text[max(0, start - 800) : start + len(locator) + 1200])
    text = "\n…\n".join(piece for piece in pieces if piece).strip()[:_PAGE_REVIEW_TEXT_LIMIT]
    return {"result_index": page.result_index, "title": page.title, "url": page.url, "text": text}


def _open_web_roots(document: Document) -> tuple[CitationRecord, ...]:
    return tuple(
        record
        for record in document.active_citations
        if record.is_root
        and isinstance(record.fields, (DocketCitation, FullCaseCitation))
        and record.judgement(Question.IDENTITY).outcome
        == LocatorIdentityResolutionOutcome.DEFERRED_TO_OPEN_WEB_SEARCH.value
    )


def _queries(record: CitationRecord) -> tuple[str, ...]:
    """Build the small source-grounded search plan for one root."""
    locator_query = _locator_query(record)
    terms = _case_name_terms(_stated_case_name(record))
    if not terms:
        return (locator_query,)
    name_query = " ".join(f'"{term}"' for term in terms)
    return (name_query, locator_query)


def _locator_query(record: CitationRecord) -> str:
    citation = record.fields
    if isinstance(citation, DocketCitation) and citation.docket_number:
        return f'"{citation.docket_number}"'
    return f'"{record.matched_text.strip()}"'


def _case_name_terms(case_name: str) -> tuple[str, ...]:
    """Return two broad retrieval terms copied from the stated case name."""
    words = tuple(match.group() for match in re.finditer(r"[A-Za-z][A-Za-z0-9'-]*", case_name))
    distinctive = tuple(word for word in words if len(word) >= 3)
    if not distinctive:
        return ()
    if len(distinctive) == 1:
        return distinctive
    return distinctive[0], distinctive[-1]


def _search(
    queries: tuple[str, ...],
    service: OpenWebSearchClient,
) -> tuple[OpenWebSearchResult, tuple[str, ...]]:
    """Run each small query and retain a stable, de-duplicated result set."""
    result_lists: list[list[OpenWebResult]] = []
    seen_urls: set[str] = set()
    errors: list[str] = []
    for query in queries:
        try:
            response = service.search(query)
        except Exception as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
            continue
        query_results: list[OpenWebResult] = []
        for item in response.results:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            query_results.append(item)
            if len(query_results) >= _RESULT_LIMIT:
                break
        result_lists.append(query_results)
    results = tuple(item for row in zip_longest(*result_lists) for item in row if item is not None)[
        :_RESULT_LIMIT
    ]
    return OpenWebSearchResult(query=queries[0], results=results), tuple(errors)


def _lookup_question(record: CitationRecord) -> Question:
    return Question.DOCKET_LOOKUP if isinstance(record.fields, DocketCitation) else Question.LOCATOR_LOOKUP


def _node(
    record: CitationRecord,
    *,
    outcome: str,
    query: str,
    queries: tuple[str, ...],
    results: tuple[OpenWebResult, ...],
    message: str,
    error: str | None = None,
) -> Node:
    return Node(
        node_id=f"{record.citation_id}:open_web_search",
        reads=Reads.RECORD,
        stage=OPEN_WEB_ROOT_SEARCH_STAGE,
        made_by=_MADE_BY,
        outcome=outcome,
        message=message,
        details={
            "query": query,
            "queries": list(queries),
            "result_count": len(results),
            "results": [serialize_dataclass(result) for result in results],
            "error": error,
        },
    )


def _require_stage(document: Document, stage: str) -> None:
    if stage not in document.passes:
        msg = f"Open-web root search requires {stage} before retrieval."
        raise ValueError(msg)


def _reject_partial_stage(document: Document) -> None:
    if any(
        node.stage == OPEN_WEB_ROOT_SEARCH_STAGE for record in document.citations for node in record.trace
    ):
        msg = "Cannot resume a partial open-web root search; restart from the preceding Document checkpoint."
        raise ValueError(msg)


class _BingResultParser(HTMLParser):
    """Extract stable title, link, and snippet fields from Bing result cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[OpenWebResult] = []
        self._current: dict[str, object] | None = None
        self._heading_depth = 0
        self._title_depth = 0
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "li" and "b_algo" in classes:
            self._current = {"title": [], "url": None, "snippet": []}
            return
        if self._current is None:
            return
        if tag == "h2":
            self._heading_depth += 1
        elif tag == "a" and self._heading_depth:
            self._title_depth += 1
            self._current["url"] = attributes.get("href")
        elif tag == "p" and ("b_lineclamp2" in classes or "b_paractl" in classes):
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if tag == "a" and self._title_depth:
            self._title_depth -= 1
        elif tag == "h2" and self._heading_depth:
            self._heading_depth -= 1
        elif tag == "p" and self._snippet_depth:
            self._snippet_depth -= 1
        if tag == "li":
            title = _clean("".join(self._current["title"]))
            href = self._current["url"]
            if title and isinstance(href, str) and href:
                self.results.append(
                    OpenWebResult(
                        title=title,
                        url=_bing_destination(href),
                        snippet=_clean("".join(self._current["snippet"])) or None,
                    )
                )
            self._current = None

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        if self._title_depth:
            self._current["title"].append(data)
        if self._snippet_depth:
            self._current["snippet"].append(data)


class _VisibleTextParser(HTMLParser):
    """Extract readable page text while excluding executable and styling content."""

    _HIDDEN = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._HIDDEN:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self._parts.append(data)


def _parse_bing_results(html: str) -> tuple[OpenWebResult, ...]:
    parser = _BingResultParser()
    parser.feed(html)
    parser.close()
    return tuple(parser.results[:_RESULT_LIMIT])


def _visible_page_text(html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(html)
    parser.close()
    return _clean(" ".join(parser._parts))


def _bing_destination(href: str) -> str:
    """Unwrap Bing's base64 redirect when it exposes the public target URL."""
    parsed = urlparse(href)
    encoded = parse_qs(parsed.query).get("u", [None])[0]
    if not encoded or not encoded.startswith("a1"):
        return href
    try:
        payload = encoded[2:]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return href
    return decoded if urlparse(decoded).scheme in {"http", "https"} else href


def _clean(value: str) -> str:
    return " ".join(unescape(value).split())
