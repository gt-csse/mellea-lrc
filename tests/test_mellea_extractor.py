"""Unit test for Mellea extraction."""

import pytest
from pathlib import Path
from collections.abc import Callable

from mellea_lrc.core import Span
from mellea_lrc.extraction.mellea import MelleaExtractor


@pytest.fixture
def korematsu_text() -> str:
    """Return the Korematsu text."""
    path = Path(__file__).parent / "data" / "Korematsu_v_US.txt"
    if not (path.exists() and path.is_file()):
        msg = f"The Korematsu file doesn't exist or is not a file: {path}"
        raise Exception(msg)
    return path.read_text()


@pytest.fixture
def incorrect_korematsu_citations() -> dict:
    """Return a dict with all full-case citations and their spans."""
    path = Path(__file__).parent / "data" / "Incorrect_Citations_Korematsu_v_US.txt"
    if not (path.exists() and path.is_file()):
        msg = f"The Korematsu file with citations doesn't exist or is not a file: {path}"
        raise Exception(msg)
    citations = None
    with path.open(mode="r") as file:
        citations = file.read()
    if citations is None:
        msg = "No citations found"
        raise Exception(msg)
    citations_components: list[list[str]] = [
        [item.strip() for item in citation.split("---")] for citation in citations.splitlines() if citation
    ]
    return {
        citation: [int(span.strip()) for span in spans.split()] for citation, spans in citations_components
    }


@pytest.fixture
def incorrect_spans_korematsu_citations() -> dict:
    """Return a dict with all full-case citations and their spans."""
    path = Path(__file__).parent / "data" / "Incorrect_Span_Korematsu_v_US.txt"
    if not (path.exists() and path.is_file()):
        msg = f"The Korematsu file with citations doesn't exist or is not a file: {path}"
        raise Exception(msg)
    citations = None
    with path.open(mode="r") as file:
        citations = file.read()
    if citations is None:
        msg = "No citations found"
        raise Exception(msg)
    citations_components: list[list[str]] = [
        [item.strip() for item in citation.split("---")] for citation in citations.splitlines() if citation
    ]
    return {
        citation: [int(span.strip()) for span in spans.split()] for citation, spans in citations_components
    }


@pytest.fixture
def correct_korematsu_citations() -> dict:
    """Return a dict with all full-case citations and their spans."""
    path = Path(__file__).parent / "data" / "Correct_Citations_Korematsu_v_US.txt"
    if not (path.exists() and path.is_file()):
        msg = f"The Korematsu file with citations doesn't exist or is not a file: {path}"
        raise Exception(msg)
    citations = None
    with path.open(mode="r") as file:
        citations = file.read()
    if citations is None:
        msg = "No citations found"
        raise Exception(msg)
    citations_components: list[list[str]] = [
        [item.strip() for item in citation.split("---")] for citation in citations.splitlines() if citation
    ]
    return {
        citation: [int(span.strip()) for span in spans.split()] for citation, spans in citations_components
    }


@pytest.fixture
def extractor() -> MelleaExtractor:
    """Instantiate MelleaExtractor."""
    return MelleaExtractor()


@pytest.fixture
def fake_citations(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Change MelleaExtractor's _naive_strategy to a dummy function."""

    def _set(lines: list[str]) -> None:
        monkeypatch.setattr(target=MelleaExtractor, name="_naive_strategy", value=lambda self, text: lines)  # noqa: ARG005

    return _set


# --- Test span ---
def test_locate_span_find_first_occurrence(
    extractor: MelleaExtractor,
    correct_korematsu_citations: dict,
    incorrect_korematsu_citations: dict,
    incorrect_spans_korematsu_citations: dict,
    korematsu_text: str,
) -> None:
    """Test the Mellea's Test function."""
    for citation, spans in correct_korematsu_citations.items():
        found_span: Span | None = extractor._locate_span(text=korematsu_text, matched_text=citation)
        start, end = spans
        assert found_span is not None
        assert found_span == Span(start=start, end=end)

    for citation, spans in incorrect_korematsu_citations.items():
        found_span: Span | None = extractor._locate_span(text=korematsu_text, matched_text=citation)
        assert found_span is None

    for citation, spans in incorrect_spans_korematsu_citations.items():
        found_span: Span | None = extractor._locate_span(text=korematsu_text, matched_text=citation)
        start, end = spans
        assert found_span != Span(start=start, end=end)
