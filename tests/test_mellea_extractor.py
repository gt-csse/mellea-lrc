"""Unit tests for the Mellea naive extraction strategy (and the shared base)."""

from collections.abc import Callable
from pathlib import Path

import pytest

from mellea_lrc.core import Span
from mellea_lrc.experimental.mellea_extractors import MelleaNaive


@pytest.fixture
def korematsu_text() -> str:
    """Return the Korematsu text."""
    path = Path(__file__).parent / "data" / "Korematsu_v_US.txt"
    if not (path.exists() and path.is_file()):
        msg = f"The Korematsu file doesn't exist or is not a file: {path}"
        raise FileNotFoundError(msg)
    return path.read_text()


def load_test_citations(file_name: Path | str) -> list[tuple[str, list[int]]]:
    """Return each fixture citation with its expected [start, end] span."""
    path = Path(__file__).parent / "data" / file_name
    if not (path.exists() and path.is_file()):
        msg = f"The citations fixture doesn't exist or is not a file: {path}"
        raise FileNotFoundError(msg)
    citations = path.read_text()
    citations_components: list[list[str]] = [
        [item.strip() for item in citation.split(sep="---")]
        for citation in citations.splitlines()
        if citation
    ]
    return [
        (citation, [int(span.strip()) for span in spans.split()]) for citation, spans in citations_components
    ]


@pytest.fixture
def incorrect_korematsu_citations() -> list:
    """Return the corrupted citations that must NOT locate in the text."""
    return load_test_citations("Incorrect_Citations_Korematsu_v_US.txt")


@pytest.fixture
def correct_korematsu_citations() -> list:
    """Return the correct citations with their first-occurrence spans."""
    return load_test_citations("Correct_Citations_Korematsu_v_US.txt")


@pytest.fixture
def extractor() -> MelleaNaive:
    """Instantiate the naive strategy."""
    return MelleaNaive()


@pytest.fixture
def fake_citations(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Replace MelleaNaive's _naive_strategy with a canned response."""

    def _set(lines: list[str]) -> None:
        monkeypatch.setattr(target=MelleaNaive, name="_naive_strategy", value=lambda self, text: lines)  # noqa: ARG005

    return _set


# --- span location --
def test_locate_span_find_first_occurrence(
    extractor: MelleaNaive,
    correct_korematsu_citations: list,
    korematsu_text: str,
) -> None:
    """Each correct citation locates at its known first-occurrence span."""
    for citation, spans in correct_korematsu_citations:
        found_span: Span | None = extractor._locate_span(text=korematsu_text, matched_text=citation)
        start, end = spans
        assert found_span is not None
        assert found_span == Span(start=start, end=end)


# --- extract_citations naive ---
def test_extract_citations(
    extractor: MelleaNaive,
    fake_citations: Callable,
    korematsu_text: str,
    correct_korematsu_citations: list,
) -> None:
    """With an honest model, every fixture citation comes back, span-consistent.

    Spans are checked against the document (matched_text == text[span]), not
    against the fixture spans: the fixture repeats 'Hirabayashi ... 320 U.S. 81'
    with its first-occurrence span, while the cursor may legitimately resolve
    repeats to later occurrences.
    """
    fake_citations([citation for citation, _ in correct_korematsu_citations])

    extracted = extractor.extract_citations(text=korematsu_text)

    expected = [citation for citation, _ in correct_korematsu_citations]
    assert [c.matched_text for c in extracted.citations] == expected
    for citation in extracted.citations:
        assert korematsu_text[citation.span.start : citation.span.end] == citation.matched_text


# --- session policy (shared base behavior) ---
def test_session_requires_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session fails loudly instead of silently falling back to localhost."""
    import mellea_lrc.experimental.mellea_extractors.base as base

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(base, "find_dotenv", lambda: "")
    monkeypatch.setattr(base, "load_dotenv", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="OLLAMA_HOST"):
        _ = MelleaNaive().session
