"""Static prompt examples must not reproduce our development filings.

This scans the five active diagnostic corpora. It intentionally does not read
the untouched final held-out set. Prompts should describe general citation
shapes or use invented examples, regardless of which development set first
exposed a behavior.

Every string that reaches a model is checked, however it is built: a module
constant whose name reads like a prompt, an argument to `req`, `check` or
`instruct`, the prompt-bearing keywords of `InstructIvrSpec`, and the docstring
of a `@generative` function, which mellea sends as the prompt.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src"
STORE = SOURCE.parent / "data"
DEVELOPMENT_SETS = {
    "primary": "documents_txt",
    "hallucination-set-1": "filings_txt",
    "hallucination-set-2": "filings_txt",
    "reliable-high-profile": "filings_txt",
    "reliable-low-profile": "filings_txt",
}
#: Shorter than this and a match is a coincidence of ordinary English.
SHORTEST = 12

#: What an example looks like inside a prompt: something quoted, a case name, a
#: citation, a docket number.
EXAMPLE = (
    re.compile(r"`([^`\n]{6,})`"),
    re.compile(r'"([^"\n]{10,})"'),
    re.compile(r"'([^'\n]{10,})'"),
    re.compile(r"([A-Z][\w.'’&-]*(?:\s+[\w.'’&-]+){0,6}\s+v\.?\s+[A-Z][\w.'’&-]*(?:\s+[\w.'’&-]+){0,6})"),
    re.compile(r"(\d{1,4}\s+[A-Z][A-Za-z.' ]{1,18}\s+\d{1,4})"),
    re.compile(r"(\d{4}\s+WL\s+\d+)"),
    re.compile(r"(In re [A-Z][\w.'’ &-]{4,40})"),
    re.compile(r"(Ex parte [A-Z][\w.'’-]{2,20})"),
)
#: A module constant holding a prompt, by the name it is given.
PROMPT_NAME = re.compile(
    r"(PROMPT|PREFIX|INSTRUCTION|TEMPLATE|REQUIREMENT|GUIDANCE|RULE|ORDERING|DESCRIPTION"
    r"|REACH|ANSWER|EXAMPLE|SYSTEM|TASK|HEADER|BODY|NOTE|WINDOW)"
)
#: Calls that send what they are given to a model.
SENDS = frozenset({"req", "simple_validate", "check", "InstructIvrSpec", "instruct", "chat", "query", "act"})
SENDS_KEYWORD = frozenset(
    {
        "description",
        "requirements",
        "prompt",
        "prefix",
        "instruction",
        "grounding_context",
        "user_variables",
        "system_prompt",
    }
)


def _collapsed(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _literals(node: ast.AST) -> Iterator[str]:
    """Every string literal inside an expression, however it is assembled."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Constant) and isinstance(inner.value, str) and len(inner.value) > 15:
            yield inner.value


def _sent_to_a_model(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in SENDS:
                for argument in node.args:
                    yield from _literals(argument)
                for keyword in node.keywords:
                    if keyword.arg in SENDS_KEYWORD:
                        yield from _literals(keyword.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {getattr(d, "id", None) or getattr(d, "attr", None) for d in node.decorator_list}
            if "generative" in decorators:
                docstring = ast.get_docstring(node, clean=False)
                if docstring:
                    yield docstring
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and PROMPT_NAME.search(target.id):
                    yield from _literals(node.value)


def _examples(text: str) -> set[str]:
    found = set()
    for pattern in EXAMPLE:
        for match in pattern.finditer(text):
            fragment = _collapsed(match.group(1)).strip(" .,;:")
            if len(fragment) >= SHORTEST and re.search(r"[A-Za-z]{3}", fragment):
                found.add(fragment)
    return found


def test_prompt_scanner_includes_prefix_constants() -> None:
    tree = ast.parse('LOOKUP_PREFIX = "Compare Exampleton v. Sampleford carefully."')
    assert any("Exampleton v. Sampleford" in text for text in _sent_to_a_model(tree))


@pytest.mark.skipif(not STORE.is_dir(), reason="the dataset store is not checked out here")
def test_no_prompt_example_reproduces_a_development_filing() -> None:
    filings = {
        f"{folder}/{path.name}": _collapsed(path.read_text())
        for folder, text_folder in DEVELOPMENT_SETS.items()
        for path in (STORE / folder / text_folder).glob("*.txt")
    }
    for folder, text_folder in DEVELOPMENT_SETS.items():
        assert list((STORE / folder / text_folder).glob("*.txt")), f"missing diagnostic corpus: {folder}"

    prompts = [
        (path, text)
        for path in sorted(SOURCE.rglob("*.py"))
        for text in _sent_to_a_model(ast.parse(path.read_text()))
    ]
    assert prompts, "no prompt was found to check, which means this test is checking nothing"

    leaked: dict[str, list[str]] = {}
    for path, text in prompts:
        for fragment in _examples(text):
            where = sorted(name for name, filing in filings.items() if fragment in filing)
            if where:
                leaked[f"{path.name}: {fragment!r}"] = where
    assert not leaked, f"prompt examples reproduced in a development filing: {leaked}"
