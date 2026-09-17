"""No prompt may carry text that only a held-out filing contains.

A prompt written from watching `evaluation_set_1` or `evaluation_set_2` fail
makes every score on that set meaningless. Examples come from `corpus/`, the 26
filings this project develops against; a fragment found in a held-out filing and
in no corpus one is a leak, and this test names it. A fragment in both is fine --
`Bell Atl. Corp. v. Twombly` is in the corpus and in half the briefs in the
country.

Every string that reaches a model is checked, however it is built: a module
constant whose name reads like a prompt, an argument to `req`, `check` or
`instruct`, the prompt-bearing keywords of `InstructIvrSpec`, and the docstring
of a `@generative` function, which mellea sends as the prompt.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src"
STORE = Path.home() / "CodingProjects/mellea-lrc-datasets"
HELD_OUT = ("evaluation_set_1", "evaluation_set_2")
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
    r"(PROMPT|INSTRUCTION|TEMPLATE|REQUIREMENT|GUIDANCE|RULE|ORDERING|DESCRIPTION"
    r"|REACH|ANSWER|EXAMPLE|SYSTEM|TASK|HEADER|BODY|NOTE|WINDOW)"
)
#: Calls that send what they are given to a model.
SENDS = frozenset({"req", "simple_validate", "check", "InstructIvrSpec", "instruct", "chat", "query", "act"})
SENDS_KEYWORD = frozenset(
    {
        "description",
        "requirements",
        "prompt",
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


@pytest.mark.skipif(not STORE.is_dir(), reason="the dataset store is not checked out here")
def test_no_prompt_example_comes_only_from_a_held_out_filing() -> None:
    held_out = {
        f"{folder}/{path.name}": _collapsed(path.read_text())
        for folder in HELD_OUT
        for path in (STORE / folder / "filings_txt").glob("*.txt")
    }
    corpus = [_collapsed(path.read_text()) for path in (STORE / "corpus/documents_txt").glob("*.txt")]
    if not held_out or not corpus:
        pytest.skip("the held-out sets and the corpus must both be present to check")

    prompts = [
        (path, text)
        for path in sorted(SOURCE.rglob("*.py"))
        for text in _sent_to_a_model(ast.parse(path.read_text()))
    ]
    assert prompts, "no prompt was found to check, which means this test is checking nothing"

    leaked: dict[str, list[str]] = {}
    for path, text in prompts:
        for fragment in _examples(text):
            where = sorted(name for name, filing in held_out.items() if fragment in filing)
            if where and not any(fragment in filing for filing in corpus):
                leaked[f"{path.name}: {fragment!r}"] = where
    assert not leaked, f"prompt examples taken from a held-out filing: {leaked}"
