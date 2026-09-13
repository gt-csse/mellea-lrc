"""The reader's prompt must not carry the held-out set's text.

A prompt written from watching `extraction-eval-1` fail would make every score
on it meaningless. Examples come from `extraction-v3.0`, which is the
development corpus; a fragment that appears only in a held-out filing is a
leak. A fragment that appears in both is fine -- `Twombly` is in the corpus and
also in half the briefs in the country.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mellea_lrc.extraction.adjudication.review.case_name import INSTRUCTION

STORE = Path.home() / "CodingProjects/mellea-lrc-datasets"
HELD_OUT = STORE / "evaluation_set_1/filings_txt"
CORPUS = STORE / "corpus/documents_txt"
#: Shorter than this and a match is a coincidence of ordinary English.
SHORTEST = 12


def _collapsed(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _fragments() -> set[str]:
    quoted = re.findall(r"`([^`]+)`", INSTRUCTION) + re.findall(r'"([^"]{6,})"', INSTRUCTION)
    return {_collapsed(fragment) for fragment in quoted if len(_collapsed(fragment)) >= SHORTEST}


@pytest.mark.skipif(not HELD_OUT.is_dir(), reason="the dataset store is not checked out here")
def test_no_prompt_example_comes_only_from_the_held_out_set() -> None:
    held_out = {path.name: _collapsed(path.read_text()) for path in HELD_OUT.glob("*.txt")}
    corpus = [_collapsed(path.read_text()) for path in CORPUS.glob("*.txt")]
    leaked = {
        fragment: sorted(name for name, text in held_out.items() if fragment in text)
        for fragment in _fragments()
        if any(fragment in text for text in held_out.values())
        and not any(fragment in text for text in corpus)
    }
    assert not leaked, f"prompt examples taken from the held-out filings: {leaked}"
