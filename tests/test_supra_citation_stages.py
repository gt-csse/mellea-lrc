"""Supra discovery shares relaxed source reading with typed normalization."""

from __future__ import annotations

import asyncio

import pytest

from mellea_lrc.extraction.supra_attribution_rule import attribute_supra_citations_rule
from mellea_lrc.extraction.supra_case_names import resolve_supra_case_names
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.parsing.supra import supra_readings
from mellea_lrc.extraction.supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.model.citations import SupraCitation, latest
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.document import Document
from mellea_lrc.workflows.grow_roots import grow_roots


def _roots(source: str) -> Document:
    return asyncio.run(grow_roots(Document.from_source(source)))


@pytest.mark.parametrize(
    "quote,name,volume,pin",
    [
        ("Smith, supra.", "Smith", None, None),
        ("Smith, supra, at 495.", "Smith", None, "495"),
        ("Smith,supra,at495.", "Smith", None, "495"),
        ("Smith,\n supra ,\n at\n 495 .", "Smith", None, "495"),
        ("Smith, 42 supra, at 495.", "Smith", 42, "495"),
        ("Smith,42supra,at495.", "Smith", 42, "495"),
        ("Smith supra at 495.", "Smith", None, "495"),
        ("Smith, supra note 4, at 495.", "Smith", None, "495"),
        ("Acme Industries, Inc., supra, at 495.", "Acme Industries, Inc.", None, "495"),
        ("Alpha v. Beta, supra, at 495.", "Alpha v. Beta", None, "495"),
    ],
)
def test_reader_and_normalizer_use_the_same_exact_source(quote, name, volume, pin):
    readings = supra_readings(quote)
    assert len(readings) == 1
    reading = readings[0]
    assert quote[reading.span[0] : reading.span[1]] == quote
    assert quote[reading.antecedent_span[0] : reading.antecedent_span[1]] == name
    assert reading.volume == volume
    assert (quote[reading.pin_span[0] : reading.pin_span[1]] if reading.pin_span else None) == pin

    discovered = find_supra_citations(
        Document.from_source(quote).complete_substage("grow_roots.root_formation.rule")
    )
    citation = discovered.citations[0]
    assert citation.case_name == ()
    assert citation.pin_cite is None
    assert citation.root_id == ()
    assert citation.supra_reference[-1].normalizable
    normalized = citation.supra_reference[-1].get_normalized()
    assert normalized.antecedent == " ".join(name.split())
    assert normalized.volume == volume


@pytest.mark.parametrize("name", ["Acme Industries, Inc.", "Alpha v. Beta"])
def test_known_complete_name_boundary_is_retained_before_separate_field_reads(name):
    root_name = name if " v. " in name else name + " v. Gamma"
    source = f"{root_name}, 347 U.S. 483 (1954). Unrelated discussion. See {name},\n supra,at495."
    roots = _roots(source)
    discovered = find_supra_citations(roots)
    supra = next(c for c in discovered.short_citations if isinstance(c, SupraCitation))
    assert supra.supra_reference[-1].quote == f"{name},\n supra,at495."

    named = resolve_supra_case_names(discovered)
    pinned = resolve_supra_pin_cites(named)
    attributed = attribute_supra_citations_rule(pinned)
    supra = next(c for c in attributed.short_citations if isinstance(c, SupraCitation))
    assert supra.case_name[-1].quote == name
    assert supra.pin_cite[-1].quote == "495"
    assert latest(supra.root_id) == roots.roots[0].id
    assert supra.nodes[-1].substage == "grow_leaves.supra_citations.rule_attribution"
    restored = Document.model_validate_json(attributed.model_dump_json())
    assert restored == attributed
    assert restored.get_substage("grow_leaves.supra_citations.discovery") == discovered
    assert restored.get_substage("grow_leaves.supra_citations.case_names") == named
    assert restored.get_substage("grow_leaves.supra_citations.pin_cites") == pinned


@pytest.mark.parametrize(
    "text",
    ["supra Part III.", "See supra Section 2.", "supra, at 5.", "See supra.", "supraordinary"],
)
def test_unanchored_internal_references_do_not_create_case_citations(text):
    document = find_supra_citations(
        Document.from_source(text).complete_substage("grow_roots.root_formation.rule")
    )
    assert document.citations == ()


def test_named_supra_is_discovered_even_when_its_antecedent_root_is_missing():
    source = "Missing, supra, at 5."
    document = find_supra_citations(
        Document.from_source(source).complete_substage("grow_roots.root_formation.rule")
    )
    document = resolve_supra_case_names(document)
    document = resolve_supra_pin_cites(document)
    document = attribute_supra_citations_rule(document)
    assert len(document.citations) == 1
    assert latest(document.citations[0].root_id) == WITHDRAWN_ROOT_ID
    assert document.citations[0].routes[-1].value == "grow_leaves.supra_citations.llm_attribution"


def test_supra_pin_stops_before_another_recognized_authority():
    source = "Smith, supra, at 5, 200 F.3d 2 (2001)."
    reading = supra_readings(source)[0]
    assert source[reading.pin_span[0] : reading.pin_span[1]] == "5"
    assert source[reading.span[0] : reading.span[1]] == "Smith, supra, at 5"


def test_each_supra_stage_rejects_duplicate_execution():
    document = find_supra_citations(
        Document.from_source("Smith, supra.").complete_substage("grow_roots.root_formation.rule")
    )
    with pytest.raises(ValueError, match="already completed"):
        find_supra_citations(document)
    document = resolve_supra_case_names(document)
    with pytest.raises(ValueError, match="already completed"):
        resolve_supra_case_names(document)
    document = resolve_supra_pin_cites(document)
    with pytest.raises(ValueError, match="already completed"):
        resolve_supra_pin_cites(document)
