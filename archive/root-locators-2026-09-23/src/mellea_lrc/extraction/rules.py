"""Composable rules for locator and metadata-reading stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from mellea_lrc.model.extraction_metadata import Relaxation

CaseNameReader = Callable[..., Any]
CitationPass = Callable[..., Any]
TokenizerFactory = Callable[[Relaxation], Any]


@dataclass(frozen=True, slots=True)
class ExtractionRules:
    """Overrides for individual extraction readers.

    Empty rules select eyecite's tokenizer and metadata reads in individual
    readers. The stable profile adds this project's docket, case-name,
    pin-cite, colocation, and bounded context rules; each non-None field passed
    in is an explicit replacement. Docket audit remains a separate optional
    reader and is not called by the public root-growth composition.
    """

    relaxation: Relaxation | None = None
    read_dockets: bool | None = None
    tokenizer_factory: TokenizerFactory | None = None
    case_name_reader: CaseNameReader | None = None
    case_name_field_reader: CitationPass | None = None
    pin_cite_reader: CitationPass | None = None
    colocation_reader: CitationPass | None = None
    docket_auditor: CitationPass | None = None
    court_reader: CitationPass | None = None
    date_reader: CitationPass | None = None


def stable(
    rules: ExtractionRules | None = None,
    *,
    relaxation: Relaxation = Relaxation.FULL,
) -> ExtractionRules:
    """Return production rules with optional stage overrides.

    The public ``api.grow_roots`` selects this profile by default. Individual
    low-level readers can still be called with no rules for their eyecite
    baseline.
    """
    from mellea_lrc.extraction.eyecite_extractor import _case_name
    from mellea_lrc.extraction.reading.case_names import reread_case_names
    from mellea_lrc.extraction.reading.docket_audit import audit_docket_citations
    from mellea_lrc.extraction.reading.dockets import with_dockets
    from mellea_lrc.extraction.reading.pin_cite_spans import read_pin_cites
    from mellea_lrc.extraction.reading.post_citation import reread_courts, reread_dates
    from mellea_lrc.extraction.reading.relaxation import tokenizer_for
    from mellea_lrc.extraction.structure.colocation import assign_colocation

    defaults = ExtractionRules(
        relaxation=relaxation,
        read_dockets=True,
        tokenizer_factory=lambda level: with_dockets(tokenizer_for(level)),
        case_name_reader=_case_name,
        case_name_field_reader=reread_case_names,
        pin_cite_reader=read_pin_cites,
        colocation_reader=assign_colocation,
        docket_auditor=audit_docket_citations,
        court_reader=reread_courts,
        date_reader=reread_dates,
    )
    if rules is None:
        return defaults
    overrides = {
        field: getattr(rules, field)
        for field in ExtractionRules.__dataclass_fields__
        if getattr(rules, field) is not None
    }
    if rules.read_dockets is False and rules.tokenizer_factory is None:
        overrides["tokenizer_factory"] = None
    return replace(defaults, **overrides)
