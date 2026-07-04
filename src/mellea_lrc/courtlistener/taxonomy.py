"""CourtListener court taxonomy mapping and lookup."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CLCourtTaxonomy:
    """CourtListener authority taxonomy for a specific court."""

    court_id: str
    system: str | None
    jurisdiction: str | None
    type: str | None


_TAXONOMY_CACHE: dict[str, CLCourtTaxonomy] | None = None


def get_court_taxonomy(court_id: str) -> CLCourtTaxonomy | None:
    """Return the CourtListener taxonomy for a court ID."""
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        _TAXONOMY_CACHE = {}
        json_path = Path(__file__).parent / "cl_court_taxonomy.json"
        if json_path.exists():
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
                for cid, cdata in data.items():
                    _TAXONOMY_CACHE[cid] = CLCourtTaxonomy(
                        court_id=cid,
                        system=cdata.get("system"),
                        jurisdiction=cdata.get("jurisdiction"),
                        type=cdata.get("type"),
                    )

    return _TAXONOMY_CACHE.get(court_id)


def is_recognized_court(court_id: str) -> bool:
    """Return True if the court ID is recognized in the CourtListener mapping."""
    return get_court_taxonomy(court_id) is not None
