"""Courts-DB classification snapshot for known court slugs.

`courts_db_classification.json` is a snapshot of the Free Law Project
[`courts-db`](https://github.com/freelawproject/courts-db) package. It extracts
the `system`, `type`, and `jurisdiction` fields for each court slug so the
jurisdiction inference layer can answer questions about a court slug without
making a network call.

The snapshot is a lookup table for court slugs that have already been resolved.
It is not a live fetch and is not authoritative about whether a citation's
locator will resolve. The only thing it asserts is that the slug exists in
`courts-db` and carries the recorded classification.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CourtsDBClassification:
    """Free Law Project `courts-db` classification for a specific court slug."""

    court_id: str
    system: str | None
    jurisdiction: str | None
    type: str | None


@lru_cache(maxsize=1)
def _court_classifications() -> dict[str, CourtsDBClassification]:
    json_path = Path(__file__).parent / "courts_db_classification.json"
    if not json_path.exists():
        return {}
    with json_path.open(encoding="utf-8") as file:
        data = json.load(file)
    return {
        court_id: CourtsDBClassification(
            court_id=court_id,
            system=classification.get("system"),
            jurisdiction=classification.get("jurisdiction"),
            type=classification.get("type"),
        )
        for court_id, classification in data.items()
    }


def get_courts_db_classification(court_id: str) -> CourtsDBClassification | None:
    """Return the `courts-db` classification for a court slug."""
    return _court_classifications().get(court_id)


def is_recognized_court(court_id: str) -> bool:
    """Return True if the court slug is present in the `courts-db` snapshot."""
    return get_courts_db_classification(court_id) is not None
