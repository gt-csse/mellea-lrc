# Reporter root exact lookup

`reporter_root_exact_lookup(document)` is the first validation stage and requires the `roots` checkpoint. It looks up each full reporter root once by normalized volume, reporter edition, and first page. Docket roots and repeated reporter occurrences are untouched. The citation stores one `ReporterExactLookup` object containing the query and complete CourtListener response. Provider failures raise; they are not recorded as lookup misses.

When the response has one cluster, the stage compares the citation's latest case-name, court, and date readings with that cluster. Case-name comparison requires both parties to appear separately in `caseNameFull`, allowing abbreviations from `reporters-db` without treating different full words as synonyms. Courts are compared by recognized court ID when the response has one. CourtListener exact-lookup clusters can omit court metadata: in that case, a court inferred only from the reporter is left unjudged, while an explicitly written court routes to review. A written full date requires the same full date; a written year requires the same year. If either side has no date, the stage makes no date judgment and does not penalize identity for it.

Each available comparison appends a field-specific judgment referencing the absolute index of its citation reading and the index of the cluster in the saved response. It does not copy either value. A unique cluster is marked `CORRECT_IDENTITY` when the case name and all applicable court and date checks match and its listed locator does not conflict. A mismatch or unreadable comparison routes to `REVIEW`; the rule stage does not declare a wrong identity. Multiple clusters route to `AMBIGUITY` without selecting one, and no cluster routes to `SEARCH`. The latest identity judgment carries this route for the next stage.

```python
from pathlib import Path

from mellea_lrc.api import Document, reporter_root_exact_lookup

checkpoint = Path("reporter-roots.json")
document = Document.model_validate_json(checkpoint.read_text())
document = reporter_root_exact_lookup(document)
checkpoint.write_text(document.model_dump_json())

before_lookup = document.get_stage("roots")
after_lookup = document.get_stage("reporter_root_exact_lookup")
```

The single lookup object, field judgments, and identity judgment are part of the citation's append-only history. A later checkpoint retains them, while `get_stage("roots")` removes them from the recovered earlier view.
