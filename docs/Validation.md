# Validation

`reporter_root_exact_lookup(document)` is the first validation stage and requires the `roots` checkpoint. It looks up each full reporter **root** by its normalized volume, reporter edition, and first page. Docket roots and repeated reporter occurrences are untouched.

The stage is rule-based. It preserves every CourtListener cluster in the reporter root's `reporter_exact_lookup` log and checks whether both source parties occur as separate whole-token phrases in the cluster's `caseNameFull`. It uses `reporters-db` case-name and state abbreviations, allowing written forms such as `Atl.` and `Atlantic` to agree without treating distinct full words as synonyms. For *In re* and *Ex parte*, it checks the subject and the same caption form. When the cluster lists reporter citations, the stage also checks that one equals the queried locator. Missing or failed full-name evidence leaves a candidate unqualified; it does not issue a wrong-identity judgment. Court, date, candidate ambiguity, and final identity remain for later stages.

```python
from mellea_lrc.api import Document, reporter_root_exact_lookup

document = reporter_root_exact_lookup(document)
saved = document.model_dump_json()
restored = Document.model_validate_json(saved)
assert restored.get_stage("roots") == document.get_stage("roots")
```

The client uses the configured `COURTLISTENER_BASE_URL` proxy. An upstream request or per-item error raises and does not complete the stage as a false lookup miss. Pass a client with `lookup_citation(volume, reporter, page)` for offline tests.
