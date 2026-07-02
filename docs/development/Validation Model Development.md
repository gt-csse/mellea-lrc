# Validation Model Development

Validation operates on the **1st layer**: canonical citations labeled `Real`/`False`, verified independently of the source document. Three layers of increasing scope; current focus is Layers 1–2.

| Layer | Input | Question |
|---|---|---|
| 1 — Existence | volume + reporter + page | Does this case exist? |
| 2 — Bibliographic | full canonical representation | Are the metadata fields correct? |
| 3 — Contextual | canonical + document context | Is the citation used correctly? |

## Layer 1 — Existence

The locator (`volume`, `reporter`, `page`, e.g. `531 U.S. 98`) is enough to look a case up in CourtListener; other fields are for Layer 2. The pipeline is deterministic (no LLM):

```mermaid
flowchart TD
    E[Extracted citation] --> F{FullCaseCitation?}
    F -->|no| SKIP[skipped]
    F -->|yes| L{volume + reporter + page?}
    L -->|no| INV[invalid]
    L -->|yes| Q[CourtListener lookup by locator]
    Q --> S{status}
    S -->|200| FOUND[found]
    S -->|300| AMB[ambiguous]
    S -->|404| NF[not_found]
    S -->|400| BAD[invalid]
    S -->|429| THROTTLED[throttled]
    S -->|other| FAIL[lookup_failed]
```

Each outcome is a `CitationValidation` variant preserving lookup status, cache/key, error/`ValidationFailureDetail`, and typed `CitationMatch` records; unmodeled upstream fields stay in `extra_data`. Statuses: `found`, `ambiguous`, `not_found`, `invalid`, `throttled`, `lookup_failed`, `skipped`.

**Principle: validation retrieves, it never compares.** It resolves data (the CourtListener court, case-name search candidates) and attaches it; all comparison/opinion is assessment's job.

- CourtListener coverage and the RECAP pipeline: [Data Source](../knowledge/Data%20Source.md).
- **Not-found handling** (case-name search): [Not Found Candidate Search](./Not%20Found%20Candidate%20Search.md). A coverage gap is not the same as a hallucination, so a not-found locator triggers a case-name search that reports how many CourtListener cases share the name.
- **Ambiguous handling** (multiple clusters, HTTP 300): [Ambiguous Resolution](./Ambiguous%20Resolution.md). Validation returns all candidates; assessment runs the found-branch assessment on each (gated above 5).

## Layer 2 — Bibliographic cross-reference

Uses the case retrieved in Layer 1 to check the remaining canonical fields — party names, year, court, pin cite in range — nothing that requires reading document content (that is Layer 3).

### Court field assessment

Validation only *retrieves* the CourtListener court. The comparison lives in assessment: it compares the raw eyecite `court` slug against the retrieved court, and when the initial comparison is `missing` and the reporter unambiguously identifies SCOTUS, it applies **reporter inference** as a field-local follow-up (mirroring the case-name `initial` + `followup` trace). Implementation: `assessment/fields/court/{assess,inference}.py`; trace shape `CourtAssessmentRun` with `CourtInferredFromReporter`.

| Reporter | eyecite `court` | inference? |
|---|---|---|
| `U.S.`, `S. Ct.` | `scotus` | No — initial `exact_match` |
| `L. Ed.`, `L. Ed. 2d` | *(absent)* | Yes — follow-up `inferred_from_reporter` |
| `F.3d` | varies | No — ambiguous reporter |

## Layer 3 — Contextual verification (lower priority)

Checks requiring input beyond the canonical representation: does a quoted passage appear in the opinion; is a cited subsection real; is a non-precedential case cited as binding; is the proposition actually supported. The harder, subjective cases may ship as structured retrieved context for downstream use rather than fully automated verdicts.
