# Citation-node retrieval handoff — 2026-07-12

## Direction agreed

- `ExtractedDocument` remains the upstream boundary. Everything downstream is
  represented as a per-citation graph and the frontend reviews one final
  `citation_nodes.json` artifact.
- Do not preserve old snapshots, fixture modes, schema aliases, Modal, Label
  Studio, or synchronous retrieval compatibility paths.
- Retrieval is evidence collection only. A surfaced case-name candidate is not
  an identity, locator-correctness, appropriateness, or proposition-support
  conclusion.
- Exact locator lookup remains first. Only a not-found locator enters
  case-name/date preparation and candidate search.
- Keep LLM behavior bounded and honest. Do not overfit prompts or procedural
  rules to individual fixtures merely to improve a corpus score.

## Current retrieval chain

For a not-found full case citation:

1. `retrieval.exact_lookup`
2. Parallel evidence branches from the fallback decision:
   - `retrieval.case_name_reextraction_before_retrieval`
   - `retrieval.date_reextraction_before_retrieval`
3. `retrieval.preparation_validation`
4. `retrieval.search_query_proposal`
5. opinion and RECAP `retrieval.search_query_execution`
6. optional docket/document evidence branches
7. `retrieval.candidate_results`

Jurisdiction inference is an always-run side branch; assessment consumes the
retrieval result, not jurisdiction inference.

## Date policy

`CaseNameSearchPreparation` now carries independent date evidence:

- `decision_date`, `decision_year`, `decision_date_precision`
  (`complete_date`, `year_only`, or `no_date`);
- `date_reextraction_status` and `date_error_message`.

The policy is deliberately asymmetric:

- If Eyecite populated `ExtractedCitation.asserted_decision_date`, use it as
  `complete_date` with basis `eyecite_extracted`. Do **not** call or revalidate
  with the date LLM.
- Only when Eyecite did not extract a date does the independent date LLM run.
  It may return a full date, a copied year only, or no date. A year must never
  be fabricated into a full ISO date.
- A date-lane failure never prevents party recovery, query planning, or
  candidate search; conversely, party failure does not erase date evidence.

The final graph explicitly says when an Eyecite-extracted date was used.

## Relevant implementation files

- `src/mellea_lrc/retrieval/case_name_reextract_before_retrieval.py`
  - party IVR, independent date IVR, search-query IVR, grounding validators.
- `src/mellea_lrc/retrieval/pipeline.py`
  - async-only retrieval; mandatory preparation after not-found.
- `src/mellea_lrc/retrieval/types.py`
  - `DateReextractionStatus`, `DecisionDatePrecision`, preparation evidence.
- `src/mellea_lrc/citation_nodes/projections.py`
  - graph projection, independent date branch, transport outcome labels.
- `src/mellea_lrc/serialization/json.py` and `transport.py`
  - current-only serialization for new date fields.
- `scripts/e2e_backend/snapshot_corpus.py`
  - in-memory validation of intermediates; writes only final
    `citation_nodes.json` per run.

## Verification completed

- `ruff check src scripts tests` passed.
- Full unit suite previously passed: 206 passed, 10 opt-in skips.
- Focused retrieval, serialization, and citation-node suites passed after date
  precision work.
- `local/snapshots/2/citation_nodes.json` was regenerated successfully after
  the final date-policy correction. Its not-found citation (`2021 WL 3081160`)
  records `2021-07-21`, `complete_date`, `eyecite_extracted`, and does not run
  date re-extraction.

## Snapshot / fixture conventions

- Generated artifacts live under `local/snapshots/<fixture>/citation_nodes.json`
  and are ignored by Git.
- Named bookmark fixtures live in `fixtures/bookmarked/sets/` as one `.txt`,
  one `.json`, and one README per set. Current sets include research,
  date-extraction, and date-recovery.
- The bookmark hook is `scripts/bookmark_fixture.py`; it requires an explicit
  named set.
- Current runs use only final citation-node artifacts. No legacy stage
  snapshots should be recreated.

## Operational note

The CourtListener access environment variable is now `CL_ACCESS_URL`. The
local `.env` was migrated from the old Modal-named variable; do not add a code
fallback for the old name.

Long local model runs should use a persistent execution session and an
unbuffered log if diagnostics are needed. The log is operational only, not a
second review serialization.

## Sensible next step

Regenerate a small varied set (for example files 3–5 plus the named date
fixtures), then review the date-precision distribution. Accept isolated LLM
failures unless a repeated structural error appears. In particular, do not add
party-extraction exclusions solely because one output included nearby docket
text; query planning already normalized that example successfully.
