# Reporter Jurisdiction Inference [in progress]

## Strategic role

Jurisdiction inference sits as a dedicated pipeline stage between extraction and
retrieval:

```
Preprocessing → Extraction → Jurisdiction Inference → Retrieval → Assessment
```

This placement is driven by three observations:

1. **Search steering.** Jurisdiction inference can guide retrieval — an inferred
   court (from reporter edition or parenthetical) can be added as a query filter
   to narrow candidate sets, reducing false positives from party-token overlap.
   Currently retrieval runs without any court context.

2. **Reporter-to-court translation.** A reporter edition like `F.3d` is known to
   publish opinions from a bounded set of federal appellate courts. Translating
   reporter strings into possible court slugs requires a curated mapping that
   does not exist in any public database (see Database landscape below). The
   inference stage is the natural home for this logic when it becomes available.

3. **Document-level fitness analysis.** Beyond per-citation inference, the
   document as a whole can be examined for jurisdiction consistency. If one
   citation resolves to SCOTUS and another to a state trial court in the same
   document, the overall jurisdiction signal is ambiguous. This can serve both
   as a search-direction signal and as a fast-fail guard before retrieval I/O.

All three are purely local operations — no network calls, no LLM, no rate limits.
The inference stage is deterministic and parallelizable.

## Database landscape (research findings)

Three Free Law Project databases exist, none providing a direct reporter-to-court-slug mapping:

### `reporters-db`

Reporter metadata used by eyecite for tokenization. Key fields for our purposes:

- **`cite_type`**: `"federal"`, `"state"`, `"neutral"`, `"specialty"`, etc. Eyecite passes this through verbatim from `reporters.json` into `Reporter.cite_type` at `tokenizers.py:185-189`.
- **`name`**: Human-readable reporter name (e.g. `"United States Supreme Court Reports"`). Used by eyecite's SCOTUS heuristic in `Reporter.__post_init__()`: if `cite_type == "federal"` and `"supreme" in name.lower()`, then `is_scotus = True`.
- **`mlz_jurisdiction`**: List of MLZ jurisdiction strings. These record **observed historical associations**, not exclusive publication scope. For `U.S.`, this includes 16 entries — `us;supreme.court` plus 15 historical courts (circuit courts, state supreme courts, even a mayor's court).

**Critical caveat**: MLZ data must not be treated as a reporter-to-court mapping. The design doc for the exact-court projection (`docs/knowledge/Reporter Court Inference.md`) explicitly warns against automatic generation from this field.

### `courts-db`

Court metadata used by eyecite for parenthetical court resolution via `get_court_by_paren()`. Key characteristics:

- **`citation_string`**: Unique per entry (strictly 1:1). Maps parenthetical abbreviations like `"2d Cir."` to court IDs.
- **`regex`**: Multiple patterns per court (many-to-one) for matching court names in case text.
- **`cites`**: A vestigial reporter-to-court hint. Only 3 entries populate it (`scotus → ["U.S."]`, `nh`, `pa`). Not a general mechanism — `courts-db` has no substantive reporter-to-court mapping.
- **`id`**: CourtListener court slug (e.g. `"ca2"`, `"scotus"`).

`courts-db` has no reporter field. It maps **parenthetical strings and court names** to court slugs, not reporters.

### The bridge (`mlz_to_cl_map.json`)

A manually curated 17-entry mapping from MLZ jurisdiction strings to CL court IDs. Covers only the federal appellate circuits, SCOTUS, and a few specialty courts. Maintained in `src/mellea_lrc/jurisdiction_inference/`.

### What's missing

No established database provides `reporter → set<court_slug>`. The MLZ field is the closest research input, but it requires:
1. Filtering observational noise from scope-based data
2. Translating from MLZ taxonomy to CL court slugs
3. Manual verification of publication scope per reporter

## Design boundary

`ReporterJurisdictionInference` represents the jurisdiction information supplied
by a reporter designation. Exact court inference is one special case of this
broader result; it is not the root abstraction.

A reporter can establish one exact court, a bounded set of courts, one or more
court classes, one or more jurisdictions, recognition without a useful
constraint, or no recognized mapping. The inference records evidence derived
from the reporter. It does not overwrite the citation's extracted court,
resolve a retrieved candidate, or choose a retrieval source.

## Proposed representation

The domain model is implemented in `mellea_lrc.reporter_jurisdiction`. Its shape
is:

```python
@dataclass(frozen=True, slots=True)
class ReporterJurisdictionInference:
    reporter: str | None
    status: ReporterJurisdictionStatus
    court_ids: tuple[str, ...]
    court_classes: tuple[CourtClass, ...]
    jurisdiction_ids: tuple[str, ...]
    coverage: ReporterCoverage
    evidence: tuple[ReporterJurisdictionEvidence, ...]

    @property
    def exact_court_id(self) -> str | None:
        if self.coverage is ReporterCoverage.EXHAUSTIVE and len(self.court_ids) == 1:
            return self.court_ids[0]
        return None
```

`status` distinguishes `missing_reporter`, `unrecognized`,
`recognized_without_constraint`, and `constrained`. `coverage` distinguishes
`exhaustive`, `partial`, and `unknown`.
Candidate tuples are immutable, deduplicated, and deterministically ordered.
Evidence retains the mapping source and its applicable reporter series or date
range; a bare confidence number is not a substitute for provenance.

The model permits simultaneous constraints. For example, a reporter may supply
a bounded court set and the federal-appellate court class. Downstream code
should intersect compatible evidence rather than discard the coarser constraint.

## Exact-court projection

The currently implemented `infer_court_from_reporter()` behavior is the
singleton projection:

```python
infer_court_from_reporter(reporter) == (
    infer_reporter_jurisdiction(reporter).exact_court_id
)
```

That projection is valid only when recorded court coverage is exhaustive and
contains exactly one court. A singleton observed in sample data is not enough.
Existing court assessment may continue to consume this projection, but the
broader inference belongs before assessment so retrieval can use non-singleton
constraints without expressing a verdict.

## Retrieval use

For a correctly used citation that exact locator lookup cannot recover,
retrieval may compare the inference with each candidate's explicit court:

- an exact court can establish compatibility or incompatibility;
- an exhaustive court set can exclude candidates outside the set;
- a court class can constrain candidates without identifying one court;
- a jurisdiction constraint can guide further source investigation;
- partial or unknown coverage must not be used for deterministic exclusion.

Reporter information may identify promising source ecosystems, but source
routing is a separate derived decision. It consumes this inference alongside
the parsed court, year, docket number, parties, and locator family.

## Invariants

- Explicit parsed court information remains separate and retains its provenance.
- Retrieved candidate courts remain separate from reporter-derived constraints.
- Exact-court inference is available only through the exhaustive singleton rule.
- Empty constraints are explicit, not an inferred court of `None` whose meaning
  is ambiguous.
- Candidate compatibility is a comparison result, not a mutation of either the
  inference or candidate.
- This layer expresses jurisdiction evidence, not which case should be
  recommended to the user.

## Research still required

Before expanding the initial mapping table, verify reporter series, historical
date boundaries, court reorganizations, and the semantics of jurisdiction data
from eyecite/reporters-db. Federal, regional, state, bankruptcy, and military
reporters must not be promoted from observed associations to exhaustive court
sets without publication-scope evidence.

### Known MLZ pitfalls

The `mlz_jurisdiction` field in `reporters-db` records **observed historical
association**, not exclusive publication scope. Common patterns that make it
unsuitable for direct use:

- **Historical inclusivity**: Early volumes of *United States Reports* (pre-1875)
  included decisions from lower federal courts and state courts. The MLZ for `U.S.`
  lists 16 courts, but modern publication scope is exclusively SCOTUS.
- **Nominative reporters**: Pre-1875 nominative reporters (Dall., Cranch, Wheat.,
  etc.) are already resolved to SCOTUS by eyecite for dated citations; their MLZ
  data may include additional historical courts.
- **Regional reporters**: MLZ for regional reporters like `P.3d` covers multiple
  state supreme courts, but the exact set per volume varies by time period.
- **Federal circuit reporters**: `F.3d`, `F.4th` have MLZ entries for all circuits
  plus historical courts, but the modern scope is a bounded set of 13 appellate
  courts — a set that cannot be reliably extracted from MLZ alone.

### Future research directions

1. Build a curated `reporter → list<court_slug>` map with provenance statements,
   starting with single-court reporters (exhaustive singleton) and expanding to
   bounded multi-court reporters like `F.3d` and `F.4th`.
2. Investigate whether `reporters-db`'s edition date ranges can be used to filter
   historical MLZ entries for scope-based inference.
3. Explore automated validation: if we map `F.3d` to `{ca1..cadc,cafc}`, we can
   cross-check against all extracted citations that have a `courts-db` parenthetical
   court — no parenthetical should resolve outside the bounded set.

Current implementation:

- `types.py` owns the immutable result, evidence, coverage, and court-class
  representations;
- `registry.py` owns `VALID_REPORTERS`, powered by `cl_reporters.json`;
- `leads.py` evaluates `ReporterLead` (reporter recognition + MLZ lookup) and
  `CourtLead` (CL taxonomy lookup from extracted court string);
- `translation.py` triangulates via `MLZ_TO_CL_MAP` as fallback (17-entry
  curated bridge);
  candidate identity; and
- the existing assessment helper returns only the exhaustive-singleton
  `exact_court_id` projection.
