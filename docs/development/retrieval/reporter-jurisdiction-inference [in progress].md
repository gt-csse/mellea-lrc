# Reporter Jurisdiction Inference [in progress]

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

Current implementation:

- `types.py` owns the immutable result, evidence, coverage, court-class, and
  compatibility representations;
- `registry.py` owns curated scopes;
- `inference.py` performs the pure reporter-to-inference operation;
- `compatibility.py` compares explicit candidate metadata without resolving
  candidate identity; and
- the existing assessment helper returns only the exhaustive-singleton
  `exact_court_id` projection.
