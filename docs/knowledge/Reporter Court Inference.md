# Reporter-to-Court Inference

This page documents the currently implemented exhaustive-single-court mapping.
It is the exact-court projection of the broader
[`ReporterJurisdictionInference`](../development/retrieval/reporter-jurisdiction-inference%20%5Bin%20progress%5D.md)
design. Multi-court and court-class constraints remain useful inference results
even though the current helper cannot represent them.

Reporter-to-court inference is a narrow extraction fallback for
`FullCaseCitation.court`. Eyecite can recognize a reporter and still leave the
court empty. When the reporter publishes decisions from exactly one court, the
reporter itself supplies the missing court without an LLM.

This is extraction-level patching expressed during court assessment. It does
not depend on CourtListener supplying a court, and it does not turn missing
CourtListener data into a match or mismatch. If the comparison still lacks a
CourtListener court, the assessment remains `missing` and expresses no opinion.

## How eyecite resolves courts

Eyecite uses two completely separate mechanisms to determine `court`:

### Mechanism 1: SCOTUS heuristic (reporter-based)

In `Reporter.__post_init__()` (`eyecite/models.py`):
```python
if (self.cite_type == "federal" and "supreme" in self.name.lower()) or "scotus" in self.cite_type.lower():
    object.__setattr__(self, "is_scotus", True)
```
Both `cite_type` and `name` are pass-through values from `reporters-db` (`reporters.json`).
Eyecite reads `source["cite_type"]` at `tokenizers.py:185-189` and constructs
`Reporter(short_name, name, cite_type, source)` directly from the data.

Then `CaseCitation.guess_court()` sets `metadata.court = "scotus"` if any edition's
reporter has `is_scotus = True`. This is the **only** court-setting logic in eyecite
that operates on reporter metadata alone.

The possible `cite_type` values in `reporters-db`: `federal`, `neutral`,
`scotus_early`, `specialty`, `specialty_lexis`, `specialty_west`, `state`,
`state_regional`. The heuristic fires when `cite_type` is `federal` AND the
reporter name contains "supreme".

### Mechanism 2: Parenthetical lookup (courts-db based)

For all other courts, eyecite's `get_court_by_paren()` (`eyecite/helpers.py`) uses
the separate `courts-db` package to map parenthetical strings to CourtListener
court slugs:

```python
from courts_db import courts

def get_court_by_paren(paren_string: str) -> str | None:
    court_str = re.sub(r"[^\w]", "", paren_string).lower()
    for court in courts:
        s = re.sub(r"[^\w]", "", court["citation_string"]).lower()
        if s == court_str:
            return court["id"]
```

When a citation like `100 F.3d 100 (2d Cir. 1996)` is parsed,
`add_post_citation()` extracts the parenthetical, and `get_court_by_paren("2d Cir.")`
returns `"ca2"`. This is a pure dictionary lookup with no reporter involvement.

`courts-db` entries are strictly 1:1: each entry has exactly one `citation_string`
and one `id`. No two entries share a `citation_string`. Its `cites` field
(reporter-to-court hint) exists for only 3 entries (`scotus`, `nh`, `pa`) and
is not a general mechanism.

### The gap

No established database directly connects a reporter to a set of CourtListener
court slugs:
- **`reporters-db`** maps reporters to MLZ jurisdiction strings (a different taxonomy)
- **`courts-db`** maps parenthetical strings and name regexes to CL slugs, but has
  no reporter field (except the vestigial `cites` on 3 entries)
- **The MLZ-to-CL bridge** (`mlz_to_cl_map.json`) manually maps only 17 MLZ strings

The project therefore does not infer an exact CourtListener court from a
reporter alone. It retains reporter recognition and MLZ evidence without
turning incomplete publication-scope data into a false exact mapping.

## Admission rule

Add a mapping only when all of the following are true:

1. Eyecite emits the canonical reporter value but can leave `court` empty.
2. The reporter covers one court exclusively. A reporter associated with a
   jurisdiction, court level, or family of courts is not sufficient.
3. The destination is a verified CourtListener court slug.
4. The mapping is based on the reporter's publication scope, not on the court
   that happened to decide a sample citation.

Reporter aliases do not need separate entries when eyecite normalizes them to
the same canonical reporter value.

## Implemented mappings

The registry and formal inference live in
`src/mellea_lrc/reporter_jurisdiction/`. The assessment helper in
`src/mellea_lrc/assessment/fields/court/inference.py` is a compatibility
projection that returns only `exact_court_id`.

| Canonical reporter | CourtListener court | Eyecite behavior | Why inference is valid |
|---|---|---|---|---|
| `U.S.` | `scotus` | Eyecite's SCOTUS heuristic fires (`cite_type=federal`, `name` contains "supreme") → `guess_court()` sets `court="scotus"`. This mapping is defensive rather than an augmentation. | *United States Reports* is the official reporter of the Supreme Court of the United States and covers that court exclusively. |
| `S. Ct.` | `scotus` | Same heuristic: `cite_type=federal`, `name="Supreme Court Reporter"` contains "supreme" → `guess_court()` sets `court="scotus"`. This mapping is defensive. | *Supreme Court Reporter* reports decisions of the Supreme Court of the United States exclusively. |
| `L. Ed.` | `scotus` | Heuristic does **not** fire: `reporters-db` sets `name="Lawyer's Edition"` (no "supreme" in the name), so `is_scotus` stays `False`. Eyecite leaves `court` empty. | The first series of *United States Supreme Court Reports, Lawyers' Edition* reports decisions of the Supreme Court of the United States exclusively. |
| `L. Ed. 2d` | `scotus` | Same name as `L. Ed.` (`"Lawyer's Edition"`), same non-firing heuristic. Eyecite leaves `court` empty. | The second series of *United States Supreme Court Reports, Lawyers' Edition* reports decisions of the Supreme Court of the United States exclusively. |

## Additional implemented mappings

These reporters satisfy the exclusivity rule and are parsed locally by
eyecite 2.7.6 with an empty court. Their CourtListener slugs are covered by
reporter-inference regression tests.

| Canonical reporter | CourtListener court | Eyecite gap | Reason to add the inference |
|---|---|---|---|
| `U.S. LEXIS` | `scotus` | Eyecite recognizes `U.S. LEXIS` but leaves `court` empty. | This Lexis reporter designation covers Supreme Court of the United States decisions exclusively. |
| `T.C.` | `tax` | Eyecite recognizes `T.C.` but leaves `court` empty. | *United States Tax Court Reports* is the official bound reporter of the United States Tax Court and covers that court exclusively. |
| `B.T.A.` | `bta` | Eyecite recognizes `B.T.A.` but leaves `court` empty. | *Reports of the United States Board of Tax Appeals* covers the historical Board of Tax Appeals exclusively; CourtListener models that tribunal separately from the later Tax Court. |
| `Fed. Cl.` | `uscfc` | Eyecite recognizes `Fed. Cl.` but leaves `court` empty. | *Federal Claims Reporter* publishes decisions of the United States Court of Federal Claims exclusively. |
| `Cl. Ct.` | `uscfc` | Eyecite recognizes `Cl. Ct.` but leaves `court` empty. | *United States Claims Court Reporter* covers the same court under its 1982–1992 name exclusively; CourtListener uses `uscfc` for the continuing court. |
| `Ct. Int'l Trade` | `cit` | Eyecite recognizes `Ct. Int'l Trade` but leaves `court` empty. | *Court of International Trade Reports* publishes decisions of the United States Court of International Trade exclusively. |
| `Cust. Ct.` | `cusc` | Eyecite recognizes `Cust. Ct.` but leaves `court` empty. | *Customs Court Reports* publishes decisions of the historical United States Customs Court exclusively. |
| `C.C.P.A.` | `ccpa` | Eyecite recognizes `C.C.P.A.` but leaves `court` empty. | *Court of Customs and Patent Appeals Reports* publishes decisions of the historical Court of Customs and Patent Appeals exclusively. |
| `Vet. App.` | `cavc` | Eyecite recognizes `Vet. App.` but leaves `court` empty. | *Veterans Appeals Reporter* publishes decisions of the United States Court of Appeals for Veterans Claims exclusively. |
| `M.S.P.R.` | `mspb` | Eyecite recognizes `M.S.P.R.` but leaves `court` empty. | *Merit Systems Protection Board Reporter* publishes decisions of the Merit Systems Protection Board exclusively. |
| `C.M.A.` | `cma` | Eyecite recognizes `C.M.A.` but leaves `court` empty. | *Decisions of the United States Court of Military Appeals* covers that historical court exclusively. |

`U.S.L.W.` is not yet eligible. *United States Law Week* contains Supreme Court
material, but the publication is broader than a reporter devoted exclusively
to one court. It needs citation-format-specific evidence before a blanket
reporter mapping is safe.

## Reporters that must remain uninferred

- `F.`, `F.2d`, `F.3d`, and `F.4th` cover multiple federal appellate courts
  and, historically, additional federal courts.
- `F. Supp.`, `F. Supp. 2d`, and `F. Supp. 3d` cover multiple federal district
  courts.
- `B.R.` covers multiple bankruptcy courts and appellate panels.
- Regional reporters and state supplements cover multiple courts.
- `M.J.` covers multiple military courts.
- Bare nominative abbreviations such as `Cranch`, `Dall.`, `Wall.`, `How.`, and
  `Pet.` have historical reuse or ambiguity. Eyecite already resolves ordinary
  dated Supreme Court citations from these reporters.

Do not generate mappings automatically from `reporters-db.mlz_jurisdiction`.
That field is useful research input, but some entries describe observed or
broad jurisdiction associations rather than an exclusive publishing court.

MLZ records historical observation: for `U.S.`, the MLZ list includes 16 entries
(`us;supreme.court`, plus historical circuit courts, state supreme courts, and even
a mayor's court) because early *United States Reports* (pre-1875) compiled decisions
from many courts. Eyecite's heuristic, by contrast, is based on the reporter's
modern name (`"United States Supreme Court Reports"`) — a present-day editorial
fact, not an observational record.

## The database landscape

Three Free Law Project databases exist, each serving a different mapping:

| Database | Maps | Direction | Cardinality | Used by |
|---|---|---|---|---|
| `reporters-db` | reporter → MLZ jurisdiction | many-to-many | `U.S.` → 16 MLZ strings | eyecite tokenization, our validation |
| `courts-db` | parenthetical string → CL court slug | one-to-one | `"2d Cir."` → `"ca2"` | eyecite `get_court_by_paren()` |
| `courts-db` (regex) | court name pattern → CL court slug | many-to-one | 40 patterns → `"ca2"` | case name matching |
| `reporters-db` (via our bridge) | MLZ string → CL court slug | curated | 17 entries in `mlz_to_cl_map.json` | `triangulate_court_id()` fallback |

No database directly connects a reporter string to a CourtListener court slug.
The `courts-db.cites` field comes closest but is populated for only 3 entries
(`scotus`, `nh`, `pa`). Building this mapping is a gap this project fills.

## Sources

- [GovInfo: United States Reports](https://www.govinfo.gov/help/usreports)
- [United States Tax Court: citation guidance](https://www.ustaxcourt.gov/petitioners-after/)
- [CourtListener: available jurisdictions and canonical slugs](https://www.courtlistener.com/help/api/jurisdictions/)
- [`reporters-db`](https://github.com/freelawproject/reporters-db), the reporter metadata used by eyecite
- [`courts-db`](https://github.com/freelawproject/courts-db), the court metadata used by eyecite `get_court_by_paren()`
