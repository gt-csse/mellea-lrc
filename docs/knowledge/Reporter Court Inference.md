# Reporter-to-Court Inference

Reporter-to-court inference is a narrow extraction fallback for
`FullCaseCitation.court`. Eyecite can recognize a reporter and still leave the
court empty. When the reporter publishes decisions from exactly one court, the
reporter itself supplies the missing court without an LLM.

This is extraction-level patching expressed during court assessment. It does
not depend on CourtListener supplying a court, and it does not turn missing
CourtListener data into a match or mismatch. If the comparison still lacks a
CourtListener court, the assessment remains `missing` and expresses no opinion.

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

The implementation is
`src/mellea_lrc/assessment/fields/court/inference.py`.

| Canonical reporter | CourtListener court | Eyecite behavior | Why inference is valid |
|---|---|---|---|
| `U.S.` | `scotus` | Eyecite normally emits `scotus`; this mapping is defensive rather than an augmentation. | *United States Reports* is the official reporter of the Supreme Court of the United States and covers that court exclusively. |
| `S. Ct.` | `scotus` | Eyecite normally emits `scotus`; this mapping is defensive rather than an augmentation. | *Supreme Court Reporter* reports decisions of the Supreme Court of the United States exclusively. |
| `L. Ed.` | `scotus` | Eyecite recognizes the reporter but does not infer a court by default. | The first series of *United States Supreme Court Reports, Lawyers' Edition* reports decisions of the Supreme Court of the United States exclusively. |
| `L. Ed. 2d` | `scotus` | Eyecite recognizes the reporter but does not infer a court by default. | The second series of *United States Supreme Court Reports, Lawyers' Edition* reports decisions of the Supreme Court of the United States exclusively. |

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

## Sources

- [GovInfo: United States Reports](https://www.govinfo.gov/help/usreports)
- [United States Tax Court: citation guidance](https://www.ustaxcourt.gov/petitioners-after/)
- [CourtListener: available jurisdictions and canonical slugs](https://www.courtlistener.com/help/api/jurisdictions/)
- [`reporters-db`](https://github.com/freelawproject/reporters-db), the reporter metadata used by eyecite
