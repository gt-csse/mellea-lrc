---
tags: [case-law, publication, access, locator, westlaw, pacer, recap, identity]
status: active
created: 2026-07-08
---

# Decision Publication, Access, and Locators

This note defines the distinctions used when retrieving and identifying a cited
judicial decision. Publication, precedential force, document availability, and
locator verification are independent properties. “Not found,” “unpublished,”
and “locked” must never be used as synonyms.

## Independent dimensions

| Dimension | Example values | Question answered |
| --- | --- | --- |
| Court designation | `published`, `unpublished`, `nonprecedential`, `unknown` | How did the issuing court classify the disposition? |
| Precedential force | `binding`, `persuasive`, `restricted`, `unknown` | What legal weight may it carry in the relevant court? |
| Locator type | `reporter`, `neutral`, `commercial_database`, `docket_document` | How does a reader address the decision? |
| Access | `open`, `authenticated`, `paywalled`, `metadata_only`, `unavailable` | Can we obtain the text, and under what conditions? |
| Repository | court website, PACER, RECAP, CourtListener case law, Westlaw, Lexis | Which system holds this representation? |

“Unpublished” is a legal term of art and is not reliably literal. It can mean
that a court designated a decision nonprecedential even though the full text is
publicly available and indexed by commercial services. Conversely, a reported
decision may be published but its convenient full-text copy may be paywalled.

For federal appellate dispositions issued on or after January 1, 2007, Federal
Rule of Appellate Procedure 32.1 prohibits courts from restricting citation
because a disposition is labeled unpublished or nonprecedential. The rule does
not make the disposition binding precedent. Older federal dispositions and
state decisions remain subject to applicable court rules.

References:

- [FRAP 32.1 and an example local rule](https://www.ca4.uscourts.gov/rules/Rule32-1.html)
- [Cornell, citing unpublished cases](https://www.law.cornell.edu/citation/2-200)

## How a court order becomes a WL citation

A Westlaw locator is assigned by Westlaw, not by the court or PACER:

```text
court files an order
        |
        +--> court docket / PACER document
                    |
                    +--> Westlaw ingests the document
                                |
                                +--> Westlaw assigns YYYY WL NNNNNNN
                                     and star pagination (*1, *2, ...)
```

The court supplies the caption, court, docket number, document number, date,
text, and original pagination. Westlaw supplies a proprietary database
identifier, searchable record, star pagination, and possible editorial/citator
enhancements.

Therefore `2007 WL 1430100` is not a reporter position and cannot be calculated
from a PACER PDF. It is a commercial database address. The same decision can
also be cited without Westlaw using its docket address, for example:

```text
Boeser v. Sharp, No. 03-cv-00031-WDM-MEH,
ECF No. 331, at 7 (D. Colo. May 14, 2007).
```

These locators can address the same document but are not automatically
interchangeable. Connecting them requires citation-concordance evidence.

## CourtListener and Westlaw cover different units

CourtListener has two relevant collections:

- **Case law:** opinion clusters collected from court sites, libraries,
  publishing partnerships, CAP/vLex data, and other sources.
- **RECAP:** federal PACER dockets, entries, and documents contributed to the
  archive.

Westlaw contains reported decisions, many unreported decisions, dockets, and
editorial metadata. Neither collection is a clean superset of the other:

- a PACER document can be in RECAP without becoming a CourtListener opinion
  cluster;
- a decision can be in Westlaw with a WL locator but absent from CourtListener
  case law;
- an opinion can be in CourtListener while its parallel or WL citation metadata
  is incomplete;
- RECAP can contain litigation documents that Westlaw does not select as
  citable case law;
- a document can remain available only from PACER or a state court.

CourtListener describes its case-law collection as covering more than 99% of
published precedential American case law, while explicitly noting that its
records do not contain every citation or page number. A CourtListener miss does
not establish that a decision or asserted WL locator is false.

References:

- [CourtListener overall coverage](https://www.courtlistener.com/help/coverage/)
- [CourtListener case-law coverage and citation limitation](https://www.courtlistener.com/help/coverage/opinions/)
- [CourtListener search fields, including `citation`](https://www.courtlistener.com/help/search-operators/)
- [Westlaw docket coverage](https://legal.thomsonreuters.com/en/products/westlaw/dockets-coverage)

## Authority identity: evidence levels

Authority identity asks which judicial decision an asserted locator denotes. It
does not ask whether that decision supports the proposition for which it was
cited.

### Citation-description match

When a case-name-and-court fallback surfaces a candidate with compatible date
metadata, the narrow conclusion is:

> This candidate matches the citation description, but the available evidence
> does not yet establish that the asserted locator denotes this candidate.

Case name, court, and year do not by themselves identify a docket or decision.
The citation year is normally the decision year, while a proceeding may have
begun years earlier and may contain many decisions.

### Docket/proceeding match

A proceeding is a particular court case identified at docket level, normally by
`court + docket number`. A docket number discovered from a search result is a
bridge to more evidence; it is source evidence only when the citation context
itself supplies that number.

### Candidate-document match

A specific docket entry can be matched using document number, filing date,
document type, caption, and text. This establishes a candidate decisional
document in the proceeding, but it does not necessarily establish the incoming
reporter, WL, or Lexis locator.

### Locator mapping

The strongest identity evidence directly maps the asserted locator to the
candidate document. Examples include:

- an opinion cluster whose parallel `citation` metadata contains the locator;
- a licensed database resolving its own locator to the document;
- court or authoritative repository metadata carrying both identities;
- an independent citation concordance that identifies the same court, docket,
  date, and decisional document.

Search-result uniqueness is not locator mapping.

## WL-specific best-effort policy

WL citations deserve a distinct retrieval route because they are proprietary
database locators rather than reporter coordinates.

1. Preserve the extracted WL locator exactly as asserted source evidence.
2. Try CourtListener's citation lookup/index.
3. Search opinion clusters by citation and then by case description; inspect
   the cluster's complete parallel-citation metadata.
4. Search RECAP by case name and court, then inspect candidate dockets and
   decisional documents.
5. Search open citation concordances and court sources using the WL locator,
   docket, date, and case name.
6. Use a licensed Westlaw backend when one is configured and its license permits
   the operation.
7. Keep candidate/document confidence separate from locator verification.

Recommended conclusions include:

```text
description_match: none | possible | strong
proceeding_match: unverified | verified
document_match: unverified | plausible | verified
locator_status: asserted | probable | verified | conflict
```

Do not label a WL locator incorrect merely because CourtListener lacks it. Use
`asserted` or `probable` until a direct concordance verifies it; use `conflict`
only when affirmative evidence is incompatible with the asserted locator.

### State-trial + WL routing policy

A citation combining a state trial court with a WL locator should take a
deliberately shallow open-retrieval route unless a licensed WL-capable backend
is configured.

This combination is a strong coverage signal:

- RECAP/PACER cannot supply the state docket route because RECAP is federal;
- CourtListener state trial opinions and docket documents are uneven by state
  and court;
- the WL identifier may be the only broadly used citation concordance attached
  to an otherwise unreported trial order;
- repeated model-driven search cannot manufacture Westlaw's proprietary
  locator-to-document mapping.

The default no-Westlaw route should be bounded to inexpensive deterministic or
shallow probes:

1. Try exact CourtListener citation lookup/search for the asserted WL locator.
2. Try one case-name + court + date query against CourtListener case law.
3. Try the issuing court's official public search when a structured connector
   is already available.
4. Optionally run one exact web query containing the quoted WL locator and case
   name.
5. Stop rather than expanding into iterative or token-consuming search.

If those probes do not produce a direct locator concordance, preserve any
description-level candidates and return an explicit limitation:

```text
locator_status: asserted
validation_status: requires_commercial_backend
recommended_backend: westlaw
message: State-trial WL citation received only shallow open-source validation;
         Westlaw or another licensed citation index is recommended for full
         locator validation.
```

This is not a negative conclusion about the citation. It is a route-level
statement that available open sources are poorly aligned with the locator and
court level. A configured licensed backend may replace the stop with direct
resolution, subject to its contract and provenance requirements.

## Current research example: Boeser

For `Boeser v. Sharp, 2007 WL 1430100 (D. Colo. May 14, 2007)`:

- CourtListener exact locator lookup missed.
- Case-name-and-court search returned RECAP docket `1:03-cv-00031`.
- The docket contains ECF No. 331, an 18-page order filed May 14, 2007.
- The docket has no linked CourtListener opinion cluster carrying parallel
  citation metadata.
- An [Iowa Law Review article](https://ilr.law.uiowa.edu/sites/ilr.law.uiowa.edu/files/2024-11/ILR-110-Nourse.pdf)
  cites the same case, docket, date, and WL locator.

The safe conclusion is a strong description/document concordance with a
probable WL mapping, not a locator mapping proven by CourtListener metadata.
Direct Westlaw resolution would raise `locator_status` to `verified`.

## Significance in the current corpus

Across the 26 files in `local/test_data`:

- 64 of 525 extracted full-case citations are WL citations (`12.19%`);
- 14 of 26 documents contain at least one WL citation (`53.85%`);
- the extracted set contains 44 distinct WL locators;
- a raw-text check finds 69 WL occurrences, of which extraction currently
  captures 64 (`92.75%`).

WL handling is therefore a normal retrieval path, not an edge-case fallback.

## Licensing boundary

Westlaw is a commercial service. Institutional and individual access is common
but not universal, coverage depends on the subscription, and automated access
requires an appropriate contract/API. A human user's web subscription must not
be assumed to authorize systematic retrieval or construction of a derivative
database.

Open retrieval must remain useful without Westlaw. A licensed backend, if added,
should be optional and should record its source, access mode, and licensing
constraints without leaking proprietary full text into unrestricted artifacts.
