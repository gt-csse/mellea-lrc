---
tags: [courtlistener, courts, classification, coverage, search]
status: research
created: 2026-07-02
---

# Court Level Classification

## Terminology and classification target

The useful term is **court level** or **court tier**, but neither term alone is
precise enough. We must classify the court that issued the cited decision or
owns the retrieved docket—not an entire real-world dispute. One dispute can
produce separate trial and appellate dockets, and an appellate court can hear
an original proceeding without becoming a trial court.

The originally considered labels are useful human interpretations:

- `federal_appellate`
- `federal_trial`
- `state_appellate`
- `state_trial`

They are neither the stored knowledge model nor comprehensive across American
adjudicative bodies. CourtListener already models federal bankruptcy courts
and panels, federal and state special courts, military, tribal, territorial,
international, attorney-general, and committee records. We should retain
CourtListener's finer category instead of forcing every record into the four
labels.

## Recommended model

Use CourtListener's `Court` record as the knowledge model whenever a court slug
or name resolves. The API gives us the canonical slug, full and short names,
and jurisdiction code. Preserve those upstream values instead of building a
parallel court ontology prematurely.

```text
court slug/name
    -> CourtListener /courts lookup
    -> canonical court identity + jurisdiction category
    -> retrieval prior for post-not-found search
```

The jurisdiction category is a routing prior, not a claim that citation lookup
or either CourtListener corpus supports the court:

```text
court recognized
!= citation lookup supported
!= case-law available
!= docket available
!= docket entries available
!= document available
```

Broad labels such as `federal_appellate`, `federal_trial`, `state_appellate`,
and `state_trial` may be derived for display or simple routing, but the raw
CourtListener identity and jurisdiction category remain the knowledge.
Special, bankruptcy, military, tribal, territorial, and malformed values must
remain visible rather than being forced into four labels.

If neither exact slug nor exact/unique-prefix name lookup resolves a Court
record, delegate court identification to general court reasoning. That
fallback must preserve uncertainty and must not manufacture a CourtListener
slug.

### Important edge cases

- SCOTUS is institutionally federal appellate even though it has limited
  original jurisdiction.
- State supreme courts belong in `state_appellate`; `supreme` is rank, not a
  third trial/appellate role.
- New York's Supreme Court is generally a trial court; its Appellate Division
  is appellate. The word “Supreme” is not a reliable level signal.
- A state “Circuit Court” is often a trial court, while a current federal
  “Circuit” reference usually means a court of appeals. Historical United
  States circuit courts were primarily trial courts.
- Bankruptcy appellate panels are appellate, while bankruptcy courts are
  first-instance; the shared word “bankruptcy” does not determine role.
- Tax, claims, veterans, military, and administrative bodies should retain
  explicit specialization rather than being silently coerced into a general
  trial/appellate bucket.

## CourtListener's classification

CourtListener's `Court` record is the primary structured source. Its public
[Available Jurisdictions](https://www.courtlistener.com/help/api/jurisdictions/)
page exposes both the canonical court slug and a jurisdiction category.

On 2026-07-02, the page contained 3,359 court records distributed across these
categories:

| CourtListener category | Count | Proposed interpretation |
|---|---:|---|
| State Trial | 2,618 | state/trial, subject to data QA |
| Federal Appellate | 127 | federal/appellate |
| Federal District | 125 | federal/trial/general |
| State Appellate | 111 | state/appellate/intermediate or mixed rank |
| Federal Bankruptcy | 95 | federal/trial/bankruptcy |
| State Special | 86 | state/special; do not force |
| State Supreme | 55 | state/appellate/last resort |
| Federal Special | 42 | inspect per court |
| Tribal Appellate | 18 | tribal/appellate |
| State Attorney General | 16 | non-court |
| Tribal Trial | 14 | tribal/trial |
| Military Appellate | 11 | military/appellate |
| Federal Bankruptcy Panel | 8 | federal/appellate/bankruptcy |
| Territory Trial | 6 | territorial/trial |
| Tribal Supreme | 6 | tribal/appellate/last resort |
| Committee | 5 | non-court or special |
| Territory Supreme | 5 | territorial/appellate/last resort |
| Tribal Special | 4 | tribal/special |
| International | 3 | international; inspect per body |
| Territory Appellate | 2 | territorial/appellate |

There were also two malformed/blank category values. More importantly, the
page contains classifications that conflict with ordinary institutional
understanding—for example Massachusetts Superior Court appears as State
Appellate. CourtListener explicitly warns that some jurisdiction metadata is
incomplete. We should consume its category as strong evidence, not infallible
ground truth, and maintain tested overrides for demonstrated errors.

CourtListener dockets also expose a PACER `jurisdiction_type` field, but that
field describes subject-matter jurisdiction metadata on a docket; it is not the
court-level classification and must not be used for this feature.

## Human-readable recognition patterns

Patterns should produce evidence with a confidence level, not silently become
truth. Prefer explicit court identity over reporter-family inference.

### Strong signals

| Signal | Likely classification | Notes |
|---|---|---|
| “Supreme Court of the United States” | federal appellate, last resort | Institutional classification despite original-jurisdiction exceptions. |
| “United States Court of Appeals for the … Circuit” | federal appellate | Includes numbered, D.C., and Federal Circuits. |
| Parenthetical such as `(9th Cir. 2024)` | federal appellate | Strong when attached to a full citation. |
| “United States District Court for the … District of …” | federal trial | General federal trial court. |
| Parenthetical such as `(S.D.N.Y. 2024)` | federal trial | District abbreviation is stronger than the reporter alone. |
| State name + “Supreme Court” or known court of last resort | state appellate, last resort | Maryland and New York naming history require a court dictionary. |
| State name + “Court of Appeals”/“Appellate Court” | usually state appellate | “Court of Appeals” can be the state's highest court. Rank requires state-specific knowledge. |
| County/circuit/superior/district/court of common pleas with state context | usually state trial | Never apply without jurisdiction context. |
| Explicit CourtListener court slug | use curated court record | Strongest machine-readable signal, subject to known overrides. |

### Reporter patterns

- `U.S.`, `S. Ct.`, and `L. Ed.` identify SCOTUS and therefore federal
  appellate.
- `F.2d`, `F.3d`, and `F.4th` indicate federal appellate publication, but the
  parenthetical or retrieved court is still required to identify the circuit.
- `F. Supp.`, `F. Supp. 2d`, and `F. Supp. 3d` indicate federal district-court
  publication and therefore normally federal trial.
- Official state and regional reporters are not sufficient by themselves to
  distinguish a state supreme court from an intermediate appellate court, and
  some historically include other courts.
- State trial decisions appear in scattered official, specialty, county, and
  miscellaneous reporters. There is no safe national reporter-only pattern.
- `B.R.` and `M.J.` span multiple levels and must not determine level alone.

Reporter-family inference is broader and weaker than the exclusive
reporter-to-court inference documented in
[Reporter-to-Court Inference](Reporter%20Court%20Inference.md).

### Docket and document patterns

Federal document headers and PACER identifiers are useful corroboration:

- `…-cv-…` and `…-cr-…` usually identify federal district civil and criminal
  dockets.
- `…-bk-…` identifies bankruptcy, not a general federal trial case.
- Federal appellate numbers commonly resemble `YY-NNNN`, but this shape is too
  generic to use without a known appellate court.
- A header naming the court is stronger than the docket-number shape.

State docket-number formats are jurisdiction-specific and change over time.
They require a court-aware registry; national regexes will overfit and create
false classifications.

### Evidence precedence

1. CourtListener court record resolved directly from a slug.
2. CourtListener court record resolved uniquely from an exact normalized name.
3. Explicit court extracted from a parenthetical or document header, then
   resolved through the Court API.
4. Exclusive reporter-to-court mapping, followed by Court API resolution.
5. Reporter-family evidence plus a compatible parenthetical.
6. Docket-number or case-caption heuristics as corroboration only.
7. Otherwise delegate to general court reasoning or return `unknown`.

Every inference should preserve `source`, `confidence`, and the raw evidence so
later corrections do not erase provenance.

## CourtListener availability by level

CourtListener has two principal, overlapping data legs relevant here. They
share the `Docket` model and identifier namespace but have different children,
sources, timeliness, and missingness:

```text
Case law:       Docket -> OpinionCluster -> Opinion
PACER / RECAP:  Docket -> DocketEntry -> RECAPDocument
```

A docket ID unifies CourtListener's internal hierarchy; it is not a universal
identity for the real-world dispute. Trial and appeal proceedings correctly
have different dockets, and duplicate CourtListener records can also have
different docket IDs. The ID does not prove that both data legs contain data
for that case. An opinion-backed state docket commonly has no docket entries.
A RECAP docket can have metadata and entries but no uploaded document. Citation
lookup returns opinion clusters, so its `docket_id` is the bridge to the court
record—not proof of RECAP coverage or deduplication.

| Derived level | Case-law leg | PACER/RECAP leg | Practical search strategy |
|---|---|---|---|
| Federal appellate | Strong for published/collected opinions; CAP supplies historical reporter material and court scrapers add current opinions. | Federal appellate dockets and filings may appear, but coverage is contribution/source dependent and should not be assumed from an opinion docket. | Search citation/case law first for cited decisions; use RECAP separately for docket documents. |
| Federal trial | Selected reported opinions and orders, including historical reporter material and opinions scraped or marked in PACER. Not a complete docket corpus. | Strongest docket metadata leg. CourtListener says it regularly gathers basic metadata for new district cases, while actual entries and PDFs remain uneven. | Search RECAP for cases/filings; search case law for published or collected opinions. |
| State appellate | Strongest state category, especially published historical case law from CAP plus ongoing court scrapers. | PACER/RECAP does not cover state courts. Opinion dockets can still have valid CourtListener docket IDs with no entries. | Search case law only; do not interpret absent entries as an absent case. |
| State trial | Sparse and highly jurisdiction-dependent; includes some reported decisions and direct court partnerships, not comprehensive trial dockets. | PACER/RECAP does not cover state courts. | Treat CourtListener miss as low-evidence; external state sources would be needed for dependable recall. |
| Other/special | Varies by tribunal and source. | Federal bankruptcy has substantial PACER coverage; other special, military, tribal, and territorial systems vary. | Route by specialization instead of the four-way bucket alone. |

CourtListener itself instructs users to search case law and federal filings in
separate databases. Cross-corpus fallback is therefore a deliberate strategy,
not pagination over one universal index.

## Timeliness and lag

- CAP is a historical book-digitization source, not a live feed. CourtListener
  has incorporated and normalized it, then supplements it with other sources.
- CourtListener scrapes current opinions from court websites, but release
  timing, scraper health, court corrections, and unpublished-opinion practices
  create court-specific lag.
- For new federal district and bankruptcy matters, CourtListener reports that
  it regularly scrapes free basic PACER metadata shortly after filing.
- RECAP docket entries and PDFs remain non-random. They arrive through browser
  users, email contributions, special scrapers, RSS feeds, bulk projects, and
  fetch APIs. A current docket may exist while a desired filing does not.
- PACER opinions or orders marked by clerks are downloaded nightly, but not
  every dispositive document is necessarily marked correctly.
- Permanent reporter citations can lag slip or neutral citations. Recent cases
  may therefore be searchable by court, docket number, name, or neutral/slip
  citation before volume/reporter/page lookup succeeds.
- Sealed and restricted documents are an intentional availability boundary,
  not ingestion lag.

Availability should be modeled as an observation with `checked_at`, corpus,
query, and result—not as a permanent property of the case.

## Development implication

Court classification remains a small deterministic enrichment step:

1. Resolve a known slug directly with `/courts/{slug}/`.
2. Resolve a name with exact case-insensitive matching; accept a prefix query
   only when it returns one credible court.
3. Preserve canonical name, slug, raw jurisdiction category, lookup method, and
   lookup time.
4. Use jurisdiction only as a prior for selecting likely retrieval paths.
5. Delegate unresolved names and conflicting human-readable evidence to general
   court reasoning.

The workflow consuming this knowledge after citation lookup fails is specified
in [Not-Found Retrieval Agent](../development/retrieval/not-found-retrieval-agent%20%5Bin%20progress%5D.md).

## Primary sources

- [CourtListener available jurisdictions](https://www.courtlistener.com/help/api/jurisdictions/)
- [CourtListener case-law API hierarchy](https://wiki.free.law/c/courtlistener/help/api/rest/v4/case-law)
- [CourtListener case-law coverage](https://www.courtlistener.com/help/coverage/opinions/)
- [CourtListener RECAP coverage and ingestion sources](https://www.courtlistener.com/help/coverage/recap/)
- [CourtListener guidance on separate search databases](https://wiki.free.law/c/courtlistener/help/search/i-cant-find-something-when-i-search-courtlistener-help)
- [Free Law Project comparison of CAP and CourtListener data](https://wiki.free.law/c/courtlistener/help/general/how-does-the-data-in-harvards-caselaw-access-project-compare-to-courtlisteners-case-law-database)
- [Free Law Project Courts Database](https://free.law/projects/courts-db)
