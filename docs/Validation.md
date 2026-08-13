---
tags: [validation, courtlistener, hallucination-detection, pinpoint]
status: active
---

# Validation

Validation is the last stage. It takes the citations
[extraction](./Extraction.md) found and asks, of each one, whether the authority
it names exists and matches how the filing cites it — down to whether the page
cited actually says what it is cited for.

It never answers "this citation is fake." It reports what it could establish and
what it could not, and keeps those two apart. Why that distinction is the whole
design is the first section below.

---

## Running it

```python
import asyncio
from mellea_lrc.extraction import extract_from_raw_document
from mellea_lrc.validation import validate_document

document = extract_from_raw_document(Path("filing.pdf"))
validated = asyncio.run(validate_document(document))
```

It is async because every step is I/O — CourtListener over HTTP, then a model.
Credentials come from the environment: `COURTLISTENER_API_TOKEN` and
`MELLEA_LRC_LLM_*`. See
[the client doc](./courtlistener-client.md#getting-an-api-token) for the token,
and [its rate-limit notes](./courtlistener-client.md#rate-limits) before running
anything at scale — a free-tier quota will not carry a real workload.

Both dependencies are injectable:

```python
validate_document(document, client=my_client, session=my_session)
```

`client` is anything satisfying `CourtListenerServiceClient`, which is the seam
for a cache or a fixture. `session` is a Mellea session. Leave the model
temperature at `0.0`.

The `mellea-lrc validate` command wraps exactly this.

---

## Absence is not falsity

The single most important thing about this stage.

A citation that CourtListener cannot resolve has not been shown to be
fabricated. It may simply not be there, and the reasons are structural:

- **CourtListener's archive is crowdsourced.** RECAP mirrors PACER only for
  documents someone already paid to download. A docket can be fully indexed
  while the filing itself was never uploaded.
- **Trial-level state material is largely absent.** Coverage is strong for
  federal and state appellate opinions, thinner below that, and varies by court.
- **Sealed and restricted filings never reach PACER at all**, so they cannot
  reach RECAP.
- **Unpublished opinions** appear only if scraped from a court's own site.

So `not_found` means *not found*. Treating it as evidence of invention is the
exact error this stage is built to avoid, and it is why the outcome vocabulary
below keeps abstentions as their own answers rather than folding them into a
verdict. When you report results, report the abstention rate next to the
accuracy; recoding abstentions inflates whichever number they land in.

---

## What comes back

A `ValidatedDocument`: one `CitationValidation` per extracted citation, each
holding an ordered tuple of **nodes**.

A node is one check. It records what was asked, what came back, and what that
means:

| field | what it is |
|---|---|
| `node_id` | stable identifier, derived from the path that produced it |
| `status` | `succeeded`, `skipped`, or `failed` |
| `outcome` | the finding — vocabulary depends on the node |
| `depends_on` | the nodes this one consumed |
| `status_message` / `outcome_message` | prose for each |
| `error` | set when `status` is `failed` |

Nothing is overwritten and nothing is summarised away. A citation's history is
the list, in order, and `citation_validation.aggregation` returns its terminal
summary node when the route produced one.

**`status` and `outcome` answer different questions.** `status` is whether the
check ran; `outcome` is what it found. A `succeeded` node with outcome
`mismatch` did its job and found a problem. A `failed` node found nothing
because the step itself broke. Reading a `failed` as a negative finding is the
same category error as reading `not_found` as fabrication.

---

## The route

Everything begins with an exact locator lookup — volume, reporter, page — and
the result of that one call decides which of three paths the citation takes.

```
exact locator lookup
├── found      → the field checks, then the pinpoint check
├── not found  → recover a case name, then search
├── ambiguous  → candidate selection, then the field checks per candidate
└── unsupported / incomplete / failed → stop
```

The last row matters: a short-form citation, or one whose locator is missing a
field, cannot be looked up at all and ends immediately. That is a property of
what extraction produced, not a finding about the citation.

### When the locator resolves

The main path. The cited authority is in hand, so every remaining question is a
comparison between what the filing says and what the record says.

```
found locator
└── locator candidate evaluation
    ├── exact case-name check ── mismatch → case-name recovery
    ├── year check
    ├── docket court retrieval → court check
    ├── reporter-page retrieval
    │   └── citing-proposition extraction → pinpoint check
    └── locator candidate assessment → citation summary
```

The three field checks are independent and each reports `match`, `mismatch`, or
`unavailable`. `unavailable` means the filing or the record did not state the
field — not that they disagreed.

**Case-name recovery** is what happens when the names do not match. Rather than
concluding `mismatch`, the pipeline tries again: a model compares the names
semantically (abbreviations, party ordering, `et al.`), and if extraction gave
it nothing to work with, a second model pass re-reads the document text to
recover the parties directly. Only then is the disagreement taken at face value.

**`locator candidate assessment`** folds the field checks into one verdict —
`match`, `partial_match`, or `mismatch`.

### The pinpoint check

The substantive one, and the only step that reads the cited opinion's text.

Two model calls, in order. **Citing-proposition extraction** reads the citing
document around the citation and states the proposition being attributed to it.
**The pinpoint check** then reads the retrieved reporter page and decides
whether that page supports it — `supports` or `inconclusive`.

Neither is asked to recall anything. Both work from text supplied to them, and
the answer must be grounded: a `supports` verdict has to quote the page, and the
quote is located back in the retrieved text before the verdict is accepted. A
quote that cannot be found uniquely is rejected and the model asked again; a
verdict that never grounds is reported as `failed` rather than as a finding.

That is the reason for `evidence_span` and `evidence_match_method` on the
result — an offset into the page that was actually retrieved, and how it was
matched (`exact`, `normalized`, or `fuzzy`). The verdict is always checkable
against the source.

By design the model can only say `supports` or `inconclusive`. It is never asked
to conclude that a page *contradicts* a proposition, because one retrieved page
is not enough evidence to condemn a citation.

### When the locator does not resolve

The fallback. Without a locator there is nothing to look up, so the pipeline
works from the case name instead.

```
not-found locator
└── local party re-extraction
    └── case-name query preparation
        ├── CourtListener opinion search → candidate selection → per candidate
        └── CourtListener RECAP search   → candidate selection → per candidate
```

Two searches, because the two corpora differ: opinions covers published
decisions, RECAP covers filings. Candidates from either are assessed the same
way as a resolved locator, except the verdict vocabulary is narrower —
`possible_match` or `mismatch`. Nothing found by search is ever called a
`match`, because a search hit is a resemblance, not an identification.

This route deliberately does not run the pinpoint check. Without a confirmed
authority there is no page to read.

### When the locator is ambiguous

One locator resolving to several clusters. Each is evaluated as its own
candidate through the same field-check subtree, and selection is capped —
`deferred_over_limit` records that some candidates were not pursued rather than
silently dropping them.

---

## Reading a result

The summary vocabulary, across both routes:

| outcome | means |
|---|---|
| `match` | the record agrees with the filing |
| `possible_match` | a candidate resembles it, unconfirmed |
| `mismatch` | the record contradicts the filing |
| `not_found` | nothing was found — see *Absence is not falsity* |

Two cells deserve attention when you look at aggregate results. A `mismatch`
against a genuine citation is a false alarm and costs a reader's trust. A
`match` against a bad one is the dangerous cell — a citation waved through — and
is the number worth minimising.

The per-node detail is where the reasoning lives. A citation that came back
`mismatch` has a specific node that said so, with the retrieved value beside the
filing's; a pinpoint result carries the quote it rested on. Nothing requires you
to trust the summary.

---

## How it is measured

Against the frozen validation set, with the scoring rules, the confusion matrix,
and how to reproduce a run documented alongside the code that does it, in
[the validation evaluation](../evaluations/validation/README.md).

One property of that benchmark follows directly from the first section:
occurrences CourtListener cannot decide are **absent from it** rather than
labelled. Inferring `mismatch` from a failed lookup is the error the set exists
to expose, so it is not built into the labels.
