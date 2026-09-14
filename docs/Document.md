---
tags: [record, artifact, extraction, validation]
status: active
---

# The document every stage appends to

One document object, one artifact type, one record per citation. A stage adds
citations, adds nodes to the citations already there, and adds findings about
the document. It does not wrap the previous stage's output, and it does not
produce an artifact of its own kind.

This replaces three types that say the same thing in three shapes:

    extracted_document    text, metadata, citations
    identified_document   source: <a whole extracted_document>, records: [...]
    validated_document    source: <a whole extracted_document>, citations: [progressions]

Each wraps the one before, so the same citation is written twice, in two
shapes, and a consumer has to know which layer it is looking at to find the
current reading. That is the only thing being removed.

---

## The unit is the citation

A `CitationRecord` is one citation's whole life, and it already holds
everything the pipeline knows about it:

    citation_id     a hash of the span and the characters at it. Stable across
                    a re-read of the same text, which is what lets a later pass
                    find the same citation without being handed a mapping
    source          what the rules read. Never changes
    stated          the same citation as it is now read, after corrections
    found           what an archive holds at that identity
    root_id         which root this citation points at
    authority_id    which authority it was established to reach
    corrections     every change to `stated`, in order, each naming its node
    judgement       what the pipeline concludes, and the node that concluded it
    withdrawn_by    the node that took this citation out, if one has
    trace           every node: what was asked and what came back

Nothing else about a citation lives anywhere else. A stage that learns
something about a citation writes it here.

### State is written; the node is a pointer

The trace is a **graph** -- a node names what it depends on -- and nothing about
a citation's current state is recovered by walking it. What the pipeline
currently says is a field on the record, and the node that said it is an id
beside it:

    a correction   `Correction.node_id`
    a judgement    `Judgement.node_id`
    a withdrawal   `withdrawn_by`
    a resolution   `Resolution.node_id`   (already this shape)
    a finding      `Finding.node_id`

The corrections were inside their nodes, and the reasoning for that was that a
change could then not exist without its evidence. The invariant is kept another
way and costs nothing: `record.correct` is the only door, and it writes the node
and the correction in one call, so there is still no way to change `stated`
without leaving the reason. What it buys is that the history is a list rather
than a traversal, and the history is read far more often than the graph is.

**The judgement is there from the start**, saying `unjudged`. A reader never has
to tell an absent judgement from an unmade one, and never has to search the
trace for whichever node happened to be the aggregation. The same is true of
every field above: a question about the current state is a field access, and
the trace answers only "how did it get that way".

## What a stage may do

**Add a node.** Every reading is a node, including the first: the parse is the
node that produced `source`, and writing it down makes `source` an ordinary
output rather than a field with no author. `reads` decides what a node may
write — `DOCUMENT` evidence corrects `stated`, `RECORD` evidence settles
`found` — and a correction that lives outside the node justifying it cannot be
constructed.

**Add a citation.** Extraction adds roots in the first pass and leaves in the
second. Validation may add one if it reads a citation the rules missed. The
invariants hold for whoever adds it: a leaf without a root cannot exist.

**Withdraw a citation, never delete it.** Reclassifying is a node whose outcome
says the span is not the citation it was read as — a statute read as a case, a
docket number that is a record entry, a root that reaches nothing. The record
stays, and stays addressable, because `root_id` and `authority_id` point at
citation ids and deleting a record breaks every reference to it. A withdrawn
root withdraws its leaves with it, in the same pass, with a node on each saying
which root took it.

**Add a finding about the document.** Some things are not about any citation
and cannot be made into one: a leaf that could not be grown, a site that was
hunted and rejected, a name standing in the text that no citation covers. A
record is a citation; inventing a record for a non-citation would break the
invariant that makes the record worth having. These go in a document-level
list, each naming the span it is about and the node that produced it.

## What the document holds

    source_metadata           where the text came from
    text                      the coordinate space every span indexes
    preprocessing_metadata    what the converter did to it
    citations                 the records
    findings                  what is true of the document and of no citation
    passes                    what has run over it, in order

`text` is the reason the container exists at all. Every span is an offset into
it, the leaf pass re-reads it, and validation's evidence quotes are slices of
it. Per-citation copies would be the same container with a weaker invariant:
nothing would stop two records disagreeing about what the document says.

`passes` is what tells a consumer the tree is not finished. A document that has
had roots grown and not leaves is not a reading of the document's citations,
and the artifact has to say so rather than leave it to be inferred from the
absence of short forms.

## Ordering is mostly timing, and twice it is not

Most of what looks like sequence is batching: identity runs per root because
that is what a lookup takes, the pin-cite check runs after identity because it
needs the opinion, the site hunt runs last because it is cheapest when the mask
is fullest. Any of those could run in any order and the record would be the
same record, only slower or more expensive.

Two are structural, and a scheduler has to know them:

*   **A leaf cannot exist before its root.** Not a preference — `CitationRecord`
    refuses a leaf with no root, because a leaf's meaning is which root it
    points at. A root admitted late has no leaves until a growth runs again.
*   **A correction to `stated` changes what later readings see.** Attachment
    matches a leaf against `stated`, so a name corrected after the leaves grew
    is a name the leaves never saw. Re-running the growth is cheap; pretending
    the order does not matter is not.

## What each stage looks like under this

    extraction, first pass    adds every citation that states a complete
                              identifier, each with the node that read it
    identity                  adds nodes to those citations: the lookup, the
                              name check, the search. Corrects `stated` from
                              document evidence, settles `found` from record
                              evidence, withdraws what reaches nothing
    extraction, second pass   adds the leaves, matched against `stated` as it
                              now is, and a finding for each leaf it could not
                              grow
    validation, after that    adds nodes to the leaves: the pinpoint check on
                              each page claim
    site hunting              triggered by the findings, adds citations the
                              rules never read, each of which then needs
                              identity before anything grows on it

Every arrow is the same object with more written on it.

## What this costs to build

*   One document type and one artifact type, replacing three. The record type
    is already shared — `mellea_lrc/validation/record.py` is a re-export of
    `mellea_lrc.core.record`.
*   `Node` gains an opaque `details` mapping that round-trips verbatim and that
    `core` never interprets, so validation's typed nodes keep their own fields
    without `core` knowing any of their types. This is what lets the
    progressions in `validated_document` move onto the trace where
    `identified_document` already puts them.
*   `CitationRecord` gains a withdrawal state, and the rule that withdrawing a
    root withdraws its leaves.
*   The document gains `findings` and `passes`.
*   A schema bump, and the old artifacts stay readable or stay archived.

## Open

*   What a finding is, exactly: a span, a kind, a node, a message. The refused
    leaves are the first users, and what the site hunt needs from them decides
    the shape.
*   Whether `withdrawn` is a field or a node outcome read off the trace. The
    trace is the single source of truth, so the field would be a cache — and a
    cache that can disagree with the trace is worse than a fold over it.
