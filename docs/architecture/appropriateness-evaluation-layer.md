# Appropriateness Evaluation Layer

Status: proposed boundary; behavior deferred

## Purpose

Appropriateness evaluation asks whether a cited authority actually supports the
way it is used in the source document. This includes proposition support,
pinpoint support, treatment of the authority, jurisdictional/procedural fit, and
potential mischaracterization.

This is more deliberative than retrieval identity resolution. It must not be
embedded in bookmark candidate comments or treated as a side effect of finding
the correct docket.

## Boundary

Three conclusions must remain distinct:

1. **Candidate retrieval:** what records or documents were surfaced?
2. **Authority identity:** did we locate the cited case and decisional document?
3. **Appropriateness:** does that authority support this citation's use in the
   source document?

A result may pass identity and fail appropriateness. Conversely, an
appropriateness conclusion cannot be made reliably until the cited decisional
document has been identified.

The bookmark fixture currently records research cases and identity-resolution
evidence. Proposition-support verdicts are intentionally deferred from those
comments until this layer has a typed, reproducible contract.

## Expected inputs

The eventual layer will likely require:

- the complete source document through the shared prefix;
- grounded document inference;
- the citation's local argumentative context;
- the verified cited decisional document;
- locator and pinpoint information;
- authority-identity confidence and provenance;
- later history or treatment when relevant.

The [prefix-cached document reasoning](./prefix-cached-document-reasoning.md)
design allows this deliberation to retain full-document context without paying
the full prefill cost independently for every citation.

## Deferred questions

- How should the cited proposition be delimited from surrounding argument?
- What evidence standard distinguishes supported, partially supported,
  overstated, contradicted, and unassessable uses?
- When must the evaluator inspect cited-document context beyond the pinpoint?
- How should quotation accuracy, negative treatment, and jurisdictional fit be
  represented without flattening them into one score?
- Which conclusions require human review?

These questions should be answered before implementation or fixture-wide
appropriateness labels.
