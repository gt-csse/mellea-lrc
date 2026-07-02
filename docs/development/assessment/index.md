# Assessment Development

Assessment is the phase that interprets retrieved evidence. It compares
extracted citation fields with validation candidates and records field-level
conclusions, provenance, and follow-up work.

This is intentionally separate from [validation](../validation/index.md): validation
retrieves without expressing an opinion; assessment owns match, mismatch,
equivalence, irregularity, and other evaluative conclusions.

## Current scope

Only validation results with status `found` or `ambiguous` are currently
assessment-eligible. A found result supplies one candidate. An ambiguous result
supplies one assessment job per candidate, subject to the defensive candidate
gate. Other validation states become typed skipped or failed assessment states;
assessment does not reinterpret `not_found` as false.

For each eligible candidate, `assess_found_citation` aggregates three independent
field runs:

| Field | Inputs | Current mechanism |
|---|---|---|
| [Case name](./case-name.md) | extracted parties, retrieved name, local context | deterministic equality, then semantic assessment and grounded re-extraction |
| [Court](./court.md) | extracted court, retrieved court ID, reporter | deterministic comparison with field-local reporter inference |
| [Year](./year.md) | extracted year, retrieved filing year | deterministic comparison |

The citation aggregate does not flatten those results into a single “real” or
“false” label. Consumers can inspect each field and its provenance independently.

## Package boundaries

- `assessment/document/` schedules work and owns document-level execution
  states and citation identity.
- `assessment/citation/` aggregates field results for one candidate.
- `assessment/fields/<field>/` owns comparison and field-local follow-ups.
- `assessment/types/<field>.py` owns the corresponding data contract.

See [Assessment Package Boundaries](../../architecture/assessment-package.md)
for the type ownership rules and [Ambiguous Citations](./ambiguous-citations.md)
for candidate fan-out.

## Adding a field

Add the workflow under `assessment/fields/<field_name>/`, its domain types under
`assessment/types/<field_name>.py`, and a field document here. Keep document
identity out of field records. Preserve both extracted and retrieved values,
make missing evidence an explicit state, and retain any model or follow-up trace
needed to explain the conclusion.
