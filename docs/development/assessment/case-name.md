# Case-Name Assessment

Case-name assessment compares the party name extracted by eyecite with the name
on one retrieved CourtListener candidate. It is the only current field workflow
that may use Mellea.

## Decision path

1. Build the extracted display name from plaintiff and defendant fields.
2. Return `unassessable` if the retrieved candidate has no case name.
3. Normalize typography and whitespace; return `exact_match` when equal.
4. Ask whether the names are a normal semantic legal-citation match in local
   document context; return `semantic_match` when they are.
5. Otherwise record `not_semantic_match` and attempt faithful re-extraction from
   the local context.
6. Ground any proposal to absolute document offsets, then reassess it as
   `exact_match`, `semantic_match`, or `not_semantic_match`.

The active validation layer intentionally does not decide whether a failed
semantic match is an “irregular” abbreviation, an incomplete citation form, or a
lawyer-acceptable short form. For validation, the useful question is only
whether the local citation name and retrieved candidate semantically identify
the same proceeding. Style and abbreviation appropriateness belong to a later
proposition/support or lawyer-facing citation-quality layer.

Re-extraction must copy the text that actually appears in context. It must not
correct toward the retrieved candidate. The original eyecite citation remains
unchanged, and failed extraction or reassessment is represented as a typed
follow-up state rather than hidden.

The current workflow uses only `exact_match`, `semantic_match`,
`not_semantic_match`, and `unassessable` as case-name conclusions.

`CaseNameAssessmentRun` retains the initial conclusion and one of
`not_required`, `reassessed`, `reextraction_failed`, or `reassessment_failed`.
A successful proposal becomes `ReextractedCaseName(case_name, case_name_span)`;
the span uses absolute offsets into preprocessed document text.

Implementation: `assessment/fields/case_name/` and
`assessment/types/case_name.py`.

This is the current implementation, not the intended long-term ownership.
Retrieval now also needs grounded case-name repair before candidate search. The
proposed shared workflow is documented in
[Shared Re-extraction Workflow](../../architecture/shared-reextraction-workflow.md).
