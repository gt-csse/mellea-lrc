# Development root-stage expectations

The `dev_data/*-root-stage-expectations-v1.jsonl` files are separate,
development-only sidecars for Primary, Hallucination Set 1, Reliable High
Profile, and Reliable Low Profile. They do not change official identity labels,
extraction code, or model prompts. Each joins by `(document, kind,
locator_span)` and preserves the full identity annotation for audit. No sidecar
is built for Hallucination Set 2 because it has no validation identity labels.

The schema separates two questions. `expected_retrievals` lists each exact
provider record ID, source path/field, and retrieval stage that the annotation
shows should be findable. `expected_identity_decision_stage` is optional and is
populated only when the indexed field evidence for the exact target record and
the annotation basis support the gold verdict. Retrieval and decision labels
are therefore independent; a related record can be findable while the gold
identity decision depends on another kind of evidence.

The conservative routes are:

| Route | Annotation evidence required | Retrieval stage |
|---|---|---|
| CourtListener reporter | `FullCaseCitation`, with `case_at_locator` evidence from a cluster under the exact `/citation-lookup/{volume}/{reporter}/{page}.json` path. | `full_reporter_locator_exact_lookup` |
| CourtListener docket | `DocketCitation`, with a docket record whose path includes its exact CourtListener docket ID. Both `case_at_locator` and `independent_record` may establish this retrieval. | `docket_root_identity` CourtListener query nodes |
| GovInfo docket | `DocketCitation`, with a docket record whose summary path and source ID identify the same GovInfo package. | `docket_root_identity` GovInfo query nodes |

The optional reporter decision stage is `full_reporter_locator_identity_resolution`:
the annotation must directly adjudicate the exact cluster and its relevant
field evidence must point to that target. In the saved trace, its first actual
decision may be in unique-identity or ambiguity resolution. The optional docket
decision stage is `docket_root_identity`; it requires a `docket_key_names_case`
or `record_at_locator_disagrees` basis and field evidence tied to the expected
docket/package target. Other roots can have expected retrievals and still have
no decision-stage label. Their `identity_decision_unknown_reason` explains why;
roots without any route-specific retrieval also carry `unknown_reason`.

No annotation `shows` value denotes an independent third-party citation in a
searched opinion body. Opinion/ruling evidence alone may simply be the cited
authority, so body-corroboration expectations are currently left unknown.
Generic docket evidence for reporter roots is also not mapped to reporter
metadata search. `unknown_reason` explains roots without a route-specific
provider record and stage.

The offline scorer checks the exact ID inside the correct provider's raw
returned candidate payloads for the named stage. It ignores query strings,
messages, prompts, and shortlist-only data. Separately, it records the first
decisive identity-resolution stage and outcome, then compares that decision
with the optional gold decision-stage expectation. `target_and_decision_success`
requires both checks to pass. A stage invocation does not count as record
retrieval. No network or model calls are made.

Rebuild the sidecar and score a saved checkpoint:

```bash
uv run python -m evaluations.validation.dev_stage_expectations \
  --annotations ../mellea-lrc-datasets/primary/documents \
  --output evaluations/validation/dev_data/primary-root-stage-expectations-v1.jsonl \
  --artifacts data/run-artifacts/54-primary-durable-case-name-rerun-v1/shared_body_evidence_review \
  --report evaluations/validation/reports/primary-v54-dev-root-stage-score.json

uv run python -m evaluations.validation.dev_stage_expectations \
  --annotations ../mellea-lrc-datasets/hallucination-set-1/documents \
  --output evaluations/validation/dev_data/hallucination-set-1-root-stage-expectations-v1.jsonl

uv run python -m evaluations.validation.dev_stage_expectations \
  --annotations ../mellea-lrc-datasets/reliable-high-profile/documents \
  --output evaluations/validation/dev_data/reliable-high-profile-root-stage-expectations-v1.jsonl

uv run python -m evaluations.validation.dev_stage_expectations \
  --annotations ../mellea-lrc-datasets/reliable-low-profile/documents \
  --output evaluations/validation/dev_data/reliable-low-profile-root-stage-expectations-v1.jsonl
```
