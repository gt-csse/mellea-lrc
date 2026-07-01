# Validation Window Redesign — Design Doc

## Motivation

The old `CitationValidation` was a flat dataclass with 15 fields that intermixed existence-check outcomes (`status`, `matches`, `lookup_status`) with court-resolution data (`CitationMatch.court_id`, enriched by a docket GET hidden in `validation/pipeline.py` helpers). The frontend rendered this as a single flat block with no temporal trace — unlike the assessment stage's `AssessmentTraceStep` UI which shows a 3-step trace (initial assessment → re-extraction → reassessment).

Goals of the redesign:

- **Mirror assessment's tracing model** for the validation window (discriminated union, per-citation run/step metaphor).
- **Extract court resolution** into its own module (`validation/court_resolution.py`), so the validation pipeline stays a thin existence-lookup orchestrator.
- **Reporter→court inference** lives in assessment (`assessment/fields/court/`), not extraction or validation — see [Validation Model Development](./Validation%20Model%20Development.md#court-field-assessment). Validation only retrieves/resolves the CourtListener court; it never compares it against the raw eyecite ``court`` — that comparison, and any reporter-based inference, is assessment's job.
- **Fix silent `AttributeError` swallowing** for clients that don't implement `get_docket` (previously any client that lacked the method silently set `court_id=None`; now we do an explicit `hasattr` guard and record `NO_DOCKET_ID`).
- **Normalize cache keys** to `str` (so JSON-int `42` and JSON-string `"42"` collapse to one docket GET).
- **Schema bumps** v8→v9 (validation union), v9→v10 (removed `extracted_court_inferred`), v10→v11 (`CourtAssessmentRun` with reporter-inference follow-up).

## New data structures

### `CitationValidation` — status-discriminated union (7 variants)

Each variant is a `@dataclass(frozen=True, slots=True)` with a `ClassVar[ValidationStatus]` discriminator, carrying only the fields its outcome requires.

| Variant | Status | Carries | Trace? |
|---|---|---|---|
| `FoundCitationValidation` | `found` | `citation_id, locator, source, message, lookup_status, lookup_cache, lookup_key, matches, court_resolution, extra_data` | Yes |
| `AmbiguousCitationValidation` | `ambiguous` | same as found but no `court_resolution` | No |
| `NotFoundCitationValidation` | `not_found` | `citation_id, locator, source, message, lookup_status, lookup_cache, lookup_key, extra_data` | No |
| `InvalidCitationValidation` | `invalid` | `citation_id, source, message` | No |
| `ThrottledCitationValidation` | `throttled` | `citation_id, locator, source, message, lookup_status, lookup_cache, lookup_key, error_message, failure_detail, extra_data` | No |
| `LookupFailedCitationValidation` | `lookup_failed` | like Throttled but `lookup_status: int | None` | No |
| `SkippedCitationValidation` | `skipped` | `citation_id, source, message` | No |

The `Found` variant alone carries `court_resolution: CourtResolutionTrace`.

All variants expose `.status` (the ClassVar) and `.case_names` property and `.citation_id`, so existing consumer reads of those attributes still work (`item.status == ValidationStatus.FOUND`, etc.).

`ValidatedDocument.found` now filters with `isinstance(item, FoundCitationValidation)` instead of `item.status == ValidationStatus.FOUND`.

### `CourtResolutionTrace`

A frozen, slotted dataclass that records how the CourtListener-side court was resolved for one found citation:

```python
@dataclass(frozen=True, slots=True)
class CourtResolutionTrace:
    courtlistener_court_id: str | None
    resolved_via: CourtResolutionSource   # cluster_provided | docket_lookup | no_docket_id | docket_lookup_failed | not_attempted
    docket_id: str | None                # normalized to str
    docket_url: str | None               # /dockets/<id> path component
    cached: bool
    error_message: str | None
```

Resolution flow in `court_resolution.py:resolve_court()`:

1. If `match.court_id` is non-empty → `resolved_via = cluster_provided` (no docket GET).
2. Else check `match.extra_data["docket_id"]`. If missing/non-scalar → `resolved_via = no_docket_id`.
3. Else normalize to `str(docket_id)`. Check cache.
   - Cache hit → `resolved_via = docket_lookup`, `cached = True`.
   - Cache miss → call `client.get_docket(docket_id)` (guarded by `hasattr(client, "get_docket")`). On success, extract `docket.get("court_id")` (validated as `str`); on failure, record error and cache `None`.

Validation stops there — it never compares `courtlistener_court_id` against the citation's extracted court. There is no fallback attempt to try and no comparison to flag as missing; retrieval either succeeds or it doesn't, and the trace records only how. The assessment stage (`assess_court`) reads `courtlistener_court_id` off this trace and does its own comparison against `citation.court`, including any reporter-inference fallback.

### `CourtResolutionSource` enum

- `cluster_provided` — CL cluster response already had `court_id`
- `docket_lookup` — docket GET succeeded
- `no_docket_id` — match had no `docket_id` to look up
- `docket_lookup_failed` — docket GET raised/malformed
- `not_attempted` — citation not found (no resolution attempted)

### `ValidationMetadata`

Added `duration_ms: float | None = None` for per-run timing telemetry.

## Module: `src/mellea_lrc/validation/court_resolution.py`

CourtListener-side resolution only. Reporter inference is in `extraction/court_inference.py`.

| Export | Purpose |
|---|---|
| `resolve_court(match, *, client, cache) -> CourtResolutionTrace` | Resolve the CourtListener court only; no comparison against `citation.court` |

## Module: `src/mellea_lrc/assessment/fields/court/`

| Export | Purpose |
|---|---|
| `assess_court(*, extracted_court, courtlistener_court_id, reporter) -> CourtAssessmentRun` | Initial comparison + optional reporter inference follow-up |
| `infer_court_from_reporter(reporter) -> str \| None` | SCOTUS reporter lookup table |

## Module changes

### `src/mellea_lrc/validation/pipeline.py`

- Removed `_enrich_found_lookup_courts`, `_enrich_match_court`, the `docket_court_cache` threading through `_validate_citation`.
- `_validation_from_lookup` now builds the right variant via an `if status is ValidationStatus.FOUND` branch that calls `resolve_court(...)` from the new module.
- `run_validation` emits `ValidationMetadata(duration_ms=...)` (perf-counter delta).
- Cache key normalized to `str` internally.

### `src/mellea_lrc/assessment/document/pipeline.py`

- Consumer migrated: `citation_validation.matches[0].court_id` → `isinstance(citation_validation, FoundCitationValidation) and citation_validation.court_resolution.courtlistener_court_id`.
- Imports `FoundCitationValidation`.

### `src/mellea_lrc/serialization/transport.py`

- Schema version bumped from `Literal[8]` → `Literal[9]` on all four document payloads.
- `CitationValidationPayload` replaced with a 7-member discriminated `Annotated` union (`Field(discriminator="status")`), one payload per status variant.
- Added `CourtResolutionTracePayload`.
- `ValidationMetadataPayload` gains `duration_ms: float | None`.

### `src/mellea_lrc/serialization/json.py`

- `SCHEMA_VERSION = 8` → `9`.
- `serialize_citation_validation` / `deserialize_citation_validation` rewritten with `isinstance` branches.
- Added `_serialize_court_resolution_trace` / `_deserialize_court_resolution_trace`.
- Added `_VALIDATION_PAYLOAD_ADAPTER = TypeAdapter(CitationValidationPayload)` for the discriminated union.
- `_serialize_validation_metadata` / `_deserialize_validation_metadata` handle `duration_ms`.
- `_required_int` helper added.

### Tests (migration in progress)

All test constructs of old `CitationValidation(...)` must switch to variant constructors. The `test_validate_full_case_found` assertion that expected `CitationMatch.court_id` → now reads `FoundCitationValidation.court_resolution.courtlistener_court_id`. Schema-rejection test bumped from "reject v7" → "reject v8".

### Frontend (pending)

`ValidationPayload` TS type and `ValidationDetails` component need updating to consume the discriminated union shape and render the structured `court_resolution` trace as an `AssessmentTraceStep`-analogue breakdown (lookup → court resolution). No consistency verdict is rendered at this stage — that comparison surfaces later in the assessment window.

## Non-changes

- `CitationMatch.court_id` and `.court` retained (upstream-shaped transport fields; `court_id` populates the `cluster_provided` cluster-resolution path).
- `CitationValidationClient` protocol retained (still needed by the docket GET path; `hasattr` guard added instead of swallowing `AttributeError`).
- Label Studio upload scripts untouched (retrieval backend already removed in prior commit `97e17a8`).
