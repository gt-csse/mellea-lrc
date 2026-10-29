# Architecture

`mellea_lrc.api` exposes three levels: **workflow → stage → substage**. A workflow
composes meaningful stages; each stage composes granular substages. All take
and return a `Document`, and both stages and substages are independently callable.
The immutable [execution catalog](../src/mellea_lrc/model/execution.py) defines
the four workflows, 15 stages, and 55 available substages in execution order.
Field reading always uses the same substages. An optional ML prediction substage
saves a raw four-field result before the independently configured field readers.

| Workflow | Stages, in order |
| --- | --- |
| `grow_roots` | `discover_root_locators`, `read_root_fields`, `form_root_groups` |
| `validate_roots_citation_identity` | `lookup_reporter_roots`, `lookup_docket_roots`, `corroborate_locator_bodies`, `discover_intended_cases` |
| `grow_leaves` | `grow_short_reporter_leaves`, `grow_reference_leaves`, `grow_id_leaves`, `grow_supra_leaves`, `correct_leaf_readings` |
| `validate_all_citation_content` | `prepare_root_opinions`, `prepare_citation_evidence`, `review_citation_content` |

A stage ID is `workflow.stage`; an atomic ID adds its substage name, such as
`validate_roots_citation_identity.lookup_reporter_roots.retrieve_clusters`. Docket hunting, docket-root
reassignment, and supra model attribution are optional substages. Intended-case
discovery is an optional workflow stage whose four substages are required when
that stage runs. Other review flags may disable model calls while still recording
the substage's completion.

| Directory | Responsibility |
| --- | --- |
| `workflows/` | Compose workflows and their semantic stages |
| `extraction/` | Discover citations, read fields, and assign roots and leaves |
| `validation/` | Retrieve external evidence and append judgments |
| `model/` | Durable Documents, citation subclasses, typed field histories, evidence, and judgments |
| `parsing/` | Read raw source strings using shared grammars and return offsets or parser results |
| `matching/` | Match strings and ground model quotations under explicit fuzziness policies |
| `providers/` | CourtListener and GovInfo transport, pagination, and response models |
| `llm/` | Model profiles, reviewer session setup, and the instruct/validate/repair wrapper |
| `preprocessing/` | Load sources and prepare text before citation processing |
| `config/` | Extraction rules and field-reader configuration |
| `configuration.py` | Read and require explicit runtime settings from `.env` |

Each workflow is a package in `workflows/<workflow>/`; its semantic stage
composers live in `<stage>.py`. For example,
`workflows/validate_roots_citation_identity/lookup_reporter_roots.py` composes retrieval and judgment
substages from `validation/`. Stage composers declare `STAGE`; atomic writers
declare `SUBSTAGE`. Each atomic writer's file or package matches its public
entrypoint, such as `extraction/find_full_reporter_locators.py` or
`validation/retrieve_reporter_root_clusters.py`.

Direct files under `extraction/` and `validation/` are substage writers. A substage
with substantial local logic becomes a package: `__init__.py` holds its
Document entrypoint and `SUBSTAGE`; its reader, context, or reviewer
modules remain beside it. Shared services have an explicit role:
`extraction/context/`, `extraction/field_readers/`, `leaf_attribution_review/`, `validation/docket_retrieval/`,
`docket_review/`, `reporter_exact/`, `reporter_review/`, and `body_search/`.
Services do not complete checkpoints.

After docket record reviews, `validate_roots_citation_identity.lookup_docket_roots.judge_identity_from_fields` combines the
selected record's saved docket-number, case-name, court, and date judgments.
It appends an overall identity and route without retrieval or a model call.
Definitive identities stop; undetermined identities proceed to locator-body
review. The substage remains independently callable through `mellea_lrc.api`.

Discovery and field normalization use the same reporter, pinpoint, Id., and
supra source grammars in `parsing/`. These readers do not import Documents,
citations, or workflow stages. Citation-aware context belongs above them. For example,
`model/citation_chronology.py` combines recorded occurrences and recognized
noncase boundaries for both Id. attribution and pinpoint inheritance.
Provider clients do not make citation decisions. Model and infrastructure
modules do not import extraction or validation implementations; extraction and
validation do not import each other.

Each provider keeps response DTOs in `models.py` and HTTP handling,
configuration, and request errors in `client.py`. Citation-specific search
queries belong to the retrieval substage that chooses them.

Layout decisions such as margin line-number removal belong to `preprocessing/`.
Its Docling rules run before exported source offsets are measured. Plain text
retains its content exactly; `matching/` grounds against the text supplied by
its caller and does not interpret page layout or import preprocessing.

Citation fields and judgments remain append-only and point to citation-local
nodes identified by `substage`. Native Pydantic serialization retains their
histories, source evidence, and model repair traces. `Citation.next_substage`
exposes routing to an atomic step or a future continuation label, such as
`search_reporter_roots` or `search_open_web_roots`. These future labels are not entries
in the execution catalog. `Document.runs` is the only
persisted checkpoint log: each frozen event has `kind="substage"` or `kind="stage"`
and its `name`. `substage_runs` and `stage_runs` derive their ordered names from
that log.

`complete_substage(name)` commits an atomic run, including a run with no citation
changes. `complete_stage(name)` adds a group marker after its required substages
complete in catalog order, with no pending changes. The latest atomic run must
belong to that group, so the marker is recorded before proceeding to another
group. A completed stage cannot acquire omitted optional substages or further
member changes. Repeated completions and unknown semantic stages raise.
The optional ML prediction checkpoint is independent of the required field
checkpoints; saving a prediction never marks a consuming field as complete.
To enable an omitted optional step later, recover its earlier substage input
before the group's completion marker.

`get_substage(name)` and `get_stage(name)` reconstruct the exact Document at their
respective events, including untouched citations and earlier group markers. An
atomic snapshot excludes the group's later completion marker; the group snapshot
includes it. Missing or unfinished checkpoints raise `KeyError`. Workflows resume
completed groups and partial substage prefixes, and checkpoint callbacks receive
both atomic and group results. A group marker creates no citation node.
`validate_workflow_checkpoint` checks a saved workflow prefix without executing
substages or changing the Document. Evaluation runners use the existing workflow
composers for complete execution and resume; explicit atomic dispatch is reserved
for runs that stop at a requested substage.

Evaluation runners atomically replace one cumulative Document file per filing
after completed boundaries. Separate saved snapshots are unnecessary. Native
loading rejects legacy serialized `stage_runs` or `substage_runs` layouts; no
compatibility loader is provided. Production IDs use the dotted hierarchy.
Development substages may use ad hoc names. Three-part `workflow.stage.substage`
identifiers must belong to the current catalog; obsolete names and typos raise
during native loading or completion. Renamed checkpoints have no compatibility
aliases or migration loader.

Preprocessing serializes typed table-of-authorities components in
`index_spans`. Citation creation attaches a node-bound TOA tag to the occurrence
inside a component. Identity checking and shared root-opinion retrieval remain
eligible; citation-support substages consume the tag and skip that occurrence.
A body leaf does not inherit its root's TOA tag. Previously extracted citations
need replay through creation to acquire tags; loading JSON does not invent them.

`grow_leaves.correct_leaf_readings.correct_leaf_fields` finishes `grow_leaves` by reusing saved root
validation evidence to reread attached occurrences, including repeated full
citations. No validation evidence means no model call. Corrections quote the
leaf's own bounded source window and append named field histories; they do not
change its locator, attachment, or identity. Printed wrong pinpoints remain
printed wrong pinpoints. Root evidence and concise metadata form the reusable
prompt prefix, while exact record references and complete IVR repair traces
stay in the citation's correction-review history. This substage precedes pinpoint
page resolution, which captures field-reading indexes.

`evaluations/` scores each workflow independently and emits one Markdown/JSON
report pair, organized by stages with detailed substage scores. Each substage
has its own Document-only scorer and scoring rules; grouping does not change
their denominators.
Only identical result types and annotation-file loading are shared.
`experiments/` and `archive/` are outside the installed package and production
workflows.

LLM review records own their `IvrRun`: each attempt retains the provider
exchange, requirement results, and any typed transport error. Token counters
and failure categories are derived views of those records, rather than another
mutable usage log. New calls also retain measured elapsed time. Evaluation
summarizes this history by workflow, substage, model and endpoint, preserving
missing-counter coverage. See [LLM telemetry](../evaluations/README.md#llm-telemetry).

## Citation-field reading

Rule reading belongs to the extraction substages that own each field. Their
Document context includes colocation groups, neighboring locators, and known
case-name aliases. They read a span and append its typed normalization to the
citation's field history. These relationships are part of the rule algorithm,
not an input contract imposed on ML inference.

An optional ML prediction substage makes one joint inference for case name,
court, date, and pincite, and stores the raw result in
`citation.ml_field_reading`. The same downstream field substages select `RULE`
or `ML` independently. Normalization and reporter-based court inference stay
there. The model adapter in `extraction/field_readers/ml_cite/` handles model loading and span
inference; it does not discover locators or write Document histories.

Rule-versus-ML comparisons use the existing `grow_roots` and `grow_leaves`
workflow reports in this repository, with the same source corpus and annotation
targets. The ML-cite training repository remains independent of Document-level
colocation and rule context.
