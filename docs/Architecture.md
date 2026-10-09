# Architecture

`mellea_lrc.api` exposes three levels: **workflow → stage → substage**. A workflow
composes meaningful stages; each stage composes granular substages. All take
and return a `Document`, and both stages and substages are independently callable.
The immutable [execution catalog](../src/mellea_lrc/model/execution.py) defines
the four workflows, 15 stages, and 53 substages in execution order.

| Workflow | Stages, in order |
| --- | --- |
| `grow_roots` | `locator_discovery`, `field_reading`, `root_formation` |
| `validate_roots` | `reporter_lookup`, `docket_lookup`, `locator_body_corroboration`, `intended_case_discovery` |
| `grow_leaves` | `short_reporter_citations`, `reference_citations`, `id_citations`, `supra_citations`, `leaf_field_correction` |
| `validate_pincite` | `opinion_preparation`, `citation_preparation`, `support_review` |

A stage ID is `workflow.stage`; an atomic ID adds its substage name, such as
`validate_roots.reporter_lookup.cluster_retrieval`. Docket hunting, docket-root
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
| `config/` | Extraction rule configuration |
| `configuration.py` | Read and require explicit runtime settings from `.env` |

Each workflow is a package in `workflows/<workflow>/`; its semantic stage
composers live in `<stage>.py`. For example,
`workflows/validate_roots/reporter_lookup.py` composes retrieval and judgment
substages from `validation/`. Stage composers declare `STAGE`; atomic writers
declare `SUBSTAGE`.

Direct files under `extraction/` and `validation/` are substage writers. A substage
with substantial local logic becomes a package: `__init__.py` holds its
Document entrypoint and `SUBSTAGE`; its reader, context, or reviewer
modules remain beside it. Shared services have an explicit role:
`extraction/context/`, `leaf_attribution_review/`, `validation/docket_retrieval/`,
`docket_review/`, `reporter_exact/`, `reporter_review/`, and `body_search/`.
Services do not complete checkpoints.

After docket record reviews, `validate_roots.docket_lookup.identity_aggregation` combines the
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
exposes routing to an atomic step. `Document.runs` is the only
persisted checkpoint log: each frozen event has `kind="substage"` or `kind="stage"`
and its `name`. `substage_runs` and `stage_runs` derive their ordered names from
that log.

`complete_substage(name)` commits an atomic run, including a run with no citation
changes. `complete_stage(name)` adds a group marker after its required substages
complete in catalog order, with no pending changes. The latest atomic run must
belong to that group, so the marker is recorded before proceeding to another
group. A completed stage cannot acquire omitted optional substages or further
member changes. Repeated completions and unknown semantic stages raise.
To enable an omitted optional step later, recover its earlier substage input
before the group's completion marker.

`get_substage(name)` and `get_stage(name)` reconstruct the exact Document at their
respective events, including untouched citations and earlier group markers. An
atomic snapshot excludes the group's later completion marker; the group snapshot
includes it. Missing or unfinished checkpoints raise `KeyError`. Workflows resume
completed groups and partial substage prefixes, and checkpoint callbacks receive
both atomic and group results. A group marker creates no citation node.

Evaluation runners atomically replace one cumulative Document file per filing
after completed boundaries. Separate saved snapshots are unnecessary. Native
loading rejects legacy serialized `stage_runs` or `substage_runs` layouts; no
compatibility loader is provided. Production IDs use the dotted hierarchy.
Development substages may use ad hoc names; semantic group membership comes
from the catalog.

Preprocessing serializes typed table-of-authorities components in
`index_spans`. Citation creation attaches a node-bound TOA tag to the occurrence
inside a component. Identity checking and shared root-opinion retrieval remain
eligible; citation-support substages consume the tag and skip that occurrence.
A body leaf does not inherit its root's TOA tag. Previously extracted citations
need replay through creation to acquire tags; loading JSON does not invent them.

`grow_leaves.leaf_field_correction.review` finishes `grow_leaves` by reusing saved root
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
