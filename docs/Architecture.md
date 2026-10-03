# Architecture

`mellea_lrc.api` exposes Documents, independently callable stages, and the four
user-defined workflows. A workflow chooses execution order. A stage performs
one job and returns a Document with a completed, named checkpoint.

| Directory | Responsibility |
| --- | --- |
| `workflows/` | Compose `grow_roots`, `validate_roots`, `grow_leaves`, and `validate_pincite` |
| `extraction/` | Discover citations, read fields, and assign roots and leaves |
| `validation/` | Retrieve external evidence and append judgments |
| `model/` | Durable Documents, citation subclasses, typed field histories, evidence, and judgments |
| `parsing/` | Read raw source strings using shared grammars and return offsets or parser results |
| `matching/` | Match strings and ground model quotations under explicit fuzziness policies |
| `providers/` | CourtListener and GovInfo transport, pagination, and response models |
| `llm/` | Model profiles, reviewer session setup, and the instruct/validate/repair wrapper |
| `preprocessing/` | Load sources and prepare text before citation stages |
| `config/` | Extraction rule configuration |

Direct files under `extraction/` and `validation/` are stage writers. A stage
with substantial local logic becomes a package: `__init__.py` holds its
Document entrypoint and numbered `STAGE`; its reader, context, or reviewer
modules remain beside it. Shared services have an explicit role:
`extraction/context/`, `leaf_attribution_review/`, `validation/docket_retrieval/`,
`docket_review/`, `reporter_exact/`, `reporter_review/`, and `body_search/`.
Services do not complete stages.

Discovery and field normalization use the same reporter, pinpoint, Id., and
supra source grammars in `parsing/`. These readers do not import Documents,
citations, or stages. Citation-aware context belongs above them. For example,
`model/citation_chronology.py` combines recorded occurrences and recognized
noncase boundaries for both Id. attribution and pinpoint inheritance.
Provider clients do not make citation decisions. Model and infrastructure
modules do not import extraction or validation stages; extraction and
validation do not import each other.

Citation fields and judgments remain append-only and point to citation-local
nodes. Native Pydantic serialization retains their histories, source evidence,
and model repair traces. `Document.get_stage(name)` reconstructs the exact
Document at a completed boundary. Evaluation runners atomically replace one
cumulative Document file per filing after completed stages; separate copies
of every checkpoint are unnecessary.

`evaluations/` scores each workflow independently and emits one Markdown/JSON
report pair. Each stage has its own Document-only scorer and scoring rules.
Only identical result types and annotation-file loading are shared.
`experiments/` and `archive/` are outside the installed package and production
workflows.
