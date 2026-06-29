# Assessment Package Boundaries

Status: accepted

The assessment package is organized by domain ownership rather than execution
mechanism:

```text
assessment/
├── document/                 # whole-document scheduling and execution states
├── citation/                 # aggregation of field results for one citation
├── fields/
│   ├── case_name/            # comparison, LLM classification, and re-extraction
│   └── year/                 # deterministic year comparison
├── types/
│   ├── document.py           # citation identity and AssessedDocument
│   ├── citation.py           # completed citation aggregate
│   ├── case_name.py          # case-name values and follow-up union
│   ├── year.py               # year values
│   └── common.py             # shared provenance records
├── context.py                # offset-preserving document text windows
└── model_options.py          # shared Mellea provider configuration
```

Document-level records alone own `citation_id`. Citation-level results aggregate
field runs without repeating document identity. Field-level records contain only
field values, conclusions, provenance, and grounding information.

Case-name re-extraction produces `ReextractedCaseName(case_name,
case_name_span)`. The span is required and uses absolute offsets into preprocessed
document text. The original eyecite citation remains unchanged; its full span and
`matched_text` are not reused for a re-extracted field.

Mechanisms such as deterministic comparison and Mellea calls remain implementation
details inside the owning field package. New field workflows should be added under
`fields/<field_name>` with their domain types added under `types/<field_name>.py`.
