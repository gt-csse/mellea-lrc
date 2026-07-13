# Scripts

The supported local workflows are citation-node snapshot generation, document
preprocessing, the local frontend backend, bookmark maintenance, and the
standalone after-retrieval case-name drill.

```text
scripts/
├── bookmark_fixture.py
├── reextract_case_name_after_retrieval.py
└── e2e_backend/
    ├── snapshot_corpus.py
    ├── preprocess_test_pdfs.py
    ├── local_server.py
    ├── api.py
    └── pipeline.py
```

Run a current named bookmark set or the configured numeric corpus:

```bash
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file date-recovery --phase assessment
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus
```

Named bookmark stores live in `fixtures/bookmarked/sets`. Use
`bookmark_fixture.py` to add a citation, revise a comment, or correct migrated
provenance; do not edit the JSON/text projection pair by hand.
