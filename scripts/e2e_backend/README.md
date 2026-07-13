# Local E2E backend

`pipeline.py` assembles the local frontend review backend. `api.py` exposes its
FastAPI application, and `local_server.py` serves it for local frontend work.

The supported persisted review artifact is a `citation_node_document`. Generate
one through the snapshot corpus runner:

```bash
uv run --group pipeline python -m scripts.e2e_backend.snapshot_corpus \
  --file research --phase assessment
```

The runner accepts a numeric test-data stem or a named bookmark set. It writes
only `citation_nodes.json`; intermediate typed artifacts are validated in memory
and are not snapshot contracts.
