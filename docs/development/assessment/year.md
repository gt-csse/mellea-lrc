# Year Assessment

Year assessment compares the extracted citation year with CourtListener's
retrieved filing year. It is deterministic.

| Condition | Status |
|---|---|
| either value is absent | `missing` |
| values are equal | `exact_match` |
| values differ | `mismatch` |

The result preserves both values and a human-readable message. It does not
modify the extracted citation or infer a year from document context.

Implementation: `assessment/fields/year/assess.py` and
`assessment/types/year.py`.
