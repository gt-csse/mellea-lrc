"""Reserved semantic repair hooks for root-identity field conflicts."""

from __future__ import annotations

from typing import Literal, NoReturn


async def repair_court_or_date_semantically(
    *,
    field: Literal["court", "date"],
) -> NoReturn:
    """Reserve the correction boundary without silently changing a root.

    TODO: add a grounded semantic comparison and local-field correction for a
    stated court or date that conflicts with a retrieved authority.  The
    deterministic court and date readers are already saturated enough that
    this must not become an unmeasured heuristic inside the first identity
    evaluation.
    """
    raise NotImplementedError(
        f"Semantic {field} repair is intentionally deferred until its separate evaluation exists."
    )
