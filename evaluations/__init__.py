"""Independent workflow scoring and shared run-persistence mechanics."""

from mellea_lrc.model.document import Document
from mellea_lrc.model.execution import STAGE_CATALOG, stage_for_substage


def complete_stage_boundary(document: Document, enabled: tuple[str, ...]) -> Document:
    """Commit a group when its last enabled substage has just completed.

    Runners can stop inside a group without claiming it is complete. Calling
    this after recovery also restores a group marker omitted by an exact
    substage rewind; it never invents markers for earlier groups.
    """
    if not document.substage_runs:
        return document
    substage = document.substage_runs[-1]
    stage = stage_for_substage(substage)
    if stage is None or stage in document.stage_runs:
        return document
    definition = STAGE_CATALOG[stage]
    members = tuple(item.name for item in definition.substages if item.name in enabled)
    if (
        members
        and substage == members[-1]
        and set(definition.required_substages).issubset(document.substage_runs)
    ):
        return document.complete_stage(stage)
    return document
